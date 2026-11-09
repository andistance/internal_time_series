from __future__ import annotations
from .errors import require_success

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .models import BestOfNResult
from .record_index import RecordIndex, write_records


def _existing_records(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            key = record.get("candidate_key")
            if key is not None:
                records[str(key)] = record
    return records


def _atomic_rewrite(path: Path, records: Iterable[Dict[str, Any]], *, keys=None):
    temporary = path.with_name(path.name + ".rewrite_tmp")
    references = write_records(temporary, records, keys=keys)
    os.replace(temporary, path)
    return references


class JsonlResultStore:
    """Thread-safe paired writer for all candidates from completed BoN rounds."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        resume: bool = False,
        overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "summary.jsonl"
        self.steps_path = self.output_dir / "steps.jsonl"
        self._lock = threading.Lock()

        if overwrite:
            for path in (self.summary_path, self.steps_path):
                if path.exists():
                    path.unlink()
        elif not resume and (self.summary_path.exists() or self.steps_path.exists()):
            raise FileExistsError(
                f"BoN output already exists in {self.output_dir}; use resume or overwrite"
            )

        summary_records = RecordIndex(self.summary_path)
        steps_records = RecordIndex(self.steps_path)
        completed_keys = set(summary_records).intersection(steps_records)
        if set(summary_records) != set(steps_records):
            # Recover from interruption between the paired appends by retaining
            # only complete summary/steps pairs, matching the original script's
            # resume synchronization behavior.
            ordered_keys = sorted(completed_keys)
            summary_refs = _atomic_rewrite(self.summary_path, summary_records.read_many(ordered_keys), keys=ordered_keys)
            steps_refs = _atomic_rewrite(self.steps_path, steps_records.read_many(ordered_keys), keys=ordered_keys)
            summary_records = RecordIndex(self.summary_path, scan=False)
            steps_records = RecordIndex(self.steps_path, scan=False)
            summary_records.apply(summary_refs)
            steps_records.apply(steps_refs)
        self._summary_records = summary_records
        self._steps_records = steps_records
        self._completed_keys = completed_keys

    def is_request_complete(self, request_id: str, n: int) -> bool:
        with self._lock:
            return all(
                f"{request_id}:c{sequence_idx}" in self._completed_keys
                for sequence_idx in range(n)
            )

    @staticmethod
    def _write_lines(path: Path, records: Iterable[Dict[str, Any]], keys=None):
        return write_records(path, records, append=True, keys=keys)

    def write_result(
        self,
        result: BestOfNResult,
        *,
        selected_is_correct: bool | None,
    ) -> None:
        evaluated_candidates = {
            result.selected_candidate_key: selected_is_correct,
        }
        self.write_top_k_result(
            result,
            k=1,
            evaluated_candidates=evaluated_candidates,
        )

    def write_top_k_result(
        self,
        result: BestOfNResult,
        *,
        k: int,
        evaluated_candidates: Mapping[str, bool | None] | None = None,
    ) -> None:
        """Write all N candidates while marking the ranked Top-K subset.

        ``evaluated_candidates`` may contain labels only for candidates that were
        actually sent to a benchmark evaluator. All other labels remain unknown.
        """

        ranked = result.ranked_candidates
        if not 1 <= k <= len(ranked):
            raise ValueError(f"k must satisfy 1 <= k <= {len(ranked)}, received {k}")
        evaluated = dict(evaluated_candidates or {})
        rank_by_key = {
            candidate.candidate_key: rank
            for rank, candidate in enumerate(ranked, start=1)
        }
        top_k_keys = {
            candidate.candidate_key for candidate in ranked[:k]
        }
        unknown_keys = set(evaluated).difference(top_k_keys)
        if unknown_keys:
            raise ValueError(
                "Correctness was supplied for candidates outside Top-K: "
                + ", ".join(sorted(unknown_keys))
            )

        summaries = []
        steps = []
        keys = []
        for candidate in result.candidates:
            require_success(candidate.summary_record)
            require_success(candidate.steps_record)
            summary = dict(candidate.summary_record)
            step_record = dict(candidate.steps_record)
            rank = rank_by_key[candidate.candidate_key]
            is_selected = rank == 1
            selected_for_pass_at_k = candidate.candidate_key in top_k_keys
            correctness = evaluated.get(candidate.candidate_key)
            summary["is_correct"] = correctness
            step_record["is_correct"] = correctness
            summary["selected_by_bon"] = is_selected
            step_record["selected_by_bon"] = is_selected
            summary["bon_rank"] = rank
            step_record["bon_rank"] = rank
            summary["selected_for_pass_at_k"] = selected_for_pass_at_k
            step_record["selected_for_pass_at_k"] = selected_for_pass_at_k
            position = rank - 1 if selected_for_pass_at_k else None
            summary["pass_at_k_position"] = position
            step_record["pass_at_k_position"] = position
            summaries.append(summary)
            steps.append(step_record)
            keys.append(candidate.candidate_key)

        self._persist_candidate_records(keys, summaries, steps)

    def write_all_candidates_result(
        self,
        result: BestOfNResult,
        *,
        evaluated_candidates: Mapping[str, bool | None],
    ) -> None:
        """Write labels for every generated candidate.

        Math benchmarks evaluate every rollout before simulating nested smaller
        Best-of-n sets.  This is intentionally separate from Top-K evaluation so
        ICPC/GPQA/LiveCodeBench retain their existing unknown-label semantics.
        """

        evaluated = dict(evaluated_candidates)
        candidate_keys = {candidate.candidate_key for candidate in result.candidates}
        if set(evaluated) != candidate_keys:
            missing = candidate_keys.difference(evaluated)
            extra = set(evaluated).difference(candidate_keys)
            raise ValueError(
                "All and only result candidates must be evaluated; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        rank_by_key = {
            candidate.candidate_key: rank
            for rank, candidate in enumerate(result.ranked_candidates, start=1)
        }
        summaries = []
        steps = []
        keys = []
        for candidate in result.candidates:
            require_success(candidate.summary_record)
            require_success(candidate.steps_record)
            summary = dict(candidate.summary_record)
            step_record = dict(candidate.steps_record)
            rank = rank_by_key[candidate.candidate_key]
            correctness = evaluated[candidate.candidate_key]
            for record in (summary, step_record):
                record["is_correct"] = correctness
                record["selected_by_bon"] = rank == 1
                record["bon_rank"] = rank
                record["selected_for_pass_at_k"] = False
                record["pass_at_k_position"] = None
                record["evaluation_scope"] = "all_candidates"
            summaries.append(summary)
            steps.append(step_record)
            keys.append(candidate.candidate_key)
        self._persist_candidate_records(keys, summaries, steps)

    def _persist_candidate_records(
        self,
        keys: list[str],
        summaries: list[Dict[str, Any]],
        steps: list[Dict[str, Any]],
    ) -> None:
        with self._lock:
            replaces_existing = any(key in self._completed_keys for key in keys)
            if replaces_existing:
                summary_updates = dict(zip(keys, summaries))
                step_updates = dict(zip(keys, steps))
                ordered_keys = sorted(set(self._summary_records).union(keys))
                def rows(index, updates):
                    for key in ordered_keys:
                        yield updates[key] if key in updates else index[key]
                summary_refs = _atomic_rewrite(self.summary_path, rows(self._summary_records, summary_updates), keys=ordered_keys)
                steps_refs = _atomic_rewrite(self.steps_path, rows(self._steps_records, step_updates), keys=ordered_keys)
            else:
                summary_refs = self._write_lines(self.summary_path, summaries, keys)
                steps_refs = self._write_lines(self.steps_path, steps, keys)
            self._summary_records.apply(summary_refs)
            self._steps_records.apply(steps_refs)
            self._completed_keys.update(keys)

    def update_evaluated_candidates(
        self,
        evaluated_candidates: Mapping[str, bool | None],
    ) -> None:
        """Update evaluator labels in both paired files without losing records."""

        updates = dict(evaluated_candidates)
        if not updates:
            return
        with self._lock:
            missing = set(updates).difference(self._completed_keys)
            if missing:
                raise KeyError(
                    "Cannot label missing candidate records: "
                    + ", ".join(sorted(missing))
                )
            # Validate the whole update before changing either file. Rows are
            # decoded one at a time, including during the atomic rewrite.
            for key in updates:
                if not self._summary_records.locations[key].selected:
                    raise ValueError(f"Candidate {key} was not selected for Pass@K evaluation")
            ordered_keys = sorted(self._summary_records)
            def rows(index):
                items = index.read_items(ordered_keys)
                try:
                    for key, record in items:
                        if key in updates:
                            record["is_correct"] = updates[key]
                        yield record
                finally:
                    items.close()
            summary_refs = _atomic_rewrite(self.summary_path, rows(self._summary_records), keys=ordered_keys)
            steps_refs = _atomic_rewrite(self.steps_path, rows(self._steps_records), keys=ordered_keys)
            self._summary_records.apply(summary_refs)
            self._steps_records.apply(steps_refs)

    def load_top_k_records(
        self,
        request_id: str,
        k: int,
    ) -> list[Dict[str, Any]]:
        """Load the previously written Top-K summary records for one request."""

        with self._lock:
            records = list(self._summary_records.read_many(
                self._summary_records.keys_for_request(request_id, selected_only=True)))
        records.sort(key=lambda record: int(record["bon_rank"]))
        if len(records) != k:
            raise KeyError(
                f"Expected {k} Top-K records for {request_id}, found {len(records)}"
            )
        return records

    def load_request_records(self, request_id: str) -> list[Dict[str, Any]]:
        """Load all previously written summary records for one request."""

        with self._lock:
            records = list(self._summary_records.read_many(
                self._summary_records.keys_for_request(request_id)))
        records.sort(key=lambda record: int(record["sequence_idx"]))
        return records
