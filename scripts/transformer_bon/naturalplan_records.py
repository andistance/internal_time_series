"""Append-only NaturalPlan results with one final legacy-compatible sort.

Indexes retain byte offsets, never historical trajectory payloads. Interrupted
appends are repaired on reopening; duplicate keys keep the latest valid row,
as in the original NaturalPlan reader. No additional persistent files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .record_index import NEWLINE
from .json_codec import encode_json, decode_json, iter_encoded


def _key(record):
    return (str(record.get("sample_id", record.get("unique_id", ""))), int(record["sequence_idx"]))


def completed_request_ids(output_dir, result_suffix, rows, n, max_tokens, benchmark):
    """Resume needs keys and token counts, not historical step/text payloads."""
    def headers(path, token_counts=False):
        result = {}
        if path.exists():
            with path.open("rb") as file:
                for raw in file:
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if file.read().strip():
                            raise
                        break
                    result[_key(row)] = row.get("num_generated_tokens", 0) if token_counts else None
        return result
    directory = Path(output_dir)
    summary = headers(directory / f"summary{result_suffix}.jsonl", True)
    steps = headers(directory / f"steps{result_suffix}.jsonl")
    completed = set()
    for row in rows:
        sample_id = str(row["sample_id"])
        keys = [(sample_id, index) for index in range(n)]
        if (all(key in summary and key in steps for key in keys)
                and not any(summary[key] > max_tokens for key in keys)):
            completed.add(f"naturalplan:{benchmark}:{sample_id}")
    return completed


class _ResultFile:
    def __init__(self, path):
        self.path = path
        self.locations = {}
        self.canonical_offsets = set()
        self.dirty = False
        previous = None
        if path.exists():
            with path.open("rb+") as file:
                while True:
                    offset = file.tell()
                    raw = file.readline()
                    if not raw:
                        break
                    if not raw.strip():
                        self.dirty = True
                        continue
                    try:
                        row = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        if file.read().strip():
                            raise
                        file.seek(offset)
                        file.truncate()
                        self.dirty = True
                        break
                    key = _key(row)
                    self.dirty |= key in self.locations or (previous is not None and key < previous)
                    self.locations[key] = offset
                    previous = key
                    if not raw.endswith(b"\n"):
                        file.write(NEWLINE)
                        self.dirty = True
        self.last_key = previous
        self.file = path.open("ab")

    def append(self, records):
        encoded = iter_encoded(records)
        try:
            for row, payload in encoded:
                key = _key(row)
                offset = self.file.tell()
                self.file.write(payload + NEWLINE)
                self.dirty = True
                self.canonical_offsets.discard(self.locations.get(key))
                self.locations[key] = offset
                self.canonical_offsets.add(offset)
                self.last_key = key
        finally:
            encoded.close()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.file.closed:
            return
        self.file.close()
        if self.dirty:
            temporary = self.path.with_name(self.path.name + ".tmp")
            try:
                with self.path.open("rb") as source, temporary.open("wb") as target:
                    for key in sorted(self.locations):
                        source.seek(self.locations[key])
                        raw = source.readline()
                        if self.locations[key] in self.canonical_offsets:
                            # Appended rows already have exactly the canonical
                            # encoding; sorting need not decode/encode them again.
                            target.write(raw)
                        else:
                            target.write(encode_json(decode_json(raw)) + NEWLINE)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary.exists():
                    temporary.unlink()


class NaturalPlanResultWriter:
    def __init__(self, output_dir, result_suffix=""):
        directory = Path(output_dir)
        self.summary = _ResultFile(directory / f"summary{result_suffix}.jsonl")
        try:
            self.steps = _ResultFile(directory / f"steps{result_suffix}.jsonl")
        except BaseException:
            self.summary.close()
            raise

    def write(self, result, rows_by_id, correct):
        sample_id = str(result.request.sample_id)
        summaries, steps = [], []
        for candidate in result.candidates:
            summary = dict(candidate.summary_record)
            step = dict(candidate.steps_record)
            is_correct = bool(correct(rows_by_id[sample_id], candidate.text))
            for record in (summary, step):
                record["sample_id"] = sample_id
                record["sequence_idx"] = int(candidate.sequence_idx)
                record["is_correct"] = is_correct
                record["request_id"] = result.request.request_id
            summaries.append(summary)
            steps.append(step)
        self.summary.append(summaries)
        self.steps.append(steps)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.summary.close()
        finally:
            self.steps.close()
