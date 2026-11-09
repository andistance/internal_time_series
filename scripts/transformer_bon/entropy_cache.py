"""Phase-independent, response-token-aligned exact entropy checkpoints.

Each JSONL record contains complete prompt/response token ID arrays. At index t,
token_entropies[t] is the full-vocabulary next-token entropy (natural logs)
BEFORE response_token_ids[t], conditioned on prompt + response[:t]. No phase
parameters enter the cache identity. Records are committed one rollout at a time.
"""

from __future__ import annotations
from .errors import raise_record_error, require_success

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


def entropy_model_identity(args: Any) -> Dict[str, Any]:
    return {
        "model_path": str(getattr(args, "entropy_model_path", None) or args.model_path),
        "dtype": str(getattr(args, "dtype", "auto")),
    }


def token_fingerprint(record: Dict[str, Any]) -> str:
    value = [record.get("prompt_token_ids", []), record.get("response_token_ids", [])]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_token_entropy_record(
    generation_record: Dict[str, Any],
    entropies: Optional[Sequence[float]],
    args: Any,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    require_success(generation_record)
    if error:
        raise_record_error(generation_record, error)
    result = {
        name: generation_record[name]
        for name in (
            "request_id", "sample_idx", "sequence_idx", "problem", "gold_answer",
            "difficulty", "unique_id", "full_output", "finish_reason",
        )
        if name in generation_record
    }
    result.update(
        schema_version=1,
        status="ok",
        entropy_definition="full_vocab_next_token_before_response_token_nats",
        entropy_model=entropy_model_identity(args),
        token_fingerprint=token_fingerprint(generation_record),
        prompt_token_ids=list(generation_record.get("prompt_token_ids", [])),
        response_token_ids=list(generation_record.get("response_token_ids", [])),
        token_entropies=[float(value) for value in entropies],
    )
    if (
        len(result["token_entropies"]) != len(result["response_token_ids"])
        or any(not math.isfinite(value) for value in result["token_entropies"])
    ):
        raise ValueError("Entropy cache requires one finite entropy per response token")
    return result


def entropy_record_matches(cached: Dict[str, Any], generation: Dict[str, Any], args: Any) -> bool:
    require_success(cached)
    require_success(generation)
    if (
        cached.get("schema_version") != 1
        or cached.get("entropy_model") != entropy_model_identity(args)
        or cached.get("token_fingerprint") != token_fingerprint(generation)
        or cached.get("prompt_token_ids") != generation.get("prompt_token_ids", [])
        or cached.get("response_token_ids") != generation.get("response_token_ids", [])
    ):
        return False
    values = cached.get("token_entropies", [])
    return (
        cached.get("status") == "ok"
        and len(values) == len(generation.get("response_token_ids", []))
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
    )


def append_entropy_record(path: str, record: Dict[str, Any]) -> None:
    require_success(record)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def read_entropy_records(path: str, *, _with_offsets: bool = False, _start_offset: int = 0) -> Iterable[Dict[str, Any]]:
    """Recover only a truncated final line, rejecting earlier corruption."""
    target = Path(path)
    if not target.exists():
        return
    # Read/repair before append, so a resumed record never follows a partial line.
    with target.open("rb+") as file:
        file.seek(_start_offset)
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                return
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                if file.read().strip():
                    raise ValueError(f"Corrupt entropy JSONL before EOF: {target}")
                file.seek(offset)
                file.truncate()
                file.flush()
                os.fsync(file.fileno())
                return
            require_success(record)
            if not line.endswith(b"\n"):
                # A complete JSON value whose final newline was interrupted.
                file.seek(0, os.SEEK_END)
                file.write(b"\n")
                file.flush()
                os.fsync(file.fileno())
            yield (offset, file.tell() - offset, record) if _with_offsets else record


def repair_prepared_tail(path: str) -> None:
    """Inspect only the final append; normal stage readers validate on load.

    Re-parsing every full entropy array just to find a partial final write makes
    warm repartitioning unnecessarily proportional to the entire old output.
    """
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return
    with target.open("rb+") as file:
        size = target.stat().st_size
        start, data = size, b""
        while True:
            length = min(start, 65536)
            start -= length
            file.seek(start)
            data = file.read(length) + data
            lines = data.splitlines(keepends=True)
            nonempty = [index for index, line in enumerate(lines) if line.strip()]
            if not nonempty:
                if start == 0:
                    return
                continue
            index = nonempty[-1]
            if index == 0 and start > 0:
                continue
            offset = start + sum(map(len, lines[:index]))
            try:
                json.loads(lines[index])
            except (ValueError, UnicodeDecodeError):
                file.seek(offset)
                file.truncate()
                file.flush()
                os.fsync(file.fileno())
            else:
                file.seek(0, os.SEEK_END)
                if not data.endswith(b"\n"):
                    file.write(b"\n")
                    file.flush()
                    os.fsync(file.fileno())
            return


def run_standalone_phase_stage(
    runtime: Any,
    args: Any,
    expected: Sequence[Any],
    generation_records: Dict[Any, Dict[str, Any]],
    generation_path: str,
    prepared_path: str,
    prepared_records: Dict[Any, Dict[str, Any]],
    completed_final: Any,
    gpu_groups: Sequence[str],
) -> None:
    """Complete/save/reload all token entropies before constructing any phase."""
    from .prepared_cache import is_prepared_record
    phase_keys = [key for key in expected if key not in completed_final
                  and not is_prepared_record(prepared_records.get(key))]
    if not phase_keys:
        print("[resume] Prepared phase records complete; skip entropy/phase partition and continue scoring.")
        return
    from .entropy_index import EntropyIndex, PhaseTasks
    cache_path = str(Path(prepared_path).with_name(".token_entropies.jsonl"))
    cached = EntropyIndex(cache_path, runtime.candidate_key)
    missing_generation = [key for key in phase_keys if key not in generation_records]
    if missing_generation:
        raise RuntimeError(f"Missing generated candidates: {missing_generation[:20]}")
    pending = [
        key for key in phase_keys
        if not cached.matches(key, generation_records[key], args)
    ]
    if pending:
        # Existing DP/TP machinery now emits only phase-independent cache rows.
        if gpu_groups:
            runtime.run_data_parallel_entropy(
                args, pending, gpu_groups, generation_path, cache_path, {},
            )
        else:
            runtime.run_single_process_entropy(args, pending, generation_records, cache_path, {})
    else:
        print("[resume] Token entropy cache complete; skip entropy model.")
    # All forward workers have exited and all cache rows are on disk here.
    cached.refresh(appended=True)
    missing = [key for key in phase_keys if key not in cached]
    if missing:
        raise RuntimeError(f"Token entropy stage incomplete: {missing[:20]}")
    repair_prepared_tail(prepared_path)
    if gpu_groups:
        from .phase_parallel import run_parallel_phases

        with runtime.tqdm(total=len(phase_keys), desc="Phase partition from cached entropy", unit="rollout") as progress:
            def persist_phase(prepared: Dict[str, Any]) -> None:
                runtime.append_jsonl(prepared_path, prepared)
                prepared_records[runtime.candidate_key(prepared)] = prepared

            run_parallel_phases(
                args,
                PhaseTasks([generation_records[key] for key in phase_keys], cached, runtime.candidate_key),
                gpu_groups,
                legacy_script_path=runtime.__file__,
                on_record=persist_phase,
                on_complete=lambda: progress.update(1),
                cache_dir=str(Path(cache_path).parent / ".phase_structure_cache"),
                collect_results=False,
                validated=True,
            )
        return
    from .phase_parallel import run_serial_phases
    from .progress import DatasetProgress

    with DatasetProgress(len(phase_keys), description="Phase partition from cached entropy", unit="rollout") as progress:
        def persist_phase(prepared):
            runtime.append_jsonl(prepared_path, prepared)
            prepared_records[runtime.candidate_key(prepared)] = prepared

        run_serial_phases(args, PhaseTasks([generation_records[key] for key in phase_keys], cached, runtime.candidate_key),
                          str(Path(cache_path).parent / ".phase_structure_cache"), runtime,
                          persist_phase, progress.update, include_request_id=False)
