"""Read completed phase records before touching generation/entropy models."""
from __future__ import annotations
from .errors import require_success

from pathlib import Path

from .entropy_cache import read_entropy_records
from .prepared_index import PreparedIndex, PreparedLocation


def iter_prepared_records(path, *, _with_locations=False):
    """Accept candidate lines, request bundles and standalone flat records."""
    path = Path(path)
    for item in read_entropy_records(str(path), _with_offsets=_with_locations):
        offset, _, entry = item if _with_locations else (None, None, item)
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid prepared record in {path}")
        records = [entry["record"]] if "record" in entry else entry.get("records", [entry])
        for index, row in enumerate(records):
            if not isinstance(row, dict):
                raise ValueError(f"Invalid prepared record in {path}")
            record = dict(row)
            require_success(record)
            if "request_id" in entry:
                request_id = str(entry["request_id"])
                if "request_id" in record and str(record["request_id"]) != request_id:
                    raise ValueError(f"Prepared request_id mismatch in {path}")
                record["request_id"] = request_id
            if _with_locations:
                member = -1 if "record" in entry else index if "records" in entry else -2
                yield record, PreparedLocation(path, offset, member)
            else:
                yield record


def is_prepared_record(record):
    """Check phase structure, not scoring validity or current phase parameters.

    Missing confidences can be legitimate legacy outputs. Failed rows abort.
    """
    if not isinstance(record, dict) or "sequence_idx" not in record:
        return False
    require_success(record)
    if record.get("status") != "ok" or not isinstance(record.get("full_output"), str):
        return False
    steps, ranges = record.get("steps"), record.get("phase_ranges")
    if not isinstance(steps, list) or not isinstance(ranges, list) or record.get("num_steps") != len(steps):
        return False
    confidences = record.get("step_avg_confidences")
    if not isinstance(confidences, list) or len(confidences) != len(steps):
        return False
    if any(not isinstance(step, dict) or not {"step", "token_avg_confidence", "first_token_index", "phase_frontier"} <= step.keys()
           for step in steps):
        return False
    next_start = 1
    for pair in ranges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return False
        start, end = pair
        if not isinstance(start, int) or not isinstance(end, int) or start != next_start or not start <= end <= len(steps):
            return False
        next_start = end + 1
    return bool(ranges) if steps else not ranges


def prepared_paths(prepared_path):
    path = Path(prepared_path)
    paths = [path]
    if path.parent.name == ".staging":
        paths.append(path.parent.parent / ".bon_prepared.jsonl")
    return list(dict.fromkeys(paths))


def load_benchmark_prepared(prepared_path, requests, n, *, _indexed=False, max_tokens=None):
    by_request = {request.request_id: request for request in requests}
    by_sample = {}
    by_problem = {}
    for request in requests:
        by_sample.setdefault(str(request.sample_id), []).append(request)
        by_problem.setdefault(request.task_text, []).append(request)
    result = PreparedIndex() if _indexed else {}
    if _indexed:
        result.originals = {}
    for path in prepared_paths(prepared_path):
        for item in iter_prepared_records(path, _with_locations=_indexed):
            record, location = item if _indexed else (item, None)
            request_id = record.get("request_id")
            if request_id is None:
                candidates = by_sample.get(str(record.get("unique_id")), [])
                if not candidates:
                    candidates = by_problem.get(record.get("problem"), [])
                candidates = [r for r in candidates if record.get("problem", r.task_text) == r.task_text]
                if len(candidates) > 1:
                    raise ValueError(f"Ambiguous prepared rollout in {path}; preserve its request_id")
                if not candidates:
                    continue
                request_id = candidates[0].request_id
            request_id = str(request_id)
            if request_id not in by_request:
                continue  # e.g. other ICPC requests in the same global stage file
            request = by_request[request_id]
            if record.get("problem", request.task_text) != request.task_text:
                raise ValueError(f"Prepared rollout {request_id} in {path} does not match the current problem")
            if record.get("unique_id") is not None and str(record["unique_id"]) != str(request.sample_id):
                raise ValueError(f"Prepared rollout {request_id} in {path} does not match the current sample_id")
            try:
                sequence_idx = int(record["sequence_idx"])
            except (KeyError, ValueError, TypeError):
                continue
            if not 0 <= sequence_idx < n:
                continue
            if max_tokens is not None and record.get("num_generated_tokens", 0) > max_tokens:
                if _indexed and is_prepared_record(record):
                    result.originals[(request_id, sequence_idx)] = location
                # An imported long root row must not shadow a new short row.
                continue
            # Last record wins even when incomplete, so an interrupted/invalid
            # replacement cannot silently resurrect an older candidate.
            slots = result.setdefault(request_id, {})
            if sequence_idx in slots:
                del slots[sequence_idx]
            if is_prepared_record(record):
                slots[sequence_idx] = location if _indexed else record
    return result


def load_standalone_prepared_records(runtime, args, data, prepared_path):
    """Keep external .bon_prepared.jsonl intact; retain legacy output filenames."""
    result = {}
    if getattr(args, "overwrite", False):
        # Prevent an explicitly replaced external cache from shadowing the new
        # .prepared_tmp.jsonl on the next resumed scoring run.
        Path(prepared_path).with_name(".bon_prepared.jsonl").unlink(missing_ok=True)
        return result
    paths = [Path(prepared_path), Path(prepared_path).with_name(".bon_prepared.jsonl")]
    for path in dict.fromkeys(paths):
        for record in iter_prepared_records(path):
            if not runtime.record_matches_input(record, data, args.N):
                raise ValueError(f"Prepared record in {path} does not match current input/N")
            # Standalone outputs do not carry the service's envelope identity.
            record.pop("request_id", None)
            key = runtime.candidate_key(record)
            result.pop(key, None)
            if is_prepared_record(record):
                result[key] = record
    return result
