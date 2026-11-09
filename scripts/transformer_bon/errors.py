"""Fail before an errored candidate can enter a checkpoint or result file."""


def raise_record_error(record, error=None):
    identity = {key: record[key] for key in ("request_id", "sample_idx", "sequence_idx") if key in record}
    raise RuntimeError(f"Best-of-N record failed {identity}: {error or record.get('error') or 'upstream stage failed'}")


def require_success(record):
    if record.get("status") == "error" or record.get("error"):
        raise_record_error(record)
    for key in ("record", "candidate", "summary_record", "steps_record"):
        child = record.get(key)
        if isinstance(child, dict):
            require_success(child)
    for child in record.get("records", ()):
        if isinstance(child, dict):
            require_success(child)
