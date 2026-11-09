"""Run-local JSONL indexes. Payloads stay in the existing files, not in RAM."""

from __future__ import annotations
from .errors import require_success

import json
import os
from collections.abc import Mapping
from pathlib import Path
from itertools import chain
from typing import NamedTuple


NEWLINE = os.linesep.encode("ascii")


class Location(NamedTuple):
    offset: int
    request_id: object
    selected: bool
    order: int


class RecordIndex(Mapping):
    """Offset and request-key lookup with the old last-row-wins semantics."""

    def __init__(self, path: Path, *, scan=True):
        self.path = path
        self.locations = {}
        self.by_request = {}
        if scan and path.exists():
            with path.open("rb") as file:
                line_number = 0
                while True:
                    offset = file.tell()
                    raw = file.readline()
                    if not raw:
                        break
                    line_number += 1
                    if not raw.strip():
                        continue
                    try:
                        record = json.loads(raw.decode("utf-8"))
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
                    require_success(record)
                    key = record.get("candidate_key")
                    if key is not None:
                        self.remember(str(key), offset, record.get("request_id"),
                                      bool(record.get("selected_for_pass_at_k")))

    def remember(self, key, offset, request_id, selected):
        previous = self.locations.get(key)
        order = previous.order if previous is not None else len(self.locations)
        if previous is not None:
            self.by_request[previous.request_id].discard(key)
        self.locations[key] = Location(offset, request_id, selected, order)
        self.by_request.setdefault(request_id, set()).add(key)

    def apply(self, references):
        for key, offset, request_id, selected in references:
            self.remember(key, offset, request_id, selected)

    def keys_for_request(self, request_id, *, selected_only=False):
        keys = self.by_request.get(request_id, ())
        return sorted((key for key in keys if not selected_only or self.locations[key].selected),
                      key=lambda key: self.locations[key].order)

    def read_many(self, keys):
        items = self.read_items(keys)
        try:
            for _, record in items:
                yield record
        finally:
            items.close()

    def read_items(self, keys):
        keys = iter(keys)
        missing = object()
        first = next(keys, missing)
        if first is missing:
            return
        with self.path.open("rb") as file:
            for key in chain((first,), keys):
                file.seek(self.locations[key].offset)
                yield key, json.loads(file.readline())

    def __getitem__(self, key):
        offset = self.locations[key].offset
        with self.path.open("rb") as file:
            file.seek(offset)
            return json.loads(file.readline())

    def __contains__(self, key):
        return key in self.locations

    def __len__(self):
        return len(self.locations)

    def __iter__(self):
        return iter(self.locations)


def write_records(path, records, *, append=False, keys=None):
    """Encode each row once, retaining the original JSON and native newlines."""
    references = []
    keys = iter(keys) if keys is not None else None
    records = iter(records)
    try:
        with path.open("ab" if append else "wb") as file:
            for record in records:
                require_success(record)
                offset = file.tell()
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + NEWLINE)
                key = next(keys) if keys is not None else record.get("candidate_key")
                if keys is not None or key is not None:
                    references.append((key if keys is not None else str(key), offset, record.get("request_id"),
                                       bool(record.get("selected_for_pass_at_k"))))
            file.flush()
            os.fsync(file.fileno())
    finally:
        if hasattr(records, "close"):
            records.close()
    return references
