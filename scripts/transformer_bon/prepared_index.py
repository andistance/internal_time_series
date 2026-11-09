"""Lazy prepared slots, including wrapped, bundled and legacy flat JSONL rows."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import NamedTuple


class PreparedLocation(NamedTuple):
    path: object
    offset: int
    member: int  # -2: flat record; -1: candidate wrapper; >=0: bundle member

    def read(self):
        with self.path.open("rb") as file:
            file.seek(self.offset)
            entry = json.loads(file.readline())
        return self.decode(entry)

    def decode(self, entry):
        row = (entry if self.member == -2 else entry["record"] if self.member == -1
               else entry["records"][self.member])
        result = dict(row)
        if "request_id" in entry:
            result["request_id"] = str(entry["request_id"])
        return result


class PreparedSlots(MutableMapping):
    def __init__(self):
        self.locations = {}

    def __getitem__(self, key):
        return self.locations[key].read()

    def __setitem__(self, key, value):
        if not isinstance(value, PreparedLocation):
            raise TypeError("Prepared slots must store file locations, not trajectories")
        self.locations[key] = value

    def __delitem__(self, key):
        del self.locations[key]

    def __contains__(self, key):
        return key in self.locations

    def __iter__(self):
        return iter(self.locations)

    def __len__(self):
        return len(self.locations)

    def read_all(self, keys):
        files, result = {}, []
        previous, entry = None, None
        try:
            for key in keys:
                location = self.locations[key]
                identity = (location.path, location.offset)
                if identity != previous:
                    if location.path not in files:
                        files[location.path] = location.path.open("rb")
                    file = files[location.path]
                    file.seek(location.offset)
                    entry = json.loads(file.readline())
                    previous = identity
                result.append(location.decode(entry))
            return result
        finally:
            for file in files.values():
                file.close()


class PreparedIndex(dict):
    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = PreparedSlots()
        return self[key]

    def remember(self, record, path, offset):
        self.setdefault(str(record["request_id"]))[int(record["sequence_idx"])] = PreparedLocation(path, offset, -1)
