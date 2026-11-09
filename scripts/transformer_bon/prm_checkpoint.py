"""A run-local offset index over the unchanged, durable PRM JSONL format."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from .prms.registry import DEFAULT_PRM_NAME, normalize_prm_name
from .record_index import NEWLINE
from .errors import require_success


class PRMCheckpoint(Mapping):
    """Keep offsets, rather than a second copy of every trajectory, in RAM.

    Index construction is one sequential scan per scoring run. Existing rows
    are decoded lazily for the current request. Appends retain per-row fsync,
    serialization and order; the GPU pipeline overlaps this work with inference.
    No index files or duplicated candidate payloads are written to disk.
    """

    def __init__(self, path: Path, prm_name: str) -> None:
        self.path = path
        self.prm_name = prm_name
        self.offsets: Dict[Tuple[str, int], int] = {}
        self.lengths = {}
        self._writer = None
        if path.exists():
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    try:
                        item = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    key = self._key(item)
                    if key is not None:
                        self.offsets[key] = offset
                        self.lengths[key] = item["candidate"]["summary_record"].get("num_generated_tokens", 0)

    def _key(self, item: Any):
        if not isinstance(item, dict) or not item.get("complete"):
            return None
        try:
            if normalize_prm_name(item.get("prm_name", DEFAULT_PRM_NAME)) != self.prm_name:
                return None
        except ValueError:
            return None
        request_id, sequence_idx = item.get("request_id"), item.get("sequence_idx")
        candidate = item.get("candidate")
        if isinstance(candidate, dict):
            require_success(candidate)
        if request_id is None or sequence_idx is None or not isinstance(candidate, dict):
            return None
        if not {"sequence_idx", "candidate_key", "score", "summary_record", "steps_record"}.issubset(candidate):
            return None
        try:
            if candidate["score"] is not None and not math.isfinite(float(candidate["score"])):
                return None
            return str(request_id), int(sequence_idx)
        except (TypeError, ValueError, OverflowError):
            return None

    def __getitem__(self, key):
        offset = self.offsets[key]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())["candidate"]

    def __contains__(self, key) -> bool:
        return key in self.offsets

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        return iter(self.offsets)

    def __len__(self) -> int:
        return len(self.offsets)

    def append(self, payload: Dict[str, Any]) -> None:
        self.append_encoded(self.encode(payload), (str(payload["request_id"]), int(payload["sequence_idx"])))

    @staticmethod
    def encode(payload):
        require_success(payload)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") + NEWLINE

    def append_encoded(self, payload: bytes, key) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Repair only an interrupted last append before reusing the file.
            # All complete rows and their byte offsets remain unchanged.
            from .entropy_cache import repair_prepared_tail
            repair_prepared_tail(str(self.path))
            # Match the old text writer's native newline convention exactly.
            self._writer = self.path.open("ab")
        handle = self._writer
        offset = handle.tell()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        self.offsets[key] = offset

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
