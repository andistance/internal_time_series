"""Small validated offset index over the unchanged rollout entropy JSONL."""
from __future__ import annotations
from .errors import require_success

import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .entropy_cache import entropy_model_identity, read_entropy_records, token_fingerprint
from .progress import DatasetProgress


class EntropyIndex(Mapping):
    def __init__(self, path, key_function):
        self.path = Path(path)
        self.key_function = key_function
        directory = self.path.parent / ".phase_structure_cache"
        directory.mkdir(parents=True, exist_ok=True)
        self.index_path = directory / (self.path.name + ".index.json")
        self.entries, self.end, self.snapshot = {}, 0, None
        self.prefixes = {}
        if self.index_path.exists():
            try:
                saved = json.loads(self.index_path.read_text(encoding="utf-8"))
                if saved.get("version") == 1 and saved.get("snapshot") == self._stat():
                    self.entries = {self._key(row["metadata"]): row for row in saved["entries"]}
                    for entry in self.entries.values():
                        require_success(entry["metadata"])
                    self.end, self.snapshot = saved["end"], saved["snapshot"]
                    return
            except (ValueError, KeyError, OSError):
                pass
        self.refresh()

    def _key(self, record):
        return self.key_function(record)

    def _stat(self):
        if not self.path.exists():
            return None
        stat = self.path.stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev]

    def refresh(self, *, appended=False):
        before = self._stat()
        if before is None:
            self.entries, self.snapshot, self.end = {}, None, 0
            return
        if before == self.snapshot:
            return
        if not appended:
            self.entries, self.end = {}, 0
        with DatasetProgress(max(0, before[0] - self.end), description="Index token entropy cache", unit="byte") as progress:
            for offset, length, row in read_entropy_records(str(self.path), _with_offsets=True, _start_offset=self.end):
                require_success(row)
                metadata = {name: row[name] for name in (
                    "request_id", "sample_idx", "sequence_idx", "schema_version", "status",
                    "entropy_model", "token_fingerprint", "error",
                ) if name in row}
                values = row.get("token_entropies", [])
                metadata["_aligned"] = (
                    row.get("token_fingerprint") == token_fingerprint(row)
                    and len(values) == len(row.get("response_token_ids", []))
                    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
                )
                try:
                    metadata["_phase_entropy_digest"] = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
                except (TypeError, ValueError, OverflowError):
                    # An invalid success row is retried by the forward stage,
                    # just as entropy_record_matches did before indexing.
                    metadata["_aligned"] = False
                    metadata["_phase_entropy_digest"] = ""
                self.entries[self._key(row)] = {"offset": offset, "length": length, "metadata": metadata}
                self.end = offset + length
                progress.update(length)
        self.snapshot = self._stat()
        temporary = self.index_path.with_name(self.index_path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(dict(version=1, snapshot=self.snapshot, end=self.end, entries=list(self.entries.values())), file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.index_path)

    def metadata(self, key):
        return self.prefixes.get(key, self.entries[key]["metadata"])

    def matches(self, key, generation, args):
        require_success(generation)
        if key not in self.entries:
            return False
        row = self.entries[key]["metadata"]
        truncated = "_bon_source_structure" in generation
        fingerprint = token_fingerprint(generation)
        if truncated or (row.get("token_fingerprint") != fingerprint
                         and row.get("entropy_model") == entropy_model_identity(args)
                         and row.get("_aligned", False)):
            # An existing long entropy row is a reusable prefix, not a reason
            # to run the entropy model again. Validate its original identity.
            if (row.get("schema_version") != 1 or row.get("entropy_model") != entropy_model_identity(args)
                    or row.get("status") != "ok" or not row.get("_aligned", False)):
                raise ValueError(f"Invalid entropy cache for truncated rollout {key}; refusing an extra forward")
            cached = self._read(key)
            size = len(generation["response_token_ids"])
            if (cached.get("prompt_token_ids") != generation.get("prompt_token_ids", [])
                    or len(cached.get("response_token_ids", [])) < size
                    or cached["response_token_ids"][:size] != generation["response_token_ids"]):
                if truncated:
                    raise ValueError(f"Entropy prefix mismatch for {key}; refusing an extra forward")
                return False
            metadata = dict(row, token_fingerprint=fingerprint, _prefix_length=size,
                            _prefix_chars=len(generation["full_output"]), _prefix_finish=generation["finish_reason"])
            metadata["_phase_entropy_digest"] = hashlib.sha256(
                np.asarray(cached["token_entropies"][:size], dtype="<f8").tobytes()).hexdigest()
            self.prefixes[key] = metadata
            return True
        if (row.get("schema_version") != 1 or row.get("entropy_model") != entropy_model_identity(args)
                or row.get("token_fingerprint") != fingerprint):
            return False
        require_success(row)
        return row.get("status") == "ok" and row.get("_aligned", False)

    def __getitem__(self, key):
        row = self._read(key)
        prefix = self.prefixes.get(key)
        if prefix:
            size = prefix["_prefix_length"]
            row["response_token_ids"] = row["response_token_ids"][:size]
            row["token_entropies"] = row["token_entropies"][:size]
            row["token_fingerprint"] = prefix["token_fingerprint"]
            row["full_output"] = row.get("full_output", "")[:prefix["_prefix_chars"]]
            row["finish_reason"] = prefix["_prefix_finish"]
        row["_phase_entropy_digest"] = self.metadata(key)["_phase_entropy_digest"]
        return row

    def _read(self, key):
        entry = self.entries[key]
        with self.path.open("rb") as file:
            file.seek(entry["offset"])
            row = json.loads(file.read(entry["length"]))
        return row

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)


class PhaseTasks(Sequence):
    """Keep only generation references and tiny entropy metadata on warm runs."""
    def __init__(self, generations, entropies, key_function):
        self.generations, self.entropies, self.key_function = generations, entropies, key_function

    def __len__(self):
        return len(self.generations)

    def metadata(self, index):
        generation = self.generations[index]
        return generation, self.entropies.metadata(self.key_function(generation))

    def __getitem__(self, index):
        generation = self.generations[index]
        return generation, self.entropies[self.key_function(generation)]
