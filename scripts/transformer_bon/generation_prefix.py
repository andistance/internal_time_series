"""Lazy, lossless response-budget views; source generation JSONL stays intact."""
from __future__ import annotations

import threading
import json
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .entropy_cache import read_entropy_records
from .prepared_index import PreparedIndex, PreparedLocation


def stage_records_index(path):
    """Index wrapped/bundled candidate locations, never all trajectory payloads."""
    path = Path(path)
    result = PreparedIndex()
    for offset, _, entry in read_entropy_records(str(path), _with_offsets=True):
        if not isinstance(entry, dict) or "request_id" not in entry:
            raise ValueError(f"Invalid staged entry in {path}")
        rows = [entry["record"]] if "record" in entry else entry.get("records", [])
        slots = result.setdefault(str(entry["request_id"]))
        for member, row in enumerate(rows):
            index = int(entry.get("sequence_idx", row["sequence_idx"])) if "record" in entry else int(row["sequence_idx"])
            slots[index] = PreparedLocation(path, offset, -1 if "record" in entry else member)
    return result


def compact_stage(path, prepared_index):
    """Drop superseded rows after a budget change, preserving envelope layouts."""
    path = Path(path)
    obsolete = set()
    for (request, sequence), location in prepared_index.originals.items():
        slots = prepared_index.get(request, {})
        if location.path == path and sequence in slots and slots.locations[sequence] != location:
            obsolete.add((location.offset, location.member))
    if not obsolete:
        return
    fresh = {}
    temporary = path.with_name(path.name + ".rewrite_tmp")
    with path.open("rb") as source, temporary.open("wb") as target:
        while True:
            offset = source.tell()
            raw = source.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            entry = json.loads(raw)
            rows = [entry["record"]] if "record" in entry else entry.get("records", [entry])
            members = [-1] if "record" in entry else list(range(len(rows))) if "records" in entry else [-2]
            kept = [(member, row) for member, row in zip(members, rows) if (offset, member) not in obsolete]
            if not kept:
                continue
            position = target.tell()
            if len(kept) == len(rows):
                target.write(raw)
            else:
                entry["records"] = [row for _, row in kept]
                target.write(json.dumps(entry, ensure_ascii=False).encode("utf-8") + os.linesep.encode("ascii"))
            for member, (previous_member, _) in enumerate(kept):
                fresh[(offset, previous_member)] = PreparedLocation(
                    path, position, -1 if "record" in entry else member if "records" in entry else -2)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
    for request, slots in prepared_index.items():
        for sequence, location in slots.locations.items():
            if location.path == path:
                slots[sequence] = fresh[(location.offset, location.member)]


@dataclass
class PrefixReference:
    """Small, spawn-safe entropy job; each worker reads only its current row."""
    location: PreparedLocation
    length: int
    char_end: int
    structure: dict
    weight: int

    def get(self, name, default=None):
        # The entropy worker's model-loading predicate needs no payload read.
        return "ok" if name == "status" else default

    def read(self):
        return self.apply(self.location.read())

    def apply(self, row):
        row["response_token_ids"] = row["response_token_ids"][:self.length]
        row["token_logprobs"] = row["token_logprobs"][:self.length]
        row["full_output"] = row["full_output"][:self.char_end]
        row["finish_reason"] = "length"
        row["_bon_source_structure"] = self.structure
        return row


class GenerationRecords(Sequence):
    """Bounded payload cache over the existing generation offset index."""
    def __init__(self, index, keys, args, runtime, directory, originals=None):
        self.index, self.keys, self.args = index, list(keys), args
        self.runtime, self.directory = runtime, Path(directory)
        self.originals = originals or {}
        self.references, self.recent = {}, OrderedDict()
        self.tokenizer = self.cache = self.tokenizer_key = None
        self.lock = threading.RLock()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        with self.lock:
            if index in self.recent:
                self.recent.move_to_end(index)
                return self.recent[index]
            if index in self.references:
                row = self.references[index].read()
            else:
                request, sequence = self.keys[index]
                row = self.index[request][sequence]
                budget = int(getattr(self.args, "max_tokens", len(row.get("response_token_ids", []))))
                if len(row.get("response_token_ids", [])) > budget:
                    reference = self._prefix(row, self.index[request].locations[sequence], budget)
                    self.references[index] = reference
                    row = reference.apply(row)
            self.recent[index] = row
            while len(self.recent) > 2:
                self.recent.popitem(last=False)
            return row

    def reference(self, index):
        row = self[index]
        if index in self.references:
            return self.references[index]
        request, sequence = self.keys[index]
        return RowReference(self.index[request].locations[sequence],
                            len(row.get("prompt_token_ids", [])) + len(row.get("response_token_ids", [])))

    def _prefix(self, generation, location, budget):
        from .phase_structure import (StructureCache, compile_structure, open_arrays,
                                      structure_key, tokenizer_identity)
        if budget < 1:
            raise ValueError("max_tokens must be positive")
        if len(generation.get("token_logprobs", [])) != len(generation["response_token_ids"]):
            raise ValueError("Generation token/logprob length mismatch")
        if self.cache is None:
            self.cache = StructureCache(self.directory)
            self.tokenizer_key = tokenizer_identity(self.args)
        original = self.originals.get((str(generation["request_id"]), int(generation["sequence_idx"])))
        identity = structure_key(generation, {"_phase_entropy_digest": "original-steps"}, self.tokenizer_key)
        key = "source-steps:" + identity
        descriptor = self.cache.entries.get(key)
        if descriptor is None:
            if original is not None:
                original = original.read() if hasattr(original, "read") else original
            if self.tokenizer is None:
                self.tokenizer = self.runtime.load_tokenizer_without_vllm(self.args)
            source = dict(generation)
            source["_bon_step_source"] = True
            if original:
                source["_bon_saved_steps"] = original["steps"]
            tokens, steps, metadata = compile_structure(
                source, {"token_entropies": np.zeros(len(generation["response_token_ids"]))}, self.tokenizer)
            if metadata["decoded_output"] is not None:
                raise ValueError("Tokenizer does not reproduce the saved generation text")
            from .phase_structure import map_spans
            spans = np.column_stack((tokens["start"], tokens["end"]))
            tokens["fallback"] = map_spans(spans, list(zip(steps["start"], steps["end"])))
            descriptor = self.cache.save(key, tokens, steps, metadata)
        tokens, steps = open_arrays(descriptor)
        if self.tokenizer is None:
            self.tokenizer = self.runtime.load_tokenizer_without_vllm(self.args)
        # Decode only the affected step, including a token straddling its start.
        estimated_end = int(tokens["end"][budget - 1])
        step = max(0, int(np.searchsorted(steps["start"], estimated_end, side="left")) - 1)
        start_char = int(steps["start"][step]) if len(steps) else 0
        token_start = max(0, int(np.searchsorted(tokens["end"], start_char, side="right")))
        token_start = min(token_start, budget - 1)
        base = int(tokens["start"][token_start])
        tail = self.tokenizer.decode(generation["response_token_ids"][token_start:budget],
                                     skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text = generation["full_output"]
        if base > len(text) or text[base:base + len(tail)] != tail:
            raise ValueError("Response cutoff cannot preserve the saved text exactly (decoder/Unicode boundary mismatch)")
        char_end = base + len(tail)
        return PrefixReference(location, budget, char_end, descriptor,
                               len(generation.get("prompt_token_ids", [])) + budget)


@dataclass
class RowReference:
    location: PreparedLocation
    weight: int

    def get(self, name, default=None):
        return "ok" if name == "status" else default

    def read(self):
        return self.location.read()


class SelectedGenerations(Sequence):
    def __init__(self, records, indices):
        self.records, self.indices = records, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.records[self.indices[index]]

    def reference(self, index):
        return self.records.reference(self.indices[index])


def within_generation_limit(records, limit):
    """Cheap final-result guard used before benchmark resume skips."""
    return all(int(row.get("num_generated_tokens", 0)) <= limit for row in records)


def over_budget_requests(path, limit):
    """Last-row-wins scan of final summaries, retaining only small identities."""
    flags = {}
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    flags[(row.get("request_id"), row.get("sequence_idx"))] = row.get("num_generated_tokens", 0) > limit
    return {request for (request, _), too_long in flags.items() if too_long}
