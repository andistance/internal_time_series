"""Parameter-independent trajectory compilation and mmap-backed phase queries.

Only private caches use this format. Prepared JSONL records retain the legacy
schema, ordering, floating point reductions and boundary tie-breaking rules.
"""
from __future__ import annotations
from .errors import raise_record_error

import bisect
import codecs
import hashlib
import json
import math
import os
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .phases import adaptive_phase_count

VERSION = 1
TOKEN_DTYPE = np.dtype([
    ("entropy", "<f8"), ("start", "<i4"), ("end", "<i4"),
    ("phase_step", "<i4"), ("valid_position", "<i4"), ("fallback", "<i4"),
])
STEP_DTYPE = np.dtype([
    ("start", "<i4"), ("end", "<i4"), ("confidence", "<f8"),
    ("has_confidence", "u1"), ("first_token", "<i4"),
    ("representative", "<i4"), ("rank", "<i4"),
    ("available", "<i4"), ("fallback", "<i4"),
])


def tokenizer_identity(args: Any) -> str:
    """Detect local tokenizer changes without importing Transformers on a hit."""
    path = Path(args.model_path).expanduser()
    if not path.is_dir():
        hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get(
            "HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"))
        repo = hub / ("models--" + str(args.model_path).replace("/", "--"))
        ref = repo / "refs" / "main"
        if ref.is_file():
            path = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(str((VERSION, args.model_path, getattr(
        args, "trust_remote_code", False))).encode())
    for package in ("tokenizers", "transformers"):
        try:
            digest.update((package + ":" + version(package)).encode())
        except PackageNotFoundError:
            pass
    if path.is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "added_tokens.json", "tokenizer.model", "vocab.json", "merges.txt"):
            source = path / name
            if source.is_file():
                digest.update(name.encode())
                with source.open("rb") as file:
                    for block in iter(lambda: file.read(1024 * 1024), b""):
                        digest.update(block)
        for source in sorted(path.glob("tokenization*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


def structure_key(generation: dict, entropy: dict, tokenizer_key: str) -> str:
    digest = hashlib.sha256(tokenizer_key.encode())
    # Include entropy values and logprobs: token IDs alone do not identify the
    # invariant confidence statistics or the boundary rankings.
    digest.update(str(entropy.get("token_fingerprint", "")).encode())
    digest.update(str(generation.get("full_output")).encode("utf-8"))
    digest.update(np.asarray(generation["response_token_ids"], dtype="<i8").tobytes())
    digest.update(json.dumps(generation["token_logprobs"], separators=(",", ":")).encode())
    entropy_digest = entropy.get("_phase_entropy_digest")
    if entropy_digest is None:
        entropy_digest = hashlib.sha256(np.asarray(entropy["token_entropies"], dtype="<f8").tobytes()).hexdigest()
    digest.update(entropy_digest.encode())
    return digest.hexdigest()


class StructureCache:
    """Append-only worker shards; commit binary data before each rollout index.

    Different workers append different files. An interrupted trailing index line
    is ignored; orphaned binary tails are never referenced or reused.
    """
    def __init__(self, directory: str | Path):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "manifest.json"
        expected = {"version": VERSION, "token_dtype": TOKEN_DTYPE.descr,
                    "step_dtype": STEP_DTYPE.descr, "byte_order": "little"}
        if not manifest.exists():
            temporary = self.root / ("manifest-" + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(expected), encoding="utf-8")
            os.replace(temporary, manifest)
        elif json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(expected)):
            raise ValueError(f"Unsupported phase structure cache format: {manifest}")
        self.entries = {}
        sizes = {}
        for path in self.root.glob("rollouts-*.jsonl"):
            with path.open("rb") as file:
                while line := file.readline():
                    try:
                        record = json.loads(line)
                    except ValueError:
                        if file.read().strip():
                            raise ValueError(f"Corrupt phase structure index: {path}")
                        break
                    if self.valid(record, sizes):
                        self.entries[record["key"]] = record
        self.shard = uuid.uuid4().hex

    @staticmethod
    def valid(record: dict, sizes=None) -> bool:
        sizes = {} if sizes is None else sizes
        for name, dtype, count in (("tokens", TOKEN_DTYPE, record["num_tokens"]),
                                   ("steps", STEP_DTYPE, record["num_steps"])):
            path = Path(record[name + "_path"])
            if path not in sizes:
                sizes[path] = path.stat().st_size if path.is_file() else -1
            if sizes[path] < record[name + "_offset"] + count * dtype.itemsize:
                return False
        return True

    def save(self, key: str, tokens: Any, steps: Any, metadata: dict) -> dict:
        record = dict(metadata, key=key)
        for name, array in (("tokens", tokens), ("steps", steps)):
            path = self.root / (name + "-" + self.shard + ".bin")
            with path.open("ab") as file:
                record[name + "_offset"] = file.tell()
                file.write(array.tobytes())
                file.flush()
                os.fsync(file.fileno())
            record[name + "_path"] = str(path)
        with (self.root / ("rollouts-" + self.shard + ".jsonl")).open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        self.entries[key] = record
        return record


def open_arrays(record: dict):
    def load(name, dtype, count):
        if not count:
            return np.empty(0, dtype=dtype)
        return np.memmap(record[name + "_path"], dtype=dtype, mode="r",
                         offset=record[name + "_offset"], shape=(count,))
    return load("tokens", TOKEN_DTYPE, record["num_tokens"]), load("steps", STEP_DTYPE, record["num_steps"])


def _bytelevel_spans(ids, tokenizer):
    """Exact prefix-length differences for the standard ByteLevel decoder.

    Incomplete UTF-8 prefixes are decoded with replacement at *each token*, as
    in the legacy repeated-prefix decoder. Independent single-token decoding
    would produce different spans. Other decoders use the compatibility path.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or backend.decoder is None:
        return None
    state = json.loads(backend.decoder.__getstate__())
    if state.get("type") != "ByteLevel":
        return None
    cache = getattr(tokenizer, "_bon_bytelevel_state", None)
    if cache is None:
        # Spell out the GPT-2 byte alphabet without depending on a model class.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        mapping = {chr(value): value for value in visible}
        extra = 0
        for value in range(256):
            if value not in visible:
                mapping[chr(256 + extra)] = value
                extra += 1
        special = {int(index) for index, token in backend.get_added_tokens_decoder().items() if token.special}
        cache = (mapping, special, {})
        tokenizer._bon_bytelevel_state = cache
    mapping, special, encoded = cache
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    spans = np.empty((len(ids), 2), dtype=np.int64)
    emitted = previous = cursor = 0
    for index, token in enumerate(ids):
        token = int(token)
        if token not in special:
            if token not in encoded:
                value = backend.id_to_token(token)
                if value is None:
                    return None
                encoded[token] = b"".join(bytes((mapping[char],)) if char in mapping else char.encode("utf-8")
                                          for char in value)
            emitted += len(decoder.decode(encoded[token], final=False))
        length = emitted + len(decoder.getstate()[0].decode("utf-8", errors="replace"))
        spans[index] = (cursor, cursor + max(0, length - previous))
        cursor = int(spans[index, 1])
        previous = length
    text = tokenizer.decode(list(map(int, ids)), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if previous != len(text):
        return None
    return text, spans


def decode_spans(ids, tokenizer):
    result = _bytelevel_spans(ids, tokenizer)
    if result is None:
        result = _bytefallback_spans(ids, tokenizer)
    if result is not None:
        return result
    spans = np.empty((len(ids), 2), dtype=np.int64)
    previous = ""
    cursor = 0
    for index in range(len(ids)):
        text = tokenizer.decode(list(map(int, ids[:index + 1])), skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        end = cursor + len(text[len(previous):])
        spans[index] = cursor, end
        cursor, previous = end, text
    return previous, spans


def _bytefallback_spans(ids, tokenizer):
    """Linear prefix lengths for SentencePiece's Replace/ByteFallback/Fuse/Strip.

    ByteFallback replaces *every byte* of an invalid/incomplete contiguous byte
    run. This differs from ByteLevel's lossy UTF-8 replacement and must not use
    the same decoder. Once a run becomes valid, its apparent prefix can shrink;
    retain the legacy nonnegative prefix-length differences in that case too.
    """
    backend = getattr(tokenizer, "backend_tokenizer", getattr(tokenizer, "_tokenizer", None))
    if backend is None or backend.decoder is None:
        return None
    state = json.loads(backend.decoder.__getstate__())
    sequence = list(state.get("decoders", [])) if state.get("type") == "Sequence" else [state]
    replacements, strip = [], False
    while sequence and sequence[0].get("type") == "Replace":
        item = sequence.pop(0)
        pattern = item.get("pattern", {}).get("String")
        if not pattern:
            return None
        replacements.append((pattern, item["content"]))
    if not sequence or sequence.pop(0).get("type") != "ByteFallback":
        return None
    fused = bool(sequence and sequence[0].get("type") == "Fuse")
    if fused:
        sequence.pop(0)
    if fused and sequence and sequence[0] == {"type": "Strip", "content": " ", "start": 1, "stop": 0}:
        sequence.pop(0)
        strip = True
    if sequence:
        return None
    special = {int(index) for index, token in backend.get_added_tokens_decoder().items() if token.special}
    encoded = {}
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    committed = run_size = run_chars = 0
    previous = cursor = 0
    first = run_first = ""
    invalid = False
    spans = np.empty((len(ids), 2), dtype=np.int64)
    for index, token in enumerate(ids):
        token = int(token)
        if token not in special:
            if token not in encoded:
                value = backend.id_to_token(token)
                if value is None:
                    return None
                for pattern, content in replacements:
                    value = value.replace(pattern, content)
                byte = None
                if len(value) == 6 and value.startswith("<0x") and value.endswith(">"):
                    try:
                        byte = int(value[3:5], 16)
                        if not 0 <= byte <= 255:
                            byte = None
                    except ValueError:
                        pass
                encoded[token] = value, byte
            value, byte = encoded[token]
            if byte is not None:
                run_size += 1
                if not invalid:
                    try:
                        chunk = decoder.decode(bytes((byte,)), final=False)
                        run_chars += len(chunk)
                        if not run_first and chunk:
                            run_first = chunk[0]
                    except UnicodeDecodeError:
                        invalid = True
            else:
                valid_run = not invalid and not decoder.getstate()[0]
                committed += run_chars if valid_run else run_size
                if not first and run_size:
                    first = run_first if valid_run else "\ufffd"
                committed += len(value)
                if not first and value:
                    first = value[0]
                decoder.reset()
                run_size = run_chars = 0
                run_first, invalid = "", False
        valid_run = not invalid and not decoder.getstate()[0]
        length = committed + (run_chars if valid_run else run_size)
        current_first = first or (run_first if valid_run else "\ufffd")
        if strip and current_first == " ":
            length -= 1
        spans[index] = cursor, cursor + max(0, length - previous)
        cursor, previous = int(spans[index, 1]), length
    text = tokenizer.decode(list(map(int, ids)), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return (text, spans) if previous == len(text) else None


def split_steps(text):
    steps, cursor = [], 0
    for part in text.split("\n\n"):
        start = cursor + len(part) - len(part.lstrip())
        end = cursor + len(part.rstrip())
        if text[start:end]:
            steps.append((start, end))
        cursor += len(part) + 2
    return steps


def map_spans(spans, steps, fallback=False):
    """Search ordered disjoint intervals, visiting only actual overlaps."""
    starts, ends = [s[0] for s in steps], [s[1] for s in steps]
    result = np.full(len(spans), -1, dtype=np.int32)
    for token, (left, right) in enumerate(spans):
        best, overlap = -1, 0
        index = bisect.bisect_right(ends, left)
        while index < len(steps) and starts[index] < right:
            current = min(right, ends[index]) - max(left, starts[index])
            if current > overlap:
                best, overlap = index, current
            index += 1
        if best < 0 and fallback:
            index = bisect.bisect_left(starts, left)
            best = index if index < len(steps) else -1
        result[token] = best
    return result


def compile_structure(generation, entropy, tokenizer):
    if "_bon_source_structure" in generation:
        return compile_prefix_structure(generation, entropy)
    ids = generation["response_token_ids"]
    text, spans = decode_spans(ids, tokenizer)
    if generation.get("_bon_step_source"):
        if text != generation.get("full_output"):
            raise ValueError("Tokenizer does not reproduce the saved generation text")
        text = generation["full_output"]
    if "_bon_saved_steps" in generation:
        if text != generation.get("full_output"):
            raise ValueError("Tokenizer does not reproduce the saved generation text")
        ranges, cursor = [], 0
        for step in generation["_bon_saved_steps"]:
            start = text.find(step["step"], cursor)
            if start < 0:
                raise ValueError("Saved step does not match generation text")
            cursor = start + len(step["step"])
            ranges.append((start, cursor))
    else:
        ranges = split_steps(text)
    if len(text) > np.iinfo(np.int32).max or len(ids) > np.iinfo(np.int32).max:
        raise ValueError("Trajectory exceeds the phase cache int32 index range")
    count, reasoning_tokens = len(ranges), len(ids) if ranges else 0
    for index, (start, end) in enumerate(ranges):
        if "</think>" in text[start:end]:
            count = index + 1
            before = np.flatnonzero(spans[:, 0] < end)
            reasoning_tokens = int(before[-1]) + 1 if len(before) else 0
            break
    tokens = np.zeros(len(ids), dtype=TOKEN_DTYPE)
    tokens["entropy"] = entropy["token_entropies"]
    tokens["start"], tokens["end"] = spans[:, 0], spans[:, 1]
    tokens["phase_step"] = map_spans(spans, ranges[:count], fallback=True)
    steps = np.zeros(len(ranges), dtype=STEP_DTYPE)
    steps["first_token"] = steps["representative"] = -1
    steps["rank"] = np.iinfo(np.int32).max
    if ranges:
        steps["start"], steps["end"] = np.asarray(ranges).T
    full_mapping = map_spans(spans, ranges)
    sums, counts = [0.0] * len(ranges), [0] * len(ranges)
    for index, logprob in enumerate(generation["token_logprobs"][:len(ids)]
                                   if "_bon_saved_steps" not in generation else []):
        step = int(full_mapping[index])
        if step < 0 or spans[index, 1] <= spans[index, 0]:
            continue
        if steps["first_token"][step] < 0:
            steps["first_token"][step] = index
        if logprob is not None:
            sums[step] += math.exp(logprob)
            counts[step] += 1
    for index, size in enumerate(counts):
        if size:
            steps["confidence"][index] = float(sums[index] / size)
            steps["has_confidence"][index] = True
    for index, saved in enumerate(generation.get("_bon_saved_steps", [])):
        confidence = saved["token_avg_confidence"]
        steps["confidence"][index] = confidence if confidence is not None else 0
        steps["has_confidence"][index] = confidence is not None
        first = saved["first_token_index"]
        steps["first_token"][index] = first if first is not None else -1
    return finish_structure(generation, tokens, steps, text, count, reasoning_tokens)


def compile_prefix_structure(generation, entropy):
    """Crop original step facts; only the cut step's average is recomputed."""
    source = generation["_bon_source_structure"]
    old_tokens, old_steps = open_arrays(source)
    size, text = len(generation["response_token_ids"]), generation["full_output"]
    tokens = old_tokens[:size].copy()
    tokens["entropy"] = entropy["token_entropies"]
    count = int(np.searchsorted(old_steps["start"], len(text), side="left"))
    steps = old_steps[:count].copy()
    if count and steps["end"][-1] > len(text):
        end = len(text.rstrip())
        steps["end"][-1] = end
        if end <= steps["start"][-1]:
            steps = steps[:-1].copy()
        else:
            # Original token ownership is retained, including overlap ties.
            last = len(steps) - 1
            start = max(0, int(old_steps["first_token"][last]))
            positions = start + np.flatnonzero((old_tokens["fallback"][start:size] == last)
                & (old_tokens["end"][start:size] > old_tokens["start"][start:size]))
            total, valid_count = 0.0, 0
            for position in positions:
                value = generation["token_logprobs"][int(position)]
                if value is not None:
                    total += math.exp(value)
                    valid_count += 1
            steps["confidence"][-1] = float(total / valid_count) if valid_count else 0
            steps["has_confidence"][-1] = bool(valid_count)
            steps["first_token"][-1] = int(positions[0]) if len(positions) else -1
    phase_steps = min(len(steps), source["phase_num_steps"])
    phase_tokens = min(size, source["phase_num_tokens"]) if phase_steps else 0
    tokens["valid_position"] = tokens["fallback"] = 0
    steps["representative"] = -1
    steps["rank"] = np.iinfo(np.int32).max
    steps["available"] = steps["fallback"] = 0
    return finish_structure(generation, tokens, steps, text, phase_steps, phase_tokens)


def finish_structure(generation, tokens, steps, text, count, reasoning_tokens):
    if generation.get("_bon_step_source") or generation.get("_bon_step_scoring"):
        return tokens, steps, dict(num_tokens=len(tokens), num_steps=len(steps), phase_num_steps=count,
            phase_num_tokens=reasoning_tokens, num_valid=0, num_available=0,
            decoded_output=None if text == generation.get("full_output") else text)
    valid = np.arange(1, reasoning_tokens, dtype=np.int32)
    valid = valid[np.isfinite(tokens["entropy"][valid])]
    tokens["valid_position"][:len(valid)] = valid
    # Original mode: stable ascending entropy then reversal (later ties first).
    tokens["fallback"][:len(valid)] = np.argsort(tokens["entropy"][valid], kind="stable")[::-1]
    ranked = valid[np.argsort(-tokens["entropy"][valid], kind="stable")]
    # Highest entropy, then earliest token wins each boundary's representative.
    for rank, position in enumerate(ranked, 1):
        step = int(tokens["phase_step"][position])
        if 0 < step < count and steps["representative"][step] < 0:
            steps["representative"][step] = position
            steps["rank"][step] = rank
    available = np.flatnonzero(steps["representative"] >= 0)
    steps["available"][:len(available)] = available
    if len(available):
        scores = tokens["entropy"][steps["representative"][available]]
        steps["fallback"][:len(available)] = np.argsort(scores, kind="stable")[::-1]
    return tokens, steps, dict(
        num_tokens=len(tokens), num_steps=len(steps), phase_num_steps=count,
        phase_num_tokens=reasoning_tokens, num_valid=len(valid), num_available=len(available),
        # Normally identical to generation text; do not duplicate it in that case.
        decoded_output=None if text == generation.get("full_output") else text,
    )


def selection_arrays(record, args):
    tokens, steps = open_arrays(record)
    if getattr(args, "phase_partition_algorithm", "original") == "adaptive":
        count = adaptive_phase_count(record["phase_num_tokens"], args.target_phase_tokens, record["phase_num_steps"])
        high_count = max(1, math.ceil(args.high_entropy_ratio * record["num_valid"])) if record["num_valid"] else 0
        available = steps["available"][:record["num_available"]]
        values = steps["rank"][available].astype(np.float64)
        positions = steps["representative"][available]
        fallback = steps["fallback"][:len(available)]
        threshold = high_count
    else:
        count = args.num_segments
        available = tokens["valid_position"][:record["num_valid"]]
        values, positions = tokens["entropy"][available], available
        fallback = tokens["fallback"][:len(available)]
        threshold, high_count = args.entropy_threshold, None
    needed = min(max(0, count - 1), len(available))
    if count <= 1 or not record["num_valid"]:
        values, positions, fallback, needed, high_count = values[:0], positions[:0], fallback[:0], 0, 0
    return values, positions, fallback, needed, threshold, high_count


def select_cached(record, args):
    values, positions, fallback, needed, threshold, candidate_count = selection_arrays(record, args)
    adaptive = getattr(args, "phase_partition_algorithm", "original") == "adaptive"
    mask = values <= threshold if adaptive else values > threshold
    high = np.flatnonzero(mask)
    count = len(high) if candidate_count is None else candidate_count
    if not needed:
        return [], count
    if len(high) >= needed:
        indices = [math.ceil(j * len(high) / (needed + 1)) - 1 for j in range(1, needed + 1)]
        selected = high[indices]
    else:
        low = fallback[~mask[fallback]]
        selected = np.concatenate((high, low[:needed - len(high)]))
    # Adaptive representatives are ordered by step, not necessarily token index.
    return positions[np.sort(selected)].astype(int).tolist(), count


def build_prepared(generation, record, boundaries, candidate_count, args):
    tokens, steps = open_arrays(record)
    text = generation.get("full_output") if record["decoded_output"] is None else record["decoded_output"]
    starts = {1}
    for position in boundaries:
        step = int(tokens["phase_step"][position])
        if 0 < step < record["phase_num_steps"]:
            starts.add(step + 1)
    ordered = sorted(starts) if len(steps) else []
    phase_ranges = [[start, ordered[index + 1] - 1 if index + 1 < len(ordered) else record["phase_num_steps"]]
                    for index, start in enumerate(ordered)]
    frontiers = [1] * len(steps)
    for start, end in phase_ranges:
        frontiers[end - 1] = 2
    for start, end in phase_ranges:
        frontiers[start - 1] = 0
    if frontiers:
        frontiers[0] = 0
        if record["phase_num_steps"] > 1:
            frontiers[record["phase_num_steps"] - 1] = 2
    confidences = [value if valid else None for value, valid in
                   zip(steps["confidence"].tolist(), steps["has_confidence"].tolist())]
    # Convert each strided field in C once, rather than repeatedly constructing
    # NumPy record scalars while assembling the required legacy dictionaries.
    step_records = [dict(step=text[start:end], token_avg_confidence=confidence,
                         first_token_index=first if first >= 0 else None, phase_frontier=frontier)
                    for start, end, confidence, first, frontier in zip(
                        steps["start"].tolist(), steps["end"].tolist(), confidences,
                        steps["first_token"].tolist(), frontiers)]
    prepared = dict(status="ok", sample_idx=generation["sample_idx"], sequence_idx=generation["sequence_idx"],
                    problem=generation["problem"], gold_answer=generation.get("gold_answer"),
                    difficulty=generation.get("difficulty"), unique_id=generation.get("unique_id"),
                    full_output=text, step_avg_confidences=confidences, num_steps=len(steps),
                    num_generated_tokens=len(tokens), finish_reason=generation.get("finish_reason"),
                    phase_ranges=phase_ranges, boundary_token_positions=boundaries,
                    num_entropy_threshold_candidates=candidate_count, steps=step_records)
    if not args.no_save_token_entropies:
        prepared["token_entropies"] = tokens["entropy"].tolist()
    return prepared


def build_error(generation, error):
    raise_record_error(generation, error)
