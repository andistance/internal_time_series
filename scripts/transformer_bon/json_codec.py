"""Exact stdlib JSON in bounded, persistent CPU-only helper processes.

Small rows stay local. Large rows cross a binary pipe as pickle (lossless for
Python floats and dict order), keeping decimal formatting/parsing off the GIL
used to submit GPU work. No multiprocessing import of the benchmark main,
models, temporary payload files, unbounded task queues or new dependencies.
"""
from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import threading
import time


MIN_BYTES = 64 * 1024
MAX_BYTES = 4 * 1024 * 1024
_workers = {}
_registry_lock = threading.Lock()
_pool = None


def _read_exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        part = stream.read(size - len(parts))
        if not part:
            raise RuntimeError("Value JSON worker exited before returning its result")
        parts.extend(part)
    return bytes(parts)


class _Codec:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.calls = 0
        self.seconds = 0.0
        self.owner = os.getpid()

    def _start(self):
        options = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name("json_codec_worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            **options,
        )

    def request(self, operation, payload):
        with self.lock:
            if self.process is None:
                self._start()
            process = self.process
            started = time.perf_counter()
            try:
                process.stdin.write(struct.pack("!Q", len(payload)))
                process.stdin.write(payload)
                process.stdin.flush()
                size = struct.unpack("!Q", _read_exact(process.stdout, 8))[0]
                if size > 64 * 1024 * 1024:
                    raise RuntimeError("Value JSON worker response exceeds its byte budget")
                response = _read_exact(process.stdout, size)
            except (OSError, RuntimeError):
                self._close()
                raise
            finally:
                self.calls += 1
                self.seconds += time.perf_counter() - started
            if response[0]:
                raise pickle.loads(response[1:])
            return response[1:] if operation == "encode" else pickle.loads(response[1:])

    def _close(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()

    def close(self):
        with self.lock:
            self._close()


def _worker(operation):
    key = os.getpid(), operation
    with _registry_lock:
        if key not in _workers:
            _workers[key] = _Codec()
        return _workers[key]


def _enabled():
    return os.environ.get("TRANSFORMER_BON_JSON_WORKERS", "1") != "0"


def _encode_payload(row):
    # Avoid walking small records just to decide whether to offload them.
    def size(value):
        try:
            return len(value)
        except TypeError:
            return 0
    large = (type(row) is dict and (
        size(row.get("token_entropies")) >= 2048
        or size(row.get("steps")) >= 128
        or size(row.get("full_output")) >= MIN_BYTES
    ))
    if large and _enabled():
        try:
            payload = pickle.dumps(("encode", row), protocol=4)
        except Exception:
            payload = b""
        if MIN_BYTES <= len(payload) <= MAX_BYTES:
            return payload
    return None


def _local_encode(row):
    return json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _remote_encode(row, payload, lane=0):
    try:
        return _worker("encode" if lane == 0 else "encode-1").request("encode", payload)
    except (ImportError, AttributeError):
        # The producer may already have reused/mutated its input object.
        return _local_encode(pickle.loads(payload)[1])


def encode_json(row):
    """Return the same UTF-8 bytes as the existing result-file serializer."""
    payload = _encode_payload(row)
    return _local_encode(row) if payload is None else _remote_encode(row, payload)


def iter_encoded(records, validate=None):
    """At most two large rows in flight, in input order (<=4 MiB per request).

    Each pipe has its own stdlib-only CPU process. Small/oversized rows drain
    lookahead and stay local, so a single oversized row cannot double the peak.
    """
    global _pool
    pending = deque()
    lane = 0
    try:
        for row in records:
            if validate is not None:
                validate(row)
            payload = _encode_payload(row)
            if payload is None:
                while pending:
                    previous, future = pending.popleft()
                    yield previous, future.result()
                yield row, _local_encode(row)
                continue
            with _registry_lock:
                if _pool is None:
                    _pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="value-json-pipe")
                pool = _pool
            metadata = dict(row) if isinstance(row, dict) else row
            pending.append((metadata, pool.submit(_remote_encode, None, payload, lane)))
            lane = 1 - lane
            if len(pending) == 2:
                previous, future = pending.popleft()
                yield previous, future.result()
        while pending:
            previous, future = pending.popleft()
            yield previous, future.result()
    finally:
        for _, future in pending:
            future.cancel()
        for _, future in pending:
            if not future.cancelled():
                try:
                    future.result()
                except BaseException:
                    pass


def decode_json(raw):
    if _enabled() and MIN_BYTES <= len(raw) <= MAX_BYTES - 64:
        payload = pickle.dumps(("decode", raw), protocol=4)
        if len(payload) <= MAX_BYTES:
            return _worker("decode").request("decode", payload)
    return json.loads(raw)


def codec_stats():
    with _registry_lock:
        return {operation: dict(calls=worker.calls, wall_seconds=round(worker.seconds, 3),
                                pid=worker.process.pid if worker.process is not None else None)
                for (pid, operation), worker in _workers.items() if pid == os.getpid()}


def close_codecs():
    global _pool
    with _registry_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
    with _registry_lock:
        workers = [worker for (pid, _), worker in _workers.items() if pid == os.getpid()]
        for worker in workers:
            worker.close()
        _workers.clear()


atexit.register(close_codecs)
