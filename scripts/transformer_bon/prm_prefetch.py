"""Bounded CPU preparation, with no changes to inference batches or precision."""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager, ExitStack


def candidate_token_ids(scorer, ids, device, runtime):
    """Cache only immutable integer IDs, never model/adapter parameter copies."""
    ids = tuple(ids)
    cached = getattr(scorer, "_candidate_ids_cache", None)
    if cached is None or cached[0] != ids:
        cached = scorer._candidate_ids_cache = (ids, {})
    key = str(device)
    if key not in cached[1]:
        cached[1][key] = runtime.as_tensor(ids, dtype=runtime.long, device=device)
    return cached[1][key]


def capture_inference_context(runtime, scorer):
    """Capture caller thread-local settings without creating another model."""
    autocast, streams = [], []
    if runtime is not None:
        for device in ("cpu", "cuda"):
            if device == "cuda" and not runtime.cuda.is_available():
                continue
            try:
                enabled, dtype = runtime.is_autocast_enabled(device), runtime.get_autocast_dtype(device)
            except (TypeError, AttributeError):
                enabled = runtime.is_autocast_cpu_enabled() if device == "cpu" else runtime.is_autocast_enabled()
                dtype = runtime.get_autocast_cpu_dtype() if device == "cpu" else runtime.get_autocast_gpu_dtype()
            autocast.append((device, enabled, dtype))
        model = getattr(scorer, "model", None)
        if runtime.cuda.is_available() and runtime.cuda.is_initialized():
            devices = ({p.device for p in model.parameters() if p.is_cuda} if model is not None
                       else {runtime.device("cuda", runtime.cuda.current_device())})
            streams = [runtime.cuda.current_stream(device) for device in devices]

    @contextmanager
    def restored():
        with ExitStack() as stack:
            for device, enabled, dtype in autocast:
                stack.enter_context(runtime.autocast(device, enabled=enabled, dtype=dtype))
            for stream in streams:
                stack.enter_context(runtime.cuda.stream(stream))
            yield
    return restored


class PreparedRowsScorer:
    """CPU prefetch for backends whose reference forward is one unpadded row.

    Deliberately do not turn these into padded batches: changing GEMM/attention
    shapes can change floating point scores and hence best-of-N tie breaking.
    """

    def prepare_batch(self, texts):
        return [self._prepare_one(value) for value in texts]

    def score_prepared_batch(self, rows):
        import torch
        with torch.inference_mode():
            if not hasattr(self, "_score_prepared_tensor"):
                return [self._score_prepared_one(row) for row in rows]
            # Forward/softmax shapes and order are unchanged. Only copies are
            # coalesced; keep at most this microbatch's small score vectors.
            scores = [self._score_prepared_tensor(row) for row in rows]
            if not scores:
                return []
            flat = torch.cat(scores).detach().cpu().tolist()
            result, offset = [], 0
            for score in scores:
                size = score.numel()
                result.append(flat[offset:offset + size])
                offset += size
            return result

    def _score_prepared_one(self, row):
        return self._score_prepared_tensor(row).detach().cpu().tolist()

    def _score_one(self, value):
        return self._score_prepared_one(self._prepare_one(value))

    def score_batch(self, texts):
        return self.score_prepared_batch(self.prepare_batch(texts))


@contextmanager
def prefetch_map(source, prepare, *, stop=None):
    """Queue one CPU batch and prepare one more during consumer inference.

    The source must be finite or check ``stop`` during blocking reads. Errors
    are delivered in input order and cancellation never waits on a full queue.
    CPU preparation must not create CUDA tensors or mutate the model.
    """
    stop = stop if stop is not None else threading.Event()
    ready = queue.Queue(maxsize=1)

    def put(value):
        while not stop.is_set():
            try:
                ready.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        try:
            for item in source:
                if stop.is_set():
                    return
                put(("value", prepare(item)))
            put(("done", None))
        except BaseException as exc:
            put(("error", exc))

    def consume():
        while True:
            kind, value = ready.get()
            if kind == "done":
                return
            if kind == "error":
                raise value
            yield value

    thread = threading.Thread(target=produce, name="prm-cpu-prefetch", daemon=True)
    thread.start()
    try:
        yield consume()
    finally:
        stop.set()
        thread.join(timeout=2.0)
