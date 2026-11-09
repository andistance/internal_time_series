"""Bounded background iteration and an ordered, durable append worker."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from contextlib import contextmanager


class PipelineStopped(BaseException):
    """Internal cancellation; never converted into a failed candidate."""


def check_stopped(stop):
    if stop is not None and stop.is_set():
        raise PipelineStopped()


@contextmanager
def background_iterator(make_source):
    """One output ahead, including the output currently being constructed.

    Unlike a plain prefetch loop, acquiring the permit precedes next(source),
    so a full queue never retains an additional completed trajectory bundle.
    Sources must check stop during waits and own their resource cleanup.
    """
    stop, permit, ready = threading.Event(), threading.Semaphore(1), queue.Queue(maxsize=1)

    def put(kind, value):
        while not stop.is_set():
            try:
                ready.put((kind, value), timeout=0.1)
                return
            except queue.Full:
                pass

    def produce():
        source = None
        try:
            source = iter(make_source(stop))
            while not stop.is_set():
                if not permit.acquire(timeout=0.1):
                    continue
                check_stopped(stop)
                try:
                    value = next(source)
                except StopIteration:
                    put("done", None)
                    return
                put("value", value)
        except PipelineStopped:
            pass
        except BaseException as error:
            put("error", error)
        finally:
            if source is not None:
                source.close()

    def consume():
        while True:
            kind, value = ready.get()
            if kind == "done":
                return
            if kind == "error":
                raise value
            permit.release()
            yield value

    thread = threading.Thread(target=produce, name="prm-ordered-pipeline", daemon=True)
    thread.start()
    try:
        yield consume()
    finally:
        stop.set()
        # Wait for cancellation/worker cleanup before model or file teardown.
        thread.join()


class OrderedAppender:
    """One writer, at most two queued encodings, byte-limited backpressure.

    The producer encodes a snapshot before enqueueing. Future completion means
    that append's fsync has succeeded. A failure is sticky and later rows are
    never appended after it. There is no extra journal or payload cache.
    """
    def __init__(self, append, *, max_bytes=4 * 1024 * 1024):
        self.append = append
        self.max_bytes = max_bytes
        self.condition = threading.Condition()
        self.pending_bytes = 0
        self.error = None
        self.closed = False
        self.queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._run, name="prm-durable-writer", daemon=True)
        self.thread.start()

    def submit(self, payload, key):
        size = len(payload)
        with self.condition:
            while self.pending_bytes and self.pending_bytes + size > self.max_bytes and self.error is None:
                self.condition.wait()
            if self.error is not None:
                raise self.error
            self.pending_bytes += size
        future = Future()
        self.queue.put((payload, key, size, future))
        return future

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            payload, key, size, future = item
            try:
                if self.error is not None:
                    raise self.error
                self.append(payload, key)
                future.set_result(None)
            except BaseException as error:
                self.error = error
                future.set_exception(error)
            finally:
                with self.condition:
                    self.pending_bytes -= size
                    self.condition.notify_all()
            del payload, key, item, future

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
            self.thread.join()
        if self.error is not None:
            raise self.error
