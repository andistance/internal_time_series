"""Opt-in wall timings; forward includes its existing result-copy wait."""
import os
import threading
import time

from .json_codec import codec_stats


class ScoringProfile:
    def __init__(self):
        self.enabled = os.environ.get("TRANSFORMER_BON_PROFILE", "0") == "1"
        self.started = time.perf_counter()
        self.totals = {}
        self.lock = threading.Lock()

    def start(self):
        return time.perf_counter() if self.enabled else None

    def add(self, name, started):
        if started is not None:
            elapsed = time.perf_counter() - started
            with self.lock:
                count, seconds = self.totals.get(name, (0, 0.0))
                self.totals[name] = count + 1, seconds + elapsed

    def report(self):
        if self.enabled:
            values = ", ".join(f"{key}={seconds:.3f}s/{count}" for key, (count, seconds) in self.totals.items())
            print(f"[INFO] Transformer scoring wall timings: {values}; "
                  f"elapsed={time.perf_counter() - self.started:.3f}s. Concurrent stages overlap.")
            print(f"[INFO] JSON codec process totals: {codec_stats()}")
