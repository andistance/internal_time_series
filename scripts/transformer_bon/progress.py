"""Small, thread-safe dataset progress wrapper shared by benchmark runners."""

from __future__ import annotations

import sys
import threading
from typing import Any, Optional


class DatasetProgress:
    """Display completed benchmark samples, including resumed samples."""

    def __init__(
        self,
        total: int,
        *,
        initial: int = 0,
        description: str = "Benchmark",
        unit: str = "sample",
    ) -> None:
        self.total = max(0, int(total))
        self.current = min(max(0, int(initial)), self.total)
        self.description = description
        self.unit = unit
        self._lock = threading.Lock()
        self._bar: Optional[Any] = None
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=self.total,
                initial=self.current,
                desc=self.description,
                unit=self.unit,
                dynamic_ncols=True,
                mininterval=0.2,
                smoothing=0.1,
            )
        except ImportError:
            self._render_fallback(final=self.current >= self.total)

    def _render_fallback(self, *, final: bool = False) -> None:
        percent = 100.0 if self.total == 0 else 100.0 * self.current / self.total
        ending = "\n" if final else "\r"
        print(
            f"{self.description}: {self.current}/{self.total} "
            f"{self.unit}s ({percent:6.2f}%)",
            end=ending,
            file=sys.stderr,
            flush=True,
        )

    def update(self, count: int = 1) -> None:
        with self._lock:
            increment = min(max(0, int(count)), self.total - self.current)
            if increment <= 0:
                return
            self.current += increment
            if self._bar is not None:
                self._bar.update(increment)
            else:
                self._render_fallback(final=self.current >= self.total)

    def set_postfix(self, **values: Any) -> None:
        with self._lock:
            if self._bar is not None:
                self._bar.set_postfix(values, refresh=False)

    def close(self) -> None:
        with self._lock:
            if self._bar is not None:
                self._bar.close()
                self._bar = None
            elif self.current < self.total:
                self._render_fallback(final=True)

    def __enter__(self) -> "DatasetProgress":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

