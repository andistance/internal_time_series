from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Protocol, runtime_checkable

from ..models import BestOfNRequest


def load_records(source: Any) -> Iterator[Dict[str, Any]]:
    """Load JSON/JSONL or pass through an iterable of mappings."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("r", encoding="utf-8") as file:
            if path.suffix.lower() == ".jsonl":
                for line_number, raw_line in enumerate(file, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError(f"{path}:{line_number} is not a JSON object")
                    yield value
                return
            value = json.load(file)
        if not isinstance(value, list):
            raise TypeError(f"JSON dataset must be an array: {path}")
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("Every dataset item must be a mapping")
            yield item
        return

    for item in source:
        if not isinstance(item, dict):
            raise TypeError("Every dataset item must be a mapping")
        yield item


@runtime_checkable
class DatasetAdapter(Protocol):
    benchmark_name: str

    def load(self, source: Any) -> Iterable[Dict[str, Any]]: ...

    def build_request(
        self, sample: Dict[str, Any], sample_idx: int
    ) -> BestOfNRequest: ...

