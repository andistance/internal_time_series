from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from .base import load_records
from ..models import BestOfNRequest


@dataclass
class FieldMappedAdapter:
    question_field: str
    answer_field: str | None = None
    id_field: str | None = None
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    benchmark_name: str = "field_mapped"

    def load(self, source: Any) -> Iterable[Dict[str, Any]]:
        return load_records(source)

    def build_request(
        self, sample: Dict[str, Any], sample_idx: int
    ) -> BestOfNRequest:
        if self.question_field not in sample:
            raise KeyError(f"Missing question field {self.question_field!r}")
        task_text = str(sample[self.question_field])
        sample_id = str(
            sample.get(self.id_field, sample_idx) if self.id_field else sample_idx
        )
        gold_answer = sample.get(self.answer_field) if self.answer_field else None
        content = f"{self.prompt_prefix}{task_text}{self.prompt_suffix}"
        return BestOfNRequest(
            request_id=f"{self.benchmark_name}:{sample_id}",
            sample_id=sample_id,
            task_text=task_text,
            messages=[{"role": "user", "content": content}],
            gold_answer=gold_answer,
            metadata={
                "benchmark": self.benchmark_name,
                "sample_idx": sample_idx,
                "raw_sample": dict(sample),
            },
        )

