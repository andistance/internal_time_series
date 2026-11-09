from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from .base import load_records
from ..models import BestOfNRequest


AIME_PROMPT_SUFFIX = (
    "\nLet's reason step by step. Enclose the reasoning process within "
    "<think>...</think>, then summarize it and present the final answer within "
    "\\boxed{}."
)


@dataclass
class AIMEAdapter:
    benchmark_name: str = "aime"
    prompt_suffix: str = AIME_PROMPT_SUFFIX

    def load(self, source: Any) -> Iterable[Dict[str, Any]]:
        return load_records(source)

    def build_request(
        self, sample: Dict[str, Any], sample_idx: int
    ) -> BestOfNRequest:
        problem = str(sample["problem"])
        sample_id = str(sample.get("unique_id", sample.get("id", sample_idx)))
        return BestOfNRequest(
            request_id=f"{self.benchmark_name}:{sample_id}",
            sample_id=sample_id,
            task_text=problem,
            messages=[{"role": "user", "content": problem + self.prompt_suffix}],
            gold_answer=sample.get("answer"),
            metadata={
                "benchmark": self.benchmark_name,
                "sample_idx": sample_idx,
                "difficulty": sample.get("difficulty"),
                "raw_sample": dict(sample),
            },
        )

