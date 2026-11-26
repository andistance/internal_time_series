from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.transformer_bon import BestOfNRequest

try:
    from .gpqa_core import SYSTEM_MESSAGE, build_official_prompt
    from .gpqa_dataset import GPQAProblem
except ImportError:  # Direct execution from GPQA/baselines.
    from gpqa_core import SYSTEM_MESSAGE, build_official_prompt
    from gpqa_dataset import GPQAProblem


@dataclass(frozen=True)
class GPQATransformerBoNAdapter:
    prompt_type: str = "zero_shot"
    chat_template: str = "qwen3"
    benchmark_name: str = "gpqa_diamond"

    def build_request(self, problem: GPQAProblem) -> BestOfNRequest:
        prompt = build_official_prompt(problem.question_block, self.prompt_type)
        metadata = {
            **problem.metadata,
            "benchmark": self.benchmark_name,
            "prompt_type": self.prompt_type,
            "answer_label": problem.answer_label,
            "label_to_choice": dict(problem.label_to_choice),
        }
        request_id = (
            problem.sample_id
            if problem.sample_id.startswith(f"{self.benchmark_name}:")
            else f"{self.benchmark_name}:{problem.sample_id}"
        )
        common = {
            "request_id": request_id,
            "sample_id": problem.sample_id,
            "task_text": problem.question_block,
            "gold_answer": problem.answer_label,
            "metadata": metadata,
        }
        if self.chat_template == "raw":
            return BestOfNRequest(raw_prompt=prompt, **common)
        return BestOfNRequest(
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            **common,
        )
