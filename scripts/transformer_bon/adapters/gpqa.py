from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Iterable, List, Tuple

from .base import load_records
from ..models import BestOfNRequest


def _first(sample: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def _extract_choices(sample: Dict[str, Any]) -> Tuple[List[str], Any]:
    choices = _first(sample, "choices", "options")
    answer = _first(sample, "answer", "correct_answer", "Correct Answer")
    if isinstance(choices, dict):
        labels = sorted(choices)
        return [str(choices[label]) for label in labels], answer
    if isinstance(choices, (list, tuple)):
        return [str(choice) for choice in choices], answer

    correct = _first(sample, "Correct Answer", "correct_answer")
    incorrect = [
        _first(sample, f"Incorrect Answer {index}", f"incorrect_answer_{index}")
        for index in range(1, 4)
    ]
    values = [value for value in [correct, *incorrect] if value is not None]
    return [str(value) for value in values], correct


@dataclass
class GPQAAdapter:
    benchmark_name: str = "gpqa"

    def load(self, source: Any) -> Iterable[Dict[str, Any]]:
        return load_records(source)

    def build_request(
        self, sample: Dict[str, Any], sample_idx: int
    ) -> BestOfNRequest:
        question = _first(sample, "question", "Question")
        if question is None:
            raise KeyError("GPQA sample has neither 'question' nor 'Question'")
        choices, answer = _extract_choices(sample)
        if not choices:
            raise ValueError("GPQA sample has no choices")
        sample_id = str(sample.get("id", sample.get("unique_id", sample_idx)))
        # The common GPQA schema stores the correct answer separately. Shuffle
        # deterministically so that it is not always rendered as option A.
        if not isinstance(_first(sample, "choices", "options"), (list, tuple, dict)):
            random.Random(sample_id).shuffle(choices)
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        answer_label = answer
        if answer is not None and str(answer) in choices:
            answer_label = labels[choices.index(str(answer))]
        formatted_choices = "\n".join(
            f"{label}. {choice}" for label, choice in zip(labels, choices)
        )
        content = (
            f"{question}\n\nChoices:\n{formatted_choices}\n\n"
            "Reason carefully, then give the final answer as a single option letter."
        )
        return BestOfNRequest(
            request_id=f"{self.benchmark_name}:{sample_id}",
            sample_id=sample_id,
            task_text=str(question),
            messages=[{"role": "user", "content": content}],
            gold_answer=answer_label,
            metadata={
                "benchmark": self.benchmark_name,
                "sample_idx": sample_idx,
                "choices": choices,
                "choice_labels": labels,
                "correct_answer_text": answer,
                "raw_sample": dict(sample),
            },
        )
