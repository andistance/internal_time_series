from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping

try:
    from .gpqa_core import ANSWER_LABELS, format_online_question_block
except ImportError:  # Direct execution from GPQA/baselines.
    from gpqa_core import ANSWER_LABELS, format_online_question_block


@dataclass(frozen=True)
class GPQAProblem:
    sample_id: str
    question_block: str
    answer_label: str
    label_to_choice: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_unique(problems: list[GPQAProblem]) -> list[GPQAProblem]:
    seen = set()
    for problem in problems:
        if problem.sample_id in seen:
            raise ValueError(f"Duplicate GPQA sample id: {problem.sample_id}")
        seen.add(problem.sample_id)
    return problems


def normalize_online_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    choice_seed: int = 0,
) -> list[GPQAProblem]:
    rng = random.Random(choice_seed)
    problems = []
    for index, row in enumerate(rows):
        required = (
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(
                f"Online GPQA item {index} is missing fields: {', '.join(missing)}"
            )
        question = str(row["Question"])
        correct_answer = str(row["Correct Answer"])
        choices = [
            str(row["Incorrect Answer 1"]),
            str(row["Incorrect Answer 2"]),
            str(row["Incorrect Answer 3"]),
            correct_answer,
        ]
        rng.shuffle(choices)
        answer_label = ANSWER_LABELS[choices.index(correct_answer)]
        label_to_choice = dict(zip(ANSWER_LABELS, choices))
        raw_id = row.get("Record ID") or row.get("id") or index
        problems.append(
            GPQAProblem(
                sample_id=str(raw_id),
                question_block=format_online_question_block(question, choices),
                answer_label=answer_label,
                label_to_choice=label_to_choice,
                metadata={
                    "dataset_source": "online",
                    "sample_idx": index,
                    "question_text": question,
                    "choices": choices,
                    "choice_labels": list(ANSWER_LABELS),
                    "correct_answer_text": correct_answer,
                    "record_id": str(raw_id),
                    "high_level_domain": row.get("High-level domain"),
                    "subdomain": row.get("Subdomain"),
                    "raw_sample": dict(row),
                },
            )
        )
    return _validate_unique(problems)


def _local_choice_mapping(question: str) -> dict[str, str]:
    matches = re.findall(r"(?m)^([A-D])[.)]\s*(.+?)\s*$", question)
    mapping = {label: choice for label, choice in matches}
    if set(mapping) == set(ANSWER_LABELS):
        return {label: mapping[label] for label in ANSWER_LABELS}
    return {label: label for label in ANSWER_LABELS}


def normalize_local_rows(rows: Any, *, source: str) -> list[GPQAProblem]:
    if not isinstance(rows, list):
        raise ValueError(f"GPQA dataset {source} must be a top-level JSON array")
    problems = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Local GPQA item {index} must be a JSON object")
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Local GPQA item {index} has an empty question")
        if not isinstance(answer, str) or answer.strip().upper() not in ANSWER_LABELS:
            raise ValueError(
                f"Local GPQA item {index} answer must be one of A, B, C, D"
            )
        answer_label = answer.strip().upper()
        sample_id = f"gpqa_diamond:{index}"
        problems.append(
            GPQAProblem(
                sample_id=sample_id,
                question_block=question,
                answer_label=answer_label,
                label_to_choice=_local_choice_mapping(question),
                metadata={
                    "dataset_source": "local",
                    "sample_idx": index,
                    "choices_preformatted": True,
                    "raw_sample": dict(row),
                },
            )
        )
    return _validate_unique(problems)


def load_gpqa_dataset(
    *,
    dataset_source: str = "online",
    local_dataset_path: str | None = None,
    dataset_name: str = "Idavidrein/gpqa",
    dataset_config: str = "gpqa_diamond",
    dataset_split: str = "train",
    hf_token: str | None = None,
    choice_seed: int = 0,
    max_examples: int | None = None,
) -> list[GPQAProblem]:
    if dataset_source == "local":
        if not local_dataset_path:
            raise ValueError("local dataset source requires local_dataset_path")
        path = Path(local_dataset_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local GPQA dataset does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            problems = normalize_local_rows(json.load(file), source=str(path))
    elif dataset_source == "online":
        from datasets import load_dataset

        rows = load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
            token=hf_token,
        )
        problems = normalize_online_rows(rows, choice_seed=choice_seed)
    else:
        raise ValueError(f"Unsupported GPQA dataset source: {dataset_source}")

    if max_examples is not None:
        if max_examples < 0:
            raise ValueError("max_examples must be non-negative")
        problems = problems[:max_examples]
    return problems
