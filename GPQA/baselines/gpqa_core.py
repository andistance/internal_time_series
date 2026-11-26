from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


GPQA_ROOT = Path(__file__).resolve().parents[1]
CHAIN_OF_THOUGHT_EXAMPLES = GPQA_ROOT / "prompts" / "chain_of_thought_examples.json"
SYSTEM_MESSAGE = (
    "You are a very intelligent assistant, who follows instructions directly."
)
ANSWER_LABELS = ("A", "B", "C", "D")
SUPPORTED_PROMPT_TYPES = ("zero_shot", "chain_of_thought", "5_shot")


def format_online_question_block(
    question: str,
    choices: Sequence[str],
    *,
    blank_line_before_choices: bool = True,
) -> str:
    if len(choices) != 4:
        raise ValueError(f"GPQA requires exactly four choices, received {len(choices)}")
    separator = "\n\n" if blank_line_before_choices else "\n"
    rendered_choices = "\n".join(
        f"({label}) {choice}" for label, choice in zip(ANSWER_LABELS, choices)
    )
    return f"{question}{separator}Choices:\n{rendered_choices}"


def generate_prompt_from_examples(
    json_data: Mapping[str, Any],
    *,
    with_explanations: bool = True,
) -> str:
    output = ""
    for question in json_data["questions"]:
        output += f'Question: {question["question"]}\nChoices:\n'
        for choice, value in question["choices"].items():
            output += f"({choice}) {value}\n"
        if with_explanations:
            output += f"Let's think step by step: \n{question['explanation']}\n"
        output += f'The correct answer is ({question["correct_answer"]})\n'
    return output


def _load_chain_examples(path: str | Path | None = None) -> Mapping[str, Any]:
    examples_path = Path(path) if path is not None else CHAIN_OF_THOUGHT_EXAMPLES
    with examples_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_official_prompt(
    question_block: str,
    prompt_type: str,
    *,
    chain_examples_path: str | Path | None = None,
) -> str:
    if prompt_type not in SUPPORTED_PROMPT_TYPES:
        raise ValueError(
            f"Unsupported GPQA Transformer BoN prompt type: {prompt_type}. "
            f"Choose one of {', '.join(SUPPORTED_PROMPT_TYPES)}"
        )

    if prompt_type == "zero_shot":
        return (
            f"What is the correct answer to this question: {question_block}"
            '\n\nFormat your response as follows: "The correct answer is '
            '(insert answer here)"'
        )

    examples = _load_chain_examples(chain_examples_path)
    if prompt_type == "chain_of_thought":
        prompt = (
            "Here are some example questions from experts. An explanation is given "
            "before the final answer. Answer the final question yourself, giving "
            "your reasoning beforehand.\n"
        )
        prompt += generate_prompt_from_examples(examples, with_explanations=True)
        prompt += f"Question: {question_block}"
        prompt += (
            "\nGive step by step reasoning before you answer, and when you're ready "
            'to answer, please use the format "The correct answer is '
            '(insert answer here)":\n'
        )
        return prompt

    prompt = (
        "Here are some example questions from experts. Answer the final question "
        "yourself, following the format of the previous questions exactly.\n"
    )
    prompt += generate_prompt_from_examples(examples, with_explanations=False)
    prompt += f"Question: {question_block}"
    # Preserve the official baseline's exact 5-shot suffix.
    prompt += (
        "\nWhen you're ready to answer, please use the format "
        '"The correct answer is (insert answer here).'
    )
    return prompt


def parse_sampled_answer(answer: str) -> str | None:
    """Apply the official GPQA sampled-answer regexes in their original order."""

    patterns = (
        r"answer is \((.)\)",
        r"Answer: \((.)\)",
        r"answer: \((.)\)",
        r"answer \((.)\)",
        r"\((.)\)",
    )
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in ANSWER_LABELS:
            return match.group(1)
    return None


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    total = len(rows)
    correct = sum(bool(row.get("is_correct")) for row in rows)
    refusals = sum(bool(row.get("is_refusal")) for row in rows)
    return {
        "total": total,
        "correct": correct,
        "refusals": refusals,
        "accuracy": correct / total if total else 0.0,
        "refusal_fraction": refusals / total if total else 0.0,
    }
