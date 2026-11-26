from __future__ import annotations

"""Deterministic, boxed-answer version of the Inspect Evals BBH scorers."""

import re
from typing import Any, Mapping


def extract_last_boxed(text: Any) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}`` expression."""
    value = "" if text is None else str(text)
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", value)]
    for start in reversed(starts):
        brace_start = value.find("{", start)
        depth = 0
        escaped = False
        for index in range(brace_start, len(value)):
            char = value[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return value[brace_start + 1 : index].strip()
    return None


def _norm(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\\text{", "").replace("\\mathrm{", "")
    text = text.replace("}", "").replace("$", "")
    text = re.sub(r"\\[,;:]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text.casefold().strip(" .,:;\n\t")


def _options_from_input(input_text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?:^|\n)\s*\(?([A-Z])\)?[.)]\s*(.*?)(?=\n\s*\(?[A-Z]\)?[.)]\s*|$)", input_text, re.S))
    if not matches:
        matches = list(re.finditer(r"\(([A-Z])\)\s*([^\n]+)", input_text))
    return [(m.group(1).upper(), m.group(2).strip()) for m in matches]


def _choice_answer(boxed: str, input_text: str, target: str) -> str | None:
    value = boxed.strip()
    # Inspect's choice scorer is letter-oriented.  Prefer a standalone letter,
    # while accepting the common parenthesized/trailing-period variants.
    match = re.fullmatch(r"[\s]*(?:\(?([A-Z])\)?)(?:[.)])?[\s]*", value, re.I)
    if match:
        return match.group(1).upper()
    options = _options_from_input(input_text)
    normalized = _norm(value)
    for label, option_text in options:
        if normalized == _norm(option_text):
            return label
    # If the box contains a short answer sentence, use the first option marker
    # rather than requiring a raw exact match.
    match = re.search(r"\(?([A-Z])\)?(?:[.)]|\b)", value, re.I)
    if match and len(value) <= 80:
        return match.group(1).upper()
    target_match = re.search(r"\(?([A-Z])\)?", target, re.I)
    if target_match and normalized == _norm(target):
        return target_match.group(1).upper()
    return None


def _binary_answer(boxed: str, task_name: str) -> str | None:
    value = _norm(boxed)
    if task_name == "web_of_lies":
        if re.search(r"\b(no|not|does not tell the truth|is not telling the truth)\b", value):
            return "no"
        if re.search(r"\b(yes|tells the truth|is telling the truth)\b", value):
            return "yes"
    if task_name == "formal_fallacies":
        for word in ("invalid", "valid"):
            if re.search(rf"\b{word}\b", value):
                return word
    if task_name == "boolean_expressions":
        for word in ("true", "false"):
            if re.search(rf"\b{word}\b", value):
                return word
    for word in ("yes", "no"):
        if re.search(rf"\b{word}\b", value):
            return word
    return None


def _exact_answer(boxed: str, task_name: str) -> str:
    value = _norm(boxed)
    if task_name == "word_sorting":
        # Normalize separators in the same spirit as Inspect's end matching;
        # preserve word order, which is the actual task target.
        return " ".join(re.findall(r"[A-Za-z]+", value))
    if task_name == "dyck_languages":
        return re.sub(r"\s+", "", value)
    if task_name in {"multistep_arithmetic_two", "object_counting"}:
        # Inspect's end matcher is insensitive to presentation whitespace;
        # numeric BBH answers also commonly include thousands separators.
        return re.sub(r"[\s,]", "", value)
    return value


def canonical_target(target: Any, *, task_type: str, input_text: str, task_name: str) -> str:
    value = str(target)
    if task_type == "multiple_choice":
        parsed = _choice_answer(value, input_text, value)
        return parsed or _norm(value)
    if task_type == "binary_choice":
        parsed = _binary_answer(value, task_name)
        return parsed or _norm(value)
    return _exact_answer(value, task_name)


def score_boxed_answer(
    response: Any,
    *,
    target: Any,
    task_name: str,
    task_type: str,
    input_text: str,
) -> tuple[bool, str | None, str]:
    """Score only the last box, returning ``(correct, parsed, target)``."""
    boxed = extract_last_boxed(response)
    expected = canonical_target(
        target, task_type=task_type, input_text=input_text, task_name=task_name
    )
    if boxed is None:
        return False, None, expected
    if task_type == "multiple_choice":
        parsed = _choice_answer(boxed, input_text, str(target))
    elif task_type == "binary_choice":
        parsed = _binary_answer(boxed, task_name)
    else:
        parsed = _exact_answer(boxed, task_name)
    return parsed is not None and parsed.casefold() == expected.casefold(), parsed, expected
