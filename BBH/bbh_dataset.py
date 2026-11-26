from __future__ import annotations

"""Local BIG-Bench-Hard dataset and cot_zeroshot prompt loading.

The official BBH files are JSON objects containing an ``examples`` array.  This
loader also accepts a JSON array and JSONL so that converted local copies remain
usable.  Task descriptions are read from the official cot_zeroshot YAML files.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping


FINAL_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


MULTIPLE_CHOICE_TASKS = {
    "date_understanding", "disambiguation_qa", "geometric_shapes", "hyperbaton",
    "logical_deduction_three_objects", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "movie_recommendation",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "temporal_sequences",
    "tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
}
BINARY_TASKS = {
    "boolean_expressions", "causal_judgement", "formal_fallacies", "navigate",
    "sports_understanding", "web_of_lies",
}
EXACT_MATCH_TASKS = {"multistep_arithmetic_two", "object_counting", "word_sorting"}
DYCK_TASKS = {"dyck_languages"}


@dataclass(frozen=True)
class BBHExample:
    task_name: str
    example_index: int
    input: str
    target: str
    description: str
    task_type: str
    raw: Mapping[str, Any]

    @property
    def sample_id(self) -> str:
        return f"{self.task_name}:{self.example_index}"

    @property
    def prompt(self) -> str:
        parts = [self.description.strip(), self.input.strip(), FINAL_INSTRUCTION]
        return "\n\n".join(part for part in parts if part)


def task_type(task_name: str) -> str:
    if task_name in MULTIPLE_CHOICE_TASKS:
        return "multiple_choice"
    if task_name in BINARY_TASKS:
        return "binary_choice"
    if task_name in DYCK_TASKS:
        return "dyck"
    if task_name in EXACT_MATCH_TASKS:
        return "exact_match"
    # Unknown local BBH-compatible tasks are safest to treat as open answers.
    return "exact_match"


def _json_records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError(f"{path}:{line_number} is not a JSON object")
                values.append(value)
        return values
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, Mapping) and isinstance(value.get("examples"), list):
        return [item for item in value["examples"] if isinstance(item, Mapping)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping) and "input" in value:
        return [value]
    raise TypeError(f"BBH JSON must contain an examples array: {path}")


def _yaml_description(path: Path) -> str:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if isinstance(value, Mapping) and value.get("description") is not None:
            return str(value["description"])
    except ModuleNotFoundError:
        pass
    except Exception:
        # A malformed optional YAML should not prevent loading a valid dataset;
        # the fallback below still handles the simple official description key.
        pass
    text = path.read_text(encoding="utf-8")
    # The official lm-evaluation-harness files are valid YAML, but many local
    # copies are emitted in JSON-style YAML, e.g. ``"description": "..."``.
    # Keep a dependency-free fallback so description loading still works when
    # PyYAML is not installed.
    match = re.search(r"^\s*[\"']?description[\"']?\s*:\s*(.*?)\s*$", text, re.M)
    if match:
        raw_value = match.group(1).strip()
        try:
            decoded = json.loads(raw_value)
            if isinstance(decoded, str):
                return decoded.strip()
        except (TypeError, json.JSONDecodeError):
            pass
        if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {"'", '\"'}:
            raw_value = raw_value[1:-1]
        # Decode the escaped ``\\n`` used by the JSON-style official files,
        # while leaving ordinary backslashes intact when possible.
        return raw_value.replace(r"\n", "\n").strip()
    return ""


def _description_for(task_name: str, description_dir: Path | None) -> str:
    if description_dir is None:
        return ""
    for suffix in (".yaml", ".yml"):
        path = description_dir / f"{task_name}{suffix}"
        if path.is_file():
            return _yaml_description(path)
    return ""


def _candidate_json_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    if not data_path.is_dir():
        raise FileNotFoundError(f"BBH data path does not exist: {data_path}")
    return sorted(
        [*data_path.glob("*.json"), *data_path.glob("*.jsonl")],
        key=lambda item: item.name.lower(),
    )


def _resolve_description_dir(data_path: Path, requested: str | Path | None) -> Path | None:
    if requested:
        candidate = Path(requested).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"Description directory does not exist: {candidate}")
        return candidate
    roots = []
    root = data_path if data_path.is_dir() else data_path.parent
    roots.extend([root / "cot_zeroshot", root.parent / "cot_zeroshot"])
    for candidate in roots:
        if candidate.is_dir():
            return candidate
    return None


def load_bbh_examples(
    data_path: str | Path,
    *,
    description_dir: str | Path | None = None,
    tasks: Iterable[str] | None = None,
    max_examples: int | None = None,
) -> list[BBHExample]:
    data_path = Path(data_path).expanduser().resolve()
    yaml_dir = _resolve_description_dir(data_path, description_dir)
    requested = {str(value) for value in tasks} if tasks else None
    examples: list[BBHExample] = []
    for json_path in _candidate_json_files(data_path):
        task_name = json_path.stem
        if requested is not None and task_name not in requested:
            continue
        records = _json_records(json_path)
        description = _description_for(task_name, yaml_dir)
        for local_index, record in enumerate(records):
            if "input" not in record or "target" not in record:
                raise KeyError(f"{json_path} example {local_index} needs input and target")
            examples.append(
                BBHExample(
                    task_name=task_name,
                    example_index=local_index,
                    input=str(record["input"]),
                    target=str(record["target"]),
                    description=description,
                    task_type=task_type(task_name),
                    raw=dict(record),
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    if requested:
        present = {item.task_name for item in examples}
        missing = sorted(requested.difference(present))
        if missing:
            raise FileNotFoundError(f"Requested BBH task files were not found: {missing}")
    return examples
