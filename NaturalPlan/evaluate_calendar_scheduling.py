"""Official NaturalPlan calendar evaluator and Transformer/PRM BoN entry point."""
from __future__ import annotations
import re
from typing import Sequence

def hour_to_num(hr_str: str) -> float:
    return float(hr_str.split(":")[0]) + (0.5 if hr_str.split(":")[1] == "30" else 0.0)

def _boxed_contents(response: str):
    contents = []
    for match in re.finditer(r"\\boxed\s*\{", response):
        index = match.end()
        depth = 1
        while index < len(response) and depth:
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            contents.append(response[match.end(): index - 1])
    return contents

def _time_matches(response: str):
    return list(re.finditer(
        r"([A-Za-z]+)\s*,\s*([0-9]+:[0-9]+)\s*-\s*([0-9]+:[0-9]+)",
        response,
    ))

def _parse_response(response: str):
    boxed = _boxed_contents(response)
    matches = _time_matches(boxed[-1]) if boxed else []
    if not matches:
        matches = _time_matches(response)
    if not matches:
        return "", -1, -1
    match = matches[-1]
    return match.group(1).strip(), hour_to_num(match.group(2)), hour_to_num(match.group(3))

def compute_solve_rate(responses: list[str], solutions: list[str]):
    return sum(_parse_response(r) == _parse_response(s) for r, s in zip(responses, solutions)) / len(responses)

def main(argv: Sequence[str] | None = None) -> None:
    try:
        from .bon_runner_calendar import run_bon
    except ImportError:
        from bon_runner_calendar import run_bon
    run_bon("calendar", "data/calendar_scheduling.json", argv)

if __name__ == "__main__":
    main()
