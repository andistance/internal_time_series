"""Official NaturalPlan trip evaluator and Transformer/PRM BoN entry point."""
from __future__ import annotations
import re
from typing import Any, Sequence

def _parse_response_original(response: str):
    pattern_visit = r"\d+-\d+"
    pattern_flight = r".*Day (\d+).*from (\w+) to (\w+)"
    pattern_days = r"European cities for (\d+) days"
    days, flights, visit_cities, total_days = [], [], [], None
    for piece in response.split("\n"):
        found = re.findall(pattern_days, piece)
        if found:
            total_days = int(found[0])
        found = re.findall(pattern_visit, piece)
        if found:
            days.append(found[0])
            if total_days is not None and int(found[0].split("-")[1]) == total_days:
                break
        found = re.findall(pattern_flight, piece)
        if found:
            flights.append(found[0])
    if not days or not flights:
        return []
    flight_days = [int(x[0]) for x in flights]
    for _, begin_city, end_city in flights:
        if not visit_cities:
            visit_cities.extend([begin_city, end_city])
        else:
            visit_cities.append(end_city)
    last_day = int(days[-1].split("-")[1])
    flight_days = [1] + flight_days + [last_day]
    return [(city, flight_days[i + 1] - flight_days[i] + 1) for i, city in enumerate(visit_cities)]

def parse_response(response: str):
    if "</think>" in response:
        parsed_plan = _parse_response_original(response.rsplit("</think>", 1)[1].strip())
        if parsed_plan:
            return parsed_plan
    return _parse_response_original(response)

def compute_example_score(cities: str, durations: str, parsed_plan: list[Any]):
    stays = [x for x in cities.split("**") if x]
    days = [int(x) for x in durations.split("**") if x]
    num_match = 0
    for i in range(min(len(stays), len(parsed_plan))):
        if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break
    return 1.0 if stays and num_match / len(stays) >= 1.0 else 0.0

def compute_score(cities: list[str], durations: list[str], responses: list[str]):
    return sum(compute_example_score(c, d, parse_response(r)) for c, d, r in zip(cities, durations, responses)) / len(responses)

def main(argv: Sequence[str] | None = None) -> None:
    try:
        from .bon_runner_trip import run_bon
    except ImportError:
        from bon_runner_trip import run_bon
    run_bon("trip", "data/trip_planning.json", argv)

if __name__ == "__main__":
    main()
