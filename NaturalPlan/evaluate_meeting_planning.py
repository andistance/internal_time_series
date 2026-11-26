"""Official NaturalPlan meeting evaluator and Transformer/PRM BoN entry point."""
from __future__ import annotations
import collections
import datetime
from typing import Any, Sequence

MAX_CONSTRAINTS = 10
NUM_SAMPLES_PER_CONSTRAINTS = 100

def convert_to_time_obj(time_str: str):
    return datetime.datetime.strptime(time_str, "%I:%M%p")

def process_constraints(data: tuple[Any, ...]):
    constraints = collections.defaultdict(dict)
    for name, location, times, meeting_time in data:
        constraints[name]["location"] = location
        start_time, end_time = [convert_to_time_obj(x.strip()) for x in times.split("to")]
        constraints[name]["start_time"] = start_time
        constraints[name]["end_time"] = end_time
        constraints[name]["meeting_time"] = meeting_time
    return constraints

def validator_from_text(plan, processed_constraints, start_location, initial_time, dist_matrix):
    met_with, score = {}, 0
    cur_location, cur_time = start_location, convert_to_time_obj(initial_time)
    for step in plan:
        try:
            if step.startswith("You start"):
                continue
            if step.startswith("You travel"):
                destination = step.split("travel to ", 1)[1].split(" in", 1)[0].strip()
                cur_time += datetime.timedelta(minutes=dist_matrix[cur_location][destination])
                cur_location = destination
            elif step.startswith("You wait"):
                end_time = convert_to_time_obj(step.split("wait until ", 1)[1].split(".", 1)[0].strip())
                if end_time <= cur_time:
                    raise ValueError("Cannot go backwards in time")
                cur_time = end_time
            elif step.startswith("You meet"):
                person = step.split("meet ", 1)[1].split(" for", 1)[0].strip()
                if person in met_with:
                    raise ValueError("Person already met")
                met_with[person] = 1
                new_time = cur_time + datetime.timedelta(minutes=processed_constraints[person]["meeting_time"])
                constraint = processed_constraints[person]
                if cur_location == constraint["location"] and cur_time >= constraint["start_time"] and new_time <= constraint["end_time"]:
                    score += 1
                    cur_time = new_time
                else:
                    raise ValueError("Invalid meeting time or location")
            else:
                raise ValueError("Unknown plan format")
        except (ValueError, KeyError, IndexError):
            break
    return score

def validator_from_dict(plan, processed_constraints, start_location, initial_time, dist_matrix):
    met_with, score = {}, 0
    cur_location, cur_time = start_location, convert_to_time_obj(initial_time)
    for step in plan[1:]:
        try:
            location, raw_start_time = step["location"], step["start_time"]
            if location and location != cur_location:
                cur_time += datetime.timedelta(minutes=dist_matrix[cur_location][location])
                cur_location = location
            start_time = convert_to_time_obj(raw_start_time)
            if start_time < cur_time:
                raise ValueError("Start time too early")
            cur_time = start_time
            person = step["person_name"]
            if person not in processed_constraints:
                continue
            if person in met_with:
                raise ValueError("Person already met")
            met_with[person] = 1
            constraint = processed_constraints[person]
            new_time = cur_time + datetime.timedelta(minutes=constraint["meeting_time"])
            if cur_location == constraint["location"] and cur_time >= constraint["start_time"] and new_time <= constraint["end_time"]:
                score += 1
                cur_time = new_time
            else:
                raise ValueError("Invalid meeting time or location")
        except (ValueError, KeyError):
            break
    return score

def parse_text_plan(plan: str):
    full_plan = plan
    if "</think>" in plan:
        plan = plan.rsplit("</think>", 1)[1].strip()
        if "SOLUTION:" in plan:
            plan = plan[plan.find("SOLUTION:") + len("SOLUTION:"):].strip()
        parsed_plan = [step.strip() for step in plan.split(".") if step.strip()]
        if parsed_plan:
            return parsed_plan

    plan = full_plan
    if "SOLUTION:" in plan:
        plan = plan[plan.find("SOLUTION:") + len("SOLUTION:"):].strip()
    return [step.strip() for step in plan.split(".") if step.strip()]

def main(argv: Sequence[str] | None = None) -> None:
    try:
        from .bon_runner_meeting import run_bon
    except ImportError:
        from bon_runner_meeting import run_bon
    run_bon("meeting", "data/meeting_planning.json", argv)

if __name__ == "__main__":
    main()
