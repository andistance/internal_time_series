"""NaturalPlan Best-of-N runner backed by :mod:`scripts.transformer_bon`.

This module is intentionally kept inside NaturalPlan.  It adapts the official
NaturalPlan JSON (an object keyed by sample id) to the reusable staged
confidence-trajectory service and owns benchmark-specific correctness checks
and output files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# Running ``python NaturalPlan/evaluate_*.py`` puts NaturalPlan, rather than the
# repository root, on sys.path.  Add the root so the shared package is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.transformer_bon import (
    BestOfNConfig,
    BestOfNRequest,
    BestOfNService,
    PRMBestOfNConfig,
    PRMBestOfNService,
)
from scripts.transformer_bon.progress import DatasetProgress

AGGREGATIONS = ("product", "min", "mean", "last")
SCORE_KEYS = {name: f"{name}_score" for name in AGGREGATIONS}
PROMPT_SUFFIX = r"""

Your final answer must contain exactly one complete trip plan that follows the formatting rules below so that it can be parsed by the evaluator.

Final-answer formatting rules:
1. The final answer must include exactly one sentence containing the phrase:
   European cities for N days
   Replace N with the total number of travel days. For example: "Here is the trip plan for visiting 5 European cities for 19 days"
2. Represent the itinerary using one visit line for each city and one flight line between every two consecutive cities.
3. Every visit line must contain an inclusive day range in the following form:
   Day START-END:
   Use only ordinary ASCII hyphens (`-`) between the two day numbers. Do not use en dashes (`–`), em dashes (`—`), slashes, or words such as "to".
4. The first visit must start on day 1. The final visit must end exactly on the total number of days N.
5. Every flight line must follow this exact form:
   Day D: Fly from CITY_A to CITY_B.
   The line must:
   • begin with `Day`;

   • contain the flight day as an integer;

   • use the lowercase words `from` and `to`;

   • use the exact city names from the problem;

   • contain no alternative city names or alternative routes.

6. Present exactly one chronological itinerary as alternating visit and flight lines: begin with the first city's visit, then write one flight line followed immediately by the next city's visit, and continue until the final city.
7. Follow this structure exactly:
   Here is the trip plan for visiting K European cities for N days:
   Day 1-D1: Arriving in CITY_1 and visit CITY_1 for S1 days.
   Day D1: Fly from CITY_1 to CITY_2.
   Day D1-D2: Visit CITY_2 for S2 days.
   Day D2: Fly from CITY_2 to CITY_3.
   ...
   Day DN: Visit CITY_K for S_LAST days.
8. Use the exact spelling and capitalization of every city name. Do not abbreviate, translate, rename, or combine city names.
9. Do not use numbered lists, bullet points, tables, code fences, quotation marks, alternative plans, rejected plans, corrections, or multiple candidate itineraries in the final answer.
10. Do not include any additional line that contains a day range or a phrase matching `Day D ... from CITY_A to CITY_B` unless it is part of the single final itinerary.
11. Do not provide a second plan after the final city. The final visit line whose end day is N must be the end of the itinerary.

Output the final trip plan only after completing the reasoning.

"""


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float for ranking/metrics, otherwise ``default``."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def load_official_data(path: str) -> List[Dict[str, Any]]:
    """Load an official NaturalPlan object and preserve its string sample id."""
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("NaturalPlan input must be a JSON object keyed by sample id")
    rows: List[Dict[str, Any]] = []
    for sample_id, row in value.items():
        if not isinstance(row, dict):
            raise ValueError(f"Sample {sample_id!r} is not a JSON object")
        item = dict(row)
        item["sample_id"] = str(sample_id)
        rows.append(item)
    return rows


def _question(row: Mapping[str, Any], prompt_field: str) -> str:
    question = row.get(prompt_field)
    if not isinstance(question, str):
        raise KeyError(f"NaturalPlan row has no string field {prompt_field!r}")
    #return PROMPT_PREFIX + question
    return question + PROMPT_SUFFIX


def make_requests(
    rows: Sequence[Mapping[str, Any]],
    benchmark: str,
    prompt_field: str,
    qwen3_model: bool = True,
) -> List[BestOfNRequest]:
    requests: List[BestOfNRequest] = []
    for sample_idx, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        task_text = str(row[prompt_field])
        requests.append(
            BestOfNRequest(
                request_id=f"naturalplan:{benchmark}:{sample_id}",
                sample_id=sample_id,
                task_text=task_text,
                messages=[{"role": "user", "content": _question(row, prompt_field)}] if qwen3_model else None,
                raw_prompt=_question(row, prompt_field) if not qwen3_model else None,
                gold_answer=row.get("golden_plan"),
                metadata={
                    "benchmark": f"naturalplan_{benchmark}",
                    "sample_idx": sample_idx,
                    "sample_id": sample_id,
                    "difficulty": row.get("num_people", row.get("num_cities")),
                },
            )
        )
    return requests


def _calendar_correct(row: Mapping[str, Any], text: str) -> bool:
    from NaturalPlan.evaluate_calendar_scheduling import _parse_response

    return _parse_response(text) == _parse_response(str(row["golden_plan"]))


def _meeting_correct(row: Mapping[str, Any], text: str) -> bool:
    from NaturalPlan.evaluate_meeting_planning import parse_text_plan, process_constraints, validator_from_text

    constraints = row["constraints"]
    start_location, initial_time = constraints[0]
    processed = process_constraints(constraints[1:])
    predicted_plan = text if isinstance(text, list) else parse_text_plan(text)
    score = validator_from_text(
        predicted_plan, processed, start_location, initial_time, row["dist_matrix"]
    )
    golden_plan = row["golden_plan"]
    if isinstance(golden_plan, str):
        golden_plan = parse_text_plan(golden_plan)
    golden_score = validator_from_text(
        golden_plan, processed, start_location, initial_time, row["dist_matrix"]
    )
    return score == golden_score


def _trip_correct(row: Mapping[str, Any], text: str) -> bool:
    from NaturalPlan.evaluate_trip_planning import compute_example_score, parse_response

    return bool(
        compute_example_score(
            str(row["cities"]), str(row["durations"]), parse_response(text)
        )
    )


def correctness_fn(benchmark: str) -> Callable[[Mapping[str, Any], str], bool]:
    return {
        "calendar": _calendar_correct,
        "meeting": _meeting_correct,
        "trip": _trip_correct,
    }[benchmark]


def _read_jsonl(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not path.exists():
        return result
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if not any(item.strip() for item in lines[line_no:]):
                break
            raise
        key = (str(record.get("sample_id", record.get("unique_id", ""))), int(record["sequence_idx"]))
        result[key] = record
    return result

def _rewrite_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
    temporary.replace(path)


def _write_final_record_pair(
    output_dir: Path,
    result: Any,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    correct: Callable[[Mapping[str, Any], str], bool],
    result_suffix: str = "",
) -> None:
    summary_path = output_dir / f"summary{result_suffix}.jsonl"
    steps_path = output_dir / f"steps{result_suffix}.jsonl"
    summaries = _read_jsonl(summary_path)
    steps = _read_jsonl(steps_path)
    sample_id = str(result.request.sample_id)
    for candidate in result.candidates:
        summary = dict(candidate.summary_record)
        step_record = dict(candidate.steps_record)
        is_correct = bool(correct(rows_by_id[sample_id], candidate.text))
        for record in (summary, step_record):
            record["sample_id"] = sample_id
            record["sequence_idx"] = int(candidate.sequence_idx)
            record["is_correct"] = is_correct
            record["request_id"] = result.request.request_id
        key = (sample_id, int(candidate.sequence_idx))
        summaries[key] = summary
        steps[key] = step_record
    _rewrite_jsonl(summary_path, (summaries[key] for key in sorted(summaries)))
    _rewrite_jsonl(steps_path, (steps[key] for key in sorted(steps)))

def _select(candidates: Sequence[Mapping[str, Any]], n: int, score_key: str) -> List[Mapping[str, Any]]:
    return list(candidates[:n])


def evaluate_nested_bon(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    ns: Sequence[int],
    n_trials: int,
    eval_seed: int,
    result_suffix: str = "",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    summaries = _read_jsonl(output_dir / f"summary{result_suffix}.jsonl")
    by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for record in summaries.values():
        by_sample.setdefault(str(record["sample_id"]), []).append(record)
    for values in by_sample.values():
        values.sort(key=lambda item: int(item["sequence_idx"]))

    ns = sorted(set(int(n) for n in ns))
    metrics: Dict[int, Dict[str, List[float]]] = {
        n: {method: [] for method in AGGREGATIONS} for n in ns
    }
    lengths: Dict[int, Dict[str, List[float]]] = {
        n: {method: [] for method in AGGREGATIONS} for n in ns
    }
    details: Dict[int, List[Dict[str, Any]]] = {n: [] for n in ns}

    for trial_idx in range(n_trials):
        rng = random.Random(eval_seed + trial_idx * 1000003)
        orders: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            values = list(by_sample.get(str(row["sample_id"]), []))
            rng.shuffle(values)
            orders[str(row["sample_id"])] = values
        for n in ns:
            correct_counts = {method: 0 for method in AGGREGATIONS}
            length_values = {method: [] for method in AGGREGATIONS}
            for row in rows:
                sample_id = str(row["sample_id"])
                chosen = orders[sample_id][:n]
                detail: Dict[str, Any] = {
                    "trial_idx": trial_idx,
                    "n": n,
                    "sample_id": sample_id,
                    "selected_sequence_indices": json.dumps(
                        [int(x["sequence_idx"]) for x in chosen]
                    ),
                    "selected_candidates": json.dumps([
                        {
                            "sequence_idx": int(x["sequence_idx"]),
                            "product_score": x.get("product_score"),
                            "min_score": x.get("min_score"),
                            "mean_score": x.get("mean_score"),
                            "last_score": x.get("last_score"),
                            "is_correct": bool(x.get("is_correct")),
                        } for x in chosen
                    ], ensure_ascii=False),
                }
                for method in AGGREGATIONS:
                    key = SCORE_KEYS[method]
                    ranked = sorted(
                        chosen,
                        key=lambda x: (-_safe_float(x.get(key), float("-inf")), int(x["sequence_idx"])),
                    )
                    pick = ranked[0] if ranked else None
                    detail[f"{method}_choice_sequence_idx"] = (
                        int(pick["sequence_idx"]) if pick else None
                    )
                    detail[f"{method}_choice_score"] = _safe_float(pick.get(key)) if pick else None
                    detail[f"{method}_choice_is_correct"] = bool(pick and pick.get("is_correct"))
                    detail[f"{method}_choice_num_generated_tokens"] = (
                        int(pick.get("num_generated_tokens", 0)) if pick else 0
                    )
                    if pick:
                        correct_counts[method] += int(bool(pick.get("is_correct")))
                        length_values[method].append(_safe_float(pick.get("num_generated_tokens")))
                details[n].append(detail)
            denominator = len(rows) or 1
            for method in AGGREGATIONS:
                metrics[n][method].append(correct_counts[method] / denominator)
                lengths[n][method].append(float(np.mean(length_values[method])) if length_values[method] else 0.0)

    rows_out: List[Dict[str, Any]] = []
    for n in ns:
        row: Dict[str, Any] = {"n": n, "n_trials": n_trials, "num_samples": len(rows)}
        for method in AGGREGATIONS:
            row[f"{method}_accuracy"] = float(np.mean(metrics[n][method]))
            row[f"{method}_std"] = float(np.std(metrics[n][method]))
            row[f"{method}_mean_output_length"] = float(np.mean(lengths[n][method]))
        rows_out.append(row)
        with (output_dir / f"details_{n}{result_suffix}.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(details[n][0].keys()) if details[n] else ["n"])
            writer.writeheader()
            writer.writerows(details[n])

    with (output_dir / f"metrics{result_suffix}.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows_out[0].keys()) if rows_out else ["n"])
        writer.writeheader()
        writer.writerows(rows_out)

    colors = {"product": "tab:blue", "min": "tab:orange", "mean": "tab:green", "last": "tab:red"}
    figure, (accuracy_axis, length_axis) = plt.subplots(1, 2, figsize=(13, 5))
    x = [row["n"] for row in rows_out]
    for method in AGGREGATIONS:
        accuracy_axis.plot(x, [row[f"{method}_accuracy"] for row in rows_out], marker="o", label=method, color=colors[method])
        length_axis.plot(x, [row[f"{method}_mean_output_length"] for row in rows_out], marker="o", label=method, color=colors[method])
    accuracy_axis.set(xlabel="n", ylabel="Mean test accuracy", title="Best-of-N accuracy by phase-score aggregation")
    length_axis.set(xlabel="n", ylabel="Mean output length (generated tokens)", title="Mean selected output length by aggregation")
    for axis in (accuracy_axis, length_axis):
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"metrics{result_suffix}.png", dpi=200)
    plt.close(figure)


def run_bon(
    benchmark: str,
    default_data_path: str,
    argv: Optional[Sequence[str]] = None,
) -> None:
    parser = argparse.ArgumentParser(description=f"NaturalPlan {benchmark} Transformer/PRM Best-of-N")
    parser.add_argument("--input_json", "--data_path", default=default_data_path)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--training_metrics_json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--result_suffix", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--score_model", "--scoring_model",
        choices=["transformer", "prm"], default="transformer",
        help="Stage-three scorer: the trained confidence Transformer or a Hugging Face PRM.",
    )
    parser.add_argument("--enable_thinking", type=parse_bool, default=True)
    parser.add_argument("--qwen3_model", "--use_qwen3", "--qwen3", type=parse_bool, default=True)
    parser.add_argument("--transformer_type", choices=["auto", "bidirectional", "causal"], default="auto")
    parser.add_argument("--value_device", default="auto")
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument("--scoring_batch_size", type=int, default=8)
    # Hugging Face PRM options.  The generation and entropy options above are
    # intentionally shared with the Transformer mode.
    parser.add_argument("--prm_model_path", default=None)
    parser.add_argument(
        "--prm_name",
        default="qwen2.5-math-prm-7b",
        help="Registered PRM backend name (qwen2.5-math-prm-7b, qwen-2.5-7b-fover-prm-2026, or versa-prm)",
    )
    parser.add_argument("--prm_device", default="auto")
    parser.add_argument("--prm_dtype", default=None)
    parser.add_argument("--prm_max_length", type=int, default=8192)
    parser.add_argument("--prm_positive_token", "--positive_token", dest="prm_positive_token", default=None)
    parser.add_argument("--prm_phase_marker", "--phase_marker", dest="prm_phase_marker", default="")
    parser.add_argument("--prm_phase_separator", "--phase_separator", dest="prm_phase_separator", default="\n\n")
    parser.add_argument("--prm_gpus", default=None)
    parser.add_argument("--prm_tensor_parallel_size", "--prm_tp_size", dest="prm_tensor_parallel_size", type=int, default=None)
    parser.add_argument("--batch_size", "--candidate_batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--logprobs", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--num_segments", "--K", type=int, default=8)
    parser.add_argument("--entropy_threshold", type=float, default=1.2)
    parser.add_argument("--phase_partition_algorithm", choices=["original", "adaptive"], default="original")
    parser.add_argument("--high_entropy_ratio", type=float, default=None, help="Adaptive: top entropy token fraction in (0, 1], e.g. 0.05")
    parser.add_argument("--target_phase_tokens", type=float, default=None, help="Adaptive: desired rollout-average phase length in tokens")
    parser.add_argument("--entropy_chunk_size", type=int, default=16)
    parser.add_argument("--entropy_model_path")
    parser.add_argument("--entropy_device_map", default="auto")
    parser.add_argument("--no_save_token_entropies", action="store_true")
    parser.add_argument("--BoN_N", "--N", dest="N", type=int, required=True)
    parser.add_argument("--ns", type=int, nargs="+", required=True)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num_input_parts", type=int, default=1)
    parser.add_argument("--input_part_id", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--staging_flush_records", type=int, default=1)
    parser.add_argument("--vllm_cache_mode", choices=["auto", "isolated", "shared"], default="auto")
    parser.add_argument("--vllm_cache_root")
    parser.add_argument("--legacy_script_path")
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--performance_profile", choices=["throughput", "balanced", "memory"], default="throughput")
    parser.add_argument("--model_transition", choices=["auto", "reload", "cpu_swap"], default="auto")
    parser.add_argument("--value_residency", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--generation_max_active_jobs", type=int)
    parser.add_argument("--generation_worker_startup_timeout", type=float, default=600.0)
    parser.add_argument("--prompt_field", choices=["prompt_0shot", "prompt_5shot"], default="prompt_0shot")
    args = parser.parse_args(argv)
    if args.N < 1 or args.batch_size < 1 or args.max_tokens < 1 or args.max_model_len < 1:
        parser.error("N, batch_size, max_tokens and max_model_len must be positive")
    if args.scoring_batch_size < 1 or args.prm_max_length < 1:
        parser.error("scoring_batch_size and prm_max_length must be positive")
    if args.score_model == "transformer" and not args.checkpoint_path:
        parser.error("--checkpoint_path is required when --score_model=transformer")
    if args.score_model == "prm" and not args.prm_model_path:
        parser.error("--prm_model_path is required when --score_model=prm")
    if args.prm_tensor_parallel_size is not None and args.prm_tensor_parallel_size < 1:
        parser.error("--prm_tensor_parallel_size must be positive")
    if any(n < 1 or n > args.N for n in args.ns):
        parser.error("every --ns value must satisfy 1 <= n <= --BoN_N")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.num_input_parts < 1 or not (0 <= args.input_part_id < args.num_input_parts):
        parser.error("invalid input partition arguments")
    input_path = Path(args.input_json).expanduser()
    if not input_path.is_absolute() and not input_path.exists():
        candidate = Path(__file__).resolve().parent / input_path
        if candidate.exists():
            input_path = candidate
    rows = load_official_data(str(input_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    base, remainder = divmod(len(rows), args.num_input_parts)
    start = args.input_part_id * base + min(args.input_part_id, remainder)
    end = start + base + (1 if args.input_part_id < remainder else 0)
    rows = rows[start:end]
    requests = make_requests(rows, benchmark, args.prompt_field, args.qwen3_model)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_suffix = str(args.result_suffix or "")
    if args.overwrite:
        for name in ("summary.jsonl", "steps.jsonl", "metrics.csv", "metrics.png"):
            path = output_dir / f"{Path(name).stem}{result_suffix}{Path(name).suffix}"
            if path.exists():
                path.unlink()
        for path in output_dir.glob("details_*.csv"):
            if result_suffix and not path.name.endswith(f"{result_suffix}.csv"):
                continue
            path.unlink()
        staging_dir = output_dir / ".staging"
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir)
    elif not args.resume and any(
        (output_dir / f"{Path(name).stem}{result_suffix}{Path(name).suffix}").exists()
        for name in ("summary.jsonl", "steps.jsonl")
    ):
        parser.error("output already exists; use --resume or --overwrite")

    config_class = PRMBestOfNConfig if args.score_model == "prm" else BestOfNConfig
    checkpoint_path = args.checkpoint_path or args.prm_model_path or ""
    config_kwargs = dict(
        model_path=args.model_path,
        checkpoint_path=checkpoint_path,
        n=args.N,
        output_aggregation="product",
        chat_template="qwen3" if args.qwen3_model else "raw",
        enable_thinking=args.enable_thinking if args.qwen3_model else None,
        training_metrics_json=args.training_metrics_json,
        transformer_type=args.transformer_type,
        value_device=args.value_device,
        fold_index=args.fold_index,
        scoring_batch_size=args.scoring_batch_size,
        candidate_batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=args.logprobs,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        num_segments=args.num_segments,
        entropy_threshold=args.entropy_threshold,
        phase_partition_algorithm=getattr(args, "phase_partition_algorithm", "original"),
        high_entropy_ratio=getattr(args, "high_entropy_ratio", None),
        target_phase_tokens=getattr(args, "target_phase_tokens", None),
        entropy_chunk_size=args.entropy_chunk_size,
        entropy_model_path=args.entropy_model_path,
        entropy_device_map=args.entropy_device_map,
        save_token_entropies=not args.no_save_token_entropies,
        seed=args.seed,
        gpu_groups=args.gpus,
        performance_profile=args.performance_profile,
        model_transition=args.model_transition,
        value_residency=args.value_residency,
        generation_max_active_jobs=args.generation_max_active_jobs,
        staging_flush_records=args.staging_flush_records,
        vllm_cache_mode=args.vllm_cache_mode,
        vllm_cache_root=args.vllm_cache_root,
        legacy_script_path=args.legacy_script_path,
        vllm_enforce_eager=args.vllm_enforce_eager,
        generation_worker_startup_timeout=args.generation_worker_startup_timeout,
    )
    if args.score_model == "prm":
        config_kwargs.update({
            "prm_model_path": args.prm_model_path,
            "prm_name": args.prm_name,
            "prm_device": args.prm_device,
            "prm_dtype": args.prm_dtype or args.dtype,
            "prm_max_length": args.prm_max_length,
            "prm_positive_token": args.prm_positive_token,
            "prm_phase_marker": args.prm_phase_marker,
            "prm_phase_separator": args.prm_phase_separator,
            "prm_trust_remote_code": args.trust_remote_code,
            # Use the same GPU groups/TP size for all three stages unless a
            # PRM-specific override is supplied.
            "prm_gpus": args.prm_gpus if args.prm_gpus is not None else args.gpus,
            "prm_tensor_parallel_size": (
                args.prm_tensor_parallel_size
                if args.prm_tensor_parallel_size is not None
                else args.tensor_parallel_size
            ),
        })
    config = config_class(**config_kwargs)
    correct = correctness_fn(benchmark)
    rows_by_id = {str(row["sample_id"]): row for row in rows}
    summary_path = output_dir / f"summary{result_suffix}.jsonl"
    completed_ids = set()
    if args.resume:
        summary_existing = _read_jsonl(summary_path)
        steps_existing = _read_jsonl(output_dir / f"steps{result_suffix}.jsonl")
        for row in rows:
            sample_id = str(row["sample_id"])
            keys = {(sample_id, i) for i in range(args.N)}
            if keys.issubset(summary_existing) and keys.issubset(steps_existing):
                if any(summary_existing[key].get("num_generated_tokens", 0) > args.max_tokens for key in keys):
                    continue
                completed_ids.add(f"naturalplan:{benchmark}:{sample_id}")

    bars: Dict[str, DatasetProgress] = {}
    def on_stage_start(stage: str, total: int) -> None:
        if stage in bars:
            bars[stage].close()
        label = "PRM scoring" if args.score_model == "prm" and stage == "scoring" else stage
        bars[stage] = DatasetProgress(total, description=f"{label:>10} rollouts", unit="rollout")
    def on_rollout_complete(stage: str) -> None:
        if stage in bars:
            bars[stage].update()

    try:
        service_class = PRMBestOfNService if args.score_model == "prm" else BestOfNService
        with service_class(config) as service:
            for result in service.generate_and_select_many(
                requests,
                staging_dir=output_dir / ".staging",
                resume=args.resume,
                overwrite=args.overwrite,
                skip_request_ids=completed_ids,
                on_stage_start=on_stage_start,
                on_rollout_complete=on_rollout_complete,
            ):
                _write_final_record_pair(
                    output_dir, result, rows_by_id, correct, result_suffix
                )
    finally:
        for bar in bars.values():
            bar.close()
    evaluate_nested_bon(
        output_dir, rows, args.ns, args.n_trials, args.eval_seed, result_suffix
    )
    print(f"[DONE] {benchmark}: {len(rows)} samples; outputs in {output_dir}")

