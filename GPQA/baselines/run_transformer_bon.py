from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.transformer_bon import (
    BestOfNConfig,
    BestOfNService,
    DatasetProgress,
    JsonlResultStore,
)

try:
    from .gpqa_core import SUPPORTED_PROMPT_TYPES, compute_metrics, parse_sampled_answer
    from .gpqa_dataset import GPQAProblem, load_gpqa_dataset
    from .transformer_bon_adapter import GPQATransformerBoNAdapter
except ImportError:  # Direct execution from GPQA/baselines.
    from gpqa_core import SUPPORTED_PROMPT_TYPES, compute_metrics, parse_sampled_answer
    from gpqa_dataset import GPQAProblem, load_gpqa_dataset
    from transformer_bon_adapter import GPQATransformerBoNAdapter


OFFICIAL_COLUMNS = (
    "Question id",
    "Question",
    "Correct answer",
    "Model answer",
    "Correct",
    "Model response",
)
LINK_COLUMNS = (
    "Request id",
    "Selected candidate key",
    "Selected index",
    "Selected score",
    "N",
    "Aggregation",
    "Parsed answer",
    "Refusal",
)
AGGREGATIONS = ("product", "min", "mean", "last")
SCORE_KEYS = {
    "product": "product_score",
    "min": "min_score",
    "mean": "mean_score",
    "last": "last_score",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def _safe_model_name(model_path: str) -> str:
    name = Path(model_path.rstrip("/\\")).name or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _json_score(score: Any) -> float | None:
    if score is None:
        return None
    numeric = float(score)
    return numeric if math.isfinite(numeric) else None


def _resolve_ns(max_n: int, requested: list[int] | None) -> list[int]:
    if requested is None:
        suggested = [1, 2, 4, 8, 16, 32, max_n]
        return sorted({value for value in suggested if 1 <= value <= max_n})
    ns = [int(value) for value in requested]
    if not ns:
        raise ValueError("No valid Best-of-N n values")
    if any(value < 1 or value > max_n for value in ns):
        raise ValueError(
            f"Every --ns value must satisfy 1 <= n <= N={max_n}"
        )
    return ns


def evaluate_selected_response(
    problem: GPQAProblem,
    *,
    response: str,
    request_id: str,
    candidate_key: str,
    selected_index: int,
    selected_score: float | None,
    n: int,
    aggregation: str,
) -> dict[str, Any]:
    parsed_answer = parse_sampled_answer(response)
    is_refusal = parsed_answer is None
    is_correct = parsed_answer == problem.answer_label
    return {
        "sample_id": problem.sample_id,
        "request_id": request_id,
        "candidate_key": candidate_key,
        "selected_index": selected_index,
        "selected_score": _json_score(selected_score),
        "n": n,
        "aggregation": aggregation,
        "question": problem.question_block,
        "gold_label": problem.answer_label,
        "gold_choice": problem.label_to_choice.get(
            problem.answer_label, problem.answer_label
        ),
        "parsed_answer": parsed_answer,
        "model_choice": (
            problem.label_to_choice.get(parsed_answer, parsed_answer)
            if parsed_answer is not None
            else "Couldn't find an answer choice!"
        ),
        "is_correct": is_correct,
        "is_refusal": is_refusal,
        "response": response,
    }


def _annotate_result_records(result: Any) -> None:
    dataset_source = result.request.metadata.get("dataset_source")
    prompt_type = result.request.metadata.get("prompt_type")
    for candidate in result.candidates:
        for record in (candidate.summary_record, candidate.steps_record):
            record["dataset_source"] = dataset_source
            record["prompt_type"] = prompt_type


def _evaluation_from_result(problem: GPQAProblem, result: Any) -> dict[str, Any]:
    """Backward-compatible helper for the canonical full-N product result."""

    return evaluate_selected_response(
        problem,
        response=result.selected_text,
        request_id=result.request.request_id,
        candidate_key=result.selected_candidate_key,
        selected_index=result.selected_index,
        selected_score=result.selected_score,
        n=result.n,
        aggregation="product",
    )


def _evaluation_from_record(
    problem: GPQAProblem,
    request_id: str,
    record: Mapping[str, Any],
    *,
    n: int,
    aggregation: str,
) -> dict[str, Any]:
    """Backward-compatible helper for evaluating one persisted candidate."""

    return evaluate_selected_response(
        problem,
        response=str(record.get("full_output") or ""),
        request_id=request_id,
        candidate_key=str(record["candidate_key"]),
        selected_index=int(record["sequence_idx"]),
        selected_score=record.get(SCORE_KEYS[aggregation]),
        n=n,
        aggregation=aggregation,
    )


def _evaluation_without_selection(
    problem: GPQAProblem,
    request_id: str,
    *,
    n: int,
    aggregation: str,
) -> dict[str, Any]:
    return {
        "sample_id": problem.sample_id,
        "request_id": request_id,
        "candidate_key": None,
        "selected_index": None,
        "selected_score": None,
        "n": n,
        "aggregation": aggregation,
        "question": problem.question_block,
        "gold_label": problem.answer_label,
        "gold_choice": problem.label_to_choice.get(
            problem.answer_label, problem.answer_label
        ),
        "parsed_answer": None,
        "model_choice": "Couldn't find an answer choice!",
        "is_correct": False,
        "is_refusal": True,
        "response": "",
    }


def _candidate_view(candidate: Any) -> dict[str, Any]:
    """Normalize a live candidate or a persisted summary record."""

    if isinstance(candidate, Mapping):
        record = candidate
        text = record.get("full_output")
    else:
        record = candidate.summary_record
        text = candidate.text
    sequence_idx = record.get("sequence_idx")
    if sequence_idx is None:
        sequence_idx = candidate.sequence_idx
    valid_for_selection = record.get("valid_for_selection")
    if valid_for_selection is None:
        valid_for_selection = _json_score(record.get("product_score")) is not None
    return {
        "candidate_key": str(record.get("candidate_key", "")),
        "sequence_idx": int(sequence_idx),
        "full_output": str(text or ""),
        "valid_for_selection": bool(valid_for_selection),
        "product_score": record.get("product_score"),
        "min_score": record.get("min_score"),
        "mean_score": record.get("mean_score"),
        "last_score": record.get("last_score"),
    }


def _evaluate_nested_candidates(
    problem: GPQAProblem,
    request_id: str,
    candidates: list[Any],
    *,
    ns: list[int],
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    """Evaluate all requested n/aggregation pairs on one nested candidate pool."""

    views = [_candidate_view(candidate) for candidate in candidates]
    views.sort(key=lambda candidate: candidate["sequence_idx"])
    rng = random.Random(int(shuffle_seed))
    rng.shuffle(views)

    rows: list[dict[str, Any]] = []
    for n in ns:
        sampled = views[:n]
        sampled_sequence_indices = json.dumps(
            [candidate["sequence_idx"] for candidate in sampled],
            ensure_ascii=False,
        )
        for aggregation in AGGREGATIONS:
            score_key = SCORE_KEYS[aggregation]
            valid = []
            for candidate in sampled:
                if not candidate["valid_for_selection"]:
                    continue
                score = _json_score(candidate.get(score_key))
                if score is not None:
                    valid.append((candidate, score))
            if not valid:
                row = _evaluation_without_selection(
                    problem,
                    request_id,
                    n=n,
                    aggregation=aggregation,
                )
                row["sampled_sequence_indices"] = sampled_sequence_indices
                rows.append(row)
                continue

            # Python's max is stable, matching the original evaluator's
            # first-in-nested-order tie behavior.
            selected, selected_score = max(valid, key=lambda item: item[1])
            row = evaluate_selected_response(
                problem,
                response=selected["full_output"],
                request_id=request_id,
                candidate_key=selected["candidate_key"],
                selected_index=selected["sequence_idx"],
                selected_score=selected_score,
                n=n,
                aggregation=aggregation,
            )
            row["sampled_sequence_indices"] = sampled_sequence_indices
            rows.append(row)
    return rows


def write_evaluation_outputs(
    output_dir: str | Path,
    model_path: str,
    rows: list[dict[str, Any]],
    metrics: Mapping[str, Any],
) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"eval_results_{_safe_model_name(model_path)}.csv"
    metrics_path = directory / "metrics.json"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*OFFICIAL_COLUMNS, *LINK_COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Question id": row["sample_id"],
                    "Question": row["question"],
                    "Correct answer": row["gold_choice"],
                    "Model answer": row["model_choice"],
                    "Correct": row["is_correct"],
                    "Model response": row["response"],
                    "Request id": row["request_id"],
                    "Selected candidate key": row["candidate_key"],
                    "Selected index": row["selected_index"],
                    "Selected score": row["selected_score"],
                    "N": row["n"],
                    "Aggregation": row["aggregation"],
                    "Parsed answer": row["parsed_answer"],
                    "Refusal": row["is_refusal"],
                }
            )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(dict(metrics), file, ensure_ascii=False, indent=2)
    return csv_path, metrics_path


def compute_metrics_by_n(
    rows: list[dict[str, Any]],
    ns: list[int],
) -> dict[str, dict[str, dict[str, int | float]]]:
    metrics_by_n: dict[str, dict[str, dict[str, int | float]]] = {}
    for n in ns:
        n_key = str(n)
        metrics_by_n[n_key] = {}
        for aggregation in AGGREGATIONS:
            selected_rows = [
                row
                for row in rows
                if row["n"] == n and row["aggregation"] == aggregation
            ]
            metrics_by_n[n_key][aggregation] = compute_metrics(selected_rows)
    return metrics_by_n


def write_aggregate_outputs(
    output_dir: str | Path,
    metrics_by_n: Mapping[str, Mapping[str, Mapping[str, int | float]]],
    evaluation_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write multi-n/multi-aggregation tables and their accuracy plot."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = directory / "metrics.csv"
    metrics_png_path = directory / "metrics.png"
    normalized_metrics_by_n = {
        str(n_key): aggregation_metrics
        for n_key, aggregation_metrics in metrics_by_n.items()
    }

    fieldnames = [
        "n",
        "aggregation",
        "total",
        "correct",
        "refusals",
        "accuracy",
        "refusal_fraction",
    ]
    rows: list[dict[str, Any]] = []
    for n_key, aggregation_metrics in normalized_metrics_by_n.items():
        for aggregation in AGGREGATIONS:
            values = aggregation_metrics[aggregation]
            rows.append(
                {
                    "n": int(n_key),
                    "aggregation": aggregation,
                    **values,
                }
            )
    aggregation_order = {name: index for index, name in enumerate(AGGREGATIONS)}
    rows.sort(
        key=lambda row: (
            int(row["n"]),
            aggregation_order[str(row["aggregation"])],
        )
    )
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if evaluation_rows is not None:
        for n_key in normalized_metrics_by_n:
            n = int(n_key)
            per_request: dict[str, dict[str, Any]] = {}
            for row in evaluation_rows:
                if row["n"] != n:
                    continue
                request_id = str(row["request_id"])
                detail = per_request.setdefault(
                    request_id,
                    {
                        "sample_id": row["sample_id"],
                        "request_id": request_id,
                        "n": n,
                        "question": row["question"],
                        "gold_label": row["gold_label"],
                        "gold_choice": row["gold_choice"],
                        "sampled_sequence_indices": row.get(
                            "sampled_sequence_indices", "[]"
                        ),
                    },
                )
                aggregation = str(row["aggregation"])
                prefix = f"{aggregation}_"
                detail[f"{prefix}candidate_key"] = row["candidate_key"]
                detail[f"{prefix}selected_index"] = row["selected_index"]
                detail[f"{prefix}selected_score"] = row["selected_score"]
                detail[f"{prefix}parsed_answer"] = row["parsed_answer"]
                detail[f"{prefix}model_choice"] = row["model_choice"]
                detail[f"{prefix}correct"] = row["is_correct"]
                detail[f"{prefix}refusal"] = row["is_refusal"]

            detail_rows = list(per_request.values())
            detail_fields = [
                "sample_id",
                "request_id",
                "n",
                "question",
                "gold_label",
                "gold_choice",
                "sampled_sequence_indices",
            ]
            for aggregation in AGGREGATIONS:
                detail_fields.extend(
                    [
                        f"{aggregation}_candidate_key",
                        f"{aggregation}_selected_index",
                        f"{aggregation}_selected_score",
                        f"{aggregation}_parsed_answer",
                        f"{aggregation}_model_choice",
                        f"{aggregation}_correct",
                        f"{aggregation}_refusal",
                    ]
                )
            detail_path = directory / f"details_{n}.csv"
            with detail_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=detail_fields)
                writer.writeheader()
                writer.writerows(detail_rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, (accuracy_axis, refusal_axis) = plt.subplots(
            1,
            2,
            figsize=(12, 4.5),
        )
        colors = {
            "product": "tab:blue",
            "min": "tab:orange",
            "mean": "tab:green",
            "last": "tab:red",
        }
        ns = sorted(int(value) for value in normalized_metrics_by_n)
        for aggregation in AGGREGATIONS:
            accuracy_axis.plot(
                ns,
                [
                    float(
                        normalized_metrics_by_n[str(n)][aggregation]["accuracy"]
                    )
                    for n in ns
                ],
                marker="o",
                label=aggregation,
                color=colors[aggregation],
            )
            refusal_axis.plot(
                ns,
                [
                    float(
                        normalized_metrics_by_n[str(n)][aggregation][
                            "refusal_fraction"
                        ]
                    )
                    for n in ns
                ],
                marker="o",
                label=aggregation,
                color=colors[aggregation],
            )
        accuracy_axis.set_xlabel("n")
        accuracy_axis.set_ylabel("Accuracy")
        accuracy_axis.set_title("GPQA Best-of-N accuracy")
        accuracy_axis.grid(True, alpha=0.3)
        accuracy_axis.legend()
        refusal_axis.set_xlabel("n")
        refusal_axis.set_ylabel("Refusal fraction")
        refusal_axis.set_title("GPQA refusal fraction")
        refusal_axis.grid(True, alpha=0.3)
        refusal_axis.legend()
        figure.tight_layout()
        figure.savefig(metrics_png_path, dpi=200)
        plt.close(figure)
    except ImportError:
        # Keep the tabular metrics available when matplotlib is not installed.
        if metrics_png_path.exists():
            metrics_png_path.unlink()

    return metrics_csv_path, metrics_png_path


def _remove_aggregate_outputs(output_dir: str | Path) -> None:
    """Remove aggregate artifacts when an explicit overwrite is requested."""

    directory = Path(output_dir).expanduser().resolve()
    for name in ("metrics.csv", "metrics.png"):
        path = directory / name
        if path.exists():
            path.unlink()
    for path in directory.glob("details_*.csv"):
        path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPQA Diamond evaluation with confidence-trajectory Transformer BoN"
    )
    data = parser.add_argument_group("GPQA dataset and prompt")
    data.add_argument(
        "--dataset_source", choices=["online", "local"], default="online"
    )
    data.add_argument("--local_dataset_path", default=None)
    data.add_argument("--dataset_name", default="Idavidrein/gpqa")
    data.add_argument("--dataset_config", default="gpqa_diamond")
    data.add_argument("--dataset_split", default="test")
    data.add_argument("--hf_token", default=None)
    data.add_argument("--choice_seed", type=int, default=0)
    data.add_argument("--max_examples", type=int, default=None)
    data.add_argument(
        "--prompt_type", choices=SUPPORTED_PROMPT_TYPES, default="chain_of_thought"
    )

    bon = parser.add_argument_group("Transformer Best-of-N")
    bon.add_argument("--bon_model_path", required=True)
    bon.add_argument("--bon_checkpoint_path", required=True)
    bon.add_argument("--bon_training_metrics_json", default=None)
    bon.add_argument("--bon_output_dir", required=True)
    bon.add_argument(
        "--bon_chat_template",
        choices=["qwen3", "deepseek_r1_distill_qwen", "raw"],
        default="qwen3",
    )
    bon.add_argument("--bon_enable_thinking", type=parse_bool, default=None)
    bon.add_argument("--bon_N", type=int, default=8)
    bon.add_argument("--ns", type=int, nargs="+", default=None)
    bon.add_argument(
        "--bon_transformer_type",
        choices=["auto", "bidirectional", "causal"],
        default="auto",
    )
    bon.add_argument("--bon_fold_index", type=int, default=None)
    bon.add_argument("--bon_value_device", default="auto")
    bon.add_argument("--bon_scoring_batch_size", type=int, default=16)
    bon.add_argument("--bon_candidate_batch_size", type=int, default=8)
    bon.add_argument("--bon_max_tokens", type=int, default=8192)
    bon.add_argument("--bon_max_model_len", type=int, default=10240)
    bon.add_argument("--bon_temperature", type=float, default=0.7)
    bon.add_argument("--bon_top_p", type=float, default=0.95)
    bon.add_argument("--bon_logprobs", type=int, default=1)
    bon.add_argument("--bon_tensor_parallel_size", type=int, default=1)
    bon.add_argument(
        "--bon_gpus",
        help="DP groups separated by ';', TP GPUs inside each group by ','",
    )
    bon.add_argument(
        "--bon_performance_profile",
        choices=["throughput", "balanced", "memory"],
        default="throughput",
    )
    bon.add_argument(
        "--bon_model_transition",
        choices=["auto", "reload", "cpu_swap"],
        default="auto",
    )
    bon.add_argument(
        "--bon_value_residency",
        choices=["auto", "gpu", "cpu"],
        default="auto",
    )
    bon.add_argument("--bon_generation_max_active_jobs", type=int, default=None)
    bon.add_argument(
        "--bon_staging_flush_records",
        type=int,
        default=1,
        help="fsync generation/entropy staging after this many completed responses",
    )
    bon.add_argument("--bon_legacy_script_path", default=None)
    bon.add_argument(
        "--bon_vllm_cache_mode",
        choices=["auto", "isolated", "shared"],
        default="auto",
    )
    bon.add_argument("--bon_vllm_cache_root", default=None)
    bon.add_argument("--bon_vllm_enforce_eager", action="store_true")
    bon.add_argument("--bon_generation_worker_startup_timeout", type=float, default=600)
    bon.add_argument("--bon_gpu_memory_utilization", type=float, default=0.90)
    bon.add_argument("--bon_dtype", default="auto")
    bon.add_argument("--bon_trust_remote_code", action="store_true")
    bon.add_argument("--bon_num_segments", "--bon_K", type=int, default=8)
    bon.add_argument("--bon_entropy_threshold", type=float, default=1.2)
    bon.add_argument("--bon_phase_partition_algorithm", choices=["original", "adaptive"], default="original")
    bon.add_argument("--bon_high_entropy_ratio", type=float, default=None, help="Adaptive: top entropy token fraction in (0, 1], e.g. 0.05")
    bon.add_argument("--bon_target_phase_tokens", type=float, default=None, help="Adaptive: desired rollout-average phase length in tokens")
    bon.add_argument("--bon_entropy_chunk_size", type=int, default=16)
    bon.add_argument("--bon_entropy_model_path")
    bon.add_argument("--bon_entropy_device_map", default="auto")
    bon.add_argument("--bon_no_save_token_entropies", action="store_true")
    bon.add_argument("--bon_seed", type=int, default=42)
    bon.add_argument("--bon_resume", action="store_true")
    bon.add_argument("--bon_overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset_source == "local":
        if not args.local_dataset_path:
            parser.error("--local_dataset_path is required for --dataset_source local")
        if not Path(args.local_dataset_path).expanduser().is_file():
            parser.error(f"Local dataset file does not exist: {args.local_dataset_path}")
    if args.max_examples is not None and args.max_examples < 0:
        parser.error("--max_examples must be non-negative")
    if args.bon_N < 1:
        parser.error("--bon_N must be at least 1")
    try:
        _resolve_ns(args.bon_N, args.ns)
    except ValueError as error:
        parser.error(str(error))
    if args.bon_resume and args.bon_overwrite:
        parser.error("--bon_resume and --bon_overwrite are mutually exclusive")
    if args.bon_chat_template != "qwen3" and args.bon_enable_thinking is not None:
        parser.error("--bon_enable_thinking is only valid for qwen3")
    if args.bon_chat_template == "qwen3" and args.bon_enable_thinking is None:
        args.bon_enable_thinking = True
    return args


def _build_service(args: argparse.Namespace) -> BestOfNService:
    return BestOfNService(
        BestOfNConfig(
            model_path=args.bon_model_path,
            checkpoint_path=args.bon_checkpoint_path,
            n=args.bon_N,
            # The shared engine still needs one canonical ranking to populate
            # summary.jsonl. Multi-n/multi-aggregation evaluation happens below.
            output_aggregation="product",
            chat_template=args.bon_chat_template,
            enable_thinking=args.bon_enable_thinking,
            training_metrics_json=args.bon_training_metrics_json,
            transformer_type=args.bon_transformer_type,
            fold_index=args.bon_fold_index,
            value_device=args.bon_value_device,
            scoring_batch_size=args.bon_scoring_batch_size,
            candidate_batch_size=args.bon_candidate_batch_size,
            max_tokens=args.bon_max_tokens,
            max_model_len=args.bon_max_model_len,
            temperature=args.bon_temperature,
            top_p=args.bon_top_p,
            logprobs=args.bon_logprobs,
            tensor_parallel_size=args.bon_tensor_parallel_size,
            gpu_memory_utilization=args.bon_gpu_memory_utilization,
            dtype=args.bon_dtype,
            trust_remote_code=args.bon_trust_remote_code,
            num_segments=args.bon_num_segments,
            entropy_threshold=args.bon_entropy_threshold,
            phase_partition_algorithm=getattr(args, "bon_phase_partition_algorithm", "original"),
            high_entropy_ratio=getattr(args, "bon_high_entropy_ratio", None),
            target_phase_tokens=getattr(args, "bon_target_phase_tokens", None),
            entropy_chunk_size=args.bon_entropy_chunk_size,
            entropy_model_path=args.bon_entropy_model_path,
            entropy_device_map=args.bon_entropy_device_map,
            save_token_entropies=not args.bon_no_save_token_entropies,
            seed=args.bon_seed,
            gpu_groups=args.bon_gpus,
            performance_profile=args.bon_performance_profile,
            model_transition=args.bon_model_transition,
            value_residency=args.bon_value_residency,
            generation_max_active_jobs=args.bon_generation_max_active_jobs,
            staging_flush_records=args.bon_staging_flush_records,
            legacy_script_path=args.bon_legacy_script_path,
            vllm_cache_mode=args.bon_vllm_cache_mode,
            vllm_cache_root=args.bon_vllm_cache_root,
            vllm_enforce_eager=args.bon_vllm_enforce_eager,
            generation_worker_startup_timeout=args.bon_generation_worker_startup_timeout,
        )
    )


class _StageProgress:
    """Rollout-level progress bars for the three staged service phases."""

    DESCRIPTIONS = {
        "generation": "GPQA generation",
        "entropy": "GPQA token entropy",
        "phase": "GPQA phase partition",
        "scoring": "GPQA Transformer scoring",
    }

    def __init__(self, *, scoring_description: str | None = None) -> None:
        self._bars: dict[str, DatasetProgress] = {}
        self._descriptions = dict(self.DESCRIPTIONS)
        if scoring_description is not None:
            self._descriptions["scoring"] = scoring_description

    def start(self, stage: str, total: int) -> None:
        previous = self._bars.get(stage)
        if previous is not None:
            previous.close()
        self._bars[stage] = DatasetProgress(
            total,
            description=self._descriptions.get(stage, stage),
            unit="rollout",
        )

    def update(self, stage: str) -> None:
        progress = self._bars.get(stage)
        if progress is not None:
            progress.update()

    def close(self) -> None:
        for progress in self._bars.values():
            progress.close()
        self._bars.clear()


def run(args: argparse.Namespace, *, service: Any | None = None) -> dict[str, Any]:
    ns = _resolve_ns(args.bon_N, getattr(args, "ns", None))
    problems = load_gpqa_dataset(
        dataset_source=args.dataset_source,
        local_dataset_path=args.local_dataset_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        hf_token=args.hf_token,
        choice_seed=args.choice_seed,
        max_examples=args.max_examples,
    )
    adapter = GPQATransformerBoNAdapter(
        prompt_type=args.prompt_type,
        chat_template=args.bon_chat_template,
    )
    requests = [adapter.build_request(problem) for problem in problems]
    problem_by_request = {
        request.request_id: problem for request, problem in zip(requests, problems)
    }
    if args.bon_overwrite:
        _remove_aggregate_outputs(args.bon_output_dir)
    store = JsonlResultStore(
        args.bon_output_dir,
        resume=args.bon_resume,
        overwrite=args.bon_overwrite,
    )

    completed = set()
    evaluation_rows: list[dict[str, Any]] = []
    primary_rows_by_request: dict[str, dict[str, Any]] = {}
    resumed_updates: dict[str, bool] = {}
    if args.bon_resume:
        for request in requests:
            if not store.is_request_complete(request.request_id, args.bon_N):
                continue
            problem = problem_by_request[request.request_id]
            records = store.load_request_records(request.request_id)
            from scripts.transformer_bon.generation_prefix import within_generation_limit
            if not within_generation_limit(records, args.bon_max_tokens):
                continue
            evaluation_rows.extend(
                _evaluate_nested_candidates(
                    problem,
                    request.request_id,
                    records,
                    ns=ns,
                    shuffle_seed=(
                        args.bon_seed
                        + int(request.metadata.get("sample_idx", 0)) * 1000003
                    ),
                )
            )
            completed.add(request.request_id)
            selected_records = [
                record
                for record in records
                if record.get("selected_by_bon")
            ]
            if not selected_records:
                valid_product_records = [
                    record
                    for record in records
                    if record.get("valid_for_selection")
                    and _json_score(record.get("product_score")) is not None
                ]
                if valid_product_records:
                    selected_records = [
                        max(
                            valid_product_records,
                            key=lambda record: float(record["product_score"]),
                        )
                    ]
            if selected_records:
                selected = selected_records[0]
                primary_evaluation = evaluate_selected_response(
                    problem,
                    response=str(selected.get("full_output") or ""),
                    request_id=request.request_id,
                    candidate_key=str(selected["candidate_key"]),
                    selected_index=int(selected["sequence_idx"]),
                    selected_score=selected.get("product_score"),
                    n=args.bon_N,
                    aggregation="product",
                )
                primary_rows_by_request[request.request_id] = primary_evaluation
                if selected.get("selected_by_bon"):
                    resumed_updates[str(selected["candidate_key"])] = (
                        primary_evaluation["is_correct"]
                    )
            else:
                primary_rows_by_request[request.request_id] = (
                    _evaluation_without_selection(
                        problem,
                        request.request_id,
                        n=args.bon_N,
                        aggregation="product",
                    )
                )

    owns_service = service is None
    active_service = service or _build_service(args)
    stage_progress = _StageProgress(
        scoring_description=getattr(args, "bon_scoring_description", None)
    )
    progress = DatasetProgress(
        len(requests),
        initial=len(completed),
        description="GPQA Diamond complete",
        unit="question",
    )
    try:
        service_call = active_service.generate_and_select_many
        service_kwargs = {
            "staging_dir": args.bon_output_dir,
            "resume": args.bon_resume,
            "overwrite": args.bon_overwrite,
            "skip_request_ids": completed,
            "on_stage_start": stage_progress.start,
            "on_rollout_complete": stage_progress.update,
        }
        try:
            supported_parameters = inspect.signature(service_call).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in supported_parameters.values()
            )
            if not accepts_kwargs:
                service_kwargs = {
                    key: value
                    for key, value in service_kwargs.items()
                    if key in supported_parameters
                }
        except (TypeError, ValueError):
            pass
        for result in service_call(
            requests,
            **service_kwargs,
        ):
            problem = problem_by_request[result.request.request_id]
            _annotate_result_records(result)
            evaluation_rows.extend(
                _evaluate_nested_candidates(
                    problem,
                    result.request.request_id,
                    result.candidates,
                    ns=ns,
                    shuffle_seed=(
                        args.bon_seed
                        + int(result.request.metadata.get("sample_idx", 0))
                        * 1000003
                    ),
                )
            )
            selected = _candidate_view(result.selected_candidate)
            selected_evaluation = evaluate_selected_response(
                problem,
                response=selected["full_output"],
                request_id=result.request.request_id,
                candidate_key=selected["candidate_key"],
                selected_index=selected["sequence_idx"],
                selected_score=selected.get("product_score"),
                n=result.n,
                aggregation="product",
            )
            primary_rows_by_request[result.request.request_id] = selected_evaluation
            store.write_result(
                result,
                selected_is_correct=selected_evaluation["is_correct"],
            )
            progress.update()
    finally:
        stage_progress.close()
        progress.close()
        if owns_service:
            active_service.close()

    if resumed_updates:
        store.update_evaluated_candidates(resumed_updates)

    row_order: dict[tuple[str, int, str], int] = {}
    order_index = 0
    for request in requests:
        for n_value in ns:
            for aggregation in AGGREGATIONS:
                row_order[(request.request_id, n_value, aggregation)] = order_index
                order_index += 1
    ordered_rows = sorted(
        evaluation_rows,
        key=lambda row: row_order[
            (row["request_id"], row["n"], row["aggregation"])
        ],
    )
    metrics_by_n = compute_metrics_by_n(ordered_rows, ns)
    ordered_primary_rows = [
        primary_rows_by_request[request.request_id] for request in requests
    ]
    metric_values = compute_metrics(ordered_primary_rows)
    metrics = {
        "benchmark": "gpqa_diamond",
        **metric_values,
        "dataset_source": args.dataset_source,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "local_dataset_path": (
            str(Path(args.local_dataset_path).expanduser().resolve())
            if args.local_dataset_path
            else None
        ),
        "prompt_type": args.prompt_type,
        "choice_seed": args.choice_seed,
        "best_of_n": {
            "model_path": args.bon_model_path,
            "checkpoint_path": args.bon_checkpoint_path,
            "N": args.bon_N,
            "ns": ns,
            "aggregations": list(AGGREGATIONS),
            "reference_n": args.bon_N,
            "reference_aggregation": "product",
            "chat_template": args.bon_chat_template,
            "enable_thinking": args.bon_enable_thinking,
            "seed": args.bon_seed,
        },
        "metrics_by_n": metrics_by_n,
    }
    csv_path, metrics_path = write_evaluation_outputs(
        args.bon_output_dir,
        args.bon_model_path,
        ordered_primary_rows,
        metrics,
    )
    metrics_csv_path, metrics_png_path = write_aggregate_outputs(
        args.bon_output_dir,
        metrics_by_n,
        ordered_rows,
    )
    metrics["results_csv"] = str(csv_path)
    metrics["metrics_json"] = str(metrics_path)
    metrics["metrics_csv"] = str(metrics_csv_path)
    metrics["metrics_png"] = str(metrics_png_path)
    return metrics


def main() -> None:
    metrics = run(parse_args())
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Refusal fraction: {metrics['refusal_fraction']}")
    print(f"Results: {metrics['results_csv']}")


if __name__ == "__main__":
    main()
