from __future__ import annotations

"""BIG-Bench-Hard Transformer Best-of-N evaluator.

This runner intentionally keeps generation, entropy/phase construction and
trajectory scoring in ``scripts.transformer_bon``.  BBH-specific code here is
limited to local dataset loading, prompt construction, boxed-answer scoring and
the multi-n/multi-aggregation report.
"""

import argparse
import csv
import inspect
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

if TYPE_CHECKING:
    # Type-only imports keep Pylance aware of the real classes even though the
    # runtime imports below are deliberately optional for ``--help``.
    from scripts.transformer_bon import (  # noqa: E402
        BestOfNConfig as _BestOfNConfigType,
        BestOfNService as _BestOfNServiceType,
        DatasetProgress as _DatasetProgressType,
    )

try:  # Keep ``--help`` and dataset/parser tests usable without GPU dependencies.
    from scripts.transformer_bon import (  # noqa: E402
        BestOfNConfig as _RuntimeBestOfNConfig,
        BestOfNService as _RuntimeBestOfNService,
        DatasetProgress as _RuntimeDatasetProgress,
        JsonlResultStore as _RuntimeJsonlResultStore,
    )
    _TRANSFORMER_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as error:  # numpy/torch/vLLM are runtime dependencies.
    _RuntimeBestOfNConfig = _RuntimeBestOfNService = None  # type: ignore[assignment]
    _RuntimeDatasetProgress = _RuntimeJsonlResultStore = None  # type: ignore[assignment]
    _TRANSFORMER_IMPORT_ERROR = error

try:  # package execution
    from .bbh_dataset import BBHExample, load_bbh_examples
    from .bbh_scorer import score_boxed_answer
except ImportError:  # direct execution: python BBH/run_transformer_bon.py
    from bbh_dataset import BBHExample, load_bbh_examples
    from bbh_scorer import score_boxed_answer


AGGREGATIONS = ("product", "min", "mean", "last")
SCORE_KEYS = {name: f"{name}_score" for name in AGGREGATIONS}
OFFICIAL_COLUMNS = (
    "Question id", "Question", "Correct answer", "Model answer", "Correct", "Model response",
)
LINK_COLUMNS = (
    "Task", "Task type", "Request id", "Selected candidate key", "Selected index",
    "Selected score", "N", "Aggregation", "Parsed answer", "Refusal",
)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def _safe_model_name(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path.rstrip("/\\")).name or "model")


def _score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def resolve_ns(max_n: int, requested: Sequence[int] | None) -> list[int]:
    if requested is None:
        return sorted({n for n in (1, 2, 4, 8, 16, 32, max_n) if 1 <= n <= max_n})
    values = sorted({int(n) for n in requested})
    if not values or any(n < 1 or n > max_n for n in values):
        raise ValueError(f"Every --ns value must satisfy 1 <= n <= N={max_n}")
    return values


def build_requests(examples: Sequence[BBHExample]) -> list[Any]:
    _require_transformer_runtime()
    from scripts.transformer_bon import BestOfNRequest

    requests = []
    for global_index, example in enumerate(examples):
        requests.append(
            BestOfNRequest(
                request_id=f"bbh:{example.sample_id}",
                sample_id=example.sample_id,
                # The shared entropy/PRM adapters use this field as the problem
                # context.  Keep the exact prompt context visible to every scorer.
                task_text=example.prompt,
                messages=[{"role": "user", "content": example.prompt}],
                gold_answer=example.target,
                metadata={
                    "benchmark": "bbh",
                    "sample_idx": global_index,
                    "task_name": example.task_name,
                    "task_type": example.task_type,
                    "input": example.input,
                    "description": example.description,
                    "target": example.target,
                    "raw_sample": dict(example.raw),
                },
            )
        )
    return requests


def _require_transformer_runtime() -> None:
    if _TRANSFORMER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "BBH BoN execution requires the scripts/transformer_bon runtime "
            "and its PyTorch/numpy/vLLM dependencies."
        ) from _TRANSFORMER_IMPORT_ERROR


def evaluate_response(example: BBHExample, response: str, *, request_id: str,
                      candidate_key: str | None, selected_index: int | None,
                      selected_score: float | None, n: int, aggregation: str) -> dict[str, Any]:
    correct, parsed, expected = score_boxed_answer(
        response,
        target=example.target,
        task_name=example.task_name,
        task_type=example.task_type,
        input_text=example.input,
    )
    return {
        "sample_id": example.sample_id,
        "task_name": example.task_name,
        "task_type": example.task_type,
        "request_id": request_id,
        "candidate_key": candidate_key,
        "selected_index": selected_index,
        "selected_score": selected_score,
        "n": int(n),
        "aggregation": aggregation,
        "question": example.input,
        "description": example.description,
        "gold_answer": expected,
        "gold_raw": example.target,
        "parsed_answer": parsed,
        "model_answer": parsed if parsed is not None else "Couldn't find an answer!",
        "is_correct": bool(correct),
        "is_refusal": parsed is None,
        "response": response,
    }


def empty_evaluation(example: BBHExample, request_id: str, *, n: int, aggregation: str) -> dict[str, Any]:
    _, _, expected = score_boxed_answer(
        r"\boxed{}", target=example.target, task_name=example.task_name,
        task_type=example.task_type, input_text=example.input,
    )
    return {
        "sample_id": example.sample_id, "task_name": example.task_name,
        "task_type": example.task_type, "request_id": request_id,
        "candidate_key": None, "selected_index": None, "selected_score": None,
        "n": int(n), "aggregation": aggregation, "question": example.input,
        "description": example.description, "gold_answer": expected,
        "gold_raw": example.target, "parsed_answer": None,
        "model_answer": "Couldn't find an answer!", "is_correct": False,
        "is_refusal": True, "response": "",
    }


def _candidate_view(candidate: Any) -> dict[str, Any]:
    record = candidate if isinstance(candidate, Mapping) else candidate.summary_record
    text = record.get("full_output") if isinstance(record, Mapping) else candidate.text
    sequence_idx = record.get("sequence_idx")
    if sequence_idx is None:
        sequence_idx = candidate.sequence_idx
    valid = record.get("valid_for_selection")
    if valid is None:
        valid = _score(record.get("product_score")) is not None
    return {
        "candidate_key": str(record.get("candidate_key", "")),
        "sequence_idx": int(sequence_idx), "full_output": str(text or ""),
        "valid_for_selection": bool(valid),
        **{key: record.get(key) for key in SCORE_KEYS.values()},
    }


def evaluate_nested(example: BBHExample, request_id: str, candidates: Sequence[Any],
                    *, ns: Sequence[int], shuffle_seed: int) -> list[dict[str, Any]]:
    views = [_candidate_view(candidate) for candidate in candidates]
    views.sort(key=lambda item: item["sequence_idx"])
    random.Random(int(shuffle_seed)).shuffle(views)
    rows: list[dict[str, Any]] = []
    for n in ns:
        sampled = views[:n]
        sampled_indices = json.dumps([item["sequence_idx"] for item in sampled])
        for aggregation in AGGREGATIONS:
            valid = [
                (item, _score(item.get(SCORE_KEYS[aggregation])))
                for item in sampled
                if item["valid_for_selection"] and _score(item.get(SCORE_KEYS[aggregation])) is not None
            ]
            if not valid:
                row = empty_evaluation(example, request_id, n=n, aggregation=aggregation)
            else:
                selected, selected_score = max(valid, key=lambda pair: pair[1])
                row = evaluate_response(
                    example, selected["full_output"], request_id=request_id,
                    candidate_key=selected["candidate_key"], selected_index=selected["sequence_idx"],
                    selected_score=selected_score, n=n, aggregation=aggregation,
                )
            row["sampled_sequence_indices"] = sampled_indices
            rows.append(row)
    return rows


def metrics_for_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    total = len(rows)
    correct = sum(bool(row.get("is_correct")) for row in rows)
    refusals = sum(bool(row.get("is_refusal")) for row in rows)
    return {
        "total": total, "correct": correct, "refusals": refusals,
        "accuracy": correct / total if total else 0.0,
        "refusal_fraction": refusals / total if total else 0.0,
    }


def metrics_by_n(rows: Sequence[Mapping[str, Any]], ns: Sequence[int]) -> dict[str, dict[str, dict[str, int | float]]]:
    return {
        str(n): {
            aggregation: metrics_for_rows(
                [row for row in rows if int(row["n"]) == n and row["aggregation"] == aggregation]
            )
            for aggregation in AGGREGATIONS
        }
        for n in ns
    }


def write_evaluation_outputs(output_dir: str | Path, model_path: str,
                            rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"eval_results_{_safe_model_name(model_path)}.csv"
    metrics_path = directory / "metrics.json"
    fields = [*OFFICIAL_COLUMNS, *LINK_COLUMNS]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Question id": row["sample_id"], "Question": row["question"],
                "Correct answer": row["gold_answer"], "Model answer": row["model_answer"],
                "Correct": row["is_correct"], "Model response": row["response"],
                "Task": row["task_name"], "Task type": row["task_type"],
                "Request id": row["request_id"], "Selected candidate key": row["candidate_key"],
                "Selected index": row["selected_index"], "Selected score": row["selected_score"],
                "N": row["n"], "Aggregation": row["aggregation"],
                "Parsed answer": row["parsed_answer"], "Refusal": row["is_refusal"],
            })
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metrics), handle, ensure_ascii=False, indent=2)
    return csv_path, metrics_path


def write_aggregate_outputs(output_dir: str | Path, by_n: Mapping[str, Mapping[str, Mapping[str, Any]]],
                            rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    metrics_csv = directory / "metrics.csv"
    metrics_png = directory / "metrics.png"
    fields = ["n", "aggregation", "total", "correct", "refusals", "accuracy", "refusal_fraction"]
    flat = [{"n": int(n), "aggregation": aggregation, **dict(values)}
            for n, values_by_agg in by_n.items() for aggregation, values in values_by_agg.items()]
    flat.sort(key=lambda row: (row["n"], AGGREGATIONS.index(row["aggregation"])))
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat)
    for n in sorted(int(value) for value in by_n):
        # Match the other benchmark runners: one wide row per sample, with a
        # fixed block of columns for each aggregation, rather than one row per
        # aggregation.  This makes details_<n>.csv directly comparable across
        # GPQA/LiveCodeBench/BBH runs.
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            if int(row["n"]) != n:
                continue
            request_id = str(row["request_id"])
            detail = grouped.setdefault(request_id, {
                "sample_id": row["sample_id"], "task_name": row["task_name"],
                "task_type": row["task_type"], "request_id": request_id,
                "n": n, "question": row["question"],
                "gold_answer": row["gold_answer"], "gold_raw": row["gold_raw"],
                "sampled_sequence_indices": row.get("sampled_sequence_indices", "[]"),
            })
            prefix = str(row["aggregation"])
            detail[f"{prefix}_candidate_key"] = row["candidate_key"]
            detail[f"{prefix}_selected_index"] = row["selected_index"]
            detail[f"{prefix}_selected_score"] = row["selected_score"]
            detail[f"{prefix}_parsed_answer"] = row["parsed_answer"]
            detail[f"{prefix}_model_answer"] = row["model_answer"]
            detail[f"{prefix}_correct"] = row["is_correct"]
            detail[f"{prefix}_refusal"] = row["is_refusal"]
        detail_rows = list(grouped.values())
        detail_rows.sort(key=lambda row: str(row["request_id"]))
        detail_path = directory / f"details_{n}.csv"
        if detail_rows:
            detail_fields = ["sample_id", "task_name", "task_type", "request_id", "n",
                             "question", "gold_answer", "gold_raw", "sampled_sequence_indices"]
            for aggregation in AGGREGATIONS:
                detail_fields.extend([
                    f"{aggregation}_candidate_key", f"{aggregation}_selected_index",
                    f"{aggregation}_selected_score", f"{aggregation}_parsed_answer",
                    f"{aggregation}_model_answer", f"{aggregation}_correct",
                    f"{aggregation}_refusal",
                ])
            with detail_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=detail_fields, extrasaction="ignore")
                writer.writeheader(); writer.writerows(detail_rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = sorted(int(value) for value in by_n)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = dict(zip(AGGREGATIONS, ("tab:blue", "tab:orange", "tab:green", "tab:red")))
        for aggregation in AGGREGATIONS:
            axes[0].plot(ns, [float(by_n[str(n)][aggregation]["accuracy"]) for n in ns], marker="o", label=aggregation, color=colors[aggregation])
            axes[1].plot(ns, [float(by_n[str(n)][aggregation]["refusal_fraction"]) for n in ns], marker="o", label=aggregation, color=colors[aggregation])
        axes[0].set(xlabel="n", ylabel="Accuracy", title="BBH Best-of-N accuracy")
        axes[1].set(xlabel="n", ylabel="Refusal fraction", title="BBH refusal fraction")
        for axis in axes:
            axis.grid(True, alpha=0.3); axis.legend()
        fig.tight_layout(); fig.savefig(metrics_png, dpi=200); plt.close(fig)
    except ImportError:
        if metrics_png.exists():
            metrics_png.unlink()
    return metrics_csv, metrics_png


def _remove_aggregate_outputs(output_dir: str | Path) -> None:
    directory = Path(output_dir).expanduser().resolve()
    for name in ("metrics.csv", "metrics.png"):
        path = directory / name
        if path.exists():
            path.unlink()
    for path in directory.glob("details_*.csv"):
        path.unlink()


class StageProgress:
    descriptions = {"generation": "BBH generation", "entropy": "BBH token entropy", "phase": "BBH phase partition", "scoring": "BBH Transformer scoring"}

    def __init__(self, scoring_description: str | None = None) -> None:
        self.bars: dict[str, _DatasetProgressType] = {}
        if scoring_description:
            self.descriptions = dict(self.descriptions)
            self.descriptions["scoring"] = scoring_description

    def start(self, stage: str, total: int) -> None:
        if stage in self.bars:
            self.bars[stage].close()
        self.bars[stage] = _RuntimeDatasetProgress(total, description=self.descriptions.get(stage, stage), unit="rollout")

    def update(self, stage: str) -> None:
        if stage in self.bars:
            self.bars[stage].update()

    def close(self) -> None:
        for bar in self.bars.values():
            bar.close()
        self.bars.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BIG-Bench-Hard Transformer Best-of-N evaluation")
    data = parser.add_argument_group("BBH local dataset and prompt")
    data.add_argument("--data_dir", "--input", "--dataset_path", required=True, dest="data_dir")
    data.add_argument("--description_dir", "--task_yaml_dir", default=None)
    data.add_argument("--tasks", nargs="+", default=None)
    data.add_argument("--max_examples", type=int, default=None)

    bon = parser.add_argument_group("Transformer Best-of-N")
    bon.add_argument("--bon_model_path", required=True)
    bon.add_argument("--bon_checkpoint_path", required=True)
    bon.add_argument("--bon_training_metrics_json", default=None)
    bon.add_argument("--bon_output_dir", required=True)
    bon.add_argument("--bon_chat_template", choices=["qwen3", "deepseek_r1_distill_qwen", "raw"], default="qwen3")
    bon.add_argument("--bon_enable_thinking", type=parse_bool, default=None)
    bon.add_argument("--bon_N", "--N", type=int, default=8)
    bon.add_argument("--ns", type=int, nargs="+", default=None)
    bon.add_argument("--bon_transformer_type", choices=["auto", "bidirectional", "causal"], default="auto")
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
    bon.add_argument("--bon_gpus", "--gpus", default=None)
    bon.add_argument("--bon_performance_profile", choices=["throughput", "balanced", "memory"], default="throughput")
    bon.add_argument("--bon_model_transition", choices=["auto", "reload", "cpu_swap"], default="auto")
    bon.add_argument("--bon_value_residency", choices=["auto", "gpu", "cpu"], default="auto")
    bon.add_argument("--bon_generation_max_active_jobs", type=int, default=None)
    bon.add_argument("--bon_staging_flush_records", type=int, default=1)
    bon.add_argument("--bon_legacy_script_path", default=None)
    bon.add_argument("--bon_vllm_cache_mode", choices=["auto", "isolated", "shared"], default="auto")
    bon.add_argument("--bon_vllm_cache_root", default=None)
    bon.add_argument("--bon_vllm_enforce_eager", action="store_true")
    bon.add_argument("--bon_generation_worker_startup_timeout", type=float, default=600.0)
    bon.add_argument("--bon_gpu_memory_utilization", type=float, default=0.90)
    bon.add_argument("--bon_dtype", default="auto")
    bon.add_argument("--bon_trust_remote_code", action="store_true")
    bon.add_argument("--bon_num_segments", "--bon_K", type=int, default=8)
    bon.add_argument("--bon_entropy_threshold", type=float, default=1.2)
    bon.add_argument("--bon_phase_partition_algorithm", choices=["original", "adaptive"], default="original")
    bon.add_argument("--bon_high_entropy_ratio", type=float, default=None, help="Adaptive: top entropy token fraction in (0, 1], e.g. 0.05")
    bon.add_argument("--bon_target_phase_tokens", type=float, default=None, help="Adaptive: desired rollout-average phase length in tokens")
    bon.add_argument("--bon_entropy_chunk_size", type=int, default=16)
    bon.add_argument("--bon_entropy_model_path", default=None)
    bon.add_argument("--bon_entropy_device_map", default="auto")
    bon.add_argument("--bon_no_save_token_entropies", action="store_true")
    bon.add_argument("--bon_seed", type=int, default=42)
    bon.add_argument("--bon_resume", action="store_true")
    bon.add_argument("--bon_overwrite", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(); args = parser.parse_args(argv)
    if args.bon_N < 1:
        parser.error("--bon_N must be at least 1")
    if args.max_examples is not None and args.max_examples < 0:
        parser.error("--max_examples must be non-negative")
    try:
        resolve_ns(args.bon_N, args.ns)
    except ValueError as error:
        parser.error(str(error))
    if args.bon_resume and args.bon_overwrite:
        parser.error("--bon_resume and --bon_overwrite are mutually exclusive")
    if args.bon_chat_template != "qwen3" and args.bon_enable_thinking is not None:
        parser.error("--bon_enable_thinking is only valid for qwen3")
    if args.bon_chat_template == "qwen3" and args.bon_enable_thinking is None:
        args.bon_enable_thinking = True
    return args


def _build_service(args: argparse.Namespace) -> _BestOfNServiceType:
    _require_transformer_runtime()
    return _RuntimeBestOfNService(_RuntimeBestOfNConfig(
        model_path=args.bon_model_path, checkpoint_path=args.bon_checkpoint_path,
        n=args.bon_N, output_aggregation="product", chat_template=args.bon_chat_template,
        enable_thinking=args.bon_enable_thinking, training_metrics_json=args.bon_training_metrics_json,
        transformer_type=args.bon_transformer_type, fold_index=args.bon_fold_index,
        value_device=args.bon_value_device, scoring_batch_size=args.bon_scoring_batch_size,
        candidate_batch_size=args.bon_candidate_batch_size, max_tokens=args.bon_max_tokens,
        max_model_len=args.bon_max_model_len, temperature=args.bon_temperature,
        top_p=args.bon_top_p, logprobs=args.bon_logprobs, tensor_parallel_size=args.bon_tensor_parallel_size,
        gpu_memory_utilization=args.bon_gpu_memory_utilization, dtype=args.bon_dtype,
        trust_remote_code=args.bon_trust_remote_code, num_segments=args.bon_num_segments,
        entropy_threshold=args.bon_entropy_threshold, entropy_chunk_size=args.bon_entropy_chunk_size,
        phase_partition_algorithm=getattr(args, "bon_phase_partition_algorithm", "original"),
        high_entropy_ratio=getattr(args, "bon_high_entropy_ratio", None),
        target_phase_tokens=getattr(args, "bon_target_phase_tokens", None),
        entropy_model_path=args.bon_entropy_model_path, entropy_device_map=args.bon_entropy_device_map,
        save_token_entropies=not args.bon_no_save_token_entropies, seed=args.bon_seed,
        gpu_groups=args.bon_gpus, performance_profile=args.bon_performance_profile,
        model_transition=args.bon_model_transition, value_residency=args.bon_value_residency,
        generation_max_active_jobs=args.bon_generation_max_active_jobs,
        staging_flush_records=args.bon_staging_flush_records, legacy_script_path=args.bon_legacy_script_path,
        vllm_cache_mode=args.bon_vllm_cache_mode, vllm_cache_root=args.bon_vllm_cache_root,
        vllm_enforce_eager=args.bon_vllm_enforce_eager,
        generation_worker_startup_timeout=args.bon_generation_worker_startup_timeout,
    ))


def _primary_from_records(example: BBHExample, request_id: str, records: Sequence[Mapping[str, Any]], n: int) -> dict[str, Any]:
    selected = [record for record in records if record.get("selected_by_bon")]
    if not selected:
        selected = [record for record in records if record.get("valid_for_selection") and _score(record.get("product_score")) is not None]
        selected = [max(selected, key=lambda record: float(record["product_score"]))] if selected else []
    if not selected:
        return empty_evaluation(example, request_id, n=n, aggregation="product")
    record = selected[0]
    return evaluate_response(example, str(record.get("full_output") or ""), request_id=request_id,
                             candidate_key=str(record.get("candidate_key")), selected_index=int(record["sequence_idx"]),
                             selected_score=_score(record.get("product_score")), n=n, aggregation="product")


def run(args: argparse.Namespace, *, service: Any | None = None) -> dict[str, Any]:
    _require_transformer_runtime()
    ns = resolve_ns(args.bon_N, args.ns)
    examples = load_bbh_examples(args.data_dir, description_dir=args.description_dir, tasks=args.tasks, max_examples=args.max_examples)
    requests = build_requests(examples)
    example_by_request = {request.request_id: example for request, example in zip(requests, examples)}
    if args.bon_overwrite:
        _remove_aggregate_outputs(args.bon_output_dir)
    store = _RuntimeJsonlResultStore(args.bon_output_dir, resume=args.bon_resume, overwrite=args.bon_overwrite)

    completed: set[str] = set(); evaluation_rows: list[dict[str, Any]] = []
    primary_rows: dict[str, dict[str, Any]] = {}; resumed_updates: dict[str, bool] = {}
    for request in requests:
        if not args.bon_resume or not store.is_request_complete(request.request_id, args.bon_N):
            continue
        records = store.load_request_records(request.request_id)
        from scripts.transformer_bon.generation_prefix import within_generation_limit
        if not within_generation_limit(records, args.bon_max_tokens):
            continue
        example = example_by_request[request.request_id]
        evaluation_rows.extend(evaluate_nested(example, request.request_id, records, ns=ns,
                                               shuffle_seed=args.bon_seed + int(request.metadata["sample_idx"]) * 1000003))
        completed.add(request.request_id)
        primary = _primary_from_records(example, request.request_id, records, args.bon_N)
        primary_rows[request.request_id] = primary
        selected = next((record for record in records if record.get("selected_by_bon")), None)
        if selected:
            resumed_updates[str(selected["candidate_key"])] = bool(primary["is_correct"])

    active_service = service or _build_service(args); owns_service = service is None
    stage_progress = StageProgress(getattr(args, "bon_scoring_description", None))
    sample_progress = _RuntimeDatasetProgress(len(requests), initial=len(completed), description="BBH complete", unit="sample")
    try:
        call = active_service.generate_and_select_many
        kwargs = {"staging_dir": args.bon_output_dir, "resume": args.bon_resume, "overwrite": args.bon_overwrite,
                  "skip_request_ids": completed, "on_stage_start": stage_progress.start,
                  "on_rollout_complete": stage_progress.update}
        try:
            params = inspect.signature(call).parameters
            if not any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
                kwargs = {key: value for key, value in kwargs.items() if key in params}
        except (TypeError, ValueError):
            pass
        for result in call(requests, **kwargs):
            request_id = result.request.request_id; example = example_by_request[request_id]
            evaluation_rows.extend(evaluate_nested(example, request_id, result.candidates, ns=ns,
                                                   shuffle_seed=args.bon_seed + int(result.request.metadata["sample_idx"]) * 1000003))
            selected = _candidate_view(result.selected_candidate)
            primary = evaluate_response(example, selected["full_output"], request_id=request_id,
                                        candidate_key=selected["candidate_key"], selected_index=selected["sequence_idx"],
                                        selected_score=_score(selected.get("product_score")), n=args.bon_N, aggregation="product")
            primary_rows[request_id] = primary
            store.write_result(result, selected_is_correct=primary["is_correct"])
            sample_progress.update()
    finally:
        stage_progress.close(); sample_progress.close()
        if owns_service:
            active_service.close()
    if resumed_updates:
        store.update_evaluated_candidates(resumed_updates)

    order = {(request.request_id, n, aggregation): index
             for index, (request, n, aggregation) in enumerate((request, n, aggregation)
                 for request in requests for n in ns for aggregation in AGGREGATIONS)}
    ordered_rows = sorted(evaluation_rows, key=lambda row: order[(row["request_id"], int(row["n"]), row["aggregation"])])
    by_n = metrics_by_n(ordered_rows, ns)
    ordered_primary = [primary_rows[request.request_id] for request in requests]
    task_metrics = {
        task_name: metrics_for_rows([row for row in ordered_primary if row["task_name"] == task_name])
        for task_name in sorted({row["task_name"] for row in ordered_primary})
    }
    metrics = {"benchmark": "big_bench_hard", **metrics_for_rows(ordered_primary),
               "data_dir": str(Path(args.data_dir).expanduser().resolve()),
               "description_dir": str(Path(args.description_dir).expanduser().resolve()) if args.description_dir else None,
               "tasks": sorted({example.task_name for example in examples}),
               "num_examples": len(examples), "task_metrics": task_metrics,
               "best_of_n": {"model_path": args.bon_model_path,
               "checkpoint_path": args.bon_checkpoint_path, "N": args.bon_N, "ns": ns,
               "aggregations": list(AGGREGATIONS), "reference_n": args.bon_N,
               "reference_aggregation": "product", "chat_template": args.bon_chat_template,
               "enable_thinking": args.bon_enable_thinking, "seed": args.bon_seed},
               "metrics_by_n": by_n}
    csv_path, metrics_path = write_evaluation_outputs(args.bon_output_dir, args.bon_model_path, ordered_primary, metrics)
    metrics_csv, metrics_png = write_aggregate_outputs(args.bon_output_dir, by_n, ordered_rows)
    metrics.update({"results_csv": str(csv_path), "metrics_json": str(metrics_path),
                    "metrics_csv": str(metrics_csv), "metrics_png": str(metrics_png)})
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    metrics = run(parse_args())
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Refusal fraction: {metrics['refusal_fraction']}")
    print(f"Results: {metrics['results_csv']}")


if __name__ == "__main__":
    main()
