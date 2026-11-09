from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from .adapters import create_adapter
from .engine import BestOfNConfig, BestOfNService
from .records import JsonlResultStore


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset-adapted confidence-trajectory Transformer Best-of-N"
    )
    parser.add_argument("--input", required=True, help="JSON array or JSONL dataset")
    parser.add_argument("--dataset_adapter", required=True)
    parser.add_argument(
        "--adapter_kwargs_json",
        default="{}",
        help="JSON object passed to the dataset adapter constructor",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--training_metrics_json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--chat_template",
        choices=["qwen3", "deepseek_r1_distill_qwen", "raw"],
        default="qwen3",
    )
    parser.add_argument("--enable_thinking", type=parse_bool, default=None)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument(
        "--aggregation",
        choices=["product", "min", "mean", "last"],
        default="product",
    )
    parser.add_argument(
        "--transformer_type",
        choices=["auto", "bidirectional", "causal"],
        default="auto",
    )
    parser.add_argument("--fold_index", type=int)
    parser.add_argument("--value_device", default="auto")
    parser.add_argument("--scoring_batch_size", type=int, default=8)
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--logprobs", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpus")
    parser.add_argument(
        "--performance_profile",
        choices=["throughput", "balanced", "memory"],
        default="throughput",
    )
    parser.add_argument(
        "--reproducibility_mode",
        choices=["legacy_strict"],
        default="legacy_strict",
    )
    parser.add_argument(
        "--model_transition",
        choices=["auto", "reload", "cpu_swap"],
        default="auto",
    )
    parser.add_argument(
        "--value_residency", choices=["auto", "gpu", "cpu"], default="auto"
    )
    parser.add_argument("--generation_max_active_jobs", type=int)
    parser.add_argument(
        "--staging_flush_records",
        type=int,
        default=32,
        help="fsync generation/entropy staging after this many completed responses",
    )
    parser.add_argument("--legacy_script_path")
    parser.add_argument(
        "--vllm_cache_mode",
        choices=["auto", "isolated", "shared"],
        default="auto",
    )
    parser.add_argument("--vllm_cache_root")
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--generation_worker_startup_timeout", type=float, default=600.0)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.chat_template != "qwen3" and args.enable_thinking is not None:
        parser.error("--enable_thinking is only valid for qwen3")
    try:
        adapter_kwargs = json.loads(args.adapter_kwargs_json)
    except json.JSONDecodeError as error:
        parser.error(f"Invalid --adapter_kwargs_json: {error}")
    if not isinstance(adapter_kwargs, dict):
        parser.error("--adapter_kwargs_json must decode to an object")
    args.adapter_kwargs = adapter_kwargs
    return args


def main() -> None:
    args = parse_args()
    adapter = create_adapter(args.dataset_adapter, **args.adapter_kwargs)
    records = adapter.load(args.input)
    store = JsonlResultStore(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    enable_thinking = (
        True
        if args.chat_template == "qwen3" and args.enable_thinking is None
        else args.enable_thinking
    )
    service = BestOfNService(
        BestOfNConfig(
            model_path=args.model_path,
            checkpoint_path=args.checkpoint_path,
            n=args.N,
            output_aggregation=args.aggregation,
            chat_template=args.chat_template,
            enable_thinking=enable_thinking,
            training_metrics_json=args.training_metrics_json,
            transformer_type=args.transformer_type,
            fold_index=args.fold_index,
            value_device=args.value_device,
            scoring_batch_size=args.scoring_batch_size,
            candidate_batch_size=args.candidate_batch_size,
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
            reproducibility_mode=args.reproducibility_mode,
            model_transition=args.model_transition,
            value_residency=args.value_residency,
            generation_max_active_jobs=args.generation_max_active_jobs,
            staging_flush_records=args.staging_flush_records,
            legacy_script_path=args.legacy_script_path,
            vllm_cache_mode=args.vllm_cache_mode,
            vllm_cache_root=args.vllm_cache_root,
            vllm_enforce_eager=args.vllm_enforce_eager,
            generation_worker_startup_timeout=args.generation_worker_startup_timeout,
        )
    )
    try:
        for sample_idx, sample in enumerate(records):
            if args.limit is not None and sample_idx >= args.limit:
                break
            request = adapter.build_request(sample, sample_idx)
            if args.resume and store.is_request_complete(request.request_id, args.N):
                continue
            result = service.generate_and_select(request)
            # Generic generation has no benchmark evaluator. A benchmark-specific
            # runner may write a real label after evaluating the selected rollout.
            store.write_result(result, selected_is_correct=None)
    finally:
        service.close()


if __name__ == "__main__":
    main()
