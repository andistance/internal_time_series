from __future__ import annotations

"""BIG-Bench-Hard Hugging Face PRM Best-of-N entry point."""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

if TYPE_CHECKING:
    # Type-only imports prevent Pylance from treating the optional runtime
    # imports as ``Any`` in the function annotations below.
    from scripts.transformer_bon import (  # noqa: E402
        PRMBestOfNConfig as _PRMBestOfNConfigType,
        PRMBestOfNService as _PRMBestOfNServiceType,
    )

try:
    from scripts.transformer_bon import (  # noqa: E402
        PRMBestOfNConfig as _RuntimePRMBestOfNConfig,
        PRMBestOfNService as _RuntimePRMBestOfNService,
    )
    _PRM_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as error:
    _RuntimePRMBestOfNConfig = _RuntimePRMBestOfNService = None  # type: ignore[assignment]
    _PRM_IMPORT_ERROR = error

try:
    from .run_transformer_bon import build_parser as build_transformer_parser, run as run_bbh
except ImportError:
    from run_transformer_bon import build_parser as build_transformer_parser, run as run_bbh


_HIDDEN = {"bon_checkpoint_path", "bon_training_metrics_json", "bon_transformer_type",
           "bon_fold_index", "bon_value_device", "bon_value_residency"}


def build_parser() -> argparse.ArgumentParser:
    parser = build_transformer_parser()
    parser.description = "BIG-Bench-Hard Hugging Face PRM Best-of-N evaluation"
    for action in parser._actions:
        if action.dest == "bon_checkpoint_path":
            action.required = False
        if action.dest in _HIDDEN:
            action.help = argparse.SUPPRESS
    for group in parser._action_groups:
        if group.title == "Transformer Best-of-N":
            group.title = "PRM Best-of-N"
    prm = parser.add_argument_group("Hugging Face PRM scoring")
    prm.add_argument("--prm_name", default="qwen2.5-math-prm-7b")
    prm.add_argument("--prm_model_path", required=True)
    prm.add_argument("--prm_device", default="auto")
    prm.add_argument("--prm_gpus", "--prm_gpu_groups", default=None)
    prm.add_argument("--prm_tensor_parallel_size", "--prm_tp_size", type=int, default=1)
    prm.add_argument("--prm_dtype", default="auto")
    prm.add_argument("--prm_max_length", type=int, default=4096)
    prm.add_argument("--prm_positive_token", default=None)
    prm.add_argument("--prm_phase_marker", default="")
    prm.add_argument("--prm_phase_separator", default="\n\n")
    prm.add_argument("--prm_trust_remote_code", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser(); args = parser.parse_args(argv)
    if args.prm_max_length < 1:
        parser.error("--prm_max_length must be at least 1")
    if args.prm_tensor_parallel_size < 1:
        parser.error("--prm_tensor_parallel_size must be at least 1")
    if args.prm_gpus and args.prm_device.lower() == "cpu":
        parser.error("--prm_device=cpu cannot be combined with --prm_gpus")
    if not args.prm_phase_separator:
        parser.error("--prm_phase_separator must not be empty")
    args.bon_checkpoint_path = args.prm_model_path
    args.bon_training_metrics_json = None
    args.bon_transformer_type = "auto"
    args.bon_fold_index = None
    args.bon_value_device = "auto"
    args.bon_value_residency = "auto"
    args.bon_scoring_description = "BBH PRM scoring"
    return args


def _build_service(args: argparse.Namespace) -> _PRMBestOfNServiceType:
    if _PRM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "BBH PRM execution requires the scripts/transformer_bon runtime "
            "and its PyTorch/numpy/Transformers dependencies."
        ) from _PRM_IMPORT_ERROR
    return _RuntimePRMBestOfNService(_RuntimePRMBestOfNConfig(
        model_path=args.bon_model_path, checkpoint_path=args.prm_model_path,
        prm_model_path=args.prm_model_path, prm_name=args.prm_name,
        prm_device=args.prm_device, prm_gpus=args.prm_gpus,
        prm_tensor_parallel_size=args.prm_tensor_parallel_size, prm_dtype=args.prm_dtype,
        prm_max_length=args.prm_max_length, prm_positive_token=args.prm_positive_token,
        prm_phase_marker=args.prm_phase_marker, prm_phase_separator=args.prm_phase_separator,
        prm_trust_remote_code=args.prm_trust_remote_code, n=args.bon_N,
        output_aggregation="product", chat_template=args.bon_chat_template,
        enable_thinking=args.bon_enable_thinking, scoring_batch_size=args.bon_scoring_batch_size,
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
        model_transition=args.bon_model_transition, value_residency="auto",
        generation_max_active_jobs=args.bon_generation_max_active_jobs,
        staging_flush_records=args.bon_staging_flush_records, legacy_script_path=args.bon_legacy_script_path,
        vllm_cache_mode=args.bon_vllm_cache_mode, vllm_cache_root=args.bon_vllm_cache_root,
        vllm_enforce_eager=args.bon_vllm_enforce_eager,
        generation_worker_startup_timeout=args.bon_generation_worker_startup_timeout,
    ))


def run(args: argparse.Namespace, *, service: Any | None = None) -> dict[str, Any]:
    active_service = service or _build_service(args)
    owns_service = service is None
    try:
        return run_bbh(args, service=active_service)
    finally:
        if owns_service:
            active_service.close()


def main() -> None:
    metrics = run(parse_args())
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Refusal fraction: {metrics['refusal_fraction']}")
    print(f"Results: {metrics['results_csv']}")


if __name__ == "__main__":
    main()
