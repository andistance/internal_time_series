"""Small adapters for existing benchmark parsers and evaluators, without patching them."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from .step_prm import StepPRMBestOfNConfig, StepPRMBestOfNService

_PHASE_OPTIONS = {"num_segments", "entropy_threshold", "phase_partition_algorithm",
                  "high_entropy_ratio", "target_phase_tokens"}


def step_parser(parser):
    """Remove phase options/aliases while keeping dormant internal defaults."""
    for action in list(parser._actions):
        name = action.dest.removeprefix("bon_")
        if name in _PHASE_OPTIONS:
            parser.set_defaults(**{action.dest: action.default})
            parser._actions.remove(action)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
            for option in action.option_strings:
                parser._option_string_actions.pop(option, None)
        elif action.dest in {"score_model", "bon_scoring_backend"}:
            action.default, action.choices = "prm", ["prm"]
        elif action.dest == "generation_backend":
            action.default, action.choices = "transformer_bon", ["transformer_bon"]
    return parser


class StepArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        step_parser(self)
        return super().parse_args(args, namespace)


def load_step_entry(path, name):
    """Load a private evaluator namespace; original imports/globals stay intact.

    Separate globals let the existing functions resolve the step service and
    filtered parser. Dataset loading, evaluation, result formats and all other
    argument defaults remain the original functions' responsibility.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for symbol, value in {
        "PRMBestOfNConfig": StepPRMBestOfNConfig,
        "PRMBestOfNService": StepPRMBestOfNService,
        "_RuntimePRMBestOfNConfig": StepPRMBestOfNConfig,
        "_RuntimePRMBestOfNService": StepPRMBestOfNService,
    }.items():
        if hasattr(module, symbol):
            setattr(module, symbol, value)
    module.argparse = SimpleNamespace(**{**vars(argparse), "ArgumentParser": StepArgumentParser})
    if hasattr(module, "build_parser"):
        build = module.build_parser
        module.build_parser = lambda: step_parser(build())
    return module


def livecodebench_step_service(args):
    """Mirror TransformerBoNRunner's PRM config without duplicating its runner."""
    names = (
        "model_path checkpoint_path chat_template enable_thinking training_metrics_json transformer_type "
        "fold_index value_device scoring_batch_size candidate_batch_size max_tokens max_model_len "
        "temperature top_p logprobs tensor_parallel_size gpu_memory_utilization dtype trust_remote_code "
        "entropy_chunk_size entropy_model_path entropy_device_map seed performance_profile "
        "model_transition value_residency generation_max_active_jobs staging_flush_records legacy_script_path "
        "vllm_cache_mode vllm_cache_root vllm_enforce_eager generation_worker_startup_timeout "
        "prm_model_path prm_name prm_device prm_dtype prm_max_length prm_positive_token prm_phase_marker "
        "prm_phase_separator prm_trust_remote_code prm_gpus prm_tensor_parallel_size"
    ).split()
    kwargs = {name: getattr(args, "bon_" + name) for name in names}
    kwargs.update(n=args.bon_N, output_aggregation="product", gpu_groups=args.bon_gpus,
                  save_token_entropies=not args.bon_no_save_token_entropies)
    return StepPRMBestOfNService(StepPRMBestOfNConfig(**kwargs))
