"""Registry for selectable PRM implementations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Sequence


DEFAULT_PRM_NAME = "qwen2.5-math-prm-7b"


@dataclass(frozen=True)
class PRMBackend:
    name: str
    module: str
    aliases: Sequence[str]

    def implementation(self):
        return import_module(self.module)

    def create_scorer(self, **kwargs: Any):
        return self.implementation().create_scorer(**kwargs)

    def build_input(self, **kwargs: Any) -> Dict[str, Any]:
        return self.implementation().build_input(**kwargs)

    def input_length(self, value: Any) -> int:
        return int(self.implementation().input_length(value))


_BACKENDS = (
    PRMBackend(
        DEFAULT_PRM_NAME,
        "scripts.transformer_bon.prms.qwen_math",
        (
            "qwen2.5-math-prm-7b",
            "qwen2.5-math-prm",
            "qwen_math",
            "qwen-math",
            "qwen2.5-math",
            "qwen2.5-math-prm-7b-hf",
        ),
    ),
    PRMBackend(
        "qwen-2.5-7b-fover-prm-2026",
        "scripts.transformer_bon.prms.fover",
        (
            "qwen-2.5-7b-fover-prm-2026",
            "qwen2.5-7b-fover-prm-2026",
            "fover",
            "fover-2026",
            "ryokamoi/qwen-2.5-7b-fover-prm-2026",
        ),
    ),
    PRMBackend(
        "versa-prm",
        "scripts.transformer_bon.prms.versa",
        (
            "versa-prm",
            "versaprm",
            "versa",
            "versa_prm",
            "versa-prm-7b",
            "uw-madison-lee-lab/versaprm",
        ),
    ),
    PRMBackend(
        "math-shepherd-mistral-7b-prm",
        "scripts.transformer_bon.prms.math_shepherd",
        (
            "math-shepherd-mistral-7b-prm",
            "math-shepherd",
            "math_shepherd",
            "mathshepherd",
            "peiyi9979/math-shepherd-mistral-7b-prm",
        ),
    ),
    PRMBackend(
        "rlhflow-llama3.1-8b-prm-deepseek-data",
        "scripts.transformer_bon.prms.rlhflow_deepseek",
        (
            "rlhflow-llama3.1-8b-prm-deepseek-data",
            "rlhflow-deepseek",
            "rlhflow_deepseek",
            "llama3.1-8b-prm-deepseek-data",
            "llama31-8b-prm-deepseek-data",
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        ),
    ),
    PRMBackend(
        "rlhflow-llama3.1-8b-prm-mistral-data",
        "scripts.transformer_bon.prms.rlhflow_mistral",
        (
            "rlhflow-llama3.1-8b-prm-mistral-data",
            "rlhflow-mistral-prm",
            "rlhflow-mistral",
            "llama3.1-8b-prm-mistral-data",
            "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
        ),
    ),
)


def available_prm_names() -> tuple[str, ...]:
    return tuple(backend.name for backend in _BACKENDS)


def normalize_prm_name(name: Optional[str]) -> str:
    requested = (name or DEFAULT_PRM_NAME).strip().lower()
    for backend in _BACKENDS:
        if requested == backend.name or requested in {alias.lower() for alias in backend.aliases}:
            return backend.name
    allowed = ", ".join(available_prm_names())
    raise ValueError(f"Unknown PRM {name!r}; choose one of: {allowed}")


def get_prm_backend(name: Optional[str]) -> PRMBackend:
    normalized = normalize_prm_name(name)
    return next(backend for backend in _BACKENDS if backend.name == normalized)
