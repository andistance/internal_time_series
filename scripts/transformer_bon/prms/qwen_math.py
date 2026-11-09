"""Qwen2.5-Math-PRM-7B backend.

The scorer is kept as a thin named adapter around the battle-tested scorer in
``transformer_bon.prm``.  Keeping this alias preserves the exact tokenization,
``<extra_0>`` marker handling, reward-head checks, and probability extraction
used by the original reusable evaluator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List


def build_input(
    problem: Any,
    prefix: str,
    marker: str = "<extra_0>",
    separator: str = "\n\n",
    messages: Optional[Sequence[Dict[str, str]]] = None,
    raw_prompt: Optional[str] = None,
    phase_ends: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build the legacy Qwen process-PRM input without changing its schema."""

    from ..prm import split_steps

    problem_text = "" if problem is None else str(problem)
    parts = split_steps(prefix, separator)
    if not parts:
        raise ValueError("PRM prefix is empty")
    normalized_marker = marker or "<extra_0>"
    assistant = normalized_marker.join(parts) + normalized_marker
    return {
        "problem": problem_text,
        "assistant": assistant,
        "messages": [dict(message) for message in messages] if messages else None,
        "raw_prompt": raw_prompt,
        "phase_ends": [int(end) for end in phase_ends] if phase_ends is not None else None,
    }


def input_length(value: Any) -> int:
    from ..prm import _prm_input_length

    return _prm_input_length(value)


class QwenMathPRMScorer:
    """Named adapter for the legacy Qwen2.5-Math scorer implementation."""

    def __new__(cls, *args: Any, **kwargs: Any):
        # Import only when a PRM run actually starts.  This keeps transformer
        # generation-only commands independent of PyTorch/Transformers.
        from ..prm import HFPRMScorer

        return HFPRMScorer(*args, **kwargs)


def scorer_class():
    return QwenMathPRMScorer


def create_scorer(
    model_path: str,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    max_length: int = 4096,
    positive_token: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    return scorer_class()(
        model_path=model_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        max_length=max_length,
        positive_token=positive_token,
        system_prompt=system_prompt,
    )
