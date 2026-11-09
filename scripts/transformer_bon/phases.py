"""Lightweight phase options and rollout-adaptive boundary selection."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


def add_phase_partition_arguments(parser: Any, prefix: str = "") -> None:
    parser.add_argument(
        f"--{prefix}phase_partition_algorithm",
        choices=("original", "adaptive"),
        default="original",
        help="original: fixed K/entropy threshold; adaptive: rollout length/top entropy ratio",
    )
    parser.add_argument(
        f"--{prefix}high_entropy_ratio",
        type=float,
        default=None,
        help="adaptive only: fraction of reasoning tokens ranked as high entropy, e.g. 0.05",
    )
    parser.add_argument(
        f"--{prefix}target_phase_tokens",
        type=float,
        default=None,
        help="adaptive only: desired rollout-average phase length in tokens",
    )


def validate_phase_partition_options(args: Any) -> None:
    algorithm = getattr(args, "phase_partition_algorithm", "original")
    if algorithm not in {"original", "adaptive"}:
        raise ValueError("phase_partition_algorithm must be original or adaptive")
    if algorithm == "original":
        if args.num_segments < 1:
            raise ValueError("num_segments/K must be at least 1")
        if not math.isfinite(args.entropy_threshold) or args.entropy_threshold < 0:
            raise ValueError("entropy_threshold must be finite and non-negative")
    else:
        ratio = getattr(args, "high_entropy_ratio", None)
        length = getattr(args, "target_phase_tokens", None)
        if ratio is None or not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("adaptive requires high_entropy_ratio in (0, 1]")
        if length is None or not math.isfinite(length) or length <= 0:
            raise ValueError("adaptive requires positive, finite target_phase_tokens")


def adaptive_phase_count(num_tokens: int, target_tokens: float, num_steps: int) -> int:
    """Exact minimizer of abs(T / K - L), with fewer phases winning ties."""
    if not math.isfinite(target_tokens) or target_tokens <= 0:
        raise ValueError("target_phase_tokens must be positive and finite")
    if num_tokens <= 0 or num_steps <= 0:
        return 0
    quotient = num_tokens / target_tokens
    # Clamp before rounding, including extremely small positive target lengths.
    quotient = min(float(num_steps), max(1.0, quotient))
    choices = {max(1, math.floor(quotient)), max(1, math.ceil(quotient))}
    return min(choices, key=lambda k: (abs(num_tokens / k - target_tokens), k))


def select_adaptive_boundaries(
    response_entropies: Sequence[float],
    token_pieces: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    target_phase_tokens: float,
    high_entropy_ratio: float,
    locate_boundary_step_index: Any,
) -> Tuple[List[int], int]:
    """Rank tokens, map/deduplicate steps, then use the original rank sampler.

    Returned positions are the highest-entropy representative tokens of the
    selected unique step boundaries, so the existing phase-range builder and
    prepared-record schema can be reused without modification. Entropy ties
    in top-p selection are resolved by increasing response-token index.
    """
    if not math.isfinite(high_entropy_ratio) or not 0 < high_entropy_ratio <= 1:
        raise ValueError("high_entropy_ratio must be in (0, 1]")
    count = adaptive_phase_count(len(response_entropies), target_phase_tokens, len(steps))
    if count <= 1:
        return [], 0
    valid = [
        t for t in range(1, len(response_entropies))
        if math.isfinite(float(response_entropies[t]))
    ]
    if not valid:
        return [], 0
    ranked = sorted(valid, key=lambda t: (-float(response_entropies[t]), t))
    high_count = max(1, math.ceil(high_entropy_ratio * len(valid)))
    high_tokens = set(ranked[:high_count])
    # Keep the threshold's top-n membership explicitly: equality at the cutoff
    # cannot be represented by a strict H > tau test when entropies are tied.
    representatives: Dict[int, int] = {}
    high_steps = set()
    for t in valid:
        step = locate_boundary_step_index(t, token_pieces, steps)
        if step is None or not 0 < step < len(steps):
            continue
        previous = representatives.get(step)
        if previous is None or response_entropies[t] > response_entropies[previous]:
            representatives[step] = t
        if t in high_tokens:
            high_steps.add(step)
    high = sorted(high_steps)
    needed = min(count - 1, len(representatives))
    if len(high) >= needed:
        # Exactly the legacy ceil(j * candidate_count / (needed + 1)) - 1 rule.
        selected = [
            high[math.ceil(j * len(high) / (needed + 1)) - 1]
            for j in range(1, needed + 1)
        ]
    else:
        selected = list(high)
        low = sorted(
            (step for step in representatives if step not in high_steps),
            # Legacy stable ascending entropy sort followed by reversal gives
            # later positions priority for equal fallback entropy.
            key=lambda step: (-float(response_entropies[representatives[step]]), -step),
        )
        selected.extend(low[:needed - len(selected)])
    return [representatives[step] for step in sorted(selected)], high_count
