"""Tensor implementations of the existing phase selectors, with identical ties.

Imports are lazy so CLI parsing and CPU-only callers do not require PyTorch.
Workers pass cuda:0; tests can use the same tensor kernels on CPU for parity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from .phases import adaptive_phase_count


def _rank_sample(candidates: Any, needed: int, torch: Any) -> Any:
    ranks = torch.ceil(
        torch.arange(1, needed + 1, dtype=torch.float64, device=candidates.device)
        * candidates.numel() / (needed + 1)
    ).to(torch.int64) - 1
    return candidates[ranks.clamp(0, candidates.numel() - 1)]


def select_original_boundaries_tensor(
    response_entropies: Sequence[float],
    entropy_threshold: float,
    num_segments: int,
    device: Any,
) -> Tuple[List[int], int]:
    """GPU equivalent of select_entropy_boundary_token_positions (float64)."""
    import torch

    entropies = torch.as_tensor(response_entropies, dtype=torch.float64, device=device)
    if entropies.ndim != 1:
        raise ValueError(f"response_entropies must be 1-D, got shape {tuple(entropies.shape)}")
    if entropy_threshold < 0:
        raise ValueError("entropy_threshold must be non-negative")
    if num_segments < 1:
        raise ValueError("num_segments/K must be at least 1")
    if entropies.numel() <= 1 or num_segments == 1:
        return [], 0
    valid = torch.arange(1, entropies.numel(), device=device)
    valid = valid[torch.isfinite(entropies[valid])]
    if valid.numel() == 0:
        return [], 0
    high_mask = entropies[valid] > entropy_threshold
    high = valid[high_mask]
    high_count = high.numel()
    needed = min(num_segments - 1, valid.numel())
    if high_count >= needed:
        selected = _rank_sample(high, needed, torch)
    else:
        low = valid[~high_mask]
        # Exactly NumPy's stable ascending argsort followed by reversal:
        # later token positions win equal-entropy fallback ties.
        order = torch.argsort(entropies[low], stable=True).flip(0)
        selected = torch.cat((high, low[order[:needed - high_count]]))
    selected = torch.unique(selected, sorted=True)
    if selected.numel() < needed:
        remaining = valid[~torch.isin(valid, selected)]
        order = torch.argsort(entropies[remaining], stable=True).flip(0)
        selected = torch.cat((selected, remaining[order[:needed - selected.numel()]]))
    return torch.sort(selected).values.cpu().tolist(), int(high_count)


def _token_step_indices(
    positions: Any,
    token_pieces: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    torch: Any,
) -> Any:
    """Original maximum-overlap/next-step mapping, computed in bounded GPU blocks."""
    device = positions.device
    token_starts = torch.tensor([piece["char_start"] for piece in token_pieces], device=device)
    token_ends = torch.tensor([piece["char_end"] for piece in token_pieces], device=device)
    step_starts = torch.tensor([step["char_start"] for step in steps], device=device)
    step_ends = torch.tensor([step["char_end"] for step in steps], device=device)
    mapped = torch.empty_like(positions)
    # Bound the overlap matrix to approximately one million int64 cells rather
    # than allocating a full [trajectory tokens, reasoning steps] matrix.
    block_size = max(1, 1048576 // len(steps))
    for start in range(0, positions.numel(), block_size):
        stop = min(start + block_size, positions.numel())
        ids = positions[start:stop]
        left = token_starts[ids]
        right = token_ends[ids]
        overlaps = (
            torch.minimum(right[:, None], step_ends[None, :])
            - torch.maximum(left[:, None], step_starts[None, :])
        ).clamp_min_(0)
        overlap, best = overlaps.max(dim=1)  # first step wins an overlap tie
        following = torch.searchsorted(step_starts, left.contiguous(), right=False)
        best = torch.where(overlap > 0, best, following)
        mapped[start:stop] = torch.where(best < len(steps), best, -1)
    return mapped


def select_adaptive_boundaries_tensor(
    response_entropies: Sequence[float],
    token_pieces: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    target_phase_tokens: float,
    high_entropy_ratio: float,
    device: Any,
) -> Tuple[List[int], int]:
    """GPU top-p, step mapping, deduplication, rank sampling and fallback."""
    import torch

    if not math.isfinite(high_entropy_ratio) or not 0 < high_entropy_ratio <= 1:
        raise ValueError("high_entropy_ratio must be in (0, 1]")
    count = adaptive_phase_count(len(response_entropies), target_phase_tokens, len(steps))
    if count <= 1:
        return [], 0
    entropies = torch.as_tensor(response_entropies, dtype=torch.float64, device=device)
    valid = torch.arange(1, entropies.numel(), device=device)
    valid = valid[torch.isfinite(entropies[valid])]
    if valid.numel() == 0:
        return [], 0
    order = torch.argsort(entropies[valid], descending=True, stable=True)
    high_count = max(1, math.ceil(high_entropy_ratio * valid.numel()))
    is_high = torch.zeros(entropies.numel(), dtype=torch.bool, device=device)
    is_high[valid[order[:high_count]]] = True
    mapped = _token_step_indices(valid, token_pieces[:len(response_entropies)], steps, torch)
    internal = (mapped > 0) & (mapped < len(steps))
    positions = valid[internal]
    step_ids = mapped[internal]
    scores = torch.full((len(steps),), -float("inf"), dtype=torch.float64, device=device)
    scores.scatter_reduce_(0, step_ids, entropies[positions], reduce="amax", include_self=True)
    representatives = torch.full((len(steps),), entropies.numel(), dtype=torch.int64, device=device)
    is_maximum = entropies[positions] == scores[step_ids]
    representatives.scatter_reduce_(
        0, step_ids[is_maximum], positions[is_maximum], reduce="amin", include_self=True,
    )
    available = torch.nonzero(representatives < entropies.numel(), as_tuple=True)[0]
    high_steps = torch.bincount(step_ids[is_high[positions]], minlength=len(steps)) > 0
    high = available[high_steps[available]]
    needed = min(count - 1, available.numel())
    if needed == 0:
        return [], int(high_count)
    if high.numel() >= needed:
        selected = _rank_sample(high, needed, torch)
    else:
        low = available[~high_steps[available]]
        order = torch.argsort(scores[low], stable=True).flip(0)
        selected = torch.cat((high, low[order[:needed - high.numel()]]))
    selected = torch.sort(selected).values
    return representatives[selected].cpu().tolist(), int(high_count)


def select_phase_boundaries_tensor(
    response_entropies: Sequence[float],
    token_pieces: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    args: Any,
    device: Any,
) -> Tuple[List[int], int]:
    if getattr(args, "phase_partition_algorithm", "original") == "adaptive":
        return select_adaptive_boundaries_tensor(
            response_entropies, token_pieces, steps,
            args.target_phase_tokens, args.high_entropy_ratio, device,
        )
    return select_original_boundaries_tensor(
        response_entropies, args.entropy_threshold, args.num_segments, device,
    )


def select_cached_batch(records: Sequence[Dict[str, Any]], args: Any, device: Any):
    """One transfer/synchronization per bounded batch, no token/step matrix.

    Search cumulative masks instead of CUDA nonzero. Precomputed fallback
    orders preserve the two selectors' distinct tie rules exactly.
    """
    import numpy as np
    import torch
    from .phase_structure import selection_arrays

    arrays = [selection_arrays(record, args) for record in records]
    if not arrays:
        return []
    width = max((len(row[0]) for row in arrays), default=0)
    output_width = max((row[3] for row in arrays), default=0)
    if width == 0 or output_width == 0:
        # Even when no boundary is available, adaptive candidate metadata can
        # be nonzero. Original K=1 explicitly reports zero, as before.
        result = []
        adaptive = getattr(args, "phase_partition_algorithm", "original") == "adaptive"
        for values, _, _, _, threshold, count in arrays:
            result.append(([], int(count if count is not None else np.count_nonzero(values > threshold))))
        return result
    batch_size = len(arrays)
    adaptive = getattr(args, "phase_partition_algorithm", "original") == "adaptive"
    host_values = np.full((batch_size, width), float("inf") if adaptive else -float("inf"), dtype=np.float64)
    host_fallback = np.zeros((batch_size, width), dtype=np.int64)
    lengths, needed, thresholds = [], [], []
    for index, (values, positions, fallback, size, threshold, _) in enumerate(arrays):
        length = len(values)
        host_values[index, :length] = values
        host_fallback[index, :length] = fallback
        lengths.append(length)
        needed.append(size)
        thresholds.append(threshold)
    values = torch.as_tensor(host_values, device=device)
    fallback = torch.as_tensor(host_fallback, device=device)
    thresholds = torch.tensor(thresholds, dtype=torch.float64, device=device)[:, None]
    valid = torch.arange(width, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
    high = ((values <= thresholds) if adaptive else (values > thresholds)) & valid
    counts = high.sum(1)
    cumulative = high.to(torch.int64).cumsum(1)
    sizes = torch.tensor(needed, device=device)[:, None]
    ordinal = torch.arange(1, output_width + 1, device=device)[None, :].expand(batch_size, -1)
    ranks = torch.ceil(ordinal.to(torch.float64) * counts[:, None] / (sizes + 1)).to(torch.int64)
    uniform = torch.searchsorted(cumulative.contiguous(), ranks.contiguous()).clamp_max(width - 1)
    all_high = torch.searchsorted(cumulative.contiguous(), ordinal.contiguous()).clamp_max(width - 1)
    low_mask = (~torch.gather(high, 1, fallback)) & valid
    low_cumulative = low_mask.to(torch.int64).cumsum(1)
    low_rank = (ordinal - counts[:, None]).clamp_min(1)
    low_index = torch.searchsorted(low_cumulative.contiguous(), low_rank.contiguous()).clamp_max(width - 1)
    low = torch.gather(fallback, 1, low_index)
    chosen = torch.where(counts[:, None] >= sizes, uniform,
                         torch.where(ordinal <= counts[:, None], all_high, low))
    chosen = torch.where(ordinal <= sizes, chosen, width)
    chosen = torch.sort(chosen, dim=1).values
    # A single device-to-host synchronization for the whole batch.
    packed = torch.cat((counts[:, None], chosen), dim=1).cpu().tolist()
    results = []
    for row, (_, positions, _, size, _, candidate_count) in zip(packed, arrays):
        count = row[0] if candidate_count is None else candidate_count
        results.append(([int(positions[index]) for index in row[1:1 + size]], int(count)))
    return results
