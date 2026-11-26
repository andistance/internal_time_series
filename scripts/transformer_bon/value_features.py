"""Vectorized legacy features without changing floating-point operation order."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


_RUNTIMES = {
    "3_best_of_n_via_phases_scores.py",
    "3_best_of_n_via_phases_scores_qwen.py",
    "3_best_of_n_via_phases_scores_qwen_sampled.py",
    "3_best_of_n_via_phases_scores_qwen_vllm_multi_signal.py",
}


def build_value_features(legacy, prepared, transformer_type):
    original = legacy.build_value_features
    code = getattr(original, "__code__", None)
    filename = Path(code.co_filename).name if code is not None else ""
    if filename not in _RUNTIMES or original.__name__ != "build_value_features":
        return original(prepared, transformer_type=transformer_type)
    steps = prepared.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return original(prepared, transformer_type=transformer_type)
    confidences = np.asarray([
        legacy.safe_float(step.get("token_avg_confidence"), default=np.nan)
        for step in steps
    ], dtype=np.float64)
    finite = np.isfinite(confidences)
    if not finite.all():
        # The multi-signal runtime alone permits a missing-confidence suffix.
        invalid = np.flatnonzero(~finite)
        stop = int(invalid[0])
        if "multi_signal" not in filename or not stop or finite[stop:].any():
            return original(prepared, transformer_type=transformer_type)
        steps, confidences = steps[:stop], confidences[:stop]
    starts, ends = legacy.compute_step_start_end_positions(steps)
    if not starts or not ends or max(map(abs, ends)) > 2**52:
        return original(prepared, transformer_type=transformer_type)
    count = len(confidences)
    features = np.empty((count, 4), dtype=np.float32)
    features[:, 0] = confidences
    if count >= 3:
        windows = np.lib.stride_tricks.sliding_window_view(confidences, 3)
        target = slice(2, None) if transformer_type == "causal" else slice(1, -1)
        features[target, 1] = windows.std(axis=1, ddof=0)
        # Preserve the original mean/subtract/multiply/sum/divide sequence.
        # Replacing this with last-first changes some FP32 rounding boundaries.
        grid = np.linspace(0.0, 1.0, 3)
        centered = grid - grid.mean()
        features[target, 2] = (
            ((windows - windows.mean(axis=1)[:, None]) * centered).sum(axis=1)
            / np.sum(centered ** 2)
        )
    edges = (range(min(2, count)) if transformer_type == "causal"
             else sorted({0, count - 1}))
    for index in edges:
        features[index, 1:3] = legacy.compute_local_std_and_slope(
            confidences, index, transformer_type=transformer_type,
        )
    features[:, 3] = np.clip(np.asarray(ends, dtype=np.float64) / max(ends[-1], 1), 0.0, 1.0)
    return features


def copy_steps(steps):
    """The canonical step records contain only immutable scalar values.

    Keep deepcopy semantics for custom nested records and aliased steps.
    """
    scalar = (str, int, float, bool, type(None))
    if (len({id(step) for step in steps}) == len(steps)
            and all(type(key) is str and type(value) in scalar
                    for step in steps for key, value in step.items())):
        return [dict(step) for step in steps]
    return copy.deepcopy(steps)


def prepare_value_items(legacy, records, normalizer, transformer_type):
    items = []
    for prepared in records:
        if prepared.get("status") != "ok":
            raise ValueError(prepared.get("error", "Candidate preparation failed"))
        features = normalizer.transform_one(build_value_features(legacy, prepared, transformer_type))
        if not prepared.get("phase_ranges"):
            raise ValueError("Candidate has no phase ranges")
        items.append((prepared, features))
    items.sort(key=lambda item: item[1].shape[0])
    return items


def attach_phase_scores(legacy, steps, phase_ranges, predictions):
    """Reuse the FP64 input array, retaining each original prefix reduction."""
    original = legacy.attach_phase_scores_to_steps
    aggregate = getattr(legacy, "aggregate_phase_scores", None)
    codes = (getattr(original, "__code__", None), getattr(aggregate, "__code__", None))
    if any(code is None or Path(code.co_filename).name not in _RUNTIMES for code in codes):
        return original(steps, phase_ranges, predictions)
    scores = []
    for pair in phase_ranges:
        if len(pair) != 2:
            return original(steps, phase_ranges, predictions)
        start, end = int(pair[0]), int(pair[1])
        if not 1 <= start <= end <= min(len(steps), len(predictions)):
            return original(steps, phase_ranges, predictions)
        scores.append(float(predictions[end - 1]))
    values = np.asarray(scores, dtype=np.float64)
    if not scores or not np.isfinite(values).all():
        return original(steps, phase_ranges, predictions)
    for index, pair in enumerate(phase_ranges):
        prefix = values[:index + 1]
        # Do not use cumprod/cumsum: those change the reduction order. Keep
        # clip/log/sum/exp and every prefix shape exactly as in the legacy code.
        clipped = np.clip(prefix, 1e-12, 1.0)
        cumulative = dict(product_score=float(np.exp(np.log(clipped).sum())),
                          min_score=float(prefix.min()), mean_score=float(prefix.mean()),
                          last_score=float(prefix[-1]))
        end = steps[int(pair[1]) - 1]
        end["phase_index"] = index
        end["phase_quality_score"] = scores[index]
        end["phase_product_score"] = cumulative["product_score"]
        end["phase_min_score"] = cumulative["min_score"]
        end["phase_mean_score"] = cumulative["mean_score"]
        end["phase_last_score"] = cumulative["last_score"]
    return scores, cumulative
