#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-process DP+TP multi-source trainer (self-contained).

Usage: python THIS_FILE.py --gpu_groups "0,1;2,3" --tensor_parallel_size 2 [original arguments]
Use ordinary python, not torchrun. Indices follow CUDA_VISIBLE_DEVICES. Every
group must have the same TP size; no GPU can occur in two groups. Without new
arguments, the original single-device code path and all original defaults apply.

batch_size is GLOBAL, never per DP replica. One original DataLoader draw is
split into whole-trajectory microbatches without trimming global padding.
Predictions are reassembled in original order; the original hybrid loss is
computed ONCE over that full batch. DP gradients are SUMMED (not averaged again)
before ONE global clipping and ONE AdamW update on the canonical TP group.
Updated parameters are then broadcast to other DP replicas. The same logic is
used for ordinary validation, each CV fold, final refit and feature importance.
Checkpoint keys/shapes and metrics/CSV schemas are unchanged; replicas do not
inflate num_trainable_parameters. Global batches smaller than DP size work,
including the final partial batch; unused replicas do no work for that batch.

Equivalence boundary: this preserves the mathematical full-batch objective and
update rule, conditional on identical stochastic masks. It DOES NOT promise
bitwise-identical weights to the dense script across GPU layouts. TP/DP dropout
draws and floating-point operation order differ, even with the same seed. With
dropout disabled, forward/gradients are expected to agree within roundoff, not
bit for bit. Different validation losses can change early stopping and LR decay.
The original dense path is retained for reproducibility when no parallel layout
is selected. Do not interpret DP+TP as a guarantee of identical task accuracy.


Performance implementation preserves the sampler/iterator RNG draws, optimizer
updates, loss autograd graph, validation schedule, and output schemas. Small
fixed-label decisions and pair indices are reused; CPU metrics share records.
The single-trajectory CUDA path caches immutable inputs (32 MiB per loader),
buffers detached reporting values (8 MiB / 32 batches), and caches exact-shaped
causal masks (16 MiB per dense model). All caches are bounded and excluded from
checkpoints. Dataset arrays/metadata must not be mutated after loader creation.
Exact CPU/CUDA regression and throughput tools live in tests/*trainer_performance.py
(select the no-TP variant when running the shared tools).

Additional usage:
    python THIS_FILE.py --input_files qwen.jsonl ds.jsonl \
        --source_names Qwen DS --use_group_kfold --num_folds 5

Each input JSONL retains exactly the original schema. Source names are metadata
only and never model features. Existing CLI defaults, feature construction, model
architecture, optimizer, hybrid loss and output fields are retained. New metadata
and source metrics are appended. Single-source auto selection uses the original
validation hybrid loss; multi-source auto selection uses source-equal trajectory
RMSE. The original ReduceLROnPlateau still monitors validation hybrid loss.

Train a Small Transformer Encoder to predict step-level continuous value
from full-trajectory 4D confidence dynamics sequences.

Input:
    steps.jsonl

Each line is one sample:
{
  "sample_idx": 0,
  "problem": "...",
  "gold_answer": "...",
  "difficulty": 2.0,
  "steps": [
    {
      "step": "...",
      "token_avg_confidence": 0.8157,
      "first_token_index": 1,
      "step_prospective_correctness": 1.0
    },
    ...
  ],
  "is_correct": true
}

For every step, construct 4D features:
    1. conf_t
    2. local_conf_std: causal=[previous_previous, previous, current],
       bidirectional=[previous, current, next]
    3. local_conf_slope: causal=[previous_previous, previous, current],
       bidirectional=[previous, current, next]
    4. normalized end position, where end_position is next step's first_token_index

Model:
    Small Transformer Encoder over the full trajectory:
        input:  [num_steps, 4]
        output: [num_steps]
    Only steps with "step_prospective_correctness" are used for loss/evaluation.

Outputs:
    window_dataset.csv
    test_predictions.csv
    metrics.json
    feature_importance.csv
    curve.png

Usage:
    python train_small_transformer_value.py \
        --input_file steps.jsonl \
        --output_dir ./small_transformer_value \
        --group_by problem

Install:
    pip install torch numpy pandas scipy scikit-learn matplotlib tqdm
"""

import argparse
import copy
from collections import OrderedDict
import hashlib
from collections import defaultdict
from functools import lru_cache
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# =========================
# 1. Basic utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LinearWarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.completed_steps = 0
        self.target_lrs = [
            float(param_group["lr"])
            for param_group in optimizer.param_groups
        ]

        if self.warmup_steps > 0:
            initial_scale = 1.0 / float(self.warmup_steps)
            for param_group, target_lr in zip(
                self.optimizer.param_groups,
                self.target_lrs,
            ):
                param_group["lr"] = target_lr * initial_scale

    @property
    def is_finished(self) -> bool:
        return self.completed_steps >= self.warmup_steps

    def step(self) -> None:
        if self.is_finished:
            return

        self.completed_steps += 1
        next_scale = min(
            float(self.completed_steps + 1) / float(self.warmup_steps),
            1.0,
        )

        for param_group, target_lr in zip(
            self.optimizer.param_groups,
            self.target_lrs,
        ):
            param_group["lr"] = target_lr * next_scale


def build_warmup_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    num_epochs: int,
    log_prefix: str = "",
) -> Optional[LinearWarmupScheduler]:
    if not args.use_warmup or args.warmup_ratio <= 0:
        return None

    total_training_steps = max(1, int(num_epochs) * len(train_loader))
    warmup_steps = max(
        1,
        int(math.ceil(total_training_steps * args.warmup_ratio)),
    )

    print(
        f"[INFO] {log_prefix}Linear warmup enabled: "
        f"warmup_steps={warmup_steps}/{total_training_steps}, "
        f"warmup_ratio={args.warmup_ratio:.6f}"
    )

    return LinearWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
    )


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip invalid JSON at line {line_idx}: {e}")
                continue

            obj["_line_idx"] = line_idx
            samples.append(obj)

    return samples


def get_group_id(sample: Dict[str, Any], group_by: str) -> str:
    if group_by == "problem":
        problem = sample.get("problem", None)
        if isinstance(problem, str) and problem.strip():
            return "problem::" + problem.strip()
        return "line::" + str(sample.get("_line_idx", ""))

    if group_by == "sample_idx":
        sample_idx = sample.get("sample_idx", None)
        if sample_idx is not None:
            return "sample::" + str(sample_idx)
        return "line::" + str(sample.get("_line_idx", ""))

    if group_by == "line_idx":
        return "line::" + str(sample.get("_line_idx", ""))

    raise ValueError(f"Unknown group_by: {group_by}")


def estimate_text_token_len(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 1
    return max(1, len(re.findall(r"\S+", text)))


def linear_slope(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        return 0.0

    if len(arr) == 2:
        return float(arr[-1] - arr[0])

    x = np.linspace(0.0, 1.0, len(arr))
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = np.sum((x - x_mean) ** 2)

    if denom <= 0:
        return 0.0

    return float(np.sum((x - x_mean) * (arr - y_mean)) / denom)


# =========================
# 2. Feature construction
# =========================

def compute_step_start_end_positions(
    steps: List[Dict[str, Any]]
) -> Tuple[List[int], List[int]]:
    starts = []

    for step in steps:
        start = safe_int(step.get("first_token_index", None), default=-1)
        if start < 0:
            return [], []
        starts.append(start)

    ends = []

    for i, step in enumerate(steps):
        if i < len(steps) - 1:
            end_i = starts[i + 1]
        else:
            end_i = starts[i] + estimate_text_token_len(step.get("step", ""))

        if end_i <= starts[i]:
            end_i = starts[i] + 1

        ends.append(int(end_i))

    return starts, ends


def compute_local_std_and_slope(
    confidences: List[float],
    step_idx: int,
    transformer_type: str,
) -> Tuple[float, float]:
    n = len(confidences)
    local_indices = []

    if transformer_type == "causal":
        candidate_indices = [step_idx - 2, step_idx - 1, step_idx]
    elif transformer_type == "bidirectional":
        candidate_indices = [step_idx - 1, step_idx, step_idx + 1]
    else:
        raise ValueError(
            "transformer_type must be either 'bidirectional' or 'causal'."
        )

    for j in candidate_indices:
        if 0 <= j < n:
            local_indices.append(j)

    vals = [float(confidences[j]) for j in local_indices]

    if not vals:
        return 0.0, 0.0

    local_std = float(np.std(np.asarray(vals, dtype=float), ddof=0))
    local_slope = linear_slope(vals)

    return local_std, local_slope


def build_trajectory_sequences(
    samples: List[Dict[str, Any]],
    group_by: str,
    position_mode: str,
    transformer_type: str,
    build_frame: bool = True,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Build one sequence per sample.

    Each sequence:
        x: [num_steps, 4]
        y: [num_steps]
        label_mask: [num_steps], True only for steps with step_prospective_correctness
        metadata
    """
    sequences = []
    csv_rows = []

    for sample in samples:
        steps = sample.get("steps", [])
        if not isinstance(steps, list) or len(steps) == 0:
            continue

        sample_idx = sample.get("sample_idx", sample.get("_line_idx", None))
        line_idx = sample.get("_line_idx", None)
        problem = sample.get("problem", "")
        gold_answer = sample.get("gold_answer", "")
        difficulty = safe_float(sample.get("difficulty", np.nan), default=np.nan)
        is_correct = sample.get("is_correct", None)
        group_id = get_group_id(sample, group_by=group_by)

        confidences = []
        valid_sample = True

        for step in steps:
            conf = safe_float(step.get("token_avg_confidence", None), default=np.nan)
            if not np.isfinite(conf):
                valid_sample = False
                break
            confidences.append(float(conf))

        if not valid_sample:
            continue

        starts, ends = compute_step_start_end_positions(steps)
        if not starts or not ends:
            continue

        final_end_position = max(ends[-1], 1)
        position_normalizer = (
            final_end_position if position_mode == "relative" else 8192
        )

        x_list = []
        y_list = []
        label_mask_list = []
        step_meta = []

        for i, step in enumerate(steps):
            conf_i = float(confidences[i])
            local_std_i, local_slope_i = compute_local_std_and_slope(
                confidences=confidences,
                step_idx=i,
                transformer_type=transformer_type,
            )
            normalized_position_i = float(ends[i] / position_normalizer)
            normalized_position_i = max(0.0, min(1.0, normalized_position_i))

            features = [
                conf_i,
                local_std_i,
                local_slope_i,
                normalized_position_i,
            ]

            has_label = "step_prospective_correctness" in step
            y_i = safe_float(step.get("step_prospective_correctness", None), default=np.nan)

            if has_label and np.isfinite(y_i):
                y_value = float(y_i)
                mask_value = True
            else:
                y_value = 0.0
                mask_value = False

            x_list.append(features)
            y_list.append(y_value)
            label_mask_list.append(mask_value)

            meta_i = {
                "sample_idx": sample_idx,
                "line_idx": line_idx,
                "problem": problem,
                "gold_answer": gold_answer,
                "difficulty": difficulty,
                "is_correct": is_correct,
                "group_id": group_id,
                "step_idx": i,
                "num_steps": len(steps),
                "step_text": step.get("step", ""),
                "has_label": bool(mask_value),
                "y_true": y_value if mask_value else np.nan,
                "conf": conf_i,
                "local_conf_std": local_std_i,
                "local_conf_slope": local_slope_i,
                "normalized_position": normalized_position_i,
                "first_token_index": starts[i],
                "end_token_index": ends[i],
            }

            step_meta.append(meta_i)
            if build_frame:
                csv_rows.append(meta_i.copy())

        # 没有任何 labeled step 的样本无法提供监督，直接跳过
        if not any(label_mask_list):
            continue

        seq = {
            "sequence_id": len(sequences),
            "sample_idx": sample_idx,
            "line_idx": line_idx,
            "problem": problem,
            "gold_answer": gold_answer,
            "difficulty": difficulty,
            "is_correct": is_correct,
            "group_id": group_id,
            "x": np.asarray(x_list, dtype=np.float32),
            "y": np.asarray(y_list, dtype=np.float32),
            "label_mask": np.asarray(label_mask_list, dtype=bool),
            "step_meta": step_meta,
        }

        sequences.append(seq)

    df = pd.DataFrame(csv_rows)
    return sequences, df


def should_drop_zero_dominated_sample(
    sequence: Dict[str, Any],
    zero_label_ratio_threshold: float,
) -> bool:
    """Apply the optional sample-level zero-label filtering rule."""
    labeled_values = sequence["y"][sequence["label_mask"]]
    if labeled_values.size == 0:
        return False

    zero_mask = np.isclose(
        labeled_values,
        0.0,
        atol=1e-12,
        rtol=0.0,
    )
    first_labeled_is_zero = bool(zero_mask[0])
    zero_ratio = float(np.mean(zero_mask))

    return (
        first_labeled_is_zero
        and zero_ratio >= zero_label_ratio_threshold
    )


def select_retained_filter_candidates(sequences, candidate_indices, keep_ratio, seed, split_name, stage):
    """Keep floor(N * ratio) candidates per source, without using training RNG.

    Stable identities make the sample selection reproducible even when input
    order changes. The split and stage are independent sampling strata.
    """
    candidates_by_source = defaultdict(list)
    for index in candidate_indices:
        sequence = sequences[index]
        source = sequence.get("source_name", "source_1")
        identity = sequence.get("sample_uid")
        if identity is None:
            identity = json.dumps([sequence.get("group_id"), sequence.get("sample_idx"),
                                   sequence.get("line_idx"), sequence.get("sequence_id")], ensure_ascii=False)
        digest = hashlib.sha256(json.dumps([seed, split_name, stage, source, identity],
                                           ensure_ascii=False).encode("utf-8")).digest()
        candidates_by_source[source].append((digest, index))
    retained = set()
    for source, candidates in sorted(candidates_by_source.items()):
        count = int(math.floor(len(candidates) * keep_ratio))
        retained.update(index for _, index in sorted(candidates)[:count])
        print(f"[INFO] {split_name} {stage} source={source}: "
              f"candidates={len(candidates)}, retained={count}, "
              f"removed={len(candidates) - count}, keep_ratio={keep_ratio}")
    return retained


def remove_short_low_confidence_steps_from_sequence(
    sequence: Dict[str, Any],
    confidence_gap_threshold: float,
    short_step_token_threshold: int,
    position_mode: str,
    transformer_type: str,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Remove interior steps whose confidence is lower than both neighboring
    confidences by at least confidence_gap_threshold and whose original token
    span is strictly smaller than short_step_token_threshold.

    All qualifying steps are identified from the original sequence and removed
    simultaneously. The retained sequence features are then rebuilt.
    """
    step_meta = sequence["step_meta"]
    num_steps = len(step_meta)

    if num_steps < 3:
        return sequence, 0

    confidences = [float(meta["conf"]) for meta in step_meta]
    keep_mask = [True] * num_steps

    for step_idx in range(1, num_steps - 1):
        current_conf = confidences[step_idx]
        previous_conf = confidences[step_idx - 1]
        next_conf = confidences[step_idx + 1]

        first_token_index = safe_int(
            step_meta[step_idx].get("first_token_index", None),
            default=-1,
        )
        end_token_index = safe_int(
            step_meta[step_idx].get("end_token_index", None),
            default=-1,
        )
        token_count = max(0, end_token_index - first_token_index)

        confidence_is_low = (
            previous_conf - current_conf >= confidence_gap_threshold
            and next_conf - current_conf >= confidence_gap_threshold
        )
        step_is_short = token_count < short_step_token_threshold

        if confidence_is_low and step_is_short:
            keep_mask[step_idx] = False

    removed_steps = int(num_steps - sum(keep_mask))
    if removed_steps == 0:
        return sequence, 0

    retained_meta = [
        meta for meta, keep in zip(step_meta, keep_mask) if keep
    ]

    pseudo_steps = [
        {
            "first_token_index": meta["first_token_index"],
            "step": meta.get("step_text", ""),
        }
        for meta in retained_meta
    ]
    starts, ends = compute_step_start_end_positions(pseudo_steps)
    if not starts or not ends:
        return None, removed_steps

    retained_confidences = [float(meta["conf"]) for meta in retained_meta]
    final_end_position = max(ends[-1], 1)
    position_normalizer = (
        final_end_position if position_mode == "relative" else 8192
    )

    x_list = []
    y_list = []
    label_mask_list = []
    rebuilt_step_meta = []
    retained_num_steps = len(retained_meta)

    for new_step_idx, meta in enumerate(retained_meta):
        conf_i = retained_confidences[new_step_idx]
        local_std_i, local_slope_i = compute_local_std_and_slope(
            confidences=retained_confidences,
            step_idx=new_step_idx,
            transformer_type=transformer_type,
        )
        normalized_position_i = float(
            ends[new_step_idx] / position_normalizer
        )
        normalized_position_i = max(
            0.0,
            min(1.0, normalized_position_i),
        )

        has_label = bool(meta["has_label"])
        y_value = float(meta["y_true"]) if has_label else 0.0

        x_list.append([
            conf_i,
            local_std_i,
            local_slope_i,
            normalized_position_i,
        ])
        y_list.append(y_value)
        label_mask_list.append(has_label)

        rebuilt_meta = meta.copy()
        rebuilt_meta.update({
            "step_idx": new_step_idx,
            "num_steps": retained_num_steps,
            "has_label": has_label,
            "y_true": y_value if has_label else np.nan,
            "conf": conf_i,
            "local_conf_std": local_std_i,
            "local_conf_slope": local_slope_i,
            "normalized_position": normalized_position_i,
            "first_token_index": starts[new_step_idx],
            "end_token_index": ends[new_step_idx],
        })
        rebuilt_step_meta.append(rebuilt_meta)

    if not any(label_mask_list):
        return None, removed_steps

    filtered_sequence = sequence.copy()
    filtered_sequence["x"] = np.asarray(x_list, dtype=np.float32)
    filtered_sequence["y"] = np.asarray(y_list, dtype=np.float32)
    filtered_sequence["label_mask"] = np.asarray(
        label_mask_list,
        dtype=bool,
    )
    filtered_sequence["step_meta"] = rebuilt_step_meta

    return filtered_sequence, removed_steps


def apply_optional_sequence_filters(
    sequences: List[Dict[str, Any]],
    args: argparse.Namespace,
    split_name: str,
) -> List[Dict[str, Any]]:
    """Apply the two optional filtering rules to one data split."""
    if (
        not args.remove_short_low_confidence_steps
        and not args.remove_zero_dominated_samples
    ):
        return sequences

    keep_ratio = getattr(args, "filtered_sample_keep_ratio", 0.0)
    if not np.isfinite(keep_ratio) or not 0.0 <= keep_ratio <= 1.0:
        raise ValueError("filtered_sample_keep_ratio must be in [0, 1].")
    zero_candidates = {
        index for index, sequence in enumerate(sequences)
        if args.remove_zero_dominated_samples and should_drop_zero_dominated_sample(
            sequence, args.zero_label_ratio_threshold)
    }
    retained_zero = select_retained_filter_candidates(
        sequences, zero_candidates, keep_ratio, getattr(args, "random_state", 42),
        split_name, "zero_dominated")
    filtered_sequences = []
    removed_zero_dominated_samples = 0
    removed_no_label_samples = 0
    removed_short_low_confidence_steps = 0

    for index, sequence in enumerate(sequences):
        if index in zero_candidates and index not in retained_zero:
            removed_zero_dominated_samples += 1
            continue

        filtered_sequence = sequence
        if args.remove_short_low_confidence_steps:
            filtered_sequence, removed_steps = (
                remove_short_low_confidence_steps_from_sequence(
                    sequence=sequence,
                    confidence_gap_threshold=(
                        args.low_confidence_gap_threshold
                    ),
                    short_step_token_threshold=(
                        args.short_step_token_threshold
                    ),
                    position_mode=args.position_mode,
                    transformer_type=args.transformer_type,
                )
            )
            removed_short_low_confidence_steps += removed_steps

            if filtered_sequence is None:
                removed_no_label_samples += 1
                continue

        filtered_sequences.append(filtered_sequence)

    print(
        f"[INFO] Optional filtering on {split_name}: "
        f"sequences {len(sequences)} -> {len(filtered_sequences)}, "
        f"removed_zero_dominated_samples="
        f"{removed_zero_dominated_samples}, "
        f"removed_samples_without_labels_after_step_filter="
        f"{removed_no_label_samples}, "
        f"removed_short_low_confidence_steps="
        f"{removed_short_low_confidence_steps}"
    )

    return filtered_sequences


# =========================
# 3. Split and normalization
# =========================

def split_train_valid_test_sequences(
    sequences: List[Dict[str, Any]],
    test_size: float,
    valid_size: float,
    random_state: int,
) -> Tuple[List[int], List[int], List[int]]:
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in (0, 1).")

    if valid_size <= 0 or valid_size >= 1:
        raise ValueError("valid_size must be in (0, 1).")

    if test_size + valid_size >= 1:
        raise ValueError("test_size + valid_size must be < 1.")

    indices = np.arange(len(sequences))
    groups = np.asarray([seq["group_id"] for seq in sequences])

    splitter_test = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_valid_idx, test_idx = next(
        splitter_test.split(indices, groups=groups)
    )

    groups_train_valid = groups[train_valid_idx]
    inner_valid_size = valid_size / (1.0 - test_size)

    splitter_valid = GroupShuffleSplit(
        n_splits=1,
        test_size=inner_valid_size,
        random_state=random_state + 1,
    )

    inner_train_idx, inner_valid_idx = next(
        splitter_valid.split(
            train_valid_idx,
            groups=groups_train_valid,
        )
    )

    train_idx = train_valid_idx[inner_train_idx]
    valid_idx = train_valid_idx[inner_valid_idx]

    train_groups = set(groups[train_idx].tolist())
    valid_groups = set(groups[valid_idx].tolist())
    test_groups = set(groups[test_idx].tolist())

    if train_groups & valid_groups:
        raise RuntimeError("Data leakage: train and valid groups overlap.")
    if train_groups & test_groups:
        raise RuntimeError("Data leakage: train and test groups overlap.")
    if valid_groups & test_groups:
        raise RuntimeError("Data leakage: valid and test groups overlap.")

    return train_idx.tolist(), valid_idx.tolist(), test_idx.tolist()


class FeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, sequences: List[Dict[str, Any]]) -> None:
        xs = []
        for seq in sequences:
            xs.append(seq["x"])
        all_x = np.concatenate(xs, axis=0)

        self.mean = all_x.mean(axis=0)
        self.std = all_x.std(axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def transform_sequences(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        transformed = []
        for seq in sequences:
            # Features are replaced; labels and metadata are read-only here.
            new_seq = seq.copy()
            new_seq["x"] = self.transform_one(seq["x"])
            transformed.append(new_seq)
        return transformed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "feature_names": [
                "conf",
                "local_conf_std",
                "local_conf_slope",
                "normalized_position",
            ],
        }


# =========================
# 4. Dataset and dataloader
# =========================

class TrajectoryDataset(Dataset):
    def __init__(self, sequences: List[Dict[str, Any]], split_name: str):
        self.sequences = sequences
        self.split_name = split_name
        # CPU views share the immutable arrays instead of copying them per draw.
        self.tensors = [
            (torch.as_tensor(seq["x"], dtype=torch.float32),
             torch.as_tensor(seq["y"], dtype=torch.float32),
             torch.as_tensor(seq["label_mask"], dtype=torch.bool))
            for seq in sequences
        ]
        self.label_counts = [int(mask.sum()) for _, _, mask in self.tensors]
        # Only Python booleans are cached per trajectory/device, never graphs.
        self.loss_pair_flags = [{} for _ in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.sequences[idx]

        return {
            "x": self.tensors[idx][0],
            "y": self.tensors[idx][1],
            "label_mask": self.tensors[idx][2],
            "_dataset_index": idx,
            "_label_count": self.label_counts[idx],
            "source_name": sequence_source(seq),
            "sequence_id": seq["sequence_id"],
            "sample_idx": seq["sample_idx"],
            "line_idx": seq["line_idx"],
            "problem": seq["problem"],
            "gold_answer": seq["gold_answer"],
            "difficulty": seq["difficulty"],
            "is_correct": seq["is_correct"],
            "group_id": seq["group_id"],
            "step_meta": seq["step_meta"],
            "split": self.split_name,
        }


def collate_trajectory_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    # Keep text in the dataset, outside worker queues and the batch hot path.
    metas = [
        {key: item[key] for key in ("sequence_id", "source_name", "split", "_dataset_index")}
        if "_dataset_index" in item else item
        for item in batch
    ]
    label_counts = tuple(
        item["_label_count"] if "_label_count" in item else int(item["label_mask"].sum())
        for item in batch
    )
    if batch_size == 1:
        item = batch[0]
        return {
            "x": item["x"].unsqueeze(0),
            "y": item["y"].unsqueeze(0),
            "label_mask": item["label_mask"].unsqueeze(0),
            "attention_mask": torch.ones(1, item["x"].shape[0], dtype=torch.bool),
            "metas": metas,
            "label_counts": label_counts,
        }
    max_len = max(item["x"].shape[0] for item in batch)
    feature_dim = batch[0]["x"].shape[1]

    x = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    y = torch.zeros(batch_size, max_len, dtype=torch.float32)
    label_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item["x"].shape[0]
        x[i, :seq_len] = item["x"]
        y[i, :seq_len] = item["y"]
        label_mask[i, :seq_len] = item["label_mask"]
        attention_mask[i, :seq_len] = True

    return {
        "x": x,
        "y": y,
        "label_mask": label_mask,
        "attention_mask": attention_mask,
        "metas": metas,
        "label_counts": label_counts,
    }


# =========================
# 5. Model
# =========================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SmallTransformerEncoderValueModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.20,
        max_len: int = 2048,
        transformer_type: str = "bidirectional",
    ):
        super().__init__()

        if transformer_type not in {"bidirectional", "causal"}:
            raise ValueError(
                "transformer_type must be either 'bidirectional' or 'causal'."
            )

        self.transformer_type = transformer_type
        # Exact-shaped boolean masks, bounded by both bytes and entry count.
        # Plain tensors deliberately stay out of checkpoints.
        self._causal_masks = OrderedDict()
        self._causal_mask_bytes = 0
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            [batch_size, seq_len, input_dim]
        attention_mask:
            [batch_size, seq_len], True for valid token, False for padding

        return:
            pred: [batch_size, seq_len], value in [0, 1]
        """
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        key_padding_mask = ~attention_mask

        if self.transformer_type == "causal":
            seq_len = h.size(1)
            cache_key = (h.device, seq_len)
            causal_mask = self._causal_masks.get(cache_key)
            if causal_mask is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=h.device, dtype=torch.bool),
                    diagonal=1,
                )
                size = causal_mask.numel() * causal_mask.element_size()
                budget = 16 * 1024 * 1024
                if size <= budget:
                    while self._causal_masks and (
                        self._causal_mask_bytes + size > budget or len(self._causal_masks) >= 64
                    ):
                        _, previous = self._causal_masks.popitem(last=False)
                        self._causal_mask_bytes -= previous.numel() * previous.element_size()
                    self._causal_masks[cache_key] = causal_mask
                    self._causal_mask_bytes += size
            else:
                self._causal_masks.move_to_end(cache_key)
        else:
            causal_mask = None

        h = self.encoder(
            h,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
            is_causal=(self.transformer_type == "causal"),
        )

        logits = self.output_head(h).squeeze(-1)
        pred = torch.sigmoid(logits)

        return pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in optimization_parameters(model) if p.requires_grad)


def _snapshot_state_dict(model: nn.Module):
    """Independent CPU snapshot, without a second full GPU model allocation."""
    if isinstance(model, DataTensorParallelValueModel):
        return model.state_dict()  # Already independent, full, CPU tensors.
    state = model.state_dict()
    snapshot = OrderedDict((key, value.detach().to("cpu", copy=True))
                           for key, value in state.items())
    if hasattr(state, "_metadata"):
        snapshot._metadata = copy.deepcopy(state._metadata)
    return snapshot


# =========================
# 6. Metrics
# =========================

def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Regression/probability ECE for continuous labels in [0, 1].

    For each prediction bin:
        calibration gap = |mean(pred) - mean(true)|

    ECE:
        weighted average of calibration gaps.
    """
    if len(y_true) == 0:
        return np.nan

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]

        if i == n_bins - 1:
            mask = (y_pred >= lower) & (y_pred <= upper)
        else:
            mask = (y_pred >= lower) & (y_pred < upper)

        if not np.any(mask):
            continue

        bin_pred = float(np.mean(y_pred[mask]))
        bin_true = float(np.mean(y_true[mask]))
        bin_weight = float(np.sum(mask) / total)
        ece += bin_weight * abs(bin_pred - bin_true)

    return float(ece)


@lru_cache(maxsize=64)
def _small_metric_pair_indices(length):
    # Only called for <= 64 labels: at most about 2 MiB across the entire cache.
    return np.triu_indices(length, k=1)


def _trajectory_metric_components(record):
    y_true = np.asarray(record.get("y_true", []), dtype=float)
    y_pred = np.asarray(record.get("y_pred", []), dtype=float)
    if len(y_true) == 0:
        return None
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    has_spearman = (len(y_true) >= 2 and len(np.unique(y_true)) > 1
                    and len(np.unique(y_pred)) > 1)
    sp = float(spearmanr(y_true, y_pred).statistic) if has_spearman else np.nan
    correct = 0.0
    total = 0

    def count_pairs(true_diff, pred_diff, upper=None):
        valid = ~np.isclose(true_diff, 0.0, atol=1e-12, rtol=0.0)
        if upper is not None:
            valid &= upper
        tied = np.isclose(pred_diff, 0.0, atol=1e-12, rtol=0.0)
        return (
            0.5 * int(np.count_nonzero(valid & tied))
            + int(np.count_nonzero(valid & ~tied & (true_diff * pred_diff > 0))),
            int(np.count_nonzero(valid)),
        )

    if len(y_true) <= 64:
        i, j = _small_metric_pair_indices(len(y_true))
        correct, total = count_pairs(y_true[i] - y_true[j], y_pred[i] - y_pred[j])
    else:
        # Bound temporary arrays; do not allocate an L x L matrix for long data.
        rows_per_chunk = max(1, 65536 // len(y_true))
        columns = np.arange(len(y_true))[None, :]
        for start in range(0, len(y_true), rows_per_chunk):
            stop = min(start + rows_per_chunk, len(y_true))
            rows = np.arange(start, stop)[:, None]
            c, n = count_pairs(
                y_true[start:stop, None] - y_true[None, :],
                y_pred[start:stop, None] - y_pred[None, :],
                columns > rows,
            )
            correct += c
            total += n
    return rmse, sp, correct, total, has_spearman


def compute_trajectory_metrics(
    trajectory_predictions: Dict[Any, Dict[str, List[float]]],
    _cache: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute metrics that must be evaluated within each trajectory.

    Metrics:
        1. macro_trajectory_rmse:
           Compute RMSE separately for each trajectory, then average across
           trajectories. Every trajectory receives equal weight.

        2. within_response_spearman:
           Compute Spearman correlation separately within each trajectory,
           then average over trajectories for which Spearman is defined
           (at least two labeled steps and non-constant true/predicted values).

        3. pairwise_ranking_accuracy:
           Within each trajectory, compare every pair of labeled steps whose
           true labels differ. A correctly ordered prediction receives 1,
           an incorrectly ordered prediction receives 0, and a predicted tie
           receives 0.5. The final score is micro-averaged over all valid pairs.
    """
    if not trajectory_predictions:
        return {
            "macro_trajectory_rmse": np.nan,
            "within_response_spearman": np.nan,
            "pairwise_ranking_accuracy": np.nan,
        }

    trajectory_rmses = []
    trajectory_spearmans = []
    pairwise_correct = 0.0
    pairwise_total = 0

    for record in trajectory_predictions.values():
        key = id(record)
        if _cache is not None and key in _cache:
            components = _cache[key]
        else:
            components = _trajectory_metric_components(record)
            if _cache is not None:
                _cache[key] = components
        if components is None:
            continue
        trajectory_rmse, sp, correct, total, _ = components
        trajectory_rmses.append(trajectory_rmse)
        if np.isfinite(sp):
            trajectory_spearmans.append(sp)
        pairwise_correct += correct
        pairwise_total += total

    macro_trajectory_rmse = (
        float(np.mean(trajectory_rmses))
        if trajectory_rmses
        else np.nan
    )
    within_response_spearman = (
        float(np.mean(trajectory_spearmans))
        if trajectory_spearmans
        else np.nan
    )
    pairwise_ranking_accuracy = (
        float(pairwise_correct / pairwise_total)
        if pairwise_total > 0
        else np.nan
    )

    return {
        "macro_trajectory_rmse": macro_trajectory_rmse,
        "within_response_spearman": within_response_spearman,
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return {
            "spearman": np.nan,
            "spearman_pvalue": np.nan,
            "pearson": np.nan,
            "pearson_pvalue": np.nan,
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "bias": np.nan,
            "ece": np.nan,
        }

    if len(y_true) >= 2 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        sp = spearmanr(y_true, y_pred)
        pr = pearsonr(y_true, y_pred)
        spearman = float(sp.statistic)
        spearman_pvalue = float(sp.pvalue)
        pearson = float(pr.statistic)
        pearson_pvalue = float(pr.pvalue)
    else:
        spearman = np.nan
        spearman_pvalue = np.nan
        pearson = np.nan
        pearson_pvalue = np.nan

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    bias = float(np.mean(y_pred - y_true))
    ece = compute_ece(y_true, y_pred, n_bins=n_bins)

    return {
        "spearman": spearman,
        "spearman_pvalue": spearman_pvalue,
        "pearson": pearson,
        "pearson_pvalue": pearson_pvalue,
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "ece": ece,
    }



# Hybrid loss weights are fixed intentionally so that the command-line
# interface and hyperparameter set remain unchanged.
HYBRID_ABSOLUTE_WEIGHT = 0.8
HYBRID_TRAJECTORY_MEAN_WEIGHT = 0.0
HYBRID_CENTERED_WEIGHT = 0.0
HYBRID_PAIRWISE_WEIGHT = 0.2
HYBRID_DELTA_WEIGHT = 0.0

PAIRWISE_TEMPERATURE = 0.20
DELTA_SMOOTH_L1_BETA = 0.10


@lru_cache(maxsize=64)
def _small_loss_pair_indices(length, device):
    # Cache small index tensors only; <= 2 MiB with at most 64 labels per entry.
    return torch.triu_indices(length, length, offset=1, device=device)


def compute_hybrid_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    label_mask: torch.Tensor,
    _label_counts=None,
    _pair_flags=None,
) -> torch.Tensor:
    """
    Hybrid PRM-style objective over labeled phase-end steps.

    Components:
        1. Absolute phase-value MSE:
           Preserves calibrated phase values and cross-trajectory quality.

        2. Trajectory-mean MSE:
           Explicitly preserves ranking/quality differences between complete
           trajectories by matching the mean labeled value of each trajectory.

        3. Trajectory-centered MSE:
           Removes each trajectory's mean and learns within-trajectory phase
           deviations independently of overall problem/trajectory difficulty.

        4. Difference-weighted pairwise logistic ranking loss:
           Orders phase pairs within the same trajectory. Pairs whose Monte
           Carlo labels differ more receive more weight; equal-label pairs are
           ignored.

        5. Adjacent-delta Smooth L1:
           Learns the local direction and magnitude of value changes between
           consecutive labeled phases while remaining robust to noisy labels.
    """
    if (sum(_label_counts) if _label_counts is not None else label_mask.sum().item()) == 0:
        return pred.sum() * 0.0

    # Micro-average over all labeled phase endpoints in the batch.
    absolute_loss = F.mse_loss(
        pred[label_mask],
        y[label_mask],
        reduction="mean",
    )

    trajectory_mean_losses = []
    centered_losses = []
    pairwise_losses = []
    delta_losses = []

    batch_size = pred.size(0)

    for b_idx in range(batch_size):
        sample_mask = label_mask[b_idx]
        if (_label_counts[b_idx] if _label_counts is not None else sample_mask.sum().item()) == 0:
            continue

        sample_pred = pred[b_idx][sample_mask]
        sample_true = y[b_idx][sample_mask]

        # Between-trajectory signal: compare each trajectory's mean value.
        trajectory_mean_losses.append(
            (sample_pred.mean() - sample_true.mean()).pow(2)
        )

        if sample_pred.numel() < 2:
            continue

        # Within-trajectory magnitude signal after removing global level.
        centered_pred = sample_pred - sample_pred.mean()
        centered_true = sample_true - sample_true.mean()
        centered_losses.append(
            F.mse_loss(centered_pred, centered_true, reduction="mean")
        )

        # Local phase-to-phase quality changes.
        pred_delta = sample_pred[1:] - sample_pred[:-1]
        true_delta = sample_true[1:] - sample_true[:-1]
        delta_losses.append(
            F.smooth_l1_loss(
                pred_delta,
                true_delta,
                reduction="mean",
                beta=DELTA_SMOOTH_L1_BETA,
            )
        )

        # All valid unordered phase pairs within one trajectory.
        if sample_pred.numel() <= 64:
            pair_indices = _small_loss_pair_indices(sample_pred.numel(), sample_pred.device)
        else:
            pair_indices = torch.triu_indices(
                sample_pred.numel(), sample_pred.numel(), offset=1, device=sample_pred.device,
            )
        pred_diff = (
            sample_pred[pair_indices[0]]
            - sample_pred[pair_indices[1]]
        )
        true_diff = (
            sample_true[pair_indices[0]]
            - sample_true[pair_indices[1]]
        )

        valid_pair_mask = true_diff.abs() > 1e-12
        if _pair_flags is None:
            has_valid_pairs = bool(valid_pair_mask.any())
        else:
            flags = _pair_flags[b_idx]
            flag_key = (sample_true.device, sample_true.dtype)
            if flag_key not in flags:
                flags[flag_key] = bool(valid_pair_mask.any())
            has_valid_pairs = flags[flag_key]
        if has_valid_pairs:
            pred_diff = pred_diff[valid_pair_mask]
            true_diff = true_diff[valid_pair_mask]

            pair_weights = true_diff.abs().detach()
            signed_margin = (
                torch.sign(true_diff)
                * pred_diff
                / PAIRWISE_TEMPERATURE
            )
            pair_loss = (
                F.softplus(-signed_margin) * pair_weights
            ).sum() / pair_weights.sum().clamp(min=1e-12)

            pairwise_losses.append(pair_loss)

    zero = absolute_loss.new_zeros(())

    trajectory_mean_loss = (
        torch.stack(trajectory_mean_losses).mean()
        if trajectory_mean_losses
        else zero
    )
    centered_loss = (
        torch.stack(centered_losses).mean()
        if centered_losses
        else zero
    )
    pairwise_loss = (
        torch.stack(pairwise_losses).mean()
        if pairwise_losses
        else zero
    )
    delta_loss = (
        torch.stack(delta_losses).mean()
        if delta_losses
        else zero
    )

    total_loss = (
        HYBRID_ABSOLUTE_WEIGHT * absolute_loss
        + HYBRID_TRAJECTORY_MEAN_WEIGHT * trajectory_mean_loss
        + HYBRID_CENTERED_WEIGHT * centered_loss
        + HYBRID_PAIRWISE_WEIGHT * pairwise_loss
        + HYBRID_DELTA_WEIGHT * delta_loss
    )

    return total_loss


# =========================
# 7. Train and evaluate
# =========================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    warmup_scheduler: Optional[LinearWarmupScheduler] = None,
) -> Dict[str, float]:
    model.train()
    accumulator = _EpochMetricAccumulator()
    metric_queue = _BatchMetricQueue(accumulator)
    pbar = tqdm(total=len(loader.dataset), desc=f"Train epoch {epoch}", dynamic_ncols=True)
    for batch in loader:
        # No changes to sampler/iterator creation, tensor shapes or update order.
        x, y, label_mask, attention_mask = _batch_to_device(batch, loader, device)
        counts, flags = _batch_loss_metadata(batch, loader)
        pbar.update(x.size(0))
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, DataTensorParallelValueModel):
            model.clear_replica_gradients()
        pred = model(x, attention_mask=attention_mask)
        if not sum(counts):
            continue
        # Retain the ORIGINAL differentiable graph, including zero-weight terms:
        # deleting those branches can change gradient accumulation on CUDA.
        loss = compute_hybrid_loss(pred, y, label_mask, counts, flags)
        loss.backward()
        if isinstance(model, DataTensorParallelValueModel):
            model.reduce_replica_gradients()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(optimization_parameters(model), grad_clip)
        optimizer.step()
        if isinstance(model, DataTensorParallelValueModel):
            model.synchronize_replicas()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        metric_queue.update(pred, y, label_mask, batch, loss, flags)
    metric_queue.flush()
    pbar.close()
    return accumulator.finalize()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Eval",
    collect_predictions: bool = False,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    model.eval()
    # Preserve sampler reconstruction and every DataLoader iterator/RNG draw.
    if isinstance(loader.sampler, WeightedRandomSampler):
        loader = make_data_loader(loader.dataset.sequences, loader.dataset.split_name,
                                  loader.batch_size, False, loader.num_workers)
    accumulator = _EpochMetricAccumulator()
    metric_queue = _BatchMetricQueue(accumulator)
    pred_rows = []
    pbar = tqdm(total=len(loader.dataset), desc=desc, dynamic_ncols=True)
    for batch in loader:
        x, y, label_mask, attention_mask = _batch_to_device(batch, loader, device)
        counts, flags = _batch_loss_metadata(batch, loader)
        pbar.update(x.size(0))
        pred = model(x, attention_mask=attention_mask)
        loss = None
        if sum(counts):
            loss = compute_hybrid_loss(pred, y, label_mask, counts, flags)
        if not collect_predictions and loss is not None:
            metric_queue.update(pred, y, label_mask, batch, loss, flags)
            continue
        metric_queue.flush()
        loss_value = float(loss.item()) if loss is not None else None
        pred_cpu, mask_cpu = accumulator.update(pred, y, label_mask, batch, loss_value, flags)
        if collect_predictions:
            for b_idx, item in enumerate(batch["metas"]):
                if "_dataset_index" in item:
                    step_meta = loader.dataset.sequences[item["_dataset_index"]]["step_meta"]
                else:
                    step_meta = item["step_meta"]
                for step_idx, meta in enumerate(step_meta):
                    if not mask_cpu[b_idx, step_idx]:
                        continue
                    row = meta.copy()
                    row["split"] = item["split"]
                    row["y_pred"] = float(pred_cpu[b_idx, step_idx])
                    row["y_pred_clipped"] = float(np.clip(pred_cpu[b_idx, step_idx], 0.0, 1.0))
                    row["error"] = float(row["y_pred_clipped"] - row["y_true"])
                    pred_rows.append(row)
    metric_queue.flush()
    pbar.close()
    return accumulator.finalize(), pd.DataFrame(pred_rows) if collect_predictions else None


def save_curve(
    history: List[Dict[str, float]],
    output_path: Path,
) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    valid_loss = [h["valid_loss"] for h in history]

    best_idx = int(np.nanargmin(valid_loss))
    best_epoch = epochs[best_idx]
    best_valid_loss = valid_loss[best_idx]

    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(epochs, train_loss, label="Train Hybrid Loss", linewidth=2)
    plt.plot(epochs, valid_loss, label="Val Hybrid Loss", linewidth=2)
    plt.axvline(
        best_epoch,
        linestyle="--",
        linewidth=1.5,
        label=f"Best epoch = {best_epoch}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Hybrid Loss")
    plt.title(
        "Small Transformer train/val hybrid loss\n"
        f"Best val hybrid loss = {best_valid_loss:.6f}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =========================
# 8. Feature importance
# =========================

def make_permuted_sequences(
    sequences: List[Dict[str, Any]],
    channel_idx: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Permute one feature channel across all valid step positions in given sequences.
    This is model-agnostic permutation importance.

    The input sequences are already normalized.
    """
    rng = np.random.default_rng(seed)

    values = []
    lengths = []

    for seq in sequences:
        x = seq["x"]
        lengths.append(x.shape[0])
        values.append(x[:, channel_idx].copy())

    flat_values = np.concatenate(values, axis=0)
    rng.shuffle(flat_values)

    new_sequences = [{**seq, "x": seq["x"].copy()} for seq in sequences]

    offset = 0
    for seq, length in zip(new_sequences, lengths):
        seq["x"][:, channel_idx] = flat_values[offset:offset + length]
        offset += length

    return new_sequences


def compute_permutation_feature_importance(
    model: nn.Module,
    test_sequences: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    feature_names = [
        "conf",
        "local_conf_std",
        "local_conf_slope",
        "normalized_position",
    ]

    base_dataset = TrajectoryDataset(test_sequences, split_name="test")
    base_loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_trajectory_batch,
    )

    base_metrics, _ = evaluate(
        model=model,
        loader=base_loader,
        device=device,
        desc="Eval baseline for permutation importance",
        collect_predictions=False,
    )

    rows = []
    source_importance_rows = []

    for channel_idx, feature_name in enumerate(feature_names):
        rmse_values = []
        spearman_values = []
        source_permutations = defaultdict(list)

        for r in range(repeats):
            permuted_sequences = make_permuted_sequences(
                sequences=test_sequences,
                channel_idx=channel_idx,
                seed=seed + 1000 * channel_idx + r,
            )

            perm_dataset = TrajectoryDataset(permuted_sequences, split_name="test")
            perm_loader = DataLoader(
                perm_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_trajectory_batch,
            )

            perm_metrics, _ = evaluate(
                model=model,
                loader=perm_loader,
                device=device,
                desc=f"Permutation {feature_name} repeat {r + 1}/{repeats}",
                collect_predictions=False,
            )

            rmse_values.append(perm_metrics["rmse"])
            spearman_values.append(perm_metrics["spearman"])
            for source_name, source_result in perm_metrics.get("by_source", {}).items():
                source_permutations[source_name].append(source_result)

        row = {
            "feature": feature_name,
            "baseline_rmse": base_metrics["rmse"],
            "permuted_rmse_mean": float(np.nanmean(rmse_values)),
            "permuted_rmse_std": float(np.nanstd(rmse_values)),
            "rmse_delta_mean": float(np.nanmean(rmse_values) - base_metrics["rmse"]),
            "baseline_spearman": base_metrics["spearman"],
            "permuted_spearman_mean": float(np.nanmean(spearman_values)),
            "permuted_spearman_std": float(np.nanstd(spearman_values)),
            "spearman_delta_mean": float(base_metrics["spearman"] - np.nanmean(spearman_values)),
        }

        rows.append(row)
        for source_name, results in source_permutations.items():
            baseline = base_metrics["by_source"][source_name]
            rmses = np.asarray([result["rmse"] for result in results])
            spearmans = np.asarray([result["spearman"] for result in results])
            source_importance_rows.append({
                "source_name": source_name, "feature": feature_name,
                "baseline_rmse": baseline["rmse"],
                "permuted_rmse_mean": float(np.mean(rmses)),
                "permuted_rmse_std": float(np.std(rmses)),
                "rmse_delta_mean": float(np.mean(rmses) - baseline["rmse"]),
                "baseline_spearman": baseline["spearman"],
                "permuted_spearman_mean": finite_mean(spearmans),
                "spearman_delta_mean": float(baseline["spearman"] - finite_mean(spearmans)),
            })
    SOURCE_CONTEXT["feature_importance"] = source_importance_rows

    return pd.DataFrame(rows).sort_values(
        "rmse_delta_mean",
        ascending=False,
    )


# =========================
# 9. Training orchestration utilities
# =========================

def split_train_test_sequences(
    sequences: List[Dict[str, Any]],
    test_size: float,
    random_state: int,
) -> Tuple[List[int], List[int]]:
    """Create one independent test set for optional GroupKFold mode."""
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be in (0, 1).")

    indices = np.arange(len(sequences))
    groups = np.asarray([seq["group_id"] for seq in sequences])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_valid_idx, test_idx = next(
        splitter.split(indices, groups=groups)
    )

    train_valid_groups = set(groups[train_valid_idx].tolist())
    test_groups = set(groups[test_idx].tolist())

    if train_valid_groups & test_groups:
        raise RuntimeError("Data leakage: train/valid pool and test groups overlap.")

    return train_valid_idx.tolist(), test_idx.tolist()


def make_data_loader(
    sequences: List[Dict[str, Any]],
    split_name: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TrajectoryDataset(sequences, split_name=split_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_trajectory_batch,
    )


def tp_sum(partials: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Autograd-aware reduction; each shard contributes once, with no averaging."""
    result = partials[0].to(device, non_blocking=True)
    for partial in partials[1:]:
        result = result + partial.to(device, non_blocking=True)
    return result

class TensorParallelAttention(nn.Module):
    """Column-sharded packed QKV and head attention, row-sharded output projection."""
    def __init__(self, original: nn.MultiheadAttention, devices, register, prefix):
        super().__init__()
        self.devices = devices
        self.num_heads = original.num_heads // len(devices)
        self.head_dim = original.embed_dim // original.num_heads
        self.width = self.num_heads * self.head_dim
        self.dropout = original.dropout
        self.shards = nn.ModuleList()
        d = original.embed_dim
        for index, device in enumerate(devices):
            start = index * self.width
            shard = nn.Module()
            # Allocate without consuming RNG: preserve original initialization exactly.
            qkv_rows = torch.cat([original.in_proj_weight.detach()[offset + start:offset + start + self.width]
                                  for offset in (0, d, 2 * d)])
            qkv_bias = torch.cat([original.in_proj_bias.detach()[offset + start:offset + start + self.width]
                                  for offset in (0, d, 2 * d)])
            shard.qkv_weight = nn.Parameter(qkv_rows.to(device).clone())
            shard.qkv_bias = nn.Parameter(qkv_bias.to(device).clone())
            shard.out_weight = nn.Parameter(original.out_proj.weight.detach()[:, start:start + self.width].to(device).clone())
            self.shards.append(shard)
            for block, offset in enumerate((0, d, 2 * d)):
                global_rows = slice(offset + start, offset + start + self.width)
                local_rows = slice(block * self.width, (block + 1) * self.width)
                register(prefix + "in_proj_weight", shard.qkv_weight, (global_rows, slice(None)), (local_rows, slice(None)))
                register(prefix + "in_proj_bias", shard.qkv_bias, (global_rows,), (local_rows,))
            register(prefix + "out_proj.weight", shard.out_weight, (slice(None), slice(start, start + self.width)))
        self.out_bias = nn.Parameter(original.out_proj.bias.detach().to(devices[0]).clone())
        register(prefix + "out_proj.bias", self.out_bias)

    def forward(self, x, attention_mask, causal):
        batch, length, _ = x.shape
        outputs = []
        # Enqueue fan-out before shard kernels so CUDA devices can overlap work.
        local_inputs = [x.to(device, non_blocking=True) for device in self.devices]
        local_masks = [attention_mask.to(device, non_blocking=True) for device in self.devices]
        for device, shard, local_x, local_mask in zip(self.devices, self.shards, local_inputs, local_masks):
            # SDPA boolean masks use True for allowed attention (opposite to MHA).
            allowed = local_mask[:, None, None, :]
            if causal:
                allowed = allowed & torch.ones(length, length, device=device, dtype=torch.bool).tril()[None, None]
            qkv = F.linear(local_x, shard.qkv_weight, shard.qkv_bias)
            q, k, v = [part.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
                       for part in qkv.chunk(3, dim=-1)]
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0)
            attended = attended.transpose(1, 2).contiguous().reshape(batch, length, self.width)
            outputs.append(F.linear(attended, shard.out_weight))
        return tp_sum(outputs, self.devices[0]) + self.out_bias

class TensorParallelFeedForward(nn.Module):
    """Split Linear1 output features / Linear2 input features across devices."""
    def __init__(self, first, second, dropout, devices, register, first_key, second_key):
        super().__init__()
        self.devices, self.dropout = devices, dropout
        self.shards = nn.ModuleList()
        width = first.out_features // len(devices)
        for index, device in enumerate(devices):
            region = slice(index * width, (index + 1) * width)
            shard = nn.Module()
            shard.first_weight = nn.Parameter(first.weight.detach()[region].to(device).clone())
            shard.first_bias = nn.Parameter(first.bias.detach()[region].to(device).clone())
            shard.second_weight = nn.Parameter(second.weight.detach()[:, region].to(device).clone())
            self.shards.append(shard)
            register(first_key + ".weight", shard.first_weight, (region, slice(None)))
            register(first_key + ".bias", shard.first_bias, (region,))
            register(second_key + ".weight", shard.second_weight, (slice(None), region))
        self.second_bias = nn.Parameter(second.bias.detach().to(devices[0]).clone())
        register(second_key + ".bias", self.second_bias)

    def forward(self, x):
        outputs = []
        local_inputs = [x.to(device, non_blocking=True) for device in self.devices]
        for shard, local_x in zip(self.shards, local_inputs):
            h = F.gelu(F.linear(local_x, shard.first_weight, shard.first_bias))
            h = F.dropout(h, p=self.dropout, training=self.training)
            outputs.append(F.linear(h, shard.second_weight))
        return tp_sum(outputs, self.devices[0]) + self.second_bias

class TensorParallelEncoderLayer(nn.Module):
    def __init__(self, original, devices, register, prefix):
        super().__init__()
        self.norm1 = original.norm1.to(devices[0])
        self.norm2 = original.norm2.to(devices[0])
        for name, parameter in self.norm1.named_parameters():
            register(prefix + "norm1." + name, parameter)
        for name, parameter in self.norm2.named_parameters():
            register(prefix + "norm2." + name, parameter)
        self.attention = TensorParallelAttention(original.self_attn, devices, register, prefix + "self_attn.")
        self.ffn = TensorParallelFeedForward(original.linear1, original.linear2, original.dropout.p,
            devices, register, prefix + "linear1", prefix + "linear2")
        self.dropout1 = original.dropout1
        self.dropout2 = original.dropout2

    def forward(self, x, attention_mask, causal):
        x = x + self.dropout1(self.attention(self.norm1(x), attention_mask, causal))
        return x + self.dropout2(self.ffn(self.norm2(x)))

class TensorParallelValueModel(nn.Module):
    """Same mathematical network; tensor shards live on D GPUs in one process.

    Inputs/residuals/norms and final predictions live on devices[0]. QKV, attention
    heads, FFN intermediate activations and head hidden activations are sharded.
    PyTorch's cross-device autograd handles the reductions and input gradients.
    No data-parallel replicas, per-rank loss averaging or batch splitting is used.
    CPU devices may be supplied internally for deterministic correctness tests.
    """
    def __init__(self, original: SmallTransformerEncoderValueModel, devices):
        super().__init__()
        devices = [torch.device(device) for device in devices]
        if len(devices) < 2:
            raise ValueError("TensorParallelValueModel requires at least two devices.")
        d = original.input_proj.out_features
        heads = original.encoder.layers[0].self_attn.num_heads
        if heads % len(devices) or any(layer.linear1.out_features % len(devices) for layer in original.encoder.layers):
            raise ValueError("Attention heads and FFN intermediate size must be divisible by TP size.")
        self.devices = devices
        self.transformer_type = original.transformer_type
        dense_state = original.state_dict()
        self._dense_shapes = OrderedDict((key, value.shape) for key, value in dense_state.items())
        self._dense_metadata = getattr(dense_state, "_metadata", {})
        self._checkpoint_parts = {key: [] for key in dense_state}

        def register(key, parameter, global_slice=None, local_slice=None):
            self._checkpoint_parts[key].append((parameter, global_slice, local_slice))

        self.input_proj = original.input_proj.to(devices[0])
        self.pos_encoder = original.pos_encoder.to(devices[0])
        for name, parameter in self.input_proj.named_parameters():
            register("input_proj." + name, parameter)
        self.layers = nn.ModuleList([
            TensorParallelEncoderLayer(layer, devices, register, f"encoder.layers.{index}.")
            for index, layer in enumerate(original.encoder.layers)])
        self.output_norm = original.output_head[0].to(devices[0])
        for name, parameter in self.output_norm.named_parameters():
            register("output_head.0." + name, parameter)
        self.output_ffn = TensorParallelFeedForward(original.output_head[1], original.output_head[4],
            original.output_head[3].p, devices, register, "output_head.1", "output_head.4")
        if any(not parts for parts in self._checkpoint_parts.values()):
            raise ValueError("Unsupported model state: some parameters have no TP mapping.")

    def forward(self, x, attention_mask):
        h = self.pos_encoder(self.input_proj(x.to(self.devices[0])))
        for layer in self.layers:
            h = layer(h, attention_mask, self.transformer_type == "causal")
        return torch.sigmoid(self.output_ffn(self.output_norm(h)).squeeze(-1))

    def dense_tensors(self, gradients=False):
        """Gather ordinary CPU tensors with exactly the original checkpoint keys."""
        result = OrderedDict()
        for key, parts in self._checkpoint_parts.items():
            value = torch.zeros(self._dense_shapes[key], dtype=parts[0][0].dtype, device="cpu")
            for parameter, global_slice, local_slice in parts:
                tensor = parameter.grad if gradients else parameter.detach()
                if tensor is None:
                    continue
                tensor = tensor.detach() if gradients else tensor
                if local_slice is not None:
                    tensor = tensor[local_slice]
                if global_slice is None:
                    value.copy_(tensor.to("cpu"))
                else:
                    value[global_slice].copy_(tensor.to("cpu"))
            result[key] = value
        result._metadata = self._dense_metadata
        return result

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if args or keep_vars:
            raise ValueError("TP checkpoint export supports keyword arguments and keep_vars=False only.")
        result = self.dense_tensors()
        if destination is None and not prefix:
            return result
        destination = OrderedDict() if destination is None else destination
        destination.update((prefix + key, value) for key, value in result.items())
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if assign:
            raise ValueError("assign=True would invalidate TP shard placement; use assign=False.")
        missing = [key for key in self._dense_shapes if key not in state_dict]
        unexpected = [key for key in state_dict if key not in self._dense_shapes]
        if strict and (missing or unexpected):
            raise RuntimeError(f"TP state_dict mismatch: missing={missing}, unexpected={unexpected}")
        for key, value in state_dict.items():
            if key in self._dense_shapes and value.shape != self._dense_shapes[key]:
                raise RuntimeError(f"TP state_dict size mismatch for {key}: {value.shape} vs {self._dense_shapes[key]}")
        with torch.no_grad():
            for key, parts in self._checkpoint_parts.items():
                if key not in state_dict:
                    continue
                for parameter, global_slice, local_slice in parts:
                    value = state_dict[key] if global_slice is None else state_dict[key][global_slice]
                    target = parameter if local_slice is None else parameter[local_slice]
                    target.copy_(value)
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)

def resolve_gpu_groups(args, *, validate_cuda=True):
    """Return logical CUDA device groups; default None preserves old behavior."""
    setting = getattr(args, "gpu_groups", None)
    requested_tp = getattr(args, "tensor_parallel_size", None)
    if requested_tp is not None and requested_tp < 1:
        raise ValueError("--tensor_parallel_size must be positive.")
    if setting is None:
        if requested_tp in (None, 1):
            return None
        ids = [list(range(requested_tp))]
    else:
        try:
            tokens = [group.split(",") for group in setting.split(";")]
            if any(not token.strip() for group in tokens for token in group):
                raise ValueError("empty device or group")
            ids = [[int(token.strip()) for token in group] for group in tokens]
        except (ValueError, AttributeError) as error:
            raise ValueError('--gpu_groups must look like "0,1;2,3" (no empty groups).') from error
        width = len(ids[0])
        if any(len(group) != width for group in ids):
            raise ValueError("Every DP group must have the same tensor-parallel size.")
        if requested_tp is not None and requested_tp != width:
            raise ValueError("--tensor_parallel_size does not match --gpu_groups.")
    flat = [index for group in ids for index in group]
    if len(flat) != len(set(flat)) or any(index < 0 for index in flat):
        raise ValueError("GPU IDs must be distinct and nonnegative across all groups.")
    tp_size = len(ids[0])
    if tp_size > 1 and (args.nhead % tp_size or args.dim_feedforward % tp_size
                        or args.d_model % args.nhead):
        raise ValueError("TP requires nhead and dim_feedforward divisible by tensor_parallel_size, and d_model divisible by nhead.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one Python process for gpu_groups, not torchrun or one process per GPU.")
    if validate_cuda:
        if not torch.cuda.is_available():
            raise ValueError("--gpu_groups / tensor_parallel_size > 1 requires CUDA-enabled PyTorch.")
        if max(flat) >= torch.cuda.device_count():
            raise ValueError(f"GPU layout {ids} exceeds {torch.cuda.device_count()} visible GPUs. IDs follow CUDA_VISIBLE_DEVICES.")
        if tp_size > 1 and not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError("TP attention requires PyTorch >= 2.0.")
    return [[torch.device("cuda", index) for index in group] for group in ids]


def resolved_parallel_groups(args):
    if hasattr(args, "_resolved_gpu_groups"):
        return args._resolved_gpu_groups
    return resolve_gpu_groups(args)


def optimization_parameters(model):
    """Only one logical copy is optimized/clipped; replicas carry partial grads."""
    if isinstance(model, DataTensorParallelValueModel):
        return model.replicas[0].parameters()
    return model.parameters()


class DataTensorParallelValueModel(nn.Module):
    """Single-process DP across identical TP replicas, with one global loss.

    Cross-device autograd propagates the loss gradient through concatenation
    and tensor copies. Each DP replica therefore receives its correctly weighted
    contribution to the ORIGINAL full-batch loss, including unequal label/pair
    counts. reduce_replica_gradients sums those contributions to one canonical
    copy. Only that copy owns optimizer states. No second division by DP size.
    CPU group devices are accepted internally for layout/gradient unit tests.
    """
    def __init__(self, original, groups):
        super().__init__()
        if not groups or any(not group for group in groups):
            raise ValueError("At least one nonempty GPU group is required.")
        groups = [[torch.device(device) for device in group] for group in groups]
        if any(len(group) != len(groups[0]) for group in groups):
            raise ValueError("All groups must have the same TP size.")
        self.groups = groups
        self.output_device = groups[0][0]
        self.transformer_type = original.transformer_type
        self.dp_size = len(groups)
        self.tensor_parallel_size = len(groups[0])
        self.replicas = nn.ModuleList()
        # CPU copies consume no RNG and all start with exactly the same weights.
        # Build noncanonical copies first so moving the canonical model cannot
        # accidentally cause a full-model allocation on GPU zero for each copy.
        for group in groups[1:]:
            dense_copy = copy.deepcopy(original)
            replica = (TensorParallelValueModel(dense_copy, group) if len(group) > 1
                       else dense_copy.to(group[0]))
            self.replicas.append(replica)
        canonical = (TensorParallelValueModel(original, groups[0]) if len(groups[0]) > 1
                     else original.to(groups[0][0]))
        self.replicas.insert(0, canonical)
        self._named_replica_parameters = [dict(replica.named_parameters()) for replica in self.replicas]
        primary = self._named_replica_parameters[0]
        for replica in self._named_replica_parameters[1:]:
            if primary.keys() != replica.keys() or any(primary[key].shape != replica[key].shape for key in primary):
                raise ValueError("DP replicas have inconsistent TP parameter layouts.")
        self._active_replicas = ()
        # Initialize independent device streams without consuming the CPU sampler
        # RNG. They preserve dropout rates/distribution, not dense mask identity.
        seed = torch.initial_seed()
        for offset, device in enumerate(device for group in groups for device in group):
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed((seed + 104729 * offset) % (2**63 - 1))
        if any(device.type == "cuda" for group in groups for device in group):
            print(f"[INFO] DP={self.dp_size}, TP={self.tensor_parallel_size}, groups={groups}; "
                  "batch_size is GLOBAL. One full-batch loss, summed DP gradients and one optimizer update.")
            print("[INFO] Parallel dropout masks/roundoff differ from dense execution; bitwise-identical checkpoints are not guaranteed.")

    def forward(self, x, attention_mask):
        if x.size(0) == 0:
            raise ValueError("A global batch must contain at least one trajectory.")
        if x.size(0) != attention_mask.size(0):
            raise ValueError("Input/mask batch sizes differ.")
        # Explicit partitioning preserves every sample and the original order;
        # no DistributedSampler padding, dropped tails or source resampling.
        quotient, remainder = divmod(x.size(0), self.dp_size)
        slices = []
        start = 0
        for index in range(self.dp_size):
            count = quotient + int(index < remainder)
            if count:
                slices.append((index, start, start + count))
            start += count
        self._active_replicas = tuple(index for index, _, _ in slices)
        inputs = [(index, x[start:end].to(self.groups[index][0], non_blocking=True),
                   attention_mask[start:end].to(self.groups[index][0], non_blocking=True))
                  for index, start, end in slices]
        predictions = [self.replicas[index](local_x, attention_mask=local_mask).to(self.output_device, non_blocking=True)
                       for index, local_x, local_mask in inputs]
        return torch.cat(predictions, dim=0)

    def clear_replica_gradients(self):
        # Optimizer.zero_grad clears only canonical parameters. Clear every copy
        # here so inactive replicas/tail batches never reuse stale gradients.
        for replica in self.replicas:
            replica.zero_grad(set_to_none=True)

    @torch.no_grad()
    def reduce_replica_gradients(self):
        primary = self._named_replica_parameters[0]
        for index in self._active_replicas:
            if index == 0:
                continue
            for key, parameter in self._named_replica_parameters[index].items():
                if parameter.grad is None:
                    continue
                target = primary[key]
                contribution = parameter.grad.detach().to(target.device, non_blocking=True)
                if target.grad is None:
                    target.grad = contribution.clone()
                else:
                    target.grad.add_(contribution)

    @torch.no_grad()
    def synchronize_replicas(self):
        # Broadcast after AdamW, including inactive copies for the next batch.
        primary = self._named_replica_parameters[0]
        for replica in self._named_replica_parameters[1:]:
            for key, target in replica.items():
                target.copy_(primary[key], non_blocking=True)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if args or keep_vars:
            raise ValueError("Parallel checkpoint export requires keyword arguments and keep_vars=False.")
        state = self.replicas[0].state_dict()
        if isinstance(self.replicas[0], TensorParallelValueModel):
            full = state  # TP already assembles independent, unsharded CPU data.
        else:
            full = OrderedDict((key, value.detach().to("cpu", copy=True)) for key, value in state.items())
            if hasattr(state, "_metadata"):
                full._metadata = copy.deepcopy(state._metadata)
        if destination is None and not prefix:
            return full
        destination = OrderedDict() if destination is None else destination
        destination.update((prefix + key, value) for key, value in full.items())
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if assign:
            raise ValueError("assign=True would invalidate parallel shard placement and optimizer references.")
        result = None
        for replica in self.replicas:
            loaded = replica.load_state_dict(state_dict, strict=strict)
            if result is None:
                result = loaded
        return result


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    groups = resolved_parallel_groups(args)
    model = SmallTransformerEncoderValueModel(
        input_dim=4,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        transformer_type=args.transformer_type,
    )
    if groups and (len(groups) > 1 or len(groups[0]) > 1):
        return DataTensorParallelValueModel(model, groups)
    return model.to(groups[0][0] if groups else device)


def train_model_with_validation(
    args: argparse.Namespace,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    seed: int,
    best_model_path: Optional[Path] = None,
    log_prefix: str = "",
) -> Tuple[nn.Module, List[Dict[str, float]], int, float, float, int]:
    """Train one model with the original validation and early-stopping logic."""
    set_seed(seed)

    model = build_model(args, device)
    num_params = count_parameters(model)
    print(f"[INFO] {log_prefix}Trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(
        optimization_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup_scheduler = build_warmup_scheduler(
        args=args,
        optimizer=optimizer,
        train_loader=train_loader,
        num_epochs=args.max_epochs,
        log_prefix=log_prefix,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, args.patience // 3),
        min_lr=1e-6,
    )

    best_valid_loss = float("inf")
    best_valid_selection_score = float("inf")
    best_valid_rmse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_state_dict = None
    history: List[Dict[str, float]] = []

    print(f"[INFO] {log_prefix}Start training...")

    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
            warmup_scheduler=warmup_scheduler,
        )

        valid_metrics, _ = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            desc=f"{log_prefix}Valid epoch {epoch}",
            collect_predictions=False,
        )

        if warmup_scheduler is None or warmup_scheduler.is_finished:
            scheduler.step(valid_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": int(epoch),
            "lr": float(current_lr),
            "train_loss": float(train_metrics["loss"]),
            "valid_loss": float(valid_metrics["loss"]),
            "train_rmse": float(train_metrics["rmse"]),
            "valid_rmse": float(valid_metrics["rmse"]),
            "train_spearman": float(train_metrics["spearman"]),
            "valid_spearman": float(valid_metrics["spearman"]),
            "train_r2": float(train_metrics["r2"]),
            "valid_r2": float(valid_metrics["r2"]),
            "train_mae": float(train_metrics["mae"]),
            "valid_mae": float(valid_metrics["mae"]),
        }
        epoch_record.update(flat_source_history(train_metrics, valid_metrics))
        history.append(epoch_record)

        print(
            f"[{log_prefix}Epoch {epoch:03d}] "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_metrics['loss']:.6f}, "
            f"val_loss={valid_metrics['loss']:.6f}, "
            f"train_rmse={train_metrics['rmse']:.6f}, "
            f"val_rmse={valid_metrics['rmse']:.6f}, "
            f"train_spearman={train_metrics['spearman']:.4f}, "
            f"val_spearman={valid_metrics['spearman']:.4f}"
        )

        improved = (
            np.isfinite(valid_metrics["selection_score"])
            and valid_metrics["selection_score"] + args.min_delta < best_valid_selection_score
        )

        if improved:
            best_valid_selection_score = float(valid_metrics["selection_score"])
            best_valid_loss = float(valid_metrics["loss"])
            best_valid_rmse = float(valid_metrics["rmse"])
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = _snapshot_state_dict(model)

            if best_model_path is not None:
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state_dict, best_model_path)

            print(
                f"[INFO] {log_prefix}New best model saved at epoch {epoch}: "
                f"val_hybrid_loss={best_valid_loss:.6f}, "
                f"selection({SOURCE_CONTEXT['selection_metric']})={best_valid_selection_score:.6f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"[INFO] {log_prefix}No improvement count: "
                f"{epochs_without_improvement}/{args.patience}"
            )

        if epochs_without_improvement >= args.patience:
            print(
                f"[INFO] {log_prefix}Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    elif best_model_path is not None and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=("cpu" if isinstance(model, DataTensorParallelValueModel) else device)))
    else:
        best_epoch = max(1, len(history))
        if history:
            best_valid_loss = float(history[-1]["valid_loss"])
            best_valid_rmse = float(history[-1]["valid_rmse"])
        best_state_dict = _snapshot_state_dict(model)
        if best_model_path is not None:
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state_dict, best_model_path)

    return (
        model,
        history,
        best_epoch,
        best_valid_loss,
        best_valid_rmse,
        num_params,
    )


def aggregate_cv_histories(
    fold_histories: List[List[Dict[str, float]]],
) -> List[Dict[str, float]]:
    if not fold_histories:
        return []

    max_epoch = max(len(history) for history in fold_histories)
    value_keys = [
        "lr",
        "train_loss",
        "valid_loss",
        "train_rmse",
        "valid_rmse",
        "train_spearman",
        "valid_spearman",
        "train_r2",
        "valid_r2",
        "train_mae",
        "valid_mae",
    ]
    extra_keys = sorted({key for history in fold_histories for record in history
                         for key in record if key.startswith(("train_source", "valid_source", "train_selection", "valid_selection"))})
    value_keys.extend(key for key in extra_keys if key not in value_keys)
    rows: List[Dict[str, float]] = []

    for epoch in range(1, max_epoch + 1):
        records = [
            history[epoch - 1]
            for history in fold_histories
            if len(history) >= epoch
        ]

        row: Dict[str, float] = {
            "epoch": int(epoch),
            "num_active_folds": int(len(records)),
        }

        for key in value_keys:
            values = np.asarray([record.get(key, np.nan) for record in records], dtype=float)
            row[key] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values))

        rows.append(row)

    return rows


def summarize_fold_metrics(
    fold_metrics: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    metric_keys = [
        "loss",
        "spearman",
        "pearson",
        "mae",
        "rmse",
        "r2",
        "bias",
        "ece",
        "macro_trajectory_rmse",
        "within_response_spearman",
        "pairwise_ranking_accuracy",
    ]

    summary: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray(
            [fold["valid_metrics"].get(key, np.nan) for fold in fold_metrics],
            dtype=float,
        )
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }

    return summary


def train_model_fixed_epochs(
    args: argparse.Namespace,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    seed: int,
    best_model_path: Path,
    lr_schedule: Optional[Dict[int, float]] = None,
) -> Tuple[nn.Module, List[Dict[str, float]], int]:
    """Refit on the complete non-test pool for the CV-selected epoch count."""
    set_seed(seed)

    model = build_model(args, device)
    num_params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        optimization_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, float]] = []
    num_epochs = max(1, int(num_epochs))
    warmup_scheduler = (
        build_warmup_scheduler(
            args=args,
            optimizer=optimizer,
            train_loader=train_loader,
            num_epochs=num_epochs,
            log_prefix="Final refit ",
        )
        if lr_schedule is None
        else None
    )

    print(
        f"[INFO] Final refit on full train/valid pool for {num_epochs} epochs."
    )

    for epoch in range(1, num_epochs + 1):
        if lr_schedule is not None and epoch in lr_schedule:
            scheduled_lr = float(lr_schedule[epoch])
            for param_group in optimizer.param_groups:
                param_group["lr"] = scheduled_lr

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
            warmup_scheduler=warmup_scheduler,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": int(epoch),
            "lr": float(current_lr),
            "train_loss": float(train_metrics["loss"]),
            "train_rmse": float(train_metrics["rmse"]),
            "train_spearman": float(train_metrics["spearman"]),
            "train_r2": float(train_metrics["r2"]),
            "train_mae": float(train_metrics["mae"]),
        })

        history[-1].update(flat_source_history(train_metrics, {}))

        print(
            f"[Final refit epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.6f}, "
            f"train_rmse={train_metrics['rmse']:.6f}, "
            f"train_spearman={train_metrics['spearman']:.4f}, "
            f"lr={current_lr:.2e}"
        )

    torch.save(model.state_dict(), best_model_path)
    return model, history, num_params


def build_window_dataset_df(
    sequences: List[Dict[str, Any]],
    sequence_id_to_split: Dict[int, str],
) -> pd.DataFrame:
    step_rows = []
    for seq in sequences:
        split = sequence_id_to_split[seq["sequence_id"]]
        for meta in seq["step_meta"]:
            row = meta.copy()
            row["sequence_id"] = seq["sequence_id"]
            row["split"] = split
            step_rows.append(row)
    return pd.DataFrame(step_rows)


def compute_oof_metrics(
    predictions_df: pd.DataFrame,
    fold_metrics: List[Dict[str, Any]],
) -> Dict[str, float]:
    metrics = compute_metrics(
        predictions_df["y_true"].to_numpy(dtype=float),
        predictions_df["y_pred"].to_numpy(dtype=float),
    )

    trajectory_predictions: Dict[Any, Dict[str, List[float]]] = {}
    for line_idx, group_df in predictions_df.groupby("line_idx", dropna=False):
        trajectory_predictions[line_idx] = {
            "y_true": group_df["y_true"].to_numpy(dtype=float).tolist(),
            "y_pred": group_df["y_pred"].to_numpy(dtype=float).tolist(),
        }
    metrics.update(compute_trajectory_metrics(trajectory_predictions))

    valid_counts = np.asarray(
        [fold["num_valid_sequences"] for fold in fold_metrics],
        dtype=float,
    )
    valid_losses = np.asarray(
        [fold["valid_metrics"].get("loss", np.nan) for fold in fold_metrics],
        dtype=float,
    )
    finite_mask = np.isfinite(valid_losses) & (valid_counts > 0)
    if finite_mask.any():
        metrics["loss"] = float(
            np.average(valid_losses[finite_mask], weights=valid_counts[finite_mask])
        )
    else:
        metrics["loss"] = np.nan

    return metrics


# =========================
# 10. Main
# =========================

# =========================
# Multi-source support (metadata never enters the four model features)
# =========================

SOURCE_CONTEXT: Dict[str, Any] = {
    "names": [], "files": [], "selection_metric": "loss",
    "selection_source": None, "weights": {}, "sampling": "proportional",
    "loss_aggregation": "original", "manifest": None, "feature_importance": [], "raw_counts": {},
}

_single_read_jsonl = read_jsonl
_single_get_group_id = get_group_id
_single_build_sequences = build_trajectory_sequences
_single_split_three = split_train_valid_test_sequences
_single_split_two = split_train_test_sequences
_single_hybrid_loss = compute_hybrid_loss
_single_permutation = make_permuted_sequences
_single_oof_metrics = compute_oof_metrics
_single_save_curve = save_curve


def add_multi_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input_files", nargs="+", default=None,
                        help="D input JSONLs; each has the unchanged original schema. Overrides the default --input_file.")
    parser.add_argument("--source_names", nargs="+", default=None,
                        help="Unique metadata names, one per input file. Default: source_1, source_2, ...")
    parser.add_argument("--sampling_strategy", choices=["proportional", "balanced"], default="proportional",
                        help="Original shuffled sampling, or source-equal sampling in expectation with replacement. Training only.")
    parser.add_argument("--selection_metric", default="auto", choices=[
        "auto", "loss", "source_macro_trajectory_rmse",
        "weighted_source_trajectory_rmse", "source_trajectory_rmse"],
        help="auto: original loss for one source; source-equal trajectory RMSE for multiple sources.")
    parser.add_argument("--selection_source", default=None,
                        help="Source whose trajectory RMSE selects checkpoints with source_trajectory_rmse.")
    parser.add_argument("--source_weights", type=float, nargs="+", default=None,
                        help="Positive selection weights in source_names order; requires weighted_source_trajectory_rmse.")
    parser.add_argument("--loss_aggregation", default="original", choices=["original", "trajectory_mean"],
                        help="Original hybrid loss unchanged by default; optionally average per-trajectory hybrid losses.")
    parser.add_argument("--split_manifest", default=None,
                        help="Reuse an existing split_manifest.json. Every run writes its own manifest in output_dir.")


def configure_sources(args: argparse.Namespace) -> None:
    if args.input_files and args.input_file != "steps.jsonl":
        raise ValueError("Use either --input_file or --input_files, not both.")
    files = list(args.input_files or [args.input_file])
    resolved = [str(Path(path).resolve()) for path in files]
    if len(set(os.path.normcase(path) for path in resolved)) != len(files):
        raise ValueError("The same input file cannot be assigned to multiple sources.")
    names = args.source_names or [f"source_{i + 1}" for i in range(len(files))]
    if len(names) != len(files) or len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("--source_names must contain one distinct nonempty name per input file.")
    metric = args.selection_metric
    if metric == "auto":
        metric = "loss" if len(files) == 1 else "source_macro_trajectory_rmse"
    if args.selection_source is not None and args.selection_source not in names:
        raise ValueError("--selection_source must be one of --source_names.")
    if metric == "source_trajectory_rmse" and not args.selection_source:
        raise ValueError("source_trajectory_rmse requires --selection_source.")
    if args.selection_source and metric != "source_trajectory_rmse":
        raise ValueError("--selection_source requires --selection_metric source_trajectory_rmse.")
    weights = args.source_weights
    if weights is not None:
        if len(weights) != len(names) or any(not np.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("--source_weights requires one finite positive weight per source.")
        if metric != "weighted_source_trajectory_rmse":
            raise ValueError("--source_weights requires --selection_metric weighted_source_trajectory_rmse.")
    if metric == "weighted_source_trajectory_rmse" and weights is None:
        raise ValueError("weighted_source_trajectory_rmse requires --source_weights.")
    SOURCE_CONTEXT.update(names=list(names), files=resolved, selection_metric=metric,
                          selection_source=args.selection_source,
                          weights=dict(zip(names, weights or [1.0] * len(names))),
                          sampling=args.sampling_strategy, loss_aggregation=args.loss_aggregation,
                          manifest=None, feature_importance=[], raw_counts={})
    if args.input_files:
        args.input_file = files[0]
    print(f"[INFO] Sources: {dict(zip(names, files))}")
    print(f"[INFO] Checkpoint/early-stop metric: {metric}; LR scheduler remains valid hybrid loss.")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    files = SOURCE_CONTEXT["files"] or [str(Path(path).resolve())]
    names = SOURCE_CONTEXT["names"] or ["source_1"]
    samples = []
    occurrences = defaultdict(int)
    for name, file_path in zip(names, files):
        records = _single_read_jsonl(file_path)
        SOURCE_CONTEXT["raw_counts"][name] = len(records)
        for record in records:
            payload = {key: value for key, value in record.items() if key != "_line_idx"}
            digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
            occurrence = occurrences[(name, digest)]
            occurrences[(name, digest)] += 1
            record["_source_line_idx"] = record["_line_idx"]
            record["_source_name"] = name
            record["_source_file"] = file_path
            record["_sample_uid"] = f"{name}::{digest}::{occurrence}"
            # Keep existing line_idx outputs globally unique, even if local sample_idx overlaps.
            record["_line_idx"] = len(samples) if len(files) > 1 else record["_source_line_idx"]
            samples.append(record)
    return samples


def get_group_id(sample: Dict[str, Any], group_by: str) -> str:
    group = _single_get_group_id(sample, group_by)
    # Same problem text remains a global group across files. Local sample indices do not.
    if group.startswith("sample::") and len(SOURCE_CONTEXT["names"]) > 1:
        return sample.get("_source_name", "source_1") + "::" + group
    return group


def build_trajectory_sequences(samples, group_by, position_mode, transformer_type, build_frame=True):
    sequences, frame = _single_build_sequences(samples, group_by, position_mode, transformer_type, build_frame)
    records = {record["_line_idx"]: record for record in samples}
    for sequence in sequences:
        record = records[sequence["line_idx"]]
        metadata = {"source_name": record["_source_name"],
                    "source_file": record["_source_file"],
                    "source_line_idx": record["_source_line_idx"],
                    "sample_uid": record["_sample_uid"]}
        sequence.update(metadata)
        for step in sequence["step_meta"]:
            step.update(metadata)
    # Match the original auxiliary DataFrame, including otherwise valid unlabeled rows.
    if not frame.empty:
        for column, record_key in [("source_name", "_source_name"), ("source_file", "_source_file"),
                                   ("source_line_idx", "_source_line_idx"), ("sample_uid", "_sample_uid")]:
            frame[column] = frame["line_idx"].map(lambda index: records[index][record_key])
    return sequences, frame


def sequence_source(sequence: Dict[str, Any]) -> str:
    if "source_name" in sequence:
        return sequence["source_name"]
    return sequence["step_meta"][0].get("source_name", "source_1")


def validate_source_coverage(sequences, split_name):
    names = SOURCE_CONTEXT["names"] or sorted({sequence_source(seq) for seq in sequences})
    missing = [name for name in names if not any(sequence_source(seq) == name for seq in sequences)]
    if missing:
        raise ValueError(f"{split_name} has no supervised trajectories from {missing}. "
                         "Add data, reduce num_folds, change split ratios, or relax optional filters.")


def allocate_group_partitions(sequences, fractions, seed):
    """Partition complete groups, stratifying sources without duplicating any group.

    Disjoint-source groups get exact largest-remainder group counts. If groups
    span multiple sources, use deterministic multi-start greedy allocation and
    require source coverage in every partition. Group integrity takes priority
    over exact sample proportions when several trajectories share a problem.
    """
    fractions = np.asarray(fractions, dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Partition fractions must be positive and sum to one.")
    names = sorted({sequence_source(seq) for seq in sequences})
    source_index = {name: i for i, name in enumerate(names)}
    groups = defaultdict(list)
    for i, sequence in enumerate(sequences):
        groups[sequence["group_id"]].append(i)
    keys = sorted(groups)
    vectors = np.zeros((len(keys), len(names)), dtype=float)
    for i, key in enumerate(keys):
        for index in groups[key]:
            vectors[i, source_index[sequence_source(sequences[index])]] += 1
    num_parts = len(fractions)
    for column, name in enumerate(names):
        count = int(np.count_nonzero(vectors[:, column]))
        if count < num_parts:
            raise ValueError(f"Source {name!r} has only {count} distinct groups for {num_parts} partitions. "
                             "Reduce num_folds or provide more independent problems.")
    assignments = {}
    if np.all(np.count_nonzero(vectors, axis=1) == 1):
        for source, column in source_index.items():
            indices = np.flatnonzero(vectors[:, column] > 0)
            rng = np.random.default_rng(seed + column)
            indices = rng.permutation(indices)
            targets = fractions * len(indices)
            counts = np.floor(targets).astype(int)
            for part in np.argsort(-(targets - counts), kind="stable")[:len(indices) - counts.sum()]:
                counts[part] += 1
            for part in np.flatnonzero(counts == 0):
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[part] += 1
            start = 0
            for part, count in enumerate(counts):
                for index in indices[start:start + count]:
                    assignments[keys[index]] = part
                start += count
    else:
        target = fractions[:, None] * vectors.sum(axis=0)[None, :]
        rarity = (vectors > 0) / np.maximum(1, (vectors > 0).sum(axis=0))
        priority = rarity.sum(axis=1)
        best = None
        for attempt in range(48):
            rng = np.random.default_rng(seed + attempt)
            order = np.lexsort((rng.random(len(keys)), -vectors.sum(axis=1), -priority))
            actual = np.zeros_like(target)
            proposed = {}
            for index in order:
                vector = vectors[index]
                deltas = (((actual + vector - target) ** 2 - (actual - target) ** 2)
                          / np.maximum(target, 1.0)).sum(axis=1)
                part = int(np.argmin(deltas + rng.random(num_parts) * 1e-9))
                actual[part] += vector
                proposed[keys[index]] = part
            if np.any(actual == 0):
                continue
            score = float((((actual - target) ** 2) / np.maximum(target, 1.0)).sum())
            if best is None or score < best[0]:
                best = (score, proposed)
        if best is None:
            raise ValueError("Cannot allocate shared problem groups while covering every source in every partition. "
                             "Reduce num_folds, adjust ratios, or add independent problem groups.")
        assignments = best[1]
    result = [[] for _ in fractions]
    for index, sequence in enumerate(sequences):
        result[assignments[sequence["group_id"]]].append(index)
    return result


def split_train_test_sequences(sequences, test_size, random_state):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1).")
    manifest = SOURCE_CONTEXT.get("manifest")
    if manifest is not None:
        rows = manifest["samples_by_uid"]
        return ([i for i, seq in enumerate(sequences) if rows[seq["sample_uid"]]["split"] != "test"],
                [i for i, seq in enumerate(sequences) if rows[seq["sample_uid"]]["split"] == "test"])
    if len({sequence_source(seq) for seq in sequences}) == 1:
        return _single_split_two(sequences, test_size, random_state)
    return tuple(allocate_group_partitions(sequences, [1 - test_size, test_size], random_state))


def split_train_valid_test_sequences(sequences, test_size, valid_size, random_state):
    if not 0 < valid_size < 1 or not 0 < test_size < 1 or valid_size + test_size >= 1:
        raise ValueError("test_size and valid_size must be positive and sum to less than one.")
    manifest = SOURCE_CONTEXT.get("manifest")
    if manifest is not None:
        rows = manifest["samples_by_uid"]
        return tuple([i for i, seq in enumerate(sequences)
                      if rows[seq["sample_uid"]]["split"] == split] for split in ("train", "valid", "test"))
    if len({sequence_source(seq) for seq in sequences}) == 1:
        return _single_split_three(sequences, test_size, valid_size, random_state)
    pool, test = split_train_test_sequences(sequences, test_size, random_state)
    inner_valid = valid_size / (1 - test_size)
    train_pos, valid_pos = allocate_group_partitions([sequences[i] for i in pool],
                                                    [1 - inner_valid, inner_valid], random_state + 1)
    return [pool[i] for i in train_pos], [pool[i] for i in valid_pos], test


class SourceGroupKFold:
    def __init__(self, sequences, num_folds, seed):
        self.sequences, self.num_folds, self.seed = sequences, num_folds, seed

    def split(self, X=None, groups=None):
        manifest = SOURCE_CONTEXT.get("manifest")
        if manifest is not None:
            rows = manifest["samples_by_uid"]
            folds = [[i for i, seq in enumerate(self.sequences)
                      if rows[seq["sample_uid"]]["cv_fold"] == fold]
                     for fold in range(1, self.num_folds + 1)]
        elif len({sequence_source(seq) for seq in self.sequences}) == 1:
            yield from GroupKFold(n_splits=self.num_folds).split(X, groups=groups)
            return
        else:
            folds = allocate_group_partitions(self.sequences, [1 / self.num_folds] * self.num_folds, self.seed)
        all_indices = set(range(len(self.sequences)))
        for valid in folds:
            train = sorted(all_indices - set(valid))
            yield np.asarray(train, dtype=int), np.asarray(valid, dtype=int)


def prepare_source_splits(sequences, args, output_dir):
    validate_source_coverage(sequences, "input")
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or manifest.get("group_by") != args.group_by:
            raise ValueError("Split manifest version/group_by differs from this run.")
        if bool(manifest.get("use_group_kfold")) != bool(args.use_group_kfold):
            raise ValueError("Split manifest belongs to a different validation mode.")
        if args.use_group_kfold and manifest.get("num_folds") != args.num_folds:
            raise ValueError("Split manifest num_folds differs from this run.")
        rows = {row["sample_uid"]: row for row in manifest["samples"]}
        if len(rows) != len(manifest["samples"]):
            raise ValueError("Duplicate sample_uid in split manifest.")
        for sequence in sequences:
            row = rows.get(sequence["sample_uid"])
            if row is None or row["group_id"] != sequence["group_id"] or row["source_name"] != sequence_source(sequence):
                raise ValueError("Split manifest does not match the current source names/sample contents/groups.")
            allowed = {"pool", "test"} if args.use_group_kfold else {"train", "valid", "test"}
            if row["split"] not in allowed:
                raise ValueError("Invalid split in manifest.")
            if args.use_group_kfold and row["split"] == "pool" and row.get("cv_fold") not in range(1, args.num_folds + 1):
                raise ValueError("Missing/invalid cv_fold in manifest.")
        retained_names = set(SOURCE_CONTEXT["names"])
        expected_uids = {uid for uid, row in rows.items() if row["source_name"] in retained_names}
        if expected_uids != {seq["sample_uid"] for seq in sequences}:
            raise ValueError("Input sample set changed. Manifest reuse requires identical samples for each retained source; "
                             "do not silently remove validation/test trajectories by subsampling input files.")
        SOURCE_CONTEXT["manifest"] = {**manifest, "samples_by_uid": rows}
    if args.use_group_kfold:
        pool, test = split_train_test_sequences(sequences, args.test_size, args.random_state)
        pool_sequences = [sequences[i] for i in pool]
        validate_source_coverage([sequences[i] for i in test], "test")
        validate_source_coverage(pool_sequences, "train/valid pool")
        folds = list(SourceGroupKFold(pool_sequences, args.num_folds, args.random_state).split(
            np.zeros((len(pool), 1)), np.asarray([seq["group_id"] for seq in pool_sequences])))
        assignments = {sequences[i]["sample_uid"]: ("test", None) for i in test}
        for fold, (train_pos, valid_pos) in enumerate(folds, start=1):
            validate_source_coverage([pool_sequences[i] for i in train_pos], f"fold {fold} train")
            validate_source_coverage([pool_sequences[i] for i in valid_pos], f"fold {fold} valid")
            for pos in valid_pos:
                assignments[pool_sequences[pos]["sample_uid"]] = ("pool", fold)
    else:
        train, valid, test = split_train_valid_test_sequences(sequences, args.test_size, args.valid_size, args.random_state)
        assignments = {}
        for split, indices in [("train", train), ("valid", valid), ("test", test)]:
            validate_source_coverage([sequences[i] for i in indices], split)
            for i in indices:
                assignments[sequences[i]["sample_uid"]] = (split, None)
    if len(assignments) != len(sequences):
        raise ValueError("Split assignment is incomplete or sample identities overlap.")
    group_assignments = defaultdict(set)
    rows = []
    for seq in sequences:
        split, fold = assignments[seq["sample_uid"]]
        group_assignments[seq["group_id"]].add((split, fold))
        rows.append({"sample_uid": seq["sample_uid"], "source_name": sequence_source(seq),
                     "source_line_idx": seq["source_line_idx"], "group_id": seq["group_id"],
                     "split": split, "cv_fold": fold})
    if any(len(values) != 1 for values in group_assignments.values()):
        raise ValueError("Split manifest leaks a group across splits or folds.")
    manifest = {"version": 1, "group_by": args.group_by, "random_state": args.random_state,
                "test_size": args.test_size, "valid_size": args.valid_size,
                "use_group_kfold": args.use_group_kfold, "num_folds": args.num_folds,
                "sources": dict(zip(SOURCE_CONTEXT["names"], SOURCE_CONTEXT["files"])), "samples": rows}
    if args.split_manifest:
        previous = SOURCE_CONTEXT["manifest"]
        manifest["reused_from"] = str(Path(args.split_manifest).resolve())
        manifest["split_random_state"] = previous.get("split_random_state", previous["random_state"])
        manifest["test_size"] = previous["test_size"]
        manifest["valid_size"] = previous["valid_size"]
    path = output_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    SOURCE_CONTEXT["manifest"] = {**manifest, "samples_by_uid": {row["sample_uid"]: row for row in rows}}
    SOURCE_CONTEXT["manifest_path"] = str(path)
    SOURCE_CONTEXT["valid_counts"] = {name: sum(sequence_source(seq) == name for seq in sequences)
                                      for name in SOURCE_CONTEXT["names"]}
    print(f"[INFO] Saved source-stratified group split: {path}")


def compute_hybrid_loss(pred, y, label_mask, _label_counts=None, _pair_flags=None):
    if SOURCE_CONTEXT["loss_aggregation"] == "original":
        return _single_hybrid_loss(pred, y, label_mask, _label_counts, _pair_flags)
    losses = [_single_hybrid_loss(
        pred[i:i + 1], y[i:i + 1], label_mask[i:i + 1],
        None if _label_counts is None else _label_counts[i:i + 1],
        None if _pair_flags is None else _pair_flags[i:i + 1],
    ) for i in range(pred.size(0))
        if (_label_counts[i] if _label_counts is not None else label_mask[i].any())]
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def make_data_loader(sequences, split_name, batch_size, shuffle, num_workers):
    dataset = TrajectoryDataset(sequences, split_name=split_name)
    sampler = None
    if shuffle and SOURCE_CONTEXT["sampling"] == "balanced":
        counts = defaultdict(int)
        for seq in sequences:
            counts[sequence_source(seq)] += 1
        weights = [1.0 / counts[sequence_source(seq)] for seq in sequences]
        sampler = WeightedRandomSampler(weights, num_samples=len(sequences), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=num_workers, collate_fn=collate_trajectory_batch)


def unique_trajectory_draw_key(records, sequence_id):
    if sequence_id not in records:
        return sequence_id
    occurrence = 1
    while (sequence_id, occurrence) in records:
        occurrence += 1
    return sequence_id, occurrence


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def source_aggregates(by_source):
    aggregate = {"by_source": by_source}
    keys = {key for result in by_source.values() for key, value in result.items()
            if key not in {"n", "num_sequences", "num_unique_sequences", "num_spearman_trajectories"}
            and not key.endswith("_pvalue")
            and isinstance(value, (int, float, np.number))}
    for key in sorted(keys):
        aggregate_key = "source_macro_trajectory_rmse" if key == "macro_trajectory_rmse" else f"source_macro_{key}"
        aggregate[aggregate_key] = finite_mean([result.get(key, np.nan) for result in by_source.values()])
    weights, values = [], []
    for name, result in by_source.items():
        value = result["macro_trajectory_rmse"]
        if np.isfinite(value):
            weights.append(SOURCE_CONTEXT["weights"].get(name, 1.0))
            values.append(value)
    aggregate["weighted_source_trajectory_rmse"] = float(np.average(values, weights=weights)) if values else float("nan")
    return aggregate


def selection_score(metrics):
    metric = SOURCE_CONTEXT["selection_metric"]
    if metric == "source_trajectory_rmse":
        return float(metrics.get("by_source", {}).get(SOURCE_CONTEXT["selection_source"], {}).get("macro_trajectory_rmse", np.nan))
    return float(metrics.get(metric, np.nan))


class SourceMetricAccumulator:
    def __init__(self):
        self.records = defaultdict(dict)
        self.unique_ids = defaultdict(set)
        self.loss_sums = defaultdict(float)
        self.counts = defaultdict(int)

    @torch.no_grad()
    def update(self, pred, y, mask, metas, *, pred_cpu=None, y_cpu=None,
               mask_cpu=None, batch_loss=None, pair_flags=None):
        # One device-to-host prediction copy is shared with all epoch metrics.
        if pred_cpu is None:
            pred_cpu = pred.detach().cpu().numpy()
        if y_cpu is None:
            y_cpu = y.detach().cpu().numpy()
        if mask_cpu is None:
            mask_cpu = mask.detach().cpu().numpy()
        indices_by_source = defaultdict(list)
        batch_records = []
        for index, item in enumerate(metas):
            if not mask_cpu[index].any():
                continue
            name = sequence_source(item)
            record = {
                "y_true": y_cpu[index][mask_cpu[index]].tolist(),
                "y_pred": pred_cpu[index][mask_cpu[index]].tolist(),
            }
            self.records[name][len(self.records[name])] = record
            batch_records.append((index, record))
            self.unique_ids[name].add(item["sequence_id"])
            indices_by_source[name].append(index)
        for name, indices in indices_by_source.items():
            if pred.size(0) == 1 and batch_loss is not None:
                value = batch_loss
            else:
                loss = compute_hybrid_loss(
                    pred[indices].detach(), y[indices], mask[indices],
                    _label_counts=tuple(int(mask_cpu[i].sum()) for i in indices),
                    _pair_flags=None if pair_flags is None else [pair_flags[i] for i in indices],
                )
                value = float(loss.item())
            self.loss_sums[name] += value * len(indices)
            self.counts[name] += len(indices)
        return batch_records

    def finalize(self, _trajectory_cache=None):
        by_source = {}
        for name, records in sorted(self.records.items()):
            true = np.concatenate([record["y_true"] for record in records.values()])
            pred = np.concatenate([record["y_pred"] for record in records.values()])
            result = compute_metrics(true, pred)
            result.update(compute_trajectory_metrics(records, _cache=_trajectory_cache))
            result["loss"] = self.loss_sums[name] / self.counts[name]
            result["num_sequences"] = self.counts[name]
            result["num_unique_sequences"] = len(self.unique_ids[name])
            result["num_spearman_trajectories"] = sum(
                len(record["y_true"]) >= 2 and len(set(record["y_true"])) > 1 and len(set(record["y_pred"])) > 1
                for record in records.values())
            by_source[name] = result
        return source_aggregates(by_source)


def _batch_to_device(batch, loader, device, _cache_on_cpu=False):
    """Reuse immutable single-trajectory inputs, with a hard per-loader budget.

    Never put a CUDA cache inside a multiprocessing dataset: Windows spawn and
    fork workers must continue to receive CPU-only dataset objects.
    """
    keys = ("x", "y", "label_mask", "attention_mask")
    cacheable = (loader.num_workers == 0 and batch["x"].size(0) == 1
                 and "_dataset_index" in batch["metas"][0]
                 and (device.type == "cuda" or _cache_on_cpu))
    if not cacheable:
        return tuple(batch[key].to(device, non_blocking=True) for key in keys)
    dataset = loader.dataset
    if not hasattr(dataset, "_device_batches"):
        dataset._device_batches = OrderedDict()
        dataset._device_batch_bytes = 0
    cache = dataset._device_batches
    cache_key = (device, batch["metas"][0]["_dataset_index"])
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key][0]
    values = tuple(batch[key].to(device, non_blocking=True) for key in keys)
    # Account conservatively for 512-byte allocator rounding on small tensors.
    size = sum(((value.numel() * value.element_size() + 511) // 512) * 512
               for value in values)
    budget = 32 * 1024 * 1024
    if size <= budget:
        while cache and (dataset._device_batch_bytes + size > budget or len(cache) >= 8192):
            _, (_, previous_size) = cache.popitem(last=False)
            dataset._device_batch_bytes -= previous_size
        cache[cache_key] = (values, size)
        dataset._device_batch_bytes += size
    return values


def _batch_loss_metadata(batch, loader):
    counts = batch.get("label_counts")
    if counts is None:
        counts = tuple(int(row.sum()) for row in batch["label_mask"])
    flags = getattr(loader.dataset, "loss_pair_flags", None)
    if flags is not None and all("_dataset_index" in item for item in batch["metas"]):
        flags = [flags[item["_dataset_index"]] for item in batch["metas"]]
    else:
        flags = None
    return counts, flags


class _EpochMetricAccumulator:
    """Keep original aggregation/reduction order while sharing CPU records."""
    def __init__(self):
        self.all_true = []
        self.all_pred = []
        self.trajectories = {}
        self.sources = SourceMetricAccumulator()
        self.total_loss = 0.0
        self.total_sequences = 0

    def update(self, pred, y, mask, batch, loss_value, pair_flags, pred_cpu=None):
        if pred_cpu is None:
            pred_cpu = pred.detach().cpu().numpy()
        y_cpu = batch["y"].numpy()
        mask_cpu = batch["label_mask"].numpy()
        records = self.sources.update(
            pred, y, mask, batch["metas"], pred_cpu=pred_cpu,
            y_cpu=y_cpu, mask_cpu=mask_cpu, batch_loss=loss_value,
            pair_flags=pair_flags,
        )
        if loss_value is not None:
            self.all_true.append(y_cpu[mask_cpu])
            self.all_pred.append(pred_cpu[mask_cpu])
            for index, record in records:
                key = unique_trajectory_draw_key(self.trajectories, batch["metas"][index]["sequence_id"])
                self.trajectories[key] = record
            self.total_loss += loss_value * pred.size(0)
            self.total_sequences += pred.size(0)
        return pred_cpu, mask_cpu

    def finalize(self):
        true = np.concatenate(self.all_true) if self.all_true else np.asarray([])
        pred = np.concatenate(self.all_pred) if self.all_pred else np.asarray([])
        metrics = compute_metrics(true, pred)
        cache = {}
        metrics.update(compute_trajectory_metrics(self.trajectories, _cache=cache))
        metrics.update(self.sources.finalize(_trajectory_cache=cache))
        metrics["loss"] = (self.total_loss / self.total_sequences
                           if self.total_sequences else np.nan)
        metrics["selection_score"] = selection_score(metrics)
        return metrics


class _BatchMetricQueue:
    """Bounded, detached reporting buffers for the single-trajectory CUDA path.

    Copies and Python loss accumulation are batched, not the optimizer updates.
    No GPU reduction is used for epoch loss; original float/Python summation
    order is retained. The CPU device option allows exact tests without CUDA.
    """
    def __init__(self, accumulator, devices=("cuda",)):
        self.accumulator = accumulator
        self.devices = devices
        self.pending = []
        self.pending_bytes = 0
        self.byte_limit = 8 * 1024 * 1024
        self.batch_limit = 32

    def update(self, pred, y, mask, batch, loss, flags):
        size = (pred.numel() + 1) * pred.element_size()
        if (pred.device.type not in self.devices or pred.size(0) != 1
                or pred.dtype != loss.dtype or size > self.byte_limit):
            self.flush()
            self.accumulator.update(pred, y, mask, batch, float(loss.item()), flags)
            return
        if self.pending_bytes + size > self.byte_limit:
            self.flush()
        # Detach immediately: keeping loss/pred graphs here would retain full
        # activations for multiple updates and defeat the memory bound.
        self.pending.append((pred.detach(), loss.detach(), batch, flags))
        self.pending_bytes += size
        if len(self.pending) >= self.batch_limit:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        parts = []
        for pred, loss, _, _ in self.pending:
            parts.extend((loss.reshape(1), pred.reshape(-1)))
        values = torch.cat(parts).cpu().numpy()
        offset = 0
        for pred, _, batch, flags in self.pending:
            loss_value = float(values[offset])
            offset += 1
            pred_cpu = values[offset:offset + pred.numel()].reshape(tuple(pred.shape))
            offset += pred.numel()
            # Single-item source loss is already represented by loss_value;
            # no y/mask device copies need to stay alive in this queue.
            self.accumulator.update(pred, None, None, batch, loss_value, flags, pred_cpu=pred_cpu)
        self.pending.clear()
        self.pending_bytes = 0


def flat_source_history(train_metrics, valid_metrics):
    result = {}
    for split, metrics in [("train", train_metrics), ("valid", valid_metrics)]:
        if not metrics:
            continue
        result[f"{split}_selection_score"] = metrics["selection_score"]
        for key, value in metrics.items():
            if key.startswith("source_macro_"):
                result[f"{split}_{key}"] = value
        for name, values in metrics.get("by_source", {}).items():
            for key, value in values.items():
                result[f"{split}_source::{name}::{key}"] = value
    return result


def compute_oof_metrics(predictions_df, fold_metrics):
    metrics = _single_oof_metrics(predictions_df, fold_metrics)
    by_source = {}
    for name, frame in predictions_df.groupby("source_name", sort=True):
        source_folds = [{"num_valid_sequences": fold["valid_metrics"]["by_source"][name]["num_sequences"],
                         "valid_metrics": fold["valid_metrics"]["by_source"][name]} for fold in fold_metrics]
        result = _single_oof_metrics(frame, source_folds)
        result["num_sequences"] = int(frame["line_idx"].nunique())
        result["num_unique_sequences"] = result["num_sequences"]
        result["num_spearman_trajectories"] = sum(
            len(part) >= 2 and part["y_true"].nunique() > 1 and part["y_pred"].nunique() > 1
            for _, part in frame.groupby("line_idx"))
        by_source[name] = result
    metrics.update(source_aggregates(by_source))
    metrics["selection_score"] = selection_score(metrics)
    return metrics


def make_permuted_sequences(sequences, channel_idx, seed):
    # Preserve every source's marginal distribution; never shuffle across models.
    result = list(sequences)
    indices = defaultdict(list)
    for i, sequence in enumerate(sequences):
        indices[sequence_source(sequence)].append(i)
    for source_index, name in enumerate(sorted(indices)):
        positions = indices[name]
        permuted = _single_permutation([sequences[i] for i in positions], channel_idx, seed + 100003 * source_index)
        for i, sequence in zip(positions, permuted):
            result[i] = sequence
    return result


def save_curve(history, output_path):
    _single_save_curve(history, output_path)
    if SOURCE_CONTEXT["selection_metric"] == "loss" or not history:
        return
    epochs = [row["epoch"] for row in history]
    values = [row["valid_selection_score"] for row in history]
    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(epochs, values, label=SOURCE_CONTEXT["selection_metric"])
    plt.axvline(epochs[int(np.nanargmin(values))], linestyle="--", label="Best selection score")
    plt.xlabel("Epoch")
    plt.ylabel("Validation selection score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path.with_name("selection_curve.png"))
    plt.close()


def attach_source_outputs(metrics, args, sequences, output_dir, train_metrics, valid_metrics, test_metrics,
                          cross_validation_info=None):
    rows = []
    for split, results in [("train", train_metrics), ("valid", valid_metrics), ("test", test_metrics)]:
        for name, result in results["by_source"].items():
            rows.append({"split": split, "source_name": name, **result})
    pd.DataFrame(rows).to_csv(output_dir / "source_metrics.csv", index=False, encoding="utf-8-sig")
    if cross_validation_info is not None:
        fold_rows = []
        for fold in cross_validation_info["folds"]:
            for split in ("train", "valid"):
                for name, result in fold[f"{split}_metrics"]["by_source"].items():
                    fold_rows.append({"fold": fold["fold"], "split": split,
                                      "source_name": name, **result})
        pd.DataFrame(fold_rows).to_csv(output_dir / "cross_validation_source_metrics.csv",
                                     index=False, encoding="utf-8-sig")
    importance = SOURCE_CONTEXT.get("feature_importance", [])
    if importance:
        pd.DataFrame(importance).to_csv(output_dir / "source_feature_importance.csv", index=False, encoding="utf-8-sig")
    metrics["input_files"] = SOURCE_CONTEXT["files"]
    metrics["test_size"] = SOURCE_CONTEXT["manifest"]["test_size"]
    metrics["valid_size"] = SOURCE_CONTEXT["manifest"]["valid_size"]
    metrics["sources"] = [{"source_name": name, "input_file": file,
                           "num_raw_samples": SOURCE_CONTEXT["raw_counts"].get(name, 0),
                           "num_valid_sequences_before_filters": SOURCE_CONTEXT["valid_counts"].get(name, 0)}
                          for name, file in zip(SOURCE_CONTEXT["names"], SOURCE_CONTEXT["files"])]
    metrics["multi_source"] = {
        "source_identity_is_model_input": False,
        "sampling_strategy": args.sampling_strategy,
        "balanced_sampling_epoch_size": "number of training trajectories; source-equal in expectation, with replacement",
        "loss_aggregation": args.loss_aggregation,
        "selection_metric": SOURCE_CONTEXT["selection_metric"],
        "selection_source": args.selection_source,
        "selection_weights": SOURCE_CONTEXT["weights"],
        "lr_scheduler_metric": "valid_hybrid_loss (unchanged)",
        "split_manifest": SOURCE_CONTEXT["manifest_path"],
        "validation_counts_kind": "out_of_fold" if args.use_group_kfold else "holdout",
        "permutation_scope": "within source",
        "source_loss_definition": "Original hybrid loss computed on each source subset within a batch, averaged by trajectory count; not a decomposition of the global batch loss.",
    }
    metrics["training"]["early_stopping_metric"] = (
        "valid_hybrid_loss" if SOURCE_CONTEXT["selection_metric"] == "loss"
        else "valid_" + SOURCE_CONTEXT["selection_metric"])
    metrics["training"]["best_valid_selection_score"] = valid_metrics["selection_score"]
    metrics["multi_source"]["counts_by_split"] = {
        split: {name: {"num_sequences": result["num_unique_sequences"], "num_labeled_steps": result["n"]}
                for name, result in results["by_source"].items()}
        for split, results in [("train", train_metrics), ("valid", valid_metrics), ("test", test_metrics)]}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, default="steps.jsonl")
    parser.add_argument("--output_dir", type=str, default="./small_transformer_value_output")

    parser.add_argument(
        "--group_by",
        type=str,
        default="problem",
        choices=["problem", "sample_idx", "line_idx"],
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--valid_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--position_mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help=(
            "Position feature normalization mode: 'relative' divides each step "
            "end position by the final end position of that sample; 'absolute' "
            "divides by the fixed model maximum output length 8192."
        ),
    )

    parser.add_argument(
        "--use_group_kfold",
        action="store_true",
        help=(
            "Use GroupKFold on the non-test pool. The independent test set is "
            "still created with --test_size and is evaluated only after CV."
        ),
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of GroupKFold folds when --use_group_kfold is enabled.",
    )

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument(
        "--transformer_type",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "causal"],
    )

    # Training hyperparameters
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--use_warmup",
        action="store_true",
        help="Enable linear learning-rate warmup before ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help=(
            "Fraction of the planned optimizer steps used for linear warmup "
            "when --use_warmup is enabled."
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)

    # Optional data filtering
    parser.add_argument(
        "--remove_short_low_confidence_steps",
        action="store_true",
        help=(
            "Remove an interior step when its confidence is lower than both "
            "neighboring steps by at least --low_confidence_gap_threshold "
            "and its token span is smaller than "
            "--short_step_token_threshold."
        ),
    )
    parser.add_argument(
        "--low_confidence_gap_threshold",
        type=float,
        default=0.20,
        help=(
            "Minimum absolute confidence gap from both neighboring steps for "
            "the optional short low-confidence step removal rule."
        ),
    )
    parser.add_argument(
        "--short_step_token_threshold",
        type=int,
        default=6,
        help=(
            "A qualifying low-confidence step is removed only when its token "
            "span is strictly smaller than this value."
        ),
    )
    parser.add_argument(
        "--remove_zero_dominated_samples",
        action="store_true",
        help=(
            "Remove a sample when its first labeled step has value 0 and the "
            "fraction of labeled steps with value 0 is at least "
            "--zero_label_ratio_threshold."
        ),
    )
    parser.add_argument(
        "--zero_label_ratio_threshold",
        type=float,
        default=0.80,
        help=(
            "Minimum zero-label fraction for the optional sample-level "
            "filtering rule."
        ),
    )
    parser.add_argument(
        "--filtered_sample_keep_ratio", type=float, default=0.0,
        help=("Shared retention fraction in [0, 1] for zero-dominated and truncated-incorrect "
              "sample filtering. Keep floor(candidate_count * ratio) per source and split. "
              "Default 0 retains the original zero-dominated removal behavior."),
    )
    parser.add_argument(
        "--filter_scope",
        type=str,
        default="train_only",
        choices=["train_only", "train_and_test"],
        help=(
            "Apply enabled filtering rules only to the non-test training side "
            "(including validation), or to both the training side and test set."
        ),
    )

    # Feature importance
    parser.add_argument("--permutation_repeats", type=int, default=5)

    add_multi_source_arguments(parser)
    parser.add_argument("--gpu_groups", type=str, default=None,
                        help='Single-process DP+TP layout, e.g. "0,1;2,3": TP=2 within groups, DP=2 between groups. IDs are logical CUDA indices.')
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="TP GPUs per group; inferred from gpu_groups when omitted. Without gpu_groups, use the first N GPUs as one TP group. Default effectively 1.")
    args = parser.parse_args()
    args._resolved_gpu_groups = resolve_gpu_groups(args)
    configure_sources(args)

    if args.use_group_kfold and args.num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")

    if args.low_confidence_gap_threshold < 0.0:
        raise ValueError(
            "low_confidence_gap_threshold must be non-negative."
        )

    if args.short_step_token_threshold <= 0:
        raise ValueError(
            "short_step_token_threshold must be greater than 0."
        )

    if not 0.0 <= args.zero_label_ratio_threshold <= 1.0:
        raise ValueError(
            "zero_label_ratio_threshold must be in [0, 1]."
        )
    if not np.isfinite(args.filtered_sample_keep_ratio) or not 0.0 <= args.filtered_sample_keep_ratio <= 1.0:
        raise ValueError("filtered_sample_keep_ratio must be in [0, 1].")

    set_seed(args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv_path = output_dir / "window_dataset.csv"
    predictions_csv_path = output_dir / "test_predictions.csv"
    metrics_json_path = output_dir / "metrics.json"
    feature_importance_csv_path = output_dir / "feature_importance.csv"
    curve_png_path = output_dir / "curve.png"
    curve_csv_path = output_dir / "curve.csv"
    best_model_path = output_dir / "best_model.pt"

    cv_metrics_csv_path = output_dir / "cross_validation_metrics.csv"
    cv_predictions_csv_path = output_dir / "cross_validation_predictions.csv"

    print(f"[INFO] Reading: {args.input_file}")
    samples = read_jsonl(args.input_file)
    print(f"[INFO] Loaded raw samples: {len(samples)}")


    print("[INFO] Building full-trajectory 4D sequences...")
    sequences, all_steps_df = build_trajectory_sequences(
        samples=samples,
        group_by=args.group_by,
        position_mode=args.position_mode,
        transformer_type=args.transformer_type,
        build_frame=False,
    )
    del samples, all_steps_df


    if not sequences:
        raise ValueError(
            "No valid sequence was built. Please check steps, token_avg_confidence, "
            "first_token_index, and step_prospective_correctness."
        )

    print(f"[INFO] Valid supervised sequences: {len(sequences)}")
    print(
        f"[INFO] Total labeled steps: "
        f"{sum(int(seq['label_mask'].sum()) for seq in sequences)}"
    )
    print(f"[INFO] Total groups: {len(set(seq['group_id'] for seq in sequences))}")

    prepare_source_splits(sequences, args, output_dir)

    device = args._resolved_gpu_groups[0][0] if args._resolved_gpu_groups else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if not args.use_group_kfold:
        train_idx, valid_idx, test_idx = split_train_valid_test_sequences(
            sequences=sequences,
            test_size=args.test_size,
            valid_size=args.valid_size,
            random_state=args.random_state,
        )

        train_sequences_raw = [sequences[i] for i in train_idx]
        valid_sequences_raw = [sequences[i] for i in valid_idx]
        test_sequences_raw = [sequences[i] for i in test_idx]

        train_sequences_raw = apply_optional_sequence_filters(
            sequences=train_sequences_raw,
            args=args,
            split_name="train",
        )
        valid_sequences_raw = apply_optional_sequence_filters(
            sequences=valid_sequences_raw,
            args=args,
            split_name="valid",
        )
        if args.filter_scope == "train_and_test":
            test_sequences_raw = apply_optional_sequence_filters(
                sequences=test_sequences_raw,
                args=args,
                split_name="test",
            )

        if not train_sequences_raw:
            raise ValueError(
                "No training sequence remains after optional filtering."
            )
        if not valid_sequences_raw:
            raise ValueError(
                "No validation sequence remains after optional filtering."
            )
        if not test_sequences_raw:
            raise ValueError(
                "No test sequence remains after optional filtering."
            )

        train_groups = set(seq["group_id"] for seq in train_sequences_raw)
        valid_groups = set(seq["group_id"] for seq in valid_sequences_raw)
        test_groups = set(seq["group_id"] for seq in test_sequences_raw)


        print(f"[INFO] Train sequences: {len(train_sequences_raw)}")
        print(f"[INFO] Valid sequences: {len(valid_sequences_raw)}")
        print(f"[INFO] Test sequences:  {len(test_sequences_raw)}")
        print(
            f"[INFO] Train labeled steps: "
            f"{sum(int(seq['label_mask'].sum()) for seq in train_sequences_raw)}"
        )
        print(
            f"[INFO] Valid labeled steps: "
            f"{sum(int(seq['label_mask'].sum()) for seq in valid_sequences_raw)}"
        )
        print(
            f"[INFO] Test labeled steps:  "
            f"{sum(int(seq['label_mask'].sum()) for seq in test_sequences_raw)}"
        )

        sequence_id_to_split: Dict[int, str] = {}
        for seq in train_sequences_raw:
            sequence_id_to_split[seq["sequence_id"]] = "train"
        for seq in valid_sequences_raw:
            sequence_id_to_split[seq["sequence_id"]] = "valid"
        for seq in test_sequences_raw:
            sequence_id_to_split[seq["sequence_id"]] = "test"

        sequences_for_output = (
            train_sequences_raw
            + valid_sequences_raw
            + test_sequences_raw
        )
        window_dataset_df = build_window_dataset_df(
            sequences=sequences_for_output,
            sequence_id_to_split=sequence_id_to_split,
        )
        window_dataset_df.to_csv(
            dataset_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[INFO] Saved dataset CSV: {dataset_csv_path}")
        del window_dataset_df

        validate_source_coverage(train_sequences_raw, "train")
        validate_source_coverage(valid_sequences_raw, "valid")
        validate_source_coverage(test_sequences_raw, "test")
        normalizer = FeatureNormalizer()
        normalizer.fit(train_sequences_raw)
        train_sequences = normalizer.transform_sequences(train_sequences_raw)
        valid_sequences = normalizer.transform_sequences(valid_sequences_raw)
        test_sequences = normalizer.transform_sequences(test_sequences_raw)

        train_loader = make_data_loader(
            train_sequences,
            "train",
            args.batch_size,
            True,
            args.num_workers,
        )
        valid_loader = make_data_loader(
            valid_sequences,
            "valid",
            args.batch_size,
            False,
            args.num_workers,
        )
        test_loader = make_data_loader(
            test_sequences,
            "test",
            args.batch_size,
            False,
            args.num_workers,
        )

        (
            model,
            history,
            best_epoch,
            best_valid_loss,
            best_valid_rmse,
            num_params,
        ) = train_model_with_validation(
            args=args,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            seed=args.random_state,
            best_model_path=best_model_path,
        )

        pd.DataFrame(history).to_csv(
            curve_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        save_curve(history=history, output_path=curve_png_path)
        print(f"[INFO] Saved curve CSV: {curve_csv_path}")
        print(f"[INFO] Saved curve PNG: {curve_png_path}")

        print("[INFO] Final evaluation on train/valid/test using best model...")
        train_metrics, _ = evaluate(
            model=model,
            loader=train_loader,
            device=device,
            desc="Final train eval",
            collect_predictions=False,
        )
        valid_metrics, _ = evaluate(
            model=model,
            loader=valid_loader,
            device=device,
            desc="Final valid eval",
            collect_predictions=False,
        )
        test_metrics, test_predictions_df = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            desc="Final test eval",
            collect_predictions=True,
        )

        test_predictions_df.to_csv(
            predictions_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[INFO] Saved test predictions: {predictions_csv_path}")

        print("[INFO] Computing permutation feature importance...")
        feature_importance_df = compute_permutation_feature_importance(
            model=model,
            test_sequences=test_sequences,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            repeats=args.permutation_repeats,
            seed=args.random_state,
        )
        feature_importance_df.to_csv(
            feature_importance_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[INFO] Saved feature importance: {feature_importance_csv_path}")

        num_train_sequences = len(train_sequences)
        num_valid_sequences = len(valid_sequences)
        num_test_sequences = len(test_sequences)
        num_train_groups = len(train_groups)
        num_valid_groups = len(valid_groups)
        num_test_groups = len(test_groups)
        num_train_labeled_steps = sum(
            int(seq["label_mask"].sum()) for seq in train_sequences_raw
        )
        num_valid_labeled_steps = sum(
            int(seq["label_mask"].sum()) for seq in valid_sequences_raw
        )
        num_test_labeled_steps = sum(
            int(seq["label_mask"].sum()) for seq in test_sequences_raw
        )
        cross_validation_info = None

    else:
        train_valid_idx, test_idx = split_train_test_sequences(
            sequences=sequences,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        train_valid_sequences_raw = [
            sequences[i] for i in train_valid_idx
        ]
        test_sequences_raw = [sequences[i] for i in test_idx]

        train_valid_sequences_raw = apply_optional_sequence_filters(
            sequences=train_valid_sequences_raw,
            args=args,
            split_name="train/valid pool",
        )
        if args.filter_scope == "train_and_test":
            test_sequences_raw = apply_optional_sequence_filters(
                sequences=test_sequences_raw,
                args=args,
                split_name="test",
            )

        if not train_valid_sequences_raw:
            raise ValueError(
                "No train/valid sequence remains after optional filtering."
            )
        if not test_sequences_raw:
            raise ValueError(
                "No test sequence remains after optional filtering."
            )

        train_valid_groups_array = np.asarray([
            seq["group_id"] for seq in train_valid_sequences_raw
        ])
        unique_train_valid_groups = np.unique(train_valid_groups_array)

        if len(unique_train_valid_groups) < args.num_folds:
            raise ValueError(
                f"GroupKFold requires at least num_folds distinct groups in the "
                f"non-test pool, but got {len(unique_train_valid_groups)} groups "
                f"for num_folds={args.num_folds}."
            )

        test_groups = set(seq["group_id"] for seq in test_sequences_raw)
        train_valid_groups = set(train_valid_groups_array.tolist())


        print("[INFO] GroupKFold mode enabled.")
        print(
            f"[INFO] Train/valid pool sequences: "
            f"{len(train_valid_sequences_raw)}"
        )
        print(f"[INFO] Independent test sequences: {len(test_sequences_raw)}")
        print(f"[INFO] Train/valid pool groups: {len(train_valid_groups)}")
        print(f"[INFO] Test groups: {len(test_groups)}")
        print(f"[INFO] Number of folds: {args.num_folds}")

        group_kfold = SourceGroupKFold(train_valid_sequences_raw, args.num_folds, args.random_state)
        fold_metrics: List[Dict[str, Any]] = []
        fold_histories: List[List[Dict[str, float]]] = []
        fold_prediction_frames: List[pd.DataFrame] = []
        fold_best_epochs: List[int] = []

        split_iterator = group_kfold.split(
            X=np.zeros(
                (len(train_valid_sequences_raw), 1),
                dtype=np.float32,
            ),
            groups=train_valid_groups_array,
        )

        for fold_idx, (fold_train_pos, fold_valid_pos) in enumerate(
            split_iterator,
            start=1,
        ):
            fold_train_raw = [
                train_valid_sequences_raw[pos]
                for pos in fold_train_pos
            ]
            fold_valid_raw = [
                train_valid_sequences_raw[pos]
                for pos in fold_valid_pos
            ]
            fold_train_groups = set(seq["group_id"] for seq in fold_train_raw)
            fold_valid_groups = set(seq["group_id"] for seq in fold_valid_raw)

            if fold_train_groups & fold_valid_groups:
                raise RuntimeError(
                    f"Data leakage in fold {fold_idx}: train and valid groups overlap."
                )

            print("\n" + "=" * 70)
            print(f"[INFO] Fold {fold_idx}/{args.num_folds}")
            print(f"[INFO] Fold train sequences: {len(fold_train_raw)}")
            print(f"[INFO] Fold valid sequences: {len(fold_valid_raw)}")
            print(f"[INFO] Fold train groups: {len(fold_train_groups)}")
            print(f"[INFO] Fold valid groups: {len(fold_valid_groups)}")

            validate_source_coverage(fold_train_raw, "fold train")
            validate_source_coverage(fold_valid_raw, "fold valid")
            fold_normalizer = FeatureNormalizer()
            fold_normalizer.fit(fold_train_raw)
            fold_train = fold_normalizer.transform_sequences(fold_train_raw)
            fold_valid = fold_normalizer.transform_sequences(fold_valid_raw)

            fold_train_loader = make_data_loader(
                fold_train,
                "train",
                args.batch_size,
                True,
                args.num_workers,
            )
            fold_valid_loader = make_data_loader(
                fold_valid,
                "valid",
                args.batch_size,
                False,
                args.num_workers,
            )

            fold_dir = output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_best_model_path = fold_dir / "best_model.pt"
            fold_curve_csv_path = fold_dir / "curve.csv"
            fold_curve_png_path = fold_dir / "curve.png"

            (
                fold_model,
                fold_history,
                fold_best_epoch,
                fold_best_loss,
                fold_best_rmse,
                fold_num_params,
            ) = train_model_with_validation(
                args=args,
                train_loader=fold_train_loader,
                valid_loader=fold_valid_loader,
                device=device,
                seed=args.random_state + fold_idx,
                best_model_path=fold_best_model_path,
                log_prefix=f"Fold {fold_idx} ",
            )

            pd.DataFrame(fold_history).to_csv(
                fold_curve_csv_path,
                index=False,
                encoding="utf-8-sig",
            )
            save_curve(fold_history, fold_curve_png_path)

            fold_train_metrics, _ = evaluate(
                model=fold_model,
                loader=fold_train_loader,
                device=device,
                desc=f"Fold {fold_idx} final train eval",
                collect_predictions=False,
            )
            fold_valid_metrics, fold_predictions_df = evaluate(
                model=fold_model,
                loader=fold_valid_loader,
                device=device,
                desc=f"Fold {fold_idx} final valid eval",
                collect_predictions=True,
            )
            fold_predictions_df["cv_fold"] = fold_idx
            fold_prediction_frames.append(fold_predictions_df)

            fold_metrics.append({
                "fold": int(fold_idx),
                "num_train_sequences": int(len(fold_train_raw)),
                "num_valid_sequences": int(len(fold_valid_raw)),
                "num_train_groups": int(len(fold_train_groups)),
                "num_valid_groups": int(len(fold_valid_groups)),
                "best_epoch": int(fold_best_epoch),
                "best_valid_loss": float(fold_best_loss),
                "best_valid_rmse": float(fold_best_rmse),
                "best_valid_selection_score": float(fold_history[fold_best_epoch - 1]["valid_selection_score"]),
                "normalization": fold_normalizer.to_dict(),
                "num_trainable_parameters": int(fold_num_params),
                "train_metrics": fold_train_metrics,
                "valid_metrics": fold_valid_metrics,
            })
            fold_histories.append(fold_history)
            fold_best_epochs.append(int(fold_best_epoch))
            # Otherwise the previous fold's GPU model survives throughout the
            # next train_model_with_validation call (the RHS runs before assignment).
            del fold_model, fold_train_loader, fold_valid_loader, fold_train, fold_valid

        cv_predictions_df = pd.concat(
            fold_prediction_frames,
            ignore_index=True,
        )
        del fold_prediction_frames, fold_predictions_df
        cv_predictions_df.to_csv(
            cv_predictions_csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        cv_oof_metrics = compute_oof_metrics(
            predictions_df=cv_predictions_df,
            fold_metrics=fold_metrics,
        )
        cv_fold_summary = summarize_fold_metrics(fold_metrics)

        cv_metric_rows = []
        for fold in fold_metrics:
            row = {
                "fold": fold["fold"],
                "num_train_sequences": fold["num_train_sequences"],
                "num_valid_sequences": fold["num_valid_sequences"],
                "num_train_groups": fold["num_train_groups"],
                "num_valid_groups": fold["num_valid_groups"],
                "best_epoch": fold["best_epoch"],
                "best_valid_loss": fold["best_valid_loss"],
                "best_valid_rmse": fold["best_valid_rmse"],
            }
            for key, value in fold["valid_metrics"].items():
                if not isinstance(value, dict):
                    row[f"valid_{key}"] = value
            cv_metric_rows.append(row)

        pd.DataFrame(cv_metric_rows).to_csv(
            cv_metrics_csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        aggregate_history = aggregate_cv_histories(fold_histories)
        pd.DataFrame(aggregate_history).to_csv(
            curve_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        save_curve(aggregate_history, curve_png_path)
        print(f"[INFO] Saved aggregate curve CSV: {curve_csv_path}")
        print(f"[INFO] Saved aggregate curve PNG: {curve_png_path}")

        selected_epochs = max(
            1,
            int(round(float(np.median(np.asarray(fold_best_epochs, dtype=float))))),
        )
        best_epoch = selected_epochs
        best_valid_loss = float(cv_oof_metrics["loss"])
        best_valid_rmse = float(cv_oof_metrics["rmse"])

        averaged_lr_schedule = {
            int(row["epoch"]): float(row["lr"])
            for row in aggregate_history
            if int(row["epoch"]) <= selected_epochs
        }

        normalizer = FeatureNormalizer()
        validate_source_coverage(train_valid_sequences_raw, "final train")
        validate_source_coverage(test_sequences_raw, "test")
        normalizer.fit(train_valid_sequences_raw)
        train_sequences = normalizer.transform_sequences(train_valid_sequences_raw)
        test_sequences = normalizer.transform_sequences(test_sequences_raw)

        sequence_id_to_split = {
            seq["sequence_id"]: "train" for seq in train_valid_sequences_raw
        }
        sequence_id_to_split.update({
            seq["sequence_id"]: "test" for seq in test_sequences_raw
        })
        sequences_for_output = (
            train_valid_sequences_raw + test_sequences_raw
        )
        window_dataset_df = build_window_dataset_df(
            sequences=sequences_for_output,
            sequence_id_to_split=sequence_id_to_split,
        )
        window_dataset_df.to_csv(
            dataset_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[INFO] Saved dataset CSV: {dataset_csv_path}")
        del window_dataset_df

        train_loader = make_data_loader(
            train_sequences,
            "train",
            args.batch_size,
            True,
            args.num_workers,
        )
        test_loader = make_data_loader(
            test_sequences,
            "test",
            args.batch_size,
            False,
            args.num_workers,
        )

        model, final_refit_history, num_params = train_model_fixed_epochs(
            args=args,
            train_loader=train_loader,
            device=device,
            num_epochs=selected_epochs,
            seed=args.random_state + 10000,
            best_model_path=best_model_path,
            lr_schedule=averaged_lr_schedule,
        )

        train_metrics, _ = evaluate(
            model=model,
            loader=train_loader,
            device=device,
            desc="Final refit train eval",
            collect_predictions=False,
        )
        valid_metrics = cv_oof_metrics
        test_metrics, test_predictions_df = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            desc="Final independent test eval",
            collect_predictions=True,
        )

        test_predictions_df.to_csv(
            predictions_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[INFO] Saved test predictions: {predictions_csv_path}")

        print("[INFO] Computing permutation feature importance...")
        feature_importance_df = compute_permutation_feature_importance(
            model=model,
            test_sequences=test_sequences,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            repeats=args.permutation_repeats,
            seed=args.random_state,
        )
        feature_importance_df.to_csv(
            feature_importance_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[INFO] Saved feature importance: {feature_importance_csv_path}")

        train_groups = train_valid_groups
        valid_groups = set()
        num_train_sequences = len(train_sequences)
        num_valid_sequences = 0
        num_test_sequences = len(test_sequences)
        num_train_groups = len(train_groups)
        num_valid_groups = 0
        num_test_groups = len(test_groups)
        num_train_labeled_steps = sum(
            int(seq["label_mask"].sum()) for seq in train_valid_sequences_raw
        )
        num_valid_labeled_steps = 0
        num_test_labeled_steps = sum(
            int(seq["label_mask"].sum()) for seq in test_sequences_raw
        )
        cross_validation_info = {
            "enabled": True,
            "num_folds": int(args.num_folds),
            "selection_rule": "rounded median of fold best epochs",
            "selected_final_refit_epochs": int(selected_epochs),
            "oof_metrics": cv_oof_metrics,
            "fold_metric_summary": cv_fold_summary,
            "folds": fold_metrics,
            "final_refit_history": final_refit_history,
            "cross_validation_metrics_csv": str(cv_metrics_csv_path),
            "cross_validation_predictions_csv": str(cv_predictions_csv_path),
        }

    metrics = {
        "input_file": args.input_file,
        "group_by": args.group_by,
        "test_size": args.test_size,
        "valid_size": args.valid_size,
        "random_state": args.random_state,
        "num_total_sequences": int(len(sequences_for_output)),
        "num_train_sequences": int(num_train_sequences),
        "num_valid_sequences": int(num_valid_sequences),
        "num_test_sequences": int(num_test_sequences),
        "num_total_groups": int(
            len(set(seq["group_id"] for seq in sequences_for_output))
        ),
        "num_train_groups": int(num_train_groups),
        "num_valid_groups": int(num_valid_groups),
        "num_test_groups": int(num_test_groups),
        "num_train_labeled_steps": int(num_train_labeled_steps),
        "num_valid_labeled_steps": int(num_valid_labeled_steps),
        "num_test_labeled_steps": int(num_test_labeled_steps),
        "feature_description": {
            "input_shape_per_sample": "[num_steps, 4]",
            "features": [
                "conf_t",
                (
                    "local_conf_std over [previous_previous, previous, current]"
                    if args.transformer_type == "causal"
                    else "local_conf_std over [previous, current, next]"
                ),
                (
                    "local_conf_slope over [previous_previous, previous, current]"
                    if args.transformer_type == "causal"
                    else "local_conf_slope over [previous, current, next]"
                ),
                "normalized end position"
            ],
            "note": (
                "Causal local features use only current and historical step confidence."
                if args.transformer_type == "causal"
                else "This is an offline centered feature setting because local features use next-step confidence."
            )
        },
        "normalization": normalizer.to_dict(),
        "model": {
            "name": "SmallTransformerEncoderValueModel",
            "input_dim": 4,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "max_len": args.max_len,
            "num_trainable_parameters": int(num_params),
            "output_activation": "sigmoid",
        },
        "training": {
            "optimizer": "AdamW",
            "lr": args.lr,
            "use_warmup": args.use_warmup,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "grad_clip": args.grad_clip,
            "loss": "Hybrid loss over labeled steps: absolute MSE + trajectory-mean MSE + centered MSE + weighted pairwise logistic ranking + adjacent-delta Smooth L1",
            "loss_weights": {
                "absolute_mse": HYBRID_ABSOLUTE_WEIGHT,
                "trajectory_mean_mse": HYBRID_TRAJECTORY_MEAN_WEIGHT,
                "centered_mse": HYBRID_CENTERED_WEIGHT,
                "pairwise_logistic": HYBRID_PAIRWISE_WEIGHT,
                "adjacent_delta_smooth_l1": HYBRID_DELTA_WEIGHT,
                "pairwise_temperature": PAIRWISE_TEMPERATURE,
                "delta_smooth_l1_beta": DELTA_SMOOTH_L1_BETA,
            },
            "early_stopping_metric": "valid_hybrid_loss",
            "best_epoch": int(best_epoch),
            "best_valid_loss": float(best_valid_loss),
            "best_valid_rmse": float(best_valid_rmse),
        },
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    attach_source_outputs(metrics, args, sequences_for_output, output_dir,
                          train_metrics, valid_metrics, test_metrics, cross_validation_info)

    if cross_validation_info is not None:
        metrics["cross_validation"] = cross_validation_info

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved metrics: {metrics_json_path}")

    print("\n========== Final Test Metrics ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    print("\n[DONE]")
    print(f"Best epoch: {best_epoch}")
    print(f"Best valid hybrid loss: {best_valid_loss}")
    print(f"Best valid RMSE at best-loss epoch: {best_valid_rmse}")
    print(f"Test Spearman: {test_metrics['spearman']}")
    print(f"Test R2: {test_metrics['r2']}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
