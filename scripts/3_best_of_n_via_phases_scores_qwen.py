#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate N Qwen3 reasoning trajectories per test problem, reproduce the original
step-confidence and exact full-vocabulary entropy phase partition, score every
complete confidence trajectory with a saved SmallTransformerEncoderValueModel,
aggregate phase scores with product/min/mean/last, grade every candidate with
`grade_answer_verl`, and evaluate random-subsampled Best-of-N.

The implementation intentionally preserves the feature and phase logic of the
four source scripts supplied by the user.  The pipeline is staged so generation,
entropy/phase construction, and value-model scoring can be resumed independently.

GPU syntax:
    --gpus '0,1;2,3' --tensor_parallel_size 2
means two data-parallel workers; each worker uses two GPUs for tensor parallelism.
TP+DP is applied to the large reasoning model during generation and exact entropy
forward.  The small trajectory value model is evaluated separately on
--value_device. Use --value_device '0,1;2,3' for value DP=2, TP=2
(commas join TP GPUs; semicolons separate DP replicas). Indices refer to
CUDA_VISIBLE_DEVICES. Scoring batch size is per DP replica. FP32 is retained;
attention kernels and TP reductions may introduce small rounding differences.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
import socket
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.transformer_bon.value_parallel import (
    ParallelValueModel, load_parallel_value_components, parse_value_devices,
)

from scripts.transformer_bon.phases import (
    add_phase_partition_arguments, select_adaptive_boundaries,
    validate_phase_partition_options,
)
from scripts.transformer_bon.gpu_phases import select_phase_boundaries_tensor
from scripts.transformer_bon.prepared_cache import load_standalone_prepared_records
from scripts.transformer_bon.entropy_cache import (
    append_entropy_record, build_token_entropy_record, run_standalone_phase_stage,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROMPT_SUFFIX = (
    "\nLet's reason step by step. Enclose the reasoning process within "
    "<think>...</think>, then summarize it and present the final answer within \\boxed{}."
)

AGGREGATIONS = ("product", "min", "mean", "last")
SCORE_KEYS = {
    "product": "product_score",
    "min": "min_score",
    "mean": "mean_score",
    "last": "last_score",
}


def load_vllm_api() -> Tuple[Any, Any]:
    """Import vLLM only after a generation worker has installed its environment."""
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def parse_bool(value: Any) -> bool:
    """解析命令行中的布尔值，支持 true/false、1/0、yes/no、on/off。"""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value!r}. Use true or false."
    )

def build_qwen3_prompt(
    problem: str,
    tokenizer: Any,
    enable_thinking: bool,
) -> str:
    """使用 Qwen3 chat template 构造生成 prompt，并控制 thinking 模式。"""
    messages = [
        {
            "role": "user",
            "content": problem + PROMPT_SUFFIX,
        }
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as e:
        raise RuntimeError(
            "The tokenizer chat template does not support enable_thinking. "
            "Please use the Qwen3-8B tokenizer and a compatible Transformers/vLLM version."
        ) from e

    if not isinstance(prompt, str):
        raise TypeError(
            "tokenizer.apply_chat_template(..., tokenize=False) must return a string"
        )

    return prompt

def load_json_array(path: str) -> List[Dict[str, Any]]:
    """读取数组格式 JSON 数据集。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input file should be a JSON array, but got {type(data)}")

    return data

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """追加写入一行 JSONL，并立即 flush 到磁盘。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """逐行读取 JSONL。"""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e

def get_sampled_token_logprob(
    logprob_dict: Optional[Dict[int, Any]],
    token_id: int,
) -> Optional[float]:
    """
    从 vLLM 每个生成位置返回的 logprobs 字典中取出实际生成 token 的 logprob。

    vLLM 中 logprob_dict 的 value 通常是 Logprob 对象，常见字段为 .logprob。
    为了兼容不同版本，也兼容 value 本身就是 float 的情况。
    """
    if logprob_dict is None:
        return None

    if token_id not in logprob_dict:
        return None

    value = logprob_dict[token_id]

    if hasattr(value, "logprob"):
        return float(value.logprob)

    return float(value)

def build_token_pieces_with_char_spans(
    token_ids: List[int],
    tokenizer,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    将生成 token 对齐到字符区间。

    不直接 tokenizer.decode([token_id])，因为很多 tokenizer 的单 token decode
    在空格、特殊符号、byte fallback 等情况下可能和整段 decode 不完全一致。

    这里使用“增量 decode 差分”的方法：
    decode(token_ids[:i+1]) - decode(token_ids[:i])
    从而得到每个 token 在最终输出文本中的字符 span。
    """
    pieces: List[Dict[str, Any]] = []

    prev_text = ""
    prev_len = 0

    for i in range(len(token_ids)):
        cur_text = tokenizer.decode(
            token_ids[: i + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        piece = cur_text[len(prev_text):]

        start = prev_len
        end = start + len(piece)

        pieces.append(
            {
                "token_index": i,
                "token_id": int(token_ids[i]),
                "text": piece,
                "char_start": start,
                "char_end": end,
            }
        )

        prev_text = cur_text
        prev_len = end

    full_text = prev_text
    return full_text, pieces

def split_text_into_steps(text: str, sep: str = "\n\n") -> List[Dict[str, Any]]:
    """
    按 sep 切分文本，返回每个 step 的文本和字符区间。
    空 step 会被过滤。
    """
    raw_parts = text.split(sep)

    steps: List[Dict[str, Any]] = []
    cursor = 0

    for part in raw_parts:
        raw_start = cursor
        raw_end = cursor + len(part)

        # 去掉 step 两端空白，但保留在原文中的准确 char span
        left_strip_len = len(part) - len(part.lstrip())
        right_strip_len = len(part) - len(part.rstrip())

        start = raw_start + left_strip_len
        end = raw_end - right_strip_len
        step_text = text[start:end]

        if step_text:
            steps.append(
                {
                    "step": step_text,
                    "char_start": start,
                    "char_end": end,
                }
            )

        cursor = raw_end + len(sep)

    return steps

def token_step_overlap(
    token_start: int,
    token_end: int,
    step_start: int,
    step_end: int,
) -> int:
    """计算 token 字符区间和 step 字符区间的重叠长度。"""
    return max(0, min(token_end, step_end) - max(token_start, step_start))

def compute_step_confidences(
    text: str,
    token_pieces: List[Dict[str, Any]],
    token_logprobs: List[Optional[float]],
    sep: str = "\n\n",
) -> Tuple[List[Optional[float]], List[Dict[str, Any]]]:
    """
    按 "\n\n" 切分 step，并计算每个 step 的 token 平均 confidence。

    confidence 定义：
        token_confidence = exp(token_logprob)

    step_confidence 定义：
        step 内所有 token confidence 的算术平均。

    同时保存：
        first_token_index:
            该 step 第一个 token 在完整生成 token 序列中的下标。
            第一个生成 token 的下标为 0。

    注意：
    - 如果某个 token 跨 step 边界，分配给重叠字符最多的 step。
    - 如果某个 step 没有有效 token logprob，则 confidence 为 None。
    - 如果某个 step 没有任何 token 被对齐到，则 first_token_index 为 None。
    """
    steps = split_text_into_steps(text, sep=sep)

    step_token_confs: List[List[float]] = [[] for _ in steps]
    step_token_indices: List[List[int]] = [[] for _ in steps]

    for piece, logprob in zip(token_pieces, token_logprobs):
        token_start = piece["char_start"]
        token_end = piece["char_end"]

        if token_end <= token_start:
            continue

        best_step_idx = None
        best_overlap = 0

        for idx, step in enumerate(steps):
            overlap = token_step_overlap(
                token_start,
                token_end,
                step["char_start"],
                step["char_end"],
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_step_idx = idx

        if best_step_idx is not None and best_overlap > 0:
            token_index = int(piece["token_index"])
            step_token_indices[best_step_idx].append(token_index)

            if logprob is not None:
                conf = math.exp(logprob)
                step_token_confs[best_step_idx].append(conf)

    step_conf_list: List[Optional[float]] = []
    step_records: List[Dict[str, Any]] = []

    for step, confs, token_indices in zip(steps, step_token_confs, step_token_indices):
        if len(confs) > 0:
            avg_conf_value: Optional[float] = float(sum(confs) / len(confs))
        else:
            avg_conf_value = None

        if len(token_indices) > 0:
            first_token_index: Optional[int] = int(min(token_indices))
        else:
            first_token_index = None

        step_conf_list.append(avg_conf_value)

        step_records.append(
            {
                "step": step["step"],
                "token_avg_confidence": avg_conf_value,
                "first_token_index": first_token_index,
            }
        )

    return step_conf_list, step_records

def release_vllm(llm: Any) -> None:
    """尽可能完整地释放不同版本 vLLM 的 engine 与 GPU 资源。"""
    shutdown_candidates = [
        getattr(llm, "shutdown", None),
        getattr(getattr(llm, "llm_engine", None), "shutdown", None),
        getattr(getattr(llm, "engine", None), "shutdown", None),
    ]

    for shutdown_fn in shutdown_candidates:
        if callable(shutdown_fn):
            try:
                shutdown_fn()
                break
            except Exception:
                # 不同版本的关闭接口差异较大，失败后继续尝试下一个。
                pass

    del llm
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def resolve_hf_torch_dtype(dtype_name: str, torch_module: Any) -> Any:
    """将原 vLLM dtype 参数映射为 Transformers 可接受的 torch dtype。"""
    normalized = str(dtype_name).lower()

    if normalized == "auto":
        return "auto"

    mapping = {
        "half": torch_module.float16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float": torch_module.float32,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }

    if normalized not in mapping:
        raise ValueError(
            f"Unsupported dtype for entropy model: {dtype_name}. "
            f"Supported: auto, float16/half, bfloat16, float32."
        )

    return mapping[normalized]

def load_entropy_model(args: argparse.Namespace):
    """
    加载用于精确 full-vocabulary entropy 计算的 Hugging Face causal LM。

    该模型在 vLLM 释放后才加载，因此不会与 vLLM 同时占用同一份模型显存。
    """
    import torch
    from transformers import AutoModelForCausalLM

    model_path = args.entropy_model_path or args.model_path
    torch_dtype = resolve_hf_torch_dtype(args.dtype, torch)

    device_map: Optional[str]
    if args.entropy_device_map.lower() in {"none", "null", ""}:
        device_map = None
    else:
        device_map = args.entropy_device_map

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    if device_map is None:
        if torch.cuda.is_available():
            model = model.cuda()

    return model

def load_entropy_model_single_gpu(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM

    model_path = args.entropy_model_path or args.model_path
    torch_dtype = resolve_hf_torch_dtype(args.dtype, torch)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map={"": 0},
    )
    model.eval()
    return model

def load_entropy_model_tensor_parallel(args: argparse.Namespace):
    """在一个 DP GPU 组内使用 Transformers 原生 TP 加载 entropy 模型。"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = args.entropy_model_path or args.model_path
    torch_dtype = resolve_hf_torch_dtype(args.dtype, torch)

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if getattr(config, "base_model_tp_plan", None) is None:
        raise ValueError(
            "The entropy model does not expose base_model_tp_plan, so native "
            "Transformers tensor parallelism cannot be enabled for this model."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        tp_plan="auto",
    )
    model.eval()
    return model

def get_model_input_device(model: Any):
    """获取输入 token 应放置的设备，兼容普通模型和 device_map 分片模型。"""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device

def compute_response_token_entropies_exact(
    model: Any,
    prompt_token_ids: List[int],
    response_token_ids: List[int],
    chunk_size: int,
) -> List[float]:
    """
    精确计算每个 response token 生成前的完整词表 next-token entropy。

    对 response token t，返回：
        H_t = -sum_v p(v | prompt, response_<t) log p(v | prompt, response_<t)

    为避免一次产生 [sequence_length, vocab_size] 的巨大 logits，本函数使用 KV cache
    按 chunk_size 分块前向，并在每个 chunk 计算完熵后立即释放 logits。
    """
    import torch

    if chunk_size < 1:
        raise ValueError("entropy_chunk_size must be at least 1")
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids is empty; cannot align response entropies")
    if len(response_token_ids) == 0:
        return []

    # 为 response 中每个 token 计算“生成前”的分布：
    # 输入只需要 prompt + response[:-1]。
    scoring_input_ids = prompt_token_ids + response_token_ids[:-1]
    target_global_start = len(prompt_token_ids) - 1
    target_global_end = target_global_start + len(response_token_ids)  # exclusive

    input_device = get_model_input_device(model)

    # Token ids are tiny compared with the model activations, so keep the whole
    # scoring sequence on the input device.  Creating a new CUDA tensor from a
    # Python list for every small chunk causes repeated host-to-device copies
    # and allocator overhead.
    scoring_ids = torch.tensor(
        scoring_input_ids,
        dtype=torch.long,
        device=input_device,
    )

    past_key_values = None
    # Keep only the one-dimensional entropy outputs on device while the next
    # chunks are being processed.  Moving every chunk to CPU with
    # .cpu().tolist() introduces a CUDA synchronization point on every loop.
    entropy_chunks: List[torch.Tensor] = []

    with torch.inference_mode():
        for chunk_start in range(0, len(scoring_input_ids), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(scoring_input_ids))
            input_ids = scoring_ids[chunk_start:chunk_end].unsqueeze(0)

            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            # outputs.logits[:, local_pos] 对应 scoring_input_ids 的全局位置
            # chunk_start + local_pos，并预测其下一个 token。
            logits = outputs.logits
            if hasattr(logits, "full_tensor"):
                logits = logits.full_tensor()
            logits = logits[0]
            overlap_start = max(chunk_start, target_global_start)
            overlap_end = min(chunk_end, target_global_end)

            if overlap_start < overlap_end:
                local_start = overlap_start - chunk_start
                local_end = overlap_end - chunk_start

                selected_logits = logits[local_start:local_end].float()
                log_probs = torch.log_softmax(selected_logits, dim=-1)
                entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
                entropy_chunks.append(entropies.detach())

                del selected_logits, log_probs, entropies

            del input_ids, outputs, logits

    # There is one unavoidable host transfer per candidate instead of one per
    # chunk.  This preserves the exact entropy values and their ordering.
    if entropy_chunks:
        entropy_values = torch.cat(entropy_chunks, dim=0).cpu().tolist()
    else:
        entropy_values = []

    if len(entropy_values) != len(response_token_ids):
        raise RuntimeError(
            "Entropy/token alignment failed: "
            f"got {len(entropy_values)} entropies for "
            f"{len(response_token_ids)} response tokens."
        )

    del past_key_values
    return entropy_values

def select_entropy_boundary_token_positions(
    response_entropies: List[float],
    entropy_threshold: float,
    num_segments: int,
) -> Tuple[List[int], int]:
    """
    参考 math_episode_generator_with_mc_advantages_entropy.py 的逻辑，
    从完整词表 token entropy 中选择 phase 内部边界。

    response_entropies[t] 表示生成 response token t 之前的分布熵。
    返回的 boundary_token_positions 是 token t 的位置，即新 phase 从 token t 开始。

    规则：
    1. 内部边界只能位于 token 1..T-1 之前；token 0 之前是固定初始边界；
    2. H_t > entropy_threshold 的位置作为候选；
    3. 候选足够时，按候选序列排名均匀下采样为 K-1 个边界；
    4. 候选不足时，保留全部阈值候选，并以剩余位置中熵最高者补齐；
    5. 实际边界数受 response 长度限制。
    """
    entropies = np.asarray(response_entropies, dtype=np.float64)
    if entropies.ndim != 1:
        raise ValueError(
            f"response_entropies must be 1-D, got shape {entropies.shape}"
        )
    if entropy_threshold < 0:
        raise ValueError("entropy_threshold must be non-negative")
    if num_segments < 1:
        raise ValueError("num_segments/K must be at least 1")

    num_response_tokens = len(entropies)
    if num_response_tokens <= 1 or num_segments == 1:
        return [], 0

    valid_positions = np.arange(1, num_response_tokens, dtype=np.int64)
    finite_mask = np.isfinite(entropies[valid_positions])
    valid_positions = valid_positions[finite_mask]
    if len(valid_positions) == 0:
        return [], 0

    threshold_candidates = valid_positions[
        entropies[valid_positions] > entropy_threshold
    ]
    num_threshold_candidates = int(len(threshold_candidates))

    target_internal_count = min(num_segments - 1, len(valid_positions))
    if target_internal_count <= 0:
        return [], num_threshold_candidates

    if len(threshold_candidates) >= target_internal_count:
        # 按候选点在序列中的排名做分位下采样，避免所有边界聚集于局部熵峰。
        candidate_ranks = np.ceil(
            np.arange(1, target_internal_count + 1, dtype=np.float64)
            * len(threshold_candidates)
            / (target_internal_count + 1)
        ).astype(np.int64) - 1
        candidate_ranks = np.clip(
            candidate_ranks,
            0,
            len(threshold_candidates) - 1,
        )
        selected_positions = threshold_candidates[candidate_ranks]
    else:
        # 候选不足时，先保留所有真正 H_t > tau 的位置，再按熵从高到低补齐。
        selected_positions_list = threshold_candidates.tolist()
        remaining_positions = np.setdiff1d(
            valid_positions,
            threshold_candidates,
            assume_unique=True,
        )
        remaining_positions = remaining_positions[
            np.argsort(entropies[remaining_positions], kind="stable")[::-1]
        ]
        missing = target_internal_count - len(selected_positions_list)
        selected_positions_list.extend(remaining_positions[:missing].tolist())
        selected_positions = np.asarray(selected_positions_list, dtype=np.int64)

    selected_positions = np.unique(selected_positions)

    # 防御式补齐，避免未来逻辑变化导致重复位置后不足 K-1 个边界。
    if len(selected_positions) < target_internal_count:
        remaining_positions = np.setdiff1d(
            valid_positions,
            selected_positions,
            assume_unique=True,
        )
        remaining_positions = remaining_positions[
            np.argsort(entropies[remaining_positions], kind="stable")[::-1]
        ]
        missing = target_internal_count - len(selected_positions)
        selected_positions = np.concatenate(
            [selected_positions, remaining_positions[:missing]]
        )

    selected_positions = np.sort(selected_positions.astype(np.int64))
    return selected_positions.tolist(), num_threshold_candidates

def get_phase_partition_limits(
    text: str,
    token_pieces: List[Dict[str, Any]],
    sep: str = "\n\n",
) -> Tuple[int, int]:
    """
    确定参与 phase 划分的 step 数和 response token 数。

    若首次出现包含 </think> 的 step，则该 step 及之前的内容参与划分；
    其后的 step 不参与划分。若没有任何 step 包含 </think>，则保持原逻辑，
    使用完整输出参与划分。

    返回：
        phase_num_steps:
            参与 phase 划分的 step 数。
        phase_num_tokens:
            参与 entropy 边界选择的 response token 前缀长度。
    """
    steps = split_text_into_steps(text, sep=sep)
    if len(steps) == 0:
        return 0, 0

    phase_num_steps = len(steps)
    found_think_end = False

    for step_idx, step in enumerate(steps):
        if "</think>" in step["step"]:
            phase_num_steps = step_idx + 1
            found_think_end = True
            break

    # 未出现 </think> 时必须完全保持原来的全输出划分逻辑，包括尾部 token。
    if not found_think_end:
        return phase_num_steps, len(token_pieces)

    phase_char_end = int(steps[phase_num_steps - 1]["char_end"])
    phase_num_tokens = 0

    # 保留所有起始字符位置位于截止 step 结束位置之前的 token。
    # 若某个 token 跨过该字符边界，也会因其 char_start 位于边界之前而被包含。
    for piece in token_pieces:
        if int(piece["char_start"]) < phase_char_end:
            phase_num_tokens = max(
                phase_num_tokens,
                int(piece["token_index"]) + 1,
            )

    return phase_num_steps, phase_num_tokens

def locate_boundary_step_index(
    boundary_token_position: int,
    token_pieces: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
) -> Optional[int]:
    """
    将“新 phase 开始前的 token 位置 t”映射到 0-based step 序号。

    优先选择与该 token 字符区间重叠最多的 step；若 token 正好位于 "\n\n"
    分隔符或被 strip 的空白区域，则选择其后最近的 step，使该 step 成为新 phase 起点。
    """
    if boundary_token_position < 0 or boundary_token_position >= len(token_pieces):
        return None
    if len(steps) == 0:
        return None

    piece = token_pieces[boundary_token_position]
    token_start = int(piece["char_start"])
    token_end = int(piece["char_end"])

    best_step_idx: Optional[int] = None
    best_overlap = 0

    for idx, step in enumerate(steps):
        overlap = token_step_overlap(
            token_start,
            token_end,
            int(step["char_start"]),
            int(step["char_end"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_step_idx = idx

    if best_step_idx is not None and best_overlap > 0:
        return best_step_idx

    # 无字符重叠时，边界位于 step 间隔中：将其归到后一个 step。
    boundary_char = token_start
    for idx, step in enumerate(steps):
        if int(step["char_start"]) >= boundary_char:
            return idx

    return None

def build_phase_ranges_and_frontiers(
    text: str,
    token_pieces: List[Dict[str, Any]],
    boundary_token_positions: List[int],
    sep: str = "\n\n",
) -> Tuple[List[List[int]], List[int]]:
    """
    将 token 级 entropy 边界映射为 step 级 phase_ranges 和 phase_frontier。

    若某个 step 包含 </think>，则 phase 只覆盖该 step 及之前的 step；
    之后的 step 保留在完整 steps 输出中，但不属于任何 phase。
    phase_ranges 中 step 序号从 1 开始。
    """
    steps = split_text_into_steps(text, sep=sep)
    num_steps = len(steps)

    if num_steps == 0:
        return [], []

    phase_num_steps, _ = get_phase_partition_limits(
        text=text,
        token_pieces=token_pieces,
        sep=sep,
    )
    phase_steps = steps[:phase_num_steps]

    # 第一个 step 固定为第一个 phase 的起点。
    phase_start_steps = {1}

    for boundary_token_position in boundary_token_positions:
        step_idx = locate_boundary_step_index(
            boundary_token_position=boundary_token_position,
            token_pieces=token_pieces,
            steps=phase_steps,
        )
        if step_idx is None:
            continue

        start_step_1based = step_idx + 1

        # step 1 已经是固定起点；边界只能落在参与划分的 step 范围内。
        if 1 < start_step_1based <= phase_num_steps:
            phase_start_steps.add(start_step_1based)

    sorted_starts = sorted(phase_start_steps)

    phase_ranges: List[List[int]] = []
    for idx, start_step in enumerate(sorted_starts):
        if idx + 1 < len(sorted_starts):
            end_step = sorted_starts[idx + 1] - 1
        else:
            end_step = phase_num_steps

        if start_step <= end_step:
            phase_ranges.append([int(start_step), int(end_step)])

    # 所有 step 仍保留 phase_frontier 字段；未参与划分的后续 step 保持非边界值 1。
    phase_frontiers = [1] * num_steps

    # 先标结束，再标开始：若某个中间 phase 只有一个 step，起点标记优先为 0。
    # 最后再强制首 step=0、参与划分的末 step=2。单 step 前缀无法用单一标量同时表达
    # “起点+终点”，此时保留首 step 的 0；phase_ranges 仍准确表达 [1, 1]。
    for start_step, end_step in phase_ranges:
        phase_frontiers[end_step - 1] = 2
    for start_step, end_step in phase_ranges:
        phase_frontiers[start_step - 1] = 0

    phase_frontiers[0] = 0
    if phase_num_steps > 1:
        phase_frontiers[phase_num_steps - 1] = 2

    return phase_ranges, phase_frontiers

def load_tokenizer_without_vllm(args: argparse.Namespace):
    """当生成阶段已完成、无需重新启动 vLLM 时，单独加载同一 tokenizer。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )

def parse_gpu_groups(
    gpu_groups: Optional[str],
    tensor_parallel_size: int,
) -> List[str]:
    """解析 DP GPU 组；每个分号分隔的组内部用于 TP。"""
    if gpu_groups is None or not gpu_groups.strip():
        return []

    groups: List[str] = []
    used_gpus: set[str] = set()

    for raw_group in gpu_groups.split(";"):
        gpu_ids = [gpu_id.strip() for gpu_id in raw_group.split(",") if gpu_id.strip()]
        if not gpu_ids:
            raise ValueError("--gpus contains an empty GPU group")
        if len(gpu_ids) != tensor_parallel_size:
            raise ValueError(
                f"GPU group '{raw_group}' contains {len(gpu_ids)} GPUs, but "
                f"--tensor_parallel_size={tensor_parallel_size}."
            )

        duplicated = used_gpus.intersection(gpu_ids)
        if duplicated:
            raise ValueError(
                "The same GPU cannot be assigned to multiple DP workers: "
                f"{sorted(duplicated)}"
            )

        used_gpus.update(gpu_ids)
        groups.append(",".join(gpu_ids))

    return groups


def effective_vllm_cache_mode(
    args: argparse.Namespace,
    gpu_groups: Sequence[str],
) -> str:
    """Resolve automatic cache isolation in the same way as transformer_bon."""
    if args.vllm_cache_mode == "auto":
        return "isolated" if len(gpu_groups) > 1 else "shared"
    return str(args.vllm_cache_mode)


def generation_cache_environment(
    args: argparse.Namespace,
    worker_index: int,
    gpu_groups: Sequence[str],
) -> Tuple[Dict[str, str], str]:
    """Return a stable compilation-cache namespace for one vLLM DP worker."""
    configured_root = args.vllm_cache_root or os.environ.get("VLLM_CACHE_ROOT")
    base_root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path.home() / ".cache" / "vllm"
    )

    if effective_vllm_cache_mode(args, gpu_groups) == "isolated":
        fingerprint_payload = {
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "trust_remote_code": args.trust_remote_code,
            "enforce_eager": args.vllm_enforce_eager,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        cache_root = (
            base_root
            / "transformer_bon"
            / fingerprint
            / f"dp_{worker_index}"
        )
        environment = {
            "VLLM_CACHE_ROOT": str(cache_root),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
        }
    else:
        cache_root = base_root
        environment = {"VLLM_CACHE_ROOT": str(cache_root)}

    return environment, str(cache_root)

def find_free_tcp_port() -> int:
    """为一个独立的 TP 进程组申请本机空闲端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default

def safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default

def estimate_text_token_len(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 1
    return max(1, len(re.findall(r"\S+", text)))

def linear_slope(values: List[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) < 2:
        return 0.0

    if len(array) == 2:
        return float(array[-1] - array[0])

    x = np.linspace(0.0, 1.0, len(array))
    x_mean = x.mean()
    y_mean = array.mean()
    denominator = np.sum((x - x_mean) ** 2)

    if denominator <= 0:
        return 0.0

    return float(np.sum((x - x_mean) * (array - y_mean)) / denominator)

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_arg!r}, but CUDA is not available."
        )
    return device

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}.")
    return data

def compute_step_start_end_positions(
    steps: List[Dict[str, Any]],
) -> Tuple[List[int], List[int]]:
    starts: List[int] = []

    for step in steps:
        start = safe_int(step.get("first_token_index"), default=-1)
        if start < 0:
            return [], []
        starts.append(start)

    ends: List[int] = []
    for index, step in enumerate(steps):
        if index < len(steps) - 1:
            end_index = starts[index + 1]
        else:
            end_index = starts[index] + estimate_text_token_len(step.get("step", ""))

        if end_index <= starts[index]:
            end_index = starts[index] + 1

        ends.append(int(end_index))

    return starts, ends

def compute_local_std_and_slope(
    confidences: List[float],
    step_idx: int,
    transformer_type: str = "bidirectional",
) -> Tuple[float, float]:
    num_steps = len(confidences)
    local_indices: List[int] = []

    if transformer_type == "causal":
        window_indices = [step_idx - 2, step_idx - 1, step_idx]
    else:
        window_indices = [step_idx - 1, step_idx, step_idx + 1]

    for index in window_indices:
        if 0 <= index < num_steps:
            local_indices.append(index)

    values = [float(confidences[index]) for index in local_indices]
    if not values:
        return 0.0, 0.0

    local_std = float(np.std(np.asarray(values, dtype=float), ddof=0))
    local_slope = linear_slope(values)
    return local_std, local_slope

class FeatureNormalizer:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureNormalizer":
        normalizer = cls()

        mean = data.get("mean")
        std = data.get("std")
        if mean is None or std is None:
            raise ValueError("Normalization metadata must contain non-null mean and std.")

        normalizer.mean = np.asarray(mean, dtype=np.float32)
        normalizer.std = np.asarray(std, dtype=np.float32)

        if normalizer.mean.shape != (4,) or normalizer.std.shape != (4,):
            raise ValueError(
                "Expected four normalization values for the 4D feature input, "
                f"got mean shape {normalizer.mean.shape} and std shape {normalizer.std.shape}."
            )

        normalizer.std = np.where(normalizer.std < 1e-8, 1.0, normalizer.std)
        return normalizer

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer has not been initialized.")
        return ((x - self.mean) / self.std).astype(np.float32)

    def transform_sequences(
        self,
        sequences: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        transformed: List[Dict[str, Any]] = []
        for sequence in sequences:
            new_sequence = copy.deepcopy(sequence)
            new_sequence["x"] = self.transform_one(sequence["x"])
            transformed.append(new_sequence)
        return transformed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "feature_names": FEATURE_NAMES.copy(),
        }

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
        sequence_length = x.size(1)
        if sequence_length > self.pe.size(1):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds model max_len "
                f"{self.pe.size(1)}. Use the same or a larger --max_len."
            )
        return x + self.pe[:, :sequence_length, :]

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
        hidden = self.input_proj(x)
        hidden = self.pos_encoder(hidden)

        key_padding_mask = ~attention_mask

        if self.transformer_type == "causal":
            sequence_length = hidden.size(1)
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    device=hidden.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        else:
            causal_mask = None

        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        logits = self.output_head(hidden).squeeze(-1)
        return torch.sigmoid(logits)

def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def infer_training_metrics_path(checkpoint_path: Path) -> Path:
    candidates = [
        checkpoint_path.parent / "metrics.json",
        checkpoint_path.parent.parent / "metrics.json",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not automatically locate the training metrics.json. Checked:\n"
        + "\n".join(f"  - {candidate}" for candidate in candidates)
        + "\nPass the file explicitly with --training_metrics_json."
    )

def infer_fold_index_from_checkpoint(checkpoint_path: Path) -> Optional[int]:
    match = re.fullmatch(r"fold_(\d+)", checkpoint_path.parent.name)
    return int(match.group(1)) if match else None

def select_normalization_metadata(
    source_metrics: Dict[str, Any],
    checkpoint_path: Path,
    explicit_fold_index: Optional[int],
) -> Tuple[Dict[str, Any], Optional[int], str]:
    fold_index = explicit_fold_index
    if fold_index is None:
        fold_index = infer_fold_index_from_checkpoint(checkpoint_path)

    if fold_index is None:
        normalization = source_metrics.get("normalization")
        if not isinstance(normalization, dict):
            raise ValueError(
                "Root normalization metadata is missing from training metrics.json."
            )
        return normalization, None, "root/final checkpoint normalization"

    cross_validation = source_metrics.get("cross_validation")
    if not isinstance(cross_validation, dict):
        raise ValueError(
            f"Checkpoint appears to be fold_{fold_index}, but metrics.json has no "
            "cross_validation section."
        )

    folds = cross_validation.get("folds")
    if not isinstance(folds, list):
        raise ValueError("cross_validation.folds is missing or is not a list.")

    for fold in folds:
        if isinstance(fold, dict) and safe_int(fold.get("fold")) == fold_index:
            normalization = fold.get("normalization")
            if not isinstance(normalization, dict):
                raise ValueError(
                    f"Fold {fold_index} normalization metadata is missing."
                )
            return normalization, fold_index, f"fold_{fold_index} normalization"

    raise ValueError(
        f"Could not find normalization metadata for fold {fold_index}."
    )

def resolve_transformer_type(
    requested_type: str,
    source_metrics: Dict[str, Any],
) -> str:
    if requested_type != "auto":
        return requested_type

    candidates = [
        source_metrics.get("model", {}).get("transformer_type"),
        source_metrics.get("training", {}).get("transformer_type"),
        source_metrics.get("transformer_type"),
    ]
    for candidate in candidates:
        if candidate in {"bidirectional", "causal"}:
            return str(candidate)

    raise ValueError(
        "The training script's metrics.json does not record transformer_type, so "
        "it cannot be inferred from the checkpoint weights. Pass exactly the same "
        "setting used during training: --transformer_type causal or "
        "--transformer_type bidirectional."
    )

def load_checkpoint_state_dict(
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint must contain a state_dict-like mapping, but got "
            f"{type(state_dict).__name__}."
        )

    cleaned_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict

def build_model_from_metadata(
    source_metrics: Dict[str, Any],
    transformer_type: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model_metadata = source_metrics.get("model", {})
    if not isinstance(model_metadata, dict):
        raise ValueError("model metadata is missing from metrics.json.")

    config = {
        "input_dim": safe_int(model_metadata.get("input_dim"), default=4),
        "d_model": safe_int(model_metadata.get("d_model"), default=32),
        "nhead": safe_int(model_metadata.get("nhead"), default=4),
        "num_layers": safe_int(model_metadata.get("num_layers"), default=2),
        "dim_feedforward": safe_int(
            model_metadata.get("dim_feedforward"),
            default=128,
        ),
        "dropout": safe_float(model_metadata.get("dropout"), default=0.20),
        "max_len": safe_int(model_metadata.get("max_len"), default=2048),
        "transformer_type": transformer_type,
    }

    if config["input_dim"] != 4:
        raise ValueError(
            f"This inference script expects the original 4D input, but metadata "
            f"reports input_dim={config['input_dim']}."
        )

    model = SmallTransformerEncoderValueModel(**config).to(device)
    return model, config


# ============================================================
# Composite candidate-key JSONL utilities
# ============================================================

CandidateKey = Tuple[int, int]


def candidate_key(record: Dict[str, Any]) -> CandidateKey:
    if "sample_idx" not in record or "sequence_idx" not in record:
        raise ValueError("Record must contain sample_idx and sequence_idx")
    return int(record["sample_idx"]), int(record["sequence_idx"])


def expected_candidate_keys(num_samples: int, n_candidates: int) -> List[CandidateKey]:
    return [
        (sample_idx, sequence_idx)
        for sample_idx in range(num_samples)
        for sequence_idx in range(n_candidates)
    ]


def read_jsonl_by_candidate(
    path: str,
    *,
    tolerate_incomplete_last_line: bool = False,
) -> Dict[CandidateKey, Dict[str, Any]]:
    records: Dict[CandidateKey, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as file:
        for zero_based_line_no, raw_line in enumerate(file):
            line_no = zero_based_line_no + 1
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                is_last = not file.read()
                if tolerate_incomplete_last_line and is_last:
                    print(f"[resume] Ignore incomplete last line at {path}:{line_no}: {error}")
                    break
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {error}") from error
            records[candidate_key(record)] = record
    return records


def atomic_rewrite_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".rewrite_tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def record_matches_input(
    record: Dict[str, Any],
    data: List[Dict[str, Any]],
    n_candidates: int,
) -> bool:
    sample_idx, sequence_idx = candidate_key(record)
    if sample_idx < 0 or sample_idx >= len(data):
        return False
    if sequence_idx < 0 or sequence_idx >= n_candidates:
        return False
    sample = data[sample_idx]
    if record.get("problem") != sample.get("problem"):
        return False
    expected_unique_id = sample.get("unique_id", str(sample_idx))
    if "unique_id" in record and record.get("unique_id") != expected_unique_id:
        return False
    return True


def compact_candidate_file(
    path: str,
    data: List[Dict[str, Any]],
    n_candidates: int,
) -> Dict[CandidateKey, Dict[str, Any]]:
    records = read_jsonl_by_candidate(path, tolerate_incomplete_last_line=True)
    valid: Dict[CandidateKey, Dict[str, Any]] = {}
    for key, record in records.items():
        if not record_matches_input(record, data, n_candidates):
            raise ValueError(
                f"Resume record {key} in {path} does not match current input/N. "
                "Use --overwrite or the original arguments."
            )
        valid[key] = record
    if os.path.exists(path):
        atomic_rewrite_jsonl(path, [valid[key] for key in sorted(valid)])
    return valid


def synchronize_final_outputs(
    summary_path: str,
    steps_path: str,
    data: List[Dict[str, Any]],
    n_candidates: int,
    max_tokens: Optional[int] = None,
    generation_records: Optional[Dict] = None,
) -> Set[CandidateKey]:
    summary = read_jsonl_by_candidate(summary_path, tolerate_incomplete_last_line=True)
    steps = read_jsonl_by_candidate(steps_path, tolerate_incomplete_last_line=True)
    completed = set(summary).intersection(steps)
    valid: List[CandidateKey] = []
    for key in sorted(completed):
        if not record_matches_input(summary[key], data, n_candidates):
            raise ValueError(f"Summary record {key} does not match current input/N")
        if not record_matches_input(steps[key], data, n_candidates):
            raise ValueError(f"Steps record {key} does not match current input/N")
        row = summary[key]
        if max_tokens is not None and row.get("num_generated_tokens", 0) > max_tokens:
            continue
        generation = generation_records.get(key) if generation_records is not None else None
        if generation is not None and ("_bon_source_structure" in generation or generation.get("_bon_prefix_view")) and (
                row.get("num_generated_tokens") != len(generation.get("response_token_ids", []))
                or row.get("full_output") != generation.get("full_output")):
            continue
        valid.append(key)
    if os.path.exists(summary_path) or os.path.exists(steps_path):
        atomic_rewrite_jsonl(summary_path, [summary[key] for key in valid])
        atomic_rewrite_jsonl(steps_path, [steps[key] for key in valid])
    return set(valid)


# ============================================================
# Generation: N candidates, batched per problem, DP across groups
# ============================================================


def make_sampling_params(
    args: argparse.Namespace,
    *,
    n: int,
    seed: Optional[int],
) -> Any:
    _, sampling_params_cls = load_vllm_api()
    kwargs: Dict[str, Any] = {
        "n": int(n),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "logprobs": args.logprobs,
        "skip_special_tokens": True,
    }
    if seed is not None:
        kwargs["seed"] = int(seed)
    try:
        return sampling_params_cls(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return sampling_params_cls(**kwargs)


def create_generation_llm(args: argparse.Namespace) -> Any:
    """Create the vLLM engine after worker-specific environment setup."""
    llm_cls, _ = load_vllm_api()
    kwargs: Dict[str, Any] = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
    }
    if args.vllm_enforce_eager:
        kwargs["enforce_eager"] = True
    return llm_cls(**kwargs)


def generation_seed(
    base_seed: Optional[int],
    sample_idx: int,
    sequence_indices: Sequence[int],
    n_candidates: int,
) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed + sample_idx * n_candidates + min(sequence_indices))


def build_generation_record(
    request_output: Any,
    output: Any,
    tokenizer: Any,
    prompt: str,
    sample_idx: int,
    sequence_idx: int,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    token_ids = [int(token_id) for token_id in output.token_ids]
    full_output, _ = build_token_pieces_with_char_spans(token_ids, tokenizer)

    output_logprobs = output.logprobs
    if output_logprobs is None:
        raise RuntimeError(
            "output.logprobs is None; use SamplingParams(logprobs>=1)."
        )
    if len(output_logprobs) != len(token_ids):
        raise RuntimeError(
            f"logprob/token mismatch: {len(output_logprobs)} vs {len(token_ids)}"
        )

    token_logprobs = [
        get_sampled_token_logprob(logprob_dict, token_id)
        for token_id, logprob_dict in zip(token_ids, output_logprobs)
    ]

    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    if prompt_token_ids is None:
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    return {
        "status": "ok",
        "sample_idx": int(sample_idx),
        "sequence_idx": int(sequence_idx),
        "problem": sample["problem"],
        "gold_answer": sample.get("answer"),
        "difficulty": sample.get("difficulty"),
        "unique_id": sample.get("unique_id", str(sample_idx)),
        "full_output": full_output,
        "prompt_token_ids": [int(value) for value in prompt_token_ids],
        "response_token_ids": token_ids,
        "token_logprobs": token_logprobs,
        "finish_reason": getattr(output, "finish_reason", None),
    }


def build_generation_error_record(
    sample_idx: int,
    sequence_idx: int,
    sample: Dict[str, Any],
    error: str,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "sample_idx": int(sample_idx),
        "sequence_idx": int(sequence_idx),
        "problem": sample["problem"],
        "gold_answer": sample.get("answer"),
        "difficulty": sample.get("difficulty"),
        "unique_id": sample.get("unique_id", str(sample_idx)),
        "error": error,
    }


def generate_missing_for_sample(
    llm: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    sample_idx: int,
    sample: Dict[str, Any],
    pending_sequence_indices: Sequence[int],
) -> Iterable[Dict[str, Any]]:
    prompt = build_qwen3_prompt(
        problem=sample["problem"],
        tokenizer=tokenizer,
        enable_thinking=args.enable_thinking,
    )

    pending = list(sorted(int(index) for index in pending_sequence_indices))
    for start in range(0, len(pending), args.batch_size):
        sequence_batch = pending[start:start + args.batch_size]
        params = make_sampling_params(
            args,
            n=len(sequence_batch),
            seed=generation_seed(args.seed, sample_idx, sequence_batch, args.N),
        )
        try:
            request_outputs = llm.generate([prompt], params)
            if len(request_outputs) != 1:
                raise RuntimeError(
                    f"Expected one request output, got {len(request_outputs)}"
                )
            request_output = request_outputs[0]
            outputs = list(request_output.outputs)
            if len(outputs) != len(sequence_batch):
                raise RuntimeError(
                    f"Expected {len(sequence_batch)} candidates, got {len(outputs)}"
                )
            for sequence_idx, output in zip(sequence_batch, outputs):
                try:
                    yield build_generation_record(
                        request_output=request_output,
                        output=output,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        sample_idx=sample_idx,
                        sequence_idx=sequence_idx,
                        sample=sample,
                    )
                except Exception:
                    yield build_generation_error_record(
                        sample_idx,
                        sequence_idx,
                        sample,
                        traceback.format_exc(),
                    )
        except Exception:
            error = traceback.format_exc()
            for sequence_idx in sequence_batch:
                yield build_generation_error_record(
                    sample_idx,
                    sequence_idx,
                    sample,
                    error,
                )


def generation_dp_worker(
    dp_rank: int,
    gpu_group: str,
    args: argparse.Namespace,
    assigned_tasks: List[Tuple[int, Dict[str, Any], List[int]]],
    worker_environment: Dict[str, str],
    cache_root: str,
    result_queue: Any,
) -> None:
    llm = None
    try:
        # vLLM reads compilation-cache settings during import. Install the
        # worker-specific namespace before the deliberately lazy import.
        os.environ.update(worker_environment)
        llm = create_generation_llm(args)
        tokenizer = llm.get_tokenizer()
        result_queue.put(
            (
                "worker_ready",
                dp_rank,
                {"gpu_group": gpu_group, "cache_root": cache_root},
            )
        )
        for sample_idx, sample, sequence_indices in assigned_tasks:
            for record in generate_missing_for_sample(
                llm, tokenizer, args, sample_idx, sample, sequence_indices
            ):
                result_queue.put(("record", dp_rank, record))
    except BaseException:
        result_queue.put(("worker_error", dp_rank, traceback.format_exc()))
    finally:
        if llm is not None:
            release_vllm(llm)
        result_queue.put(("worker_done", dp_rank, gpu_group))


def run_data_parallel_generation(
    args: argparse.Namespace,
    tasks: List[Tuple[int, Dict[str, Any], List[int]]],
    gpu_groups: List[str],
    temp_path: str,
    records_by_key: Dict[CandidateKey, Dict[str, Any]],
) -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=max(8, len(gpu_groups) * args.batch_size * 2))
    shards: List[List[Tuple[int, Dict[str, Any], List[int]]]] = [
        [] for _ in gpu_groups
    ]
    for position, task in enumerate(tasks):
        shards[position % len(gpu_groups)].append(task)

    processes = []
    worker_cache_roots: Dict[int, str] = {}
    old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for dp_rank, (gpu_group, shard) in enumerate(zip(gpu_groups, shards)):
            worker_environment, cache_root = generation_cache_environment(
                args, dp_rank, gpu_groups
            )
            for path_string in worker_environment.values():
                Path(path_string).mkdir(parents=True, exist_ok=True)
            worker_cache_roots[dp_rank] = cache_root
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
            process = ctx.Process(
                target=generation_dp_worker,
                args=(
                    dp_rank,
                    gpu_group,
                    args,
                    shard,
                    worker_environment,
                    cache_root,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)
    finally:
        if old_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda

    def terminate_workers() -> None:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10.0)

    # Do not enter the generation loop until every vLLM engine has completed
    # model loading, compilation, warm-up, and KV-cache initialization.
    ready: Set[int] = set()
    startup_errors: List[str] = []
    startup_deadline = time.monotonic() + args.generation_worker_startup_timeout
    while len(ready) < len(processes) and not startup_errors:
        remaining = startup_deadline - time.monotonic()
        if remaining <= 0:
            waiting = sorted(set(range(len(processes))).difference(ready))
            startup_errors.append(
                "Generation worker startup timed out after "
                f"{args.generation_worker_startup_timeout:.1f}s; waiting for "
                f"workers {waiting}. Cache roots: {worker_cache_roots}"
            )
            break
        try:
            message_type, dp_rank, payload = result_queue.get(
                timeout=min(1.0, remaining)
            )
        except queue.Empty:
            dead = [
                index
                for index, process in enumerate(processes)
                if index not in ready and not process.is_alive()
            ]
            if dead:
                startup_errors.append(
                    "Generation workers exited before becoming ready: "
                    f"{dead}. Cache roots: {worker_cache_roots}"
                )
            continue

        if message_type == "worker_ready":
            ready.add(int(dp_rank))
            print(
                "[generation] worker "
                f"{dp_rank} ready on GPU group {payload['gpu_group']} "
                f"(cache={payload['cache_root']})",
                flush=True,
            )
        elif message_type == "worker_error":
            startup_errors.append(
                f"Generation worker {dp_rank} failed during startup "
                f"(cache={worker_cache_roots.get(int(dp_rank))}):\n{payload}"
            )
        elif message_type == "worker_done":
            startup_errors.append(
                f"Generation worker {dp_rank} exited before becoming ready"
            )
        else:
            startup_errors.append(
                f"Unexpected generation message during startup: {message_type}"
            )

    if startup_errors:
        terminate_workers()
        raise RuntimeError(
            "DP generation failed during vLLM startup.\n"
            + "\n".join(startup_errors)
        )

    total = sum(len(task[2]) for task in tasks)
    finished: Set[int] = set()
    errors: List[str] = []
    with tqdm(total=total, desc="Generating candidates") as progress:
        while len(finished) < len(processes):
            try:
                message_type, dp_rank, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue
            if message_type == "record":
                append_jsonl(temp_path, payload)
                records_by_key[candidate_key(payload)] = payload
                progress.update(1)
            elif message_type == "worker_error":
                errors.append(
                    f"DP worker {dp_rank} failed "
                    f"(cache={worker_cache_roots.get(int(dp_rank))}):\n{payload}"
                )
                break
            elif message_type == "worker_done":
                finished.add(int(dp_rank))

    if errors:
        terminate_workers()
    else:
        for process in processes:
            process.join()
    abnormal = [index for index, process in enumerate(processes) if process.exitcode != 0]
    if abnormal:
        errors.append(f"Generation workers exited abnormally: {abnormal}")
    if errors:
        raise RuntimeError(
            "DP generation failed; completed candidates remain resumable.\n"
            + "\n".join(errors)
        )


def run_single_process_generation(
    args: argparse.Namespace,
    tasks: List[Tuple[int, Dict[str, Any], List[int]]],
    temp_path: str,
    records_by_key: Dict[CandidateKey, Dict[str, Any]],
) -> None:
    llm = create_generation_llm(args)
    try:
        tokenizer = llm.get_tokenizer()
        total = sum(len(task[2]) for task in tasks)
        with tqdm(total=total, desc="Generating candidates") as progress:
            for sample_idx, sample, sequence_indices in tasks:
                for record in generate_missing_for_sample(
                    llm, tokenizer, args, sample_idx, sample, sequence_indices
                ):
                    append_jsonl(temp_path, record)
                    records_by_key[candidate_key(record)] = record
                    progress.update(1)
    finally:
        release_vllm(llm)


# ============================================================
# Exact entropy and phase preparation, DP across groups + TP inside group
# ============================================================


def build_prepared_error(record: Dict[str, Any], error: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "sample_idx": record["sample_idx"],
        "sequence_idx": record["sequence_idx"],
        "problem": record["problem"],
        "gold_answer": record.get("gold_answer"),
        "difficulty": record.get("difficulty"),
        "unique_id": record.get("unique_id"),
        "full_output": record.get("full_output"),
        "finish_reason": record.get("finish_reason"),
        "error": error,
        "step_avg_confidences": [],
        "num_steps": 0,
        "num_generated_tokens": 0,
        "phase_ranges": [],
        "steps": [],
    }


def build_prepared_record(
    generation_record: Dict[str, Any],
    tokenizer: Any,
    response_entropies: List[float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    token_ids = [int(value) for value in generation_record["response_token_ids"]]
    token_logprobs = generation_record["token_logprobs"]
    full_output, token_pieces = build_token_pieces_with_char_spans(token_ids, tokenizer)

    step_confidences, step_records = compute_step_confidences(
        text=full_output,
        token_pieces=token_pieces,
        token_logprobs=token_logprobs,
        sep="\n\n",
    )
    phase_num_steps, phase_num_tokens = get_phase_partition_limits(
        text=full_output,
        token_pieces=token_pieces,
        sep="\n\n",
    )
    phase_entropies = response_entropies[:phase_num_tokens]
    phase_device = getattr(args, "_phase_device", None)
    if phase_device is not None:
        phase_steps = split_text_into_steps(full_output, sep="\n\n")[:phase_num_steps]
        boundary_positions, num_threshold_candidates = select_phase_boundaries_tensor(
            phase_entropies, token_pieces, phase_steps, args, phase_device,
        )
    elif getattr(args, "phase_partition_algorithm", "original") == "adaptive":
        phase_steps = split_text_into_steps(full_output, sep="\n\n")[:phase_num_steps]
        boundary_positions, num_threshold_candidates = select_adaptive_boundaries(
            response_entropies=phase_entropies,
            token_pieces=token_pieces,
            steps=phase_steps,
            target_phase_tokens=args.target_phase_tokens,
            high_entropy_ratio=args.high_entropy_ratio,
            locate_boundary_step_index=locate_boundary_step_index,
        )
    else:
        boundary_positions, num_threshold_candidates = select_entropy_boundary_token_positions(
            response_entropies=phase_entropies,
            entropy_threshold=args.entropy_threshold,
            num_segments=args.num_segments,
        )
    phase_ranges, phase_frontiers = build_phase_ranges_and_frontiers(
        text=full_output,
        token_pieces=token_pieces,
        boundary_token_positions=boundary_positions,
        sep="\n\n",
    )
    if len(phase_frontiers) != len(step_records):
        raise RuntimeError(
            f"Phase/step mismatch: {len(phase_frontiers)} vs {len(step_records)}"
        )
    for step, frontier in zip(step_records, phase_frontiers):
        step["phase_frontier"] = int(frontier)

    prepared = {
        "status": "ok",
        "sample_idx": generation_record["sample_idx"],
        "sequence_idx": generation_record["sequence_idx"],
        "problem": generation_record["problem"],
        "gold_answer": generation_record.get("gold_answer"),
        "difficulty": generation_record.get("difficulty"),
        "unique_id": generation_record.get("unique_id"),
        "full_output": full_output,
        "step_avg_confidences": step_confidences,
        "num_steps": len(step_records),
        "num_generated_tokens": len(token_ids),
        "finish_reason": generation_record.get("finish_reason"),
        "phase_ranges": phase_ranges,
        "boundary_token_positions": boundary_positions,
        "num_entropy_threshold_candidates": num_threshold_candidates,
        "steps": step_records,
    }
    if not args.no_save_token_entropies:
        prepared["token_entropies"] = [float(value) for value in response_entropies]
    return prepared


def build_prepared_record_or_error(
    generation_record: Dict[str, Any],
    tokenizer: Any,
    response_entropies: List[float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """在 CPU worker 中构造 prepared record，并保持原有错误输出格式。"""
    try:
        return build_prepared_record(
            generation_record,
            tokenizer,
            response_entropies,
            args,
        )
    except Exception:
        return build_prepared_error(generation_record, traceback.format_exc())


def prepare_record_in_background(
    generation_record: Dict[str, Any],
    tokenizer: Any,
    response_entropies: List[float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    在 rank 0 的 CPU 线程中完成 tokenizer/phase 构造。

    entropy 的数值计算仍在 TP 主线程中执行；这里只是把原来会阻塞下一次
    GPU forward 的 CPU 后处理移到后台线程。最终记录内容与同步版本完全一致。
    """
    return build_prepared_record_or_error(
        generation_record,
        tokenizer,
        response_entropies,
        args,
    )


def read_selected_generation_records(
    path: str,
    keys: Sequence[CandidateKey],
) -> Dict[CandidateKey, Dict[str, Any]]:
    selected = set(keys)
    records: Dict[CandidateKey, Dict[str, Any]] = {}
    for entry in iter_jsonl(path):
        rows = [entry["record"]] if "record" in entry else entry.get("records", [entry])
        for record in rows:
            key = candidate_key(record)
            if key in selected:
                if "request_id" in entry:
                    record = dict(record, request_id=str(entry["request_id"]))
                records[key] = record
    missing = [key for key in keys if key not in records]
    if missing:
        raise RuntimeError(f"Missing generation records: {missing[:20]}")
    return records


def entropy_tp_worker(
    dp_rank: int,
    tp_rank: int,
    gpu_group: str,
    args: argparse.Namespace,
    assigned_keys: List[CandidateKey],
    generation_temp_path: str,
    result_queue: Any,
) -> None:
    """Forward only: emit complete token-aligned entropy cache rows, no phases."""
    entropy_model = None
    distributed_initialized = False
    try:
        import torch.distributed as dist

        torch.cuda.set_device(tp_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        distributed_initialized = True
        generation_records = read_selected_generation_records(
            generation_temp_path, assigned_keys
        )
        if any(generation_records[key].get("status") == "ok" for key in assigned_keys):
            entropy_model = (
                load_entropy_model_single_gpu(args)
                if args.tensor_parallel_size == 1
                else load_entropy_model_tensor_parallel(args)
            )
        from importlib import import_module
        budget = import_module("scripts.3_best_of_n_via_phases_scores")
        for key in assigned_keys:
            record = budget.apply_worker_budget(generation_records[key], args)
            entropies = None
            local_error = None
            if record.get("status") != "ok":
                local_error = record.get("error", "Generation failed")
            else:
                try:
                    entropies = compute_response_token_entropies_exact(
                        model=entropy_model,
                        prompt_token_ids=[int(v) for v in record["prompt_token_ids"]],
                        response_token_ids=[int(v) for v in record["response_token_ids"]],
                        chunk_size=args.entropy_chunk_size,
                    )
                except Exception:
                    local_error = traceback.format_exc()
            gathered: List[Optional[str]] = [None] * args.tensor_parallel_size
            dist.all_gather_object(gathered, local_error)
            first_error = next((value for value in gathered if value is not None), None)
            if tp_rank == 0:
                try:
                    cached = build_token_entropy_record(record, entropies, args, first_error)
                except Exception:
                    cached = build_token_entropy_record(record, None, args, traceback.format_exc())
                result_queue.put(("record", dp_rank, cached))
            dist.barrier()
    except BaseException:
        result_queue.put((
            "worker_error", dp_rank,
            f"TP rank {tp_rank} on GPUs {gpu_group} failed:\n{traceback.format_exc()}",
        ))
    finally:
        if entropy_model is not None:
            del entropy_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if distributed_initialized:
            try:
                import torch.distributed as dist
                dist.destroy_process_group()
            except Exception:
                pass
        result_queue.put(("worker_done", dp_rank, (tp_rank, gpu_group)))


def run_data_parallel_entropy(
    args: argparse.Namespace,
    pending_keys: List[CandidateKey],
    gpu_groups: List[str],
    generation_temp_path: str,
    prepared_temp_path: str,
    prepared_records: Dict[CandidateKey, Dict[str, Any]],
) -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(
        maxsize=max(8, len(gpu_groups) * args.tensor_parallel_size * 2)
    )
    shards: List[List[CandidateKey]] = [[] for _ in gpu_groups]
    for position, key in enumerate(pending_keys):
        shards[position % len(gpu_groups)].append(key)

    processes: List[Tuple[int, int, Any]] = []
    env_keys = [
        "CUDA_VISIBLE_DEVICES", "MASTER_ADDR", "MASTER_PORT",
        "RANK", "WORLD_SIZE", "LOCAL_RANK",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        for dp_rank, (gpu_group, shard) in enumerate(zip(gpu_groups, shards)):
            master_port = find_free_tcp_port()
            for tp_rank in range(args.tensor_parallel_size):
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = str(master_port)
                os.environ["RANK"] = str(tp_rank)
                os.environ["WORLD_SIZE"] = str(args.tensor_parallel_size)
                os.environ["LOCAL_RANK"] = str(tp_rank)
                process = ctx.Process(
                    target=entropy_tp_worker,
                    args=(
                        dp_rank, tp_rank, gpu_group, args, shard,
                        generation_temp_path, result_queue,
                    ),
                )
                process.start()
                processes.append((dp_rank, tp_rank, process))
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    finished: Set[Tuple[int, int]] = set()
    errors: List[str] = []
    with tqdm(total=len(pending_keys), desc="Exact token entropy", unit="rollout") as progress:
        while len(finished) < len(processes):
            try:
                message_type, dp_rank, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                if all(not process.is_alive() for _, _, process in processes):
                    break
                continue
            if message_type == "record":
                append_entropy_record(prepared_temp_path, payload)
                prepared_records[candidate_key(payload)] = payload
                progress.update(1)
            elif message_type == "worker_error":
                errors.append(str(payload))
                for _, _, process in processes:
                    if process.is_alive():
                        process.terminate()
                break
            elif message_type == "worker_done":
                tp_rank, _ = payload
                finished.add((int(dp_rank), int(tp_rank)))

    for _, _, process in processes:
        process.join()
    abnormal = [
        (dp_rank, tp_rank)
        for dp_rank, tp_rank, process in processes
        if process.exitcode != 0
    ]
    if abnormal:
        errors.append(f"Entropy workers exited abnormally: {abnormal}")
    missing = [key for key in pending_keys if key not in prepared_records]
    if missing:
        errors.append(f"Entropy stage missing results: {missing[:20]}")
    if errors:
        raise RuntimeError(
            "DP+TP entropy stage failed; completed candidates remain resumable.\n"
            + "\n".join(errors)
        )


def run_single_process_entropy(
    args: argparse.Namespace,
    pending_keys: List[CandidateKey],
    generation_records: Dict[CandidateKey, Dict[str, Any]],
    prepared_temp_path: str,
    prepared_records: Dict[CandidateKey, Dict[str, Any]],
) -> None:
    """Persist each complete rollout's entropy before starting phase partition."""
    need_model = any(
        generation_records[key].get("status") == "ok" for key in pending_keys
    )
    entropy_model = load_entropy_model(args) if need_model else None
    try:
        for key in tqdm(pending_keys, desc="Exact token entropy", unit="rollout"):
            record = generation_records[key]
            if record.get("status") != "ok":
                cached = build_token_entropy_record(
                    record, None, args, record.get("error", "Generation failed")
                )
            else:
                try:
                    entropies = compute_response_token_entropies_exact(
                        model=entropy_model,
                        prompt_token_ids=[int(v) for v in record["prompt_token_ids"]],
                        response_token_ids=[int(v) for v in record["response_token_ids"]],
                        chunk_size=args.entropy_chunk_size,
                    )
                    cached = build_token_entropy_record(record, entropies, args)
                except Exception:
                    cached = build_token_entropy_record(record, None, args, traceback.format_exc())
            append_entropy_record(prepared_temp_path, cached)
            prepared_records[key] = cached
    finally:
        if entropy_model is not None:
            del entropy_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================
# Value-model feature construction and phase scoring
# ============================================================


def build_value_features(
    prepared: Dict[str, Any],
    transformer_type: str = "bidirectional",
) -> np.ndarray:
    steps = prepared.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("Candidate has no steps")

    confidences: List[float] = []
    for step in steps:
        confidence = safe_float(step.get("token_avg_confidence"), default=np.nan)
        if not np.isfinite(confidence):
            raise ValueError("Missing or invalid token_avg_confidence")
        confidences.append(float(confidence))

    starts, ends = compute_step_start_end_positions(steps)
    if not starts or not ends:
        raise ValueError("Missing or invalid first_token_index")
    final_end_position = max(ends[-1], 1)

    x_list: List[List[float]] = []
    for step_idx, confidence in enumerate(confidences):
        local_std, local_slope = compute_local_std_and_slope(
            confidences,
            step_idx,
            transformer_type=transformer_type,
        )
        normalized_position = float(ends[step_idx] / final_end_position)
        normalized_position = max(0.0, min(1.0, normalized_position))
        x_list.append([
            confidence,
            local_std,
            local_slope,
            normalized_position,
        ])
    return np.asarray(x_list, dtype=np.float32)


def import_grade_answer_verl(project_root: Optional[str]):
    try:
        from deepscaler.rewards.math_utils.utils import grade_answer_verl
        return grade_answer_verl
    except Exception as first_error:
        if project_root:
            expanded = os.path.abspath(os.path.expanduser(project_root))
            if expanded not in sys.path:
                sys.path.insert(0, expanded)
        try:
            from deepscaler.rewards.math_utils.utils import grade_answer_verl
            return grade_answer_verl
        except Exception as second_error:
            raise ImportError(
                "Could not import deepscaler.rewards.math_utils.utils.grade_answer_verl. "
                "Pass --project_root pointing to the repository that contains deepscaler.\n"
                f"Initial error: {first_error!r}\nRetry error: {second_error!r}"
            ) from second_error


def get_solution_after_think(full_output: Any) -> str:
    if full_output is None:
        return ""
    text = str(full_output)
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def aggregate_phase_scores(phase_scores: Sequence[float]) -> Dict[str, float]:
    if not phase_scores:
        raise ValueError("No phase scores to aggregate")
    values = np.asarray(phase_scores, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite phase score")
    clipped = np.clip(values, 1e-12, 1.0)
    return {
        "product_score": float(np.exp(np.log(clipped).sum())),
        "min_score": float(values.min()),
        "mean_score": float(values.mean()),
        "last_score": float(values[-1]),
    }


def attach_phase_scores_to_steps(
    steps: List[Dict[str, Any]],
    phase_ranges: Sequence[Sequence[int]],
    all_step_predictions: Sequence[float],
) -> Tuple[List[float], Dict[str, float]]:
    phase_scores: List[float] = []
    for phase_index, phase_range in enumerate(phase_ranges):
        if len(phase_range) != 2:
            raise ValueError(f"Invalid phase range: {phase_range}")
        start_step, end_step = int(phase_range[0]), int(phase_range[1])
        if not (1 <= start_step <= end_step <= len(steps)):
            raise ValueError(
                f"Phase range {phase_range} outside 1..{len(steps)}"
            )
        phase_score = float(all_step_predictions[end_step - 1])
        phase_scores.append(phase_score)
        cumulative = aggregate_phase_scores(phase_scores)
        end_record = steps[end_step - 1]
        end_record["phase_index"] = int(phase_index)
        end_record["phase_quality_score"] = phase_score
        end_record["phase_product_score"] = cumulative["product_score"]
        end_record["phase_min_score"] = cumulative["min_score"]
        end_record["phase_mean_score"] = cumulative["mean_score"]
        end_record["phase_last_score"] = cumulative["last_score"]
    return phase_scores, aggregate_phase_scores(phase_scores)


def make_final_error_records(
    prepared: Dict[str, Any],
    error: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    common = {
        "sample_idx": prepared["sample_idx"],
        "sequence_idx": prepared["sequence_idx"],
        "problem": prepared["problem"],
        "gold_answer": prepared.get("gold_answer"),
        "difficulty": prepared.get("difficulty"),
        "unique_id": prepared.get("unique_id"),
        "error": error,
        "is_correct": False,
        "valid_for_selection": False,
    }
    summary = {
        **common,
        "full_output": prepared.get("full_output"),
        "step_avg_confidences": prepared.get("step_avg_confidences", []),
        "num_steps": prepared.get("num_steps", 0),
        "num_generated_tokens": prepared.get("num_generated_tokens", 0),
        "finish_reason": prepared.get("finish_reason"),
        "phase_ranges": prepared.get("phase_ranges", []),
        "phase_scores": [],
        "product_score": None,
        "min_score": None,
        "mean_score": None,
        "last_score": None,
    }
    if "token_entropies" in prepared:
        summary["token_entropies"] = prepared["token_entropies"]
    steps = {
        **common,
        "steps": prepared.get("steps", []),
    }
    return summary, steps


def load_value_components(args: argparse.Namespace):
    groups = parse_value_devices(args.value_device)
    if groups:
        return load_parallel_value_components(sys.modules[__name__], args, groups)
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.training_metrics_json is None:
        metrics_path = infer_training_metrics_path(checkpoint_path)
    else:
        metrics_path = Path(args.training_metrics_json).expanduser().resolve()
    source_metrics = load_json(metrics_path)
    transformer_type = resolve_transformer_type(args.transformer_type, source_metrics)
    normalization_metadata, fold_index, normalization_source = select_normalization_metadata(
        source_metrics, checkpoint_path, args.fold_index
    )
    normalizer = FeatureNormalizer.from_dict(normalization_metadata)
    device = resolve_device(args.value_device)
    model, model_config = build_model_from_metadata(
        source_metrics, transformer_type, device
    )
    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[INFO] Value checkpoint: {checkpoint_path}")
    print(f"[INFO] Value metadata: {metrics_path}")
    print(f"[INFO] Transformer type: {transformer_type}")
    print(f"[INFO] Normalization source: {normalization_source}")
    print(f"[INFO] Value checkpoint load device: {device}")
    print(f"[INFO] Trainable parameters: {count_parameters(model)}")
    return model, normalizer, device, {
        "checkpoint_path": str(checkpoint_path),
        "training_metrics_json": str(metrics_path),
        "transformer_type": transformer_type,
        "fold_index": fold_index,
        "normalization_source": normalization_source,
        "model_config": model_config,
    }


def score_prepared_candidates(
    args: argparse.Namespace,
    pending_keys: List[CandidateKey],
    prepared_records: Dict[CandidateKey, Dict[str, Any]],
    summary_path: str,
    steps_path: str,
    completed_final: Set[CandidateKey],
) -> None:
    grade_answer_verl = import_grade_answer_verl(args.project_root)

    feature_transformer_type = args.transformer_type
    if feature_transformer_type == "auto":
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        if args.training_metrics_json is None:
            metrics_path = infer_training_metrics_path(checkpoint_path)
        else:
            metrics_path = Path(args.training_metrics_json).expanduser().resolve()
        feature_transformer_type = resolve_transformer_type(
            "auto",
            load_json(metrics_path),
        )

    error_keys: List[CandidateKey] = []
    valid_items: List[Tuple[CandidateKey, np.ndarray]] = []
    validation_errors: Dict[CandidateKey, str] = {}
    for key in pending_keys:
        prepared = prepared_records[key]
        if prepared.get("status") != "ok":
            error_keys.append(key)
            validation_errors[key] = prepared.get("error", "Preparation error")
            continue
        try:
            x = build_value_features(
                prepared,
                transformer_type=feature_transformer_type,
            )
            if not prepared.get("phase_ranges"):
                raise ValueError("No phase ranges")
            valid_items.append((key, x))
        except Exception:
            error_keys.append(key)
            validation_errors[key] = traceback.format_exc()

    for key in error_keys:
        summary, steps = make_final_error_records(
            prepared_records[key], validation_errors[key]
        )
        append_jsonl(summary_path, summary)
        append_jsonl(steps_path, steps)
        completed_final.add(key)

    if not valid_items:
        return

    model, normalizer, device, _ = load_value_components(args)
    scoring_start = time.perf_counter()
    forward_seconds = grading_seconds = write_seconds = 0.0
    try:
        if isinstance(model, ParallelValueModel):
            scoring_devices = tuple(
                tuple(str(shard["qkv"].device) for shard in replica.layers[0]["shards"])
                for replica in model.replicas
            )
        else:
            scoring_devices = ((str(device),),)
        print(f"[INFO] Value scoring devices (actual weights): {scoring_devices}")
        # Keep the existing per-GPU batch limit while filling every DP replica.
        dp_size = len(model.replicas) if isinstance(model, ParallelValueModel) else 1
        batch_size = args.scoring_batch_size * dp_size
        input_device = getattr(model, "input_device", device)
        for batch_start in tqdm(
            range(0, len(valid_items), batch_size),
            desc="Scoring confidence trajectories",
        ):
            batch_items = valid_items[
                batch_start:batch_start + batch_size
            ]
            arrays = [normalizer.transform_one(item[1]) for item in batch_items]
            # Bucket only this bounded window, then restore original output order.
            order = sorted(range(len(arrays)), key=lambda i: arrays[i].shape[0])
            max_len = max(array.shape[0] for array in arrays)
            feature_dim = arrays[0].shape[1]
            x_tensor = torch.zeros(
                len(arrays), max_len, feature_dim, dtype=torch.float32
            )
            attention_mask = torch.zeros(
                len(arrays), max_len, dtype=torch.bool
            )
            lengths = [array.shape[0] for array in arrays]
            for index, source_index in enumerate(order):
                array = arrays[source_index]
                length = array.shape[0]
                x_tensor[index, :length] = torch.from_numpy(array)
                attention_mask[index, :length] = True

            forward_start = time.perf_counter()
            with torch.inference_mode():
                predictions = model(
                    x_tensor.to(input_device),
                    attention_mask=attention_mask.to(input_device),
                ).detach().cpu().numpy()
            # .cpu() above waits for forward/results; no extra CUDA sync needed.
            forward_seconds += time.perf_counter() - forward_start
            predictions = predictions[np.argsort(order)]

            for batch_index, ((key, _), length) in enumerate(zip(batch_items, lengths)):
                prepared = prepared_records[key]
                try:
                    steps_list = copy.deepcopy(prepared["steps"])
                    all_step_predictions = [
                        float(value) for value in predictions[batch_index, :length]
                    ]
                    for step, prediction in zip(steps_list, all_step_predictions):
                        step["trajectory_model_step_score"] = prediction
                    phase_scores, final_scores = attach_phase_scores_to_steps(
                        steps=steps_list,
                        phase_ranges=prepared["phase_ranges"],
                        all_step_predictions=all_step_predictions,
                    )
                    solution = get_solution_after_think(prepared.get("full_output"))
                    grade_error = None
                    grading_start = time.perf_counter()
                    try:
                        is_correct = bool(
                            grade_answer_verl(solution, prepared.get("gold_answer"))
                        )
                    except Exception:
                        is_correct = False
                        grade_error = traceback.format_exc()
                    grading_seconds += time.perf_counter() - grading_start

                    common = {
                        "sample_idx": prepared["sample_idx"],
                        "sequence_idx": prepared["sequence_idx"],
                        "problem": prepared["problem"],
                        "gold_answer": prepared.get("gold_answer"),
                        "difficulty": prepared.get("difficulty"),
                        "unique_id": prepared.get("unique_id"),
                        "is_correct": is_correct,
                        "valid_for_selection": True,
                    }
                    summary = {
                        **common,
                        "full_output": prepared.get("full_output"),
                        "step_avg_confidences": prepared["step_avg_confidences"],
                        "num_steps": prepared["num_steps"],
                        "num_generated_tokens": prepared["num_generated_tokens"],
                        "finish_reason": prepared.get("finish_reason"),
                        "phase_ranges": prepared["phase_ranges"],
                        "phase_scores": phase_scores,
                        **final_scores,
                    }
                    if "token_entropies" in prepared:
                        summary["token_entropies"] = prepared["token_entropies"]
                    if grade_error is not None:
                        summary["grade_error"] = grade_error

                    steps_record = {
                        **common,
                        "steps": steps_list,
                    }
                    if grade_error is not None:
                        steps_record["grade_error"] = grade_error

                except Exception:
                    summary, steps_record = make_final_error_records(
                        prepared, traceback.format_exc()
                    )

                write_start = time.perf_counter()
                append_jsonl(summary_path, summary)
                append_jsonl(steps_path, steps_record)
                write_seconds += time.perf_counter() - write_start
                completed_final.add(key)
    finally:
        elapsed = time.perf_counter() - scoring_start
        if isinstance(model, ParallelValueModel):
            model.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        other_seconds = max(0.0, elapsed - forward_seconds - grading_seconds - write_seconds)
        print(
            f"[INFO] Scoring timings (wall seconds): forward={forward_seconds:.3f}, "
            f"grading={grading_seconds:.3f}, write={write_seconds:.3f}, "
            f"other={other_seconds:.3f}, total={elapsed:.3f}"
        )


# ============================================================
# Random-subsampled Best-of-N evaluation and artifacts
# ============================================================


def serializable_candidate_view(
    sequence_idx: int,
    record: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if record is None:
        return {
            "sequence_idx": int(sequence_idx),
            "missing": True,
            "valid_for_selection": False,
        }
    return {
        "sequence_idx": int(sequence_idx),
        "missing": False,
        "valid_for_selection": bool(record.get("valid_for_selection", False)),
        "product_score": record.get("product_score"),
        "min_score": record.get("min_score"),
        "mean_score": record.get("mean_score"),
        "last_score": record.get("last_score"),
        "is_correct": bool(record.get("is_correct", False)),
        "error": record.get("error"),
    }


def choose_by_score(
    candidates: Sequence[Tuple[int, Dict[str, Any]]],
    score_key: str,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    valid: List[Tuple[int, Dict[str, Any]]] = []
    for sequence_idx, record in candidates:
        if not record.get("valid_for_selection", False):
            continue
        score = safe_float(record.get(score_key), default=np.nan)
        if np.isfinite(score):
            valid.append((sequence_idx, record))
    if not valid:
        return None
    return max(valid, key=lambda item: float(item[1][score_key]))


def evaluate_best_of_n(
    args: argparse.Namespace,
    data: List[Dict[str, Any]],
    summary_records: Dict[CandidateKey, Dict[str, Any]],
    output_dir: Path,
) -> None:
    ns = args.ns
    if ns is None:
        suggested = [1, 2, 4, 8, 16, 32, args.N]
        ns = sorted({value for value in suggested if 1 <= value <= args.N})
    else:
        ns = [int(value) for value in ns]
    if not ns:
        raise ValueError("No valid Best-of-N n values")
    if any(value < 1 or value > args.N for value in ns):
        raise ValueError(f"Every --ns value must satisfy 1 <= n <= N={args.N}")

    trial_accuracies: Dict[int, Dict[str, List[float]]] = {
        n: {method: [] for method in AGGREGATIONS} for n in ns
    }
    trial_mean_output_lengths: Dict[int, Dict[str, List[float]]] = {
        n: {method: [] for method in AGGREGATIONS} for n in ns
    }
    details_by_n: Dict[int, List[Dict[str, Any]]] = {n: [] for n in ns}

    records_by_sample: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for (sample_idx, sequence_idx), record in summary_records.items():
        records_by_sample[sample_idx][sequence_idx] = record

    for trial_idx in range(args.n_trials):
        rng = random.Random(args.eval_seed + trial_idx * 1000003)
        shuffled_slots_by_sample: Dict[
            int, List[Tuple[int, Optional[Dict[str, Any]]]]
        ] = {}
        for sample_idx, _sample in enumerate(data):
            slots: List[Tuple[int, Optional[Dict[str, Any]]]] = [
                (sequence_idx, records_by_sample[sample_idx].get(sequence_idx))
                for sequence_idx in range(args.N)
            ]
            # Shuffle once per sample/trial; all n values reuse this order and
            # take prefixes, so smaller-n responses are nested in larger-n sets.
            rng.shuffle(slots)
            shuffled_slots_by_sample[sample_idx] = slots

        for n in ns:
            correct_counts = {method: 0 for method in AGGREGATIONS}
            selected_output_lengths: Dict[str, List[float]] = {
                method: [] for method in AGGREGATIONS
            }

            for sample_idx, sample in enumerate(data):
                sampled_slots = shuffled_slots_by_sample[sample_idx][:n]
                sampled_existing: List[Tuple[int, Dict[str, Any]]] = [
                    (sequence_idx, record)
                    for sequence_idx, record in sampled_slots
                    if record is not None
                ]

                detail: Dict[str, Any] = {
                    "trial_idx": int(trial_idx),
                    "n": int(n),
                    "sample_idx": int(sample_idx),
                    "problem": sample.get("problem"),
                    "gold_answer": sample.get("answer"),
                    "sampled_sequence_indices": json.dumps(
                        [sequence_idx for sequence_idx, _ in sampled_slots],
                        ensure_ascii=False,
                    ),
                    "sampled_candidates": json.dumps(
                        [
                            serializable_candidate_view(sequence_idx, record)
                            for sequence_idx, record in sampled_slots
                        ],
                        ensure_ascii=False,
                    ),
                }

                for method in AGGREGATIONS:
                    choice = choose_by_score(sampled_existing, SCORE_KEYS[method])
                    if choice is None:
                        detail[f"{method}_choice_sequence_idx"] = None
                        detail[f"{method}_choice_score"] = None
                        detail[f"{method}_choice_is_correct"] = False
                        selected_output_lengths[method].append(0.0)
                    else:
                        sequence_idx, record = choice
                        is_correct = bool(record.get("is_correct", False))
                        detail[f"{method}_choice_sequence_idx"] = int(sequence_idx)
                        detail[f"{method}_choice_score"] = float(
                            record[SCORE_KEYS[method]]
                        )
                        detail[f"{method}_choice_is_correct"] = is_correct
                        selected_output_lengths[method].append(
                            safe_float(record.get("num_generated_tokens"), default=0.0)
                        )
                        if is_correct:
                            correct_counts[method] += 1

                details_by_n[n].append(detail)

            denominator = len(data)
            for method in AGGREGATIONS:
                accuracy = correct_counts[method] / denominator if denominator else 0.0
                trial_accuracies[n][method].append(float(accuracy))
            for method in AGGREGATIONS:
                method_lengths = selected_output_lengths[method]
                trial_mean_output_lengths[n][method].append(
                    float(np.mean(method_lengths)) if method_lengths else 0.0
                )

    metric_rows: List[Dict[str, Any]] = []
    for n in ns:
        row: Dict[str, Any] = {
            "n": int(n),
            "n_trials": int(args.n_trials),
            "num_samples": int(len(data)),
        }
        for method in AGGREGATIONS:
            values = np.asarray(trial_accuracies[n][method], dtype=float)
            row[f"{method}_accuracy"] = float(values.mean())
            row[f"{method}_std"] = float(values.std())
        for method in AGGREGATIONS:
            output_length_values = np.asarray(
                trial_mean_output_lengths[n][method], dtype=float
            )
            row[f"{method}_mean_output_length"] = float(
                output_length_values.mean()
            )
        metric_rows.append(row)

        details_path = output_dir / f"details_{n}.csv"
        pd.DataFrame(details_by_n[n]).to_csv(
            details_path, index=False, encoding="utf-8-sig"
        )
        print(f"[INFO] Saved details: {details_path}")

    metrics_df = pd.DataFrame(metric_rows).sort_values("n")
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved metrics: {metrics_path}")

    colors = {
        "product": "tab:blue",
        "min": "tab:orange",
        "mean": "tab:green",
        "last": "tab:red",
    }
    figure, (accuracy_axis, length_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )
    for method in AGGREGATIONS:
        accuracy_axis.plot(
            metrics_df["n"],
            metrics_df[f"{method}_accuracy"],
            marker="o",
            label=method,
            color=colors[method],
        )
    accuracy_axis.set_xlabel("n")
    accuracy_axis.set_ylabel("Mean test accuracy")
    accuracy_axis.set_title("Best-of-N accuracy by phase-score aggregation")
    accuracy_axis.grid(True, alpha=0.3)
    accuracy_axis.legend()

    for method in AGGREGATIONS:
        length_axis.plot(
            metrics_df["n"],
            metrics_df[f"{method}_mean_output_length"],
            marker="o",
            label=method,
            color=colors[method],
        )
    length_axis.set_xlabel("n")
    length_axis.set_ylabel("Mean output length (generated tokens)")
    length_axis.set_title("Mean selected output length by aggregation")
    length_axis.grid(True, alpha=0.3)
    length_axis.legend()

    figure.tight_layout()
    figure_path = output_dir / "metrics.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    print(f"[INFO] Saved plot: {figure_path}")


# ============================================================
# Argument parsing and main pipeline
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3 confidence-trajectory phase reward Best-of-N evaluation"
    )

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--training_metrics_json", type=str, default=None)
    parser.add_argument(
        "--transformer_type",
        type=str,
        default="auto",
        choices=["auto", "bidirectional", "causal"],
        help="Must match value-model training.",
    )
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument(
        "--value_device", type=str, default="auto",
        help="Value device: auto/cpu/cuda:0, or '0;1' (DP), '0,1' (TP), '0,1;2,3' (DP+TP).",
    )
    parser.add_argument("--scoring_batch_size", type=int, default=8)

    parser.add_argument("--enable_thinking", type=parse_bool, default=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of candidates generated for one problem in each vLLM call.",
    )
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=10240)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--logprobs", type=int, default=1)

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Semicolon-separated DP groups and comma-separated TP GPUs inside each "
            "group; e.g. '0,1;2,3' with --tensor_parallel_size 2."
        ),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--vllm_enforce_eager",
        "--enforce_eager",
        dest="vllm_enforce_eager",
        action="store_true",
        help=(
            "Disable vLLM CUDA Graph execution and run eagerly. Use this as a "
            "compatibility fallback for NCCL failures during graph warm-up."
        ),
    )
    parser.add_argument(
        "--vllm_cache_mode",
        choices=["auto", "isolated", "shared"],
        default="auto",
        help=(
            "Compilation-cache policy for generation workers. auto isolates "
            "VLLM/TorchInductor/Triton caches when multiple DP workers run."
        ),
    )
    parser.add_argument(
        "--vllm_cache_root",
        type=str,
        default=None,
        help="Optional base directory for stable vLLM compilation caches.",
    )
    parser.add_argument(
        "--generation_worker_startup_timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for every vLLM DP engine to finish startup.",
    )

    parser.add_argument("--num_segments", "--K", dest="num_segments", type=int, default=8)
    parser.add_argument("--entropy_threshold", type=float, default=1.2)
    add_phase_partition_arguments(parser)
    parser.add_argument("--entropy_chunk_size", type=int, default=16)
    parser.add_argument("--entropy_model_path", type=str, default=None)
    parser.add_argument("--entropy_device_map", type=str, default="auto")
    parser.add_argument(
        "--no_save_token_entropies",
        action="store_true",
        help="Do not include the per-token entropy list in summary.jsonl.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--ns", type=int, nargs="+", default=None)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_input_parts", type=int, default=1)
    parser.add_argument("--input_part_id", type=int, default=0)
    parser.add_argument(
        "--project_root",
        type=str,
        default="internal_time_series",
        help="Repository root containing the deepscaler package.",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.N < 1:
        parser.error("--N must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.scoring_batch_size < 1:
        parser.error("--scoring_batch_size must be at least 1")
    try:
        parse_value_devices(args.value_device)
    except ValueError as error:
        parser.error(str(error))
    if args.tensor_parallel_size < 1:
        parser.error("--tensor_parallel_size must be at least 1")
    if args.logprobs < 1:
        parser.error("--logprobs must be at least 1")
    try:
        validate_phase_partition_options(args)
    except ValueError as error:
        parser.error(str(error))
    if args.entropy_chunk_size < 1:
        parser.error("--entropy_chunk_size must be at least 1")
    if args.generation_worker_startup_timeout <= 0:
        parser.error("--generation_worker_startup_timeout must be positive")
    if args.n_trials < 1:
        parser.error("--n_trials must be at least 1")
    if args.num_input_parts < 1:
        parser.error("--num_input_parts must be at least 1")
    if not (0 <= args.input_part_id < args.num_input_parts):
        parser.error("--input_part_id must satisfy 0 <= i < --num_input_parts")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")


def resolve_entropy_gpu_groups(
    args: argparse.Namespace,
    gpu_groups: List[str],
) -> List[str]:
    if gpu_groups:
        return gpu_groups
    if args.tensor_parallel_size <= 1:
        return []
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if visible:
        if len(visible) < args.tensor_parallel_size:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES has fewer GPUs than tensor_parallel_size"
            )
        return [",".join(visible[:args.tensor_parallel_size])]
    return [",".join(str(i) for i in range(args.tensor_parallel_size))]


def remove_output_artifacts(output_dir: Path) -> None:
    fixed_names = [
        "summary.jsonl", "steps.jsonl", "metrics.csv", "metrics.png",
    ]
    for name in fixed_names:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for path in output_dir.glob("details_*.csv"):
        path.unlink()


def main() -> None:
    from importlib import import_module
    budget = import_module("scripts.3_best_of_n_via_phases_scores")
    parser = argparse.ArgumentParser(add_help=False)
    # Reuse the complete parser while retaining parser.error in validation.
    args = parse_args()
    validation_parser = argparse.ArgumentParser()
    validate_args(validation_parser, args)
    set_seed(args.seed)

    try:
        gpu_groups = parse_gpu_groups(args.gpus, args.tensor_parallel_size)
        entropy_gpu_groups = resolve_entropy_gpu_groups(args, gpu_groups)
    except ValueError as error:
        validation_parser.error(str(error))

    data = load_json_array(args.input_json)
    if args.limit is not None:
        data = data[:args.limit]
    num_samples = len(data)
    base_size, remainder = divmod(num_samples, args.num_input_parts)
    split_start = args.input_part_id * base_size + min(args.input_part_id, remainder)
    split_end = split_start + base_size + (
        1 if args.input_part_id < remainder else 0
    )
    data = data[split_start:split_end]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = str(output_dir / "summary.jsonl")
    steps_path = str(output_dir / "steps.jsonl")
    generation_temp_path = str(output_dir / ".generation_tmp.jsonl")
    if args.resume and not os.path.exists(generation_temp_path) and (output_dir / ".bon_generation.jsonl").is_file():
        generation_temp_path = str(output_dir / ".bon_generation.jsonl")
    prepared_temp_path = str(output_dir / ".prepared_tmp.jsonl")

    if args.overwrite:
        remove_output_artifacts(output_dir)
    elif not args.resume and (
        os.path.exists(summary_path) or os.path.exists(steps_path)
    ):
        raise FileExistsError(
            "summary.jsonl or steps.jsonl already exists. Use --resume or --overwrite."
        )

    if args.resume:
        completed_final = synchronize_final_outputs(
            summary_path, steps_path, data, args.N
        )
        generation_records = budget.read_generation_checkpoint(
            sys.modules[__name__], generation_temp_path, data, args.N
        )
        prepared_records = compact_candidate_file(
            prepared_temp_path, data, args.N
        )
    else:
        completed_final: Set[CandidateKey] = set()
        generation_records: Dict[CandidateKey, Dict[str, Any]] = {}
        prepared_records: Dict[CandidateKey, Dict[str, Any]] = {}

    # A complete phase file is sufficient for scoring without generation or
    # entropy files. Import it read-only; keep the legacy output filenames.
    budget.load_budget_prepared(
        sys.modules[__name__], args, data, prepared_temp_path, prepared_records,
    )

    expected = expected_candidate_keys(len(data), args.N)
    budget.normalize_resume_budget(
        sys.modules[__name__], args, generation_records, prepared_records,
        completed_final, output_dir, expected,
    )
    if args.resume:
        completed_final.intersection_update(synchronize_final_outputs(
            summary_path, steps_path, data, args.N,
            max_tokens=args.max_tokens, generation_records=generation_records,
        ))

    print(f"[INFO] Test samples: {len(data)}")
    print(f"[INFO] Candidates per sample: {args.N}")
    print(f"[INFO] Total candidate records: {len(expected)}")
    print(f"[resume] Final completed: {len(completed_final)}")
    print(f"[resume] Generated: {len(generation_records)}")
    print(f"[resume] Entropy/phase prepared: {len(prepared_records)}")

    # Stage 1: generate every missing (sample_idx, sequence_idx).
    generation_tasks: List[Tuple[int, Dict[str, Any], List[int]]] = []
    for sample_idx, sample in enumerate(data):
        pending_sequence_indices = [
            sequence_idx
            for sequence_idx in range(args.N)
            if (sample_idx, sequence_idx) not in generation_records
            and (sample_idx, sequence_idx) not in prepared_records
            and (sample_idx, sequence_idx) not in completed_final
        ]
        if pending_sequence_indices:
            generation_tasks.append((sample_idx, sample, pending_sequence_indices))

    if generation_tasks:
        if Path(generation_temp_path).name == ".bon_generation.jsonl":
            raise ValueError("Imported .bon_generation.jsonl is incomplete; keep the source read-only and resume from the complete generation file")
        if gpu_groups:
            run_data_parallel_generation(
                args, generation_tasks, gpu_groups,
                generation_temp_path, generation_records,
            )
        else:
            run_single_process_generation(
                args, generation_tasks, generation_temp_path, generation_records
            )
    else:
        print("[resume] Generation stage complete; skip vLLM.")

    if generation_tasks and os.path.exists(generation_temp_path):
        generation_records = budget.read_generation_checkpoint(
            sys.modules[__name__], generation_temp_path, data, args.N,
        )
        budget.normalize_resume_budget(
            sys.modules[__name__], args, generation_records, prepared_records,
            completed_final, output_dir, expected,
        )

    # Stage 2a completes the independent entropy cache; stage 2b only reads it.
    budget.run_budget_phase_stage(
        runtime=sys.modules[__name__],
        args=args,
        expected=expected,
        generation_records=generation_records,
        generation_path=generation_temp_path,
        prepared_path=prepared_temp_path,
        prepared_records=prepared_records,
        completed_final=completed_final,
        gpu_groups=entropy_gpu_groups,
    )

    if os.path.exists(prepared_temp_path):
        # The phase writer already updated prepared_records. Do not drop
        # candidates imported from output_dir/.bon_prepared.jsonl.
        compact_candidate_file(prepared_temp_path, data, args.N)

    # Stage 3: value-model scoring, grading, and final paired JSONL output.
    pending_score_keys = [key for key in expected if key not in completed_final]
    missing_prepared = [key for key in pending_score_keys if key not in prepared_records]
    if missing_prepared:
        raise RuntimeError(f"Missing prepared candidates: {missing_prepared[:20]}")
    if pending_score_keys:
        score_prepared_candidates(
            args, pending_score_keys, prepared_records,
            summary_path, steps_path, completed_final,
        )
    else:
        print("[resume] Value-scoring stage complete.")

    completed_final = synchronize_final_outputs(
        summary_path, steps_path, data, args.N
    )
    if len(completed_final) != len(expected):
        missing = [key for key in expected if key not in completed_final]
        raise RuntimeError(f"Final outputs incomplete: {missing[:20]}")

    # Stable ordering before evaluation.
    summary_records = read_jsonl_by_candidate(summary_path)
    steps_records = read_jsonl_by_candidate(steps_path)
    atomic_rewrite_jsonl(
        summary_path, [summary_records[key] for key in sorted(summary_records)]
    )
    atomic_rewrite_jsonl(
        steps_path, [steps_records[key] for key in sorted(steps_records)]
    )

    evaluate_best_of_n(args, data, summary_records, output_dir)

    print(f"[DONE] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
