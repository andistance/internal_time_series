#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用 vLLM 对 MATH-500 风格 JSON 数据集逐条推理，
按 "\n\n" 切分 reasoning step，
并保存每个 step 的平均 token confidence。

在原有 step 划分基础上，额外根据“完整词表 next-token entropy”进行自适应 phase 划分：
1. 完整输出仍然先按 "\n\n" 切分 step；
2. 对每个生成 token 计算完整词表熵：
       H_t = -sum_v p(v | x_<t) log p(v | x_<t)
3. 若某个 step 包含 </think>，则只使用该 step 及之前 token 的熵参与 phase 划分；
   若不存在 </think>，则仍使用完整输出；
4. original（默认）：以固定 entropy_threshold 和 K 选择边界；
   adaptive：复用 Best-of-N 的轨迹长度/高熵比例自适应边界选择；
5. 将候选边界按其在序列中的排名均匀下采样，不足时按熵补齐；
6. 将 token 边界映射回其所在 step：该 step 是新 phase 的起点，
   上一个 step 是上一个 phase 的终点，参与划分的最后一个 step 是最后一个 phase 的终点。
   </think> 所在 step 之后的 step 仍保留在输出中，但不参与 phase 划分。

输入 JSON 格式：
[
  {
    "problem": "...",
    "answer": "...",
    "difficulty": 2.0,
    "unique_id": "test/precalculus/807.json"
  },
  ...
]

输出：
1. summary_jsonl:
   原格式不变，仅新增：
       "phase_ranges": [[phase_start_step, phase_end_step], ...]
   step 序号从 1 开始。

2. step_jsonl:
   原格式不变，仅在 steps 列表的每个元素中新增：
       "phase_frontier": 0 / 1 / 2
   0：某个 phase 的起点；
   1：phase 内部；
   2：某个 phase 的终点。

实现说明：
- vLLM 仍只负责原来的逐样本生成和 sampled-token logprob 获取；
- 为避免让 vLLM 返回每个位置的完整词表 logprobs（输出规模极大），生成结束后释放
  vLLM，再用同一模型的 Hugging Face forward 进行一次分块精确熵计算；
- 因而程序分为“生成阶段”和“熵/phase 标注阶段”，最终输出格式保持不变。
"""

import argparse
import bisect
import codecs
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import socket
import sys
import traceback
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.transformer_bon.phases import (
    add_phase_partition_arguments, select_adaptive_boundaries,
    validate_phase_partition_options,
)

import numpy as np
from tqdm import tqdm


PROMPT_SUFFIX = (
    "\nLet's reason step by step. Enclose the reasoning process within "
    "<think>...</think>, then summarize it and present the final answer within \\boxed{}."
)


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


def _bytelevel_token_pieces(token_ids, tokenizer):
    """Linear equivalent of prefix decoding for Qwen's plain ByteLevel decoder.

    Decode incomplete UTF-8 at every token, including apparent prefix shrinkage,
    so even legacy piece text and nonnegative character spans stay identical.
    Other decoder configurations retain the original compatibility path.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or backend.decoder is None:
        return None
    state = json.loads(backend.decoder.__getstate__())
    if state.get("type") != "ByteLevel":
        return None
    cached = getattr(tokenizer, "_phase_bytelevel_state", None)
    if cached is None:
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        byte_map = {chr(value): value for value in visible}
        extra = 0
        for value in range(256):
            if value not in visible:
                byte_map[chr(256 + extra)] = value
                extra += 1
        special = {int(index) for index, token in backend.get_added_tokens_decoder().items()
                   if token.special}
        cached = byte_map, special
        tokenizer._phase_bytelevel_state = cached
    byte_map, special = cached
    encoded, pieces, chunks = {}, [], []
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    previous_pending = cursor = 0
    pending = ""
    for index, token_id in enumerate(token_ids):
        token_id = int(token_id)
        chunk = ""
        if token_id not in special:
            if token_id not in encoded:
                value = backend.id_to_token(token_id)
                if value is None:
                    return None
                encoded[token_id] = b"".join(
                    bytes((byte_map[char],)) if char in byte_map else char.encode("utf-8")
                    for char in value
                )
            chunk = decoder.decode(encoded[token_id], final=False)
        pending = decoder.getstate()[0].decode("utf-8", errors="replace")
        piece = (chunk + pending)[previous_pending:]
        end = cursor + len(piece)
        pieces.append(dict(token_index=index, token_id=token_id, text=piece,
                           char_start=cursor, char_end=end))
        chunks.append(chunk)
        previous_pending, cursor = len(pending), end
    full_text = "".join(chunks) + pending
    # Verify the complete decode before trusting the specialized decoder.
    expected = tokenizer.decode(token_ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    return (full_text, pieces) if full_text == expected else None


def _bytefallback_token_pieces(token_ids, tokenizer):
    """Exact linear prefix differences for standard SentencePiece ByteFallback.

    Match Replace/ByteFallback/Fuse/leading-Strip decoding, including the legacy
    replacement of every byte in an invalid or incomplete contiguous byte run.
    Unsupported configurations fall back to the original prefix decoder.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or backend.decoder is None:
        return None
    state = json.loads(backend.decoder.__getstate__())
    sequence = list(state.get("decoders", [])) if state.get("type") == "Sequence" else [state]
    replacements, strip = [], False
    while sequence and sequence[0].get("type") == "Replace":
        item = sequence.pop(0)
        pattern = item.get("pattern", {}).get("String")
        if not pattern:
            return None
        replacements.append((pattern, item["content"]))
    if not sequence or sequence.pop(0).get("type") != "ByteFallback":
        return None
    fused = bool(sequence and sequence[0].get("type") == "Fuse")
    if fused:
        sequence.pop(0)
    if fused and sequence and sequence[0] == {
        "type": "Strip", "content": " ", "start": 1, "stop": 0,
    }:
        sequence.pop(0)
        strip = True
    if sequence:
        return None
    special = {int(index) for index, token in backend.get_added_tokens_decoder().items()
               if token.special}
    encoded, pieces = {}, []
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    committed = run_size = run_chars = previous = cursor = 0
    first = run_first = ""
    invalid = False
    for index, token_id in enumerate(token_ids):
        token_id = int(token_id)
        value, byte, chunk = "", None, ""
        if token_id not in special:
            if token_id not in encoded:
                value = backend.id_to_token(token_id)
                if value is None:
                    return None
                for pattern, content in replacements:
                    value = value.replace(pattern, content)
                if len(value) == 6 and value.startswith("<0x") and value.endswith(">"):
                    try:
                        byte = int(value[3:5], 16)
                        if not 0 <= byte <= 255:
                            byte = None
                    except ValueError:
                        pass
                encoded[token_id] = value, byte
            value, byte = encoded[token_id]
            if byte is not None:
                run_size += 1
                if not invalid:
                    try:
                        chunk = decoder.decode(bytes((byte,)), final=False)
                        run_chars += len(chunk)
                        if not run_first and chunk:
                            run_first = chunk[0]
                    except UnicodeDecodeError:
                        invalid = True
            else:
                valid_run = not invalid and not decoder.getstate()[0]
                committed += run_chars if valid_run else run_size
                if not first and run_size:
                    first = run_first if valid_run else "\ufffd"
                committed += len(value)
                if not first and value:
                    first = value[0]
                decoder.reset()
                run_size = run_chars = 0
                run_first, invalid = "", False
        valid_run = not invalid and not decoder.getstate()[0]
        length = committed + (run_chars if valid_run else run_size)
        current_first = first or (run_first if valid_run else "\ufffd")
        if strip and current_first == " ":
            length -= 1
        added = max(0, length - previous)
        if byte is not None and not valid_run:
            piece = "\ufffd" * added
        else:
            suffix = chunk if byte is not None else value
            if added > len(suffix):
                return None
            piece = suffix[-added:] if added else ""
        end = cursor + added
        pieces.append(dict(token_index=index, token_id=token_id, text=piece,
                           char_start=cursor, char_end=end))
        previous, cursor = length, end
    full_text = tokenizer.decode(token_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    return (full_text, pieces) if previous == len(full_text) else None


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
    if token_ids:
        fast = _bytelevel_token_pieces(token_ids, tokenizer)
        if fast is None:
            fast = _bytefallback_token_pieces(token_ids, tokenizer)
        if fast is not None:
            return fast
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


def _locate_step_span(left, right, starts, ends, fallback=False):
    """Visit only overlapping steps, preserving the first maximum-overlap tie."""
    index = bisect.bisect_right(ends, left)
    best, maximum = None, 0
    while index < len(starts) and starts[index] < right:
        overlap = min(right, ends[index]) - max(left, starts[index])
        if overlap > maximum:
            best, maximum = index, overlap
        index += 1
    if best is None and fallback:
        index = bisect.bisect_left(starts, left)
        return index if index < len(starts) else None
    return best


def _boundary_step_locator(token_pieces, steps):
    starts = [int(step["char_start"]) for step in steps]
    ends = [int(step["char_end"]) for step in steps]

    def locate(position, _pieces, _steps):
        if not 0 <= position < len(token_pieces):
            return None
        piece = token_pieces[position]
        return _locate_step_span(int(piece["char_start"]), int(piece["char_end"]),
                                 starts, ends, fallback=True)
    return locate


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
    first_token_indices: List[Optional[int]] = [None] * len(steps)
    starts = [step["char_start"] for step in steps]
    ends = [step["char_end"] for step in steps]

    for piece, logprob in zip(token_pieces, token_logprobs):
        token_start = piece["char_start"]
        token_end = piece["char_end"]

        if token_end <= token_start:
            continue

        best_step_idx = _locate_step_span(token_start, token_end, starts, ends)
        if best_step_idx is not None:
            token_index = int(piece["token_index"])
            previous = first_token_indices[best_step_idx]
            if previous is None or token_index < previous:
                first_token_indices[best_step_idx] = token_index

            if logprob is not None:
                conf = math.exp(logprob)
                step_token_confs[best_step_idx].append(conf)

    step_conf_list: List[Optional[float]] = []
    step_records: List[Dict[str, Any]] = []

    for step, confs, first_token_index in zip(steps, step_token_confs, first_token_indices):
        if len(confs) > 0:
            avg_conf_value: Optional[float] = float(sum(confs) / len(confs))
        else:
            avg_conf_value = None

        step_conf_list.append(avg_conf_value)

        step_records.append(
            {
                "step": step["step"],
                "token_avg_confidence": avg_conf_value,
                "first_token_index": first_token_index,
            }
        )

    return step_conf_list, step_records


def make_sampling_params(args: argparse.Namespace) -> "SamplingParams":
    """
    构造 vLLM SamplingParams。

    logprobs=1：
    返回每个输出位置 top-1 token 的 logprob，同时 vLLM 会包含实际采样 token 的 logprob。
    """
    from vllm import SamplingParams

    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "logprobs": args.logprobs,
        "skip_special_tokens": True,
    }

    # 部分 vLLM 版本支持 SamplingParams(seed=...)
    # 为了兼容旧版本，这里做一次兜底。
    if args.seed is not None:
        kwargs["seed"] = args.seed

    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


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
    scoring_ids = torch.tensor(scoring_input_ids, dtype=torch.long, device=input_device)
    past_key_values = None
    entropy_chunks = []

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
                # Keep only O(response length) scalar entropies on device;
                # transfer once instead of synchronizing every small chunk.
                entropy_chunks.append(entropies.detach())

                del selected_logits, log_probs, entropies

            del input_ids, outputs, logits

    entropy_values = torch.cat(entropy_chunks).cpu().tolist() if entropy_chunks else []
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


def _select_phase_boundaries(response_entropies, token_pieces, text, args):
    phase_num_steps, phase_num_tokens = get_phase_partition_limits(text, token_pieces)
    entropies = response_entropies[:phase_num_tokens]
    if getattr(args, "phase_partition_algorithm", "original") == "adaptive":
        steps = split_text_into_steps(text)[:phase_num_steps]
        return select_adaptive_boundaries(
            response_entropies=entropies,
            token_pieces=token_pieces,
            steps=steps,
            target_phase_tokens=args.target_phase_tokens,
            high_entropy_ratio=args.high_entropy_ratio,
            locate_boundary_step_index=_boundary_step_locator(token_pieces, steps),
        )
    return select_entropy_boundary_token_positions(
        response_entropies=entropies,
        entropy_threshold=args.entropy_threshold,
        num_segments=args.num_segments,
    )


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
    locate = _boundary_step_locator(token_pieces, phase_steps)

    # 第一个 step 固定为第一个 phase 的起点。
    phase_start_steps = {1}

    for boundary_token_position in boundary_token_positions:
        step_idx = locate(boundary_token_position, token_pieces, phase_steps)
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


def build_generation_error_records(
    sample_idx: int,
    problem: str,
    answer: Any,
    difficulty: Any,
    error: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """构造与原脚本一致的错误记录，并加入 phase_ranges。"""
    summary_record = {
        "sample_idx": sample_idx,
        "problem": problem,
        "gold_answer": answer,
        "difficulty": difficulty,
        "error": error,
        "full_output": None,
        "step_avg_confidences": [],
        "num_steps": 0,
        "num_generated_tokens": 0,
        "phase_ranges": [],
    }

    step_record = {
        "sample_idx": sample_idx,
        "problem": problem,
        "gold_answer": answer,
        "difficulty": difficulty,
        "error": error,
        "steps": [],
    }

    return summary_record, step_record



class _GenerationRecordIndex(MutableMapping):
    """In-memory byte offsets; load just one existing JSONL record on demand.

    Assignments are made immediately after append_jsonl, so indexing a newly
    generated row does not retain its token arrays or add a second cache file.
    """

    def __init__(self, path, offsets=None, *, tolerate_incomplete_last_line=False):
        self.path = path
        self.offsets = {} if offsets is None else offsets
        self.end = os.path.getsize(path) if os.path.exists(path) else 0
        if offsets is not None or not os.path.exists(path):
            return
        with open(path, "rb") as stream:
            line_no = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                line_no += 1
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    if tolerate_incomplete_last_line and not stream.readline():
                        print(f"[resume] Ignore incomplete last line at {path}:{line_no}: {error}")
                        break
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {error}") from error
                if "sample_idx" not in record:
                    raise ValueError(f"Missing sample_idx in resume file {path}:{line_no}")
                self.offsets[int(record["sample_idx"])] = offset

    def __getitem__(self, key):
        offset = self.offsets[key]
        with open(self.path, "rb") as stream:
            stream.seek(offset)
            return json.loads(stream.readline().decode("utf-8"))

    def __setitem__(self, key, value):
        self.offsets[key] = self.end
        self.end = os.path.getsize(self.path)

    def __delitem__(self, key):
        del self.offsets[key]

    def __iter__(self):
        return iter(self.offsets)

    def __len__(self):
        return len(self.offsets)

    def __contains__(self, key):
        return key in self.offsets


def read_jsonl_records_by_sample_idx(
    path: str,
    *,
    tolerate_incomplete_last_line: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """读取 JSONL，并按 sample_idx 保留每个样本最后一条记录。"""
    records: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for zero_based_line_no, raw_line in enumerate(lines):
        line_no = zero_based_line_no + 1
        line = raw_line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            is_last_line = zero_based_line_no == len(lines) - 1
            if tolerate_incomplete_last_line and is_last_line:
                # JSONL 每条记录均在 append_jsonl 中 fsync；这里只为极端情况下
                # 最后一行写到一半便异常退出提供兜底。
                print(
                    f"[resume] Ignore incomplete last line at {path}:{line_no}: {e}"
                )
                break
            raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e

        if "sample_idx" not in record:
            raise ValueError(
                f"Missing sample_idx in resume file {path}:{line_no}"
            )

        sample_idx = int(record["sample_idx"])
        records[sample_idx] = record

    return records


def atomic_rewrite_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    """原子重写 JSONL，避免整理断点文件时再次留下半写文件。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    tmp_path = path + ".rewrite_tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def record_matches_input_sample(
    record: Dict[str, Any],
    sample_idx: int,
    sample: Dict[str, Any],
) -> bool:
    """检查断点记录是否仍对应当前输入文件中的同一条样本。"""
    if int(record.get("sample_idx", -1)) != sample_idx:
        return False

    if record.get("problem") != sample.get("problem"):
        return False

    # temp 文件保存 unique_id；最终输出旧格式没有该字段，因此仅在双方都有时比较。
    expected_unique_id = sample.get("unique_id", str(sample_idx))
    if "unique_id" in record and record.get("unique_id") != expected_unique_id:
        return False

    return True


def compact_and_validate_temp_records(
    temp_path: str,
    data: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """读取、校验并整理生成阶段的断点文件。"""
    records = _GenerationRecordIndex(
        temp_path,
        tolerate_incomplete_last_line=True,
    )

    valid_offsets = {}
    for sample_idx in records:
        if sample_idx < 0 or sample_idx >= len(data):
            # 可能是上次使用了更大的 --limit；本次只保留当前数据范围。
            continue

        if not record_matches_input_sample(records[sample_idx], sample_idx, data[sample_idx]):
            raise ValueError(
                "Resume temp file does not match the current input dataset at "
                f"sample_idx={sample_idx}. Please use --overwrite to restart, or "
                "use the same --input_json/--limit as the previous run."
            )

        valid_offsets[sample_idx] = records.offsets[sample_idx]

    if os.path.exists(temp_path):
        offsets = {}
        tmp_path = temp_path + ".rewrite_tmp"
        os.makedirs(os.path.dirname(os.path.abspath(temp_path)), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as stream:
            for idx in sorted(valid_offsets):
                offsets[idx] = stream.tell()
                stream.write(json.dumps(records[idx], ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, temp_path)
        valid_offsets = offsets

    return _GenerationRecordIndex(temp_path, valid_offsets)


def synchronize_final_outputs_for_resume(
    summary_path: str,
    step_path: str,
    data: List[Dict[str, Any]],
) -> set[int]:
    """
    对齐 summary/steps 两个最终输出文件。

    每个样本只有同时存在于两个文件中才视为第二阶段已完成。若上次恰好在写完
    summary 后、写 step 前中断，恢复时会丢弃该单边记录并重新处理该样本。
    同时去除重复 sample_idx，只保留每个文件中最后一条有效记录。
    """
    summary_records = _GenerationRecordIndex(
        summary_path,
        tolerate_incomplete_last_line=True,
    )
    step_records = _GenerationRecordIndex(
        step_path,
        tolerate_incomplete_last_line=True,
    )

    completed_indices: set[int] = set(summary_records).intersection(step_records)
    valid_completed: List[int] = []

    for sample_idx in sorted(completed_indices):
        if sample_idx < 0 or sample_idx >= len(data):
            continue

        if not record_matches_input_sample(
            summary_records[sample_idx], sample_idx, data[sample_idx]
        ):
            raise ValueError(
                "Existing summary_jsonl does not match the current input dataset at "
                f"sample_idx={sample_idx}. Use --overwrite to restart."
            )
        if not record_matches_input_sample(
            step_records[sample_idx], sample_idx, data[sample_idx]
        ):
            raise ValueError(
                "Existing step_jsonl does not match the current input dataset at "
                f"sample_idx={sample_idx}. Use --overwrite to restart."
            )

        valid_completed.append(sample_idx)

    # 在 resume 模式下进行一次原子整理，清除：
    # 1. summary/steps 之间不成对的半完成记录；
    # 2. 同一样本的重复记录；
    # 3. 当前 --limit 范围之外的旧记录。
    if os.path.exists(summary_path) or os.path.exists(step_path):
        atomic_rewrite_jsonl(
            summary_path,
            (summary_records[idx] for idx in valid_completed),
        )
        atomic_rewrite_jsonl(
            step_path,
            (step_records[idx] for idx in valid_completed),
        )

    return set(valid_completed)


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


def generation_dp_worker(
    dp_rank: int,
    gpu_group: str,
    args: argparse.Namespace,
    assigned_samples: List[Tuple[int, Dict[str, Any]]],
    result_queue: Any,
) -> None:
    """一个 DP worker；其可见 GPU 组内部由 vLLM 执行 TP。"""
    llm = None
    try:
        from vllm import LLM

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.vllm_enforce_eager,
            max_model_len=args.max_model_len,
        )

        tokenizer = llm.get_tokenizer()
        sampling_params = make_sampling_params(args)

        for sample_idx, sample in assigned_samples:
            problem = sample["problem"]
            answer = sample.get("answer")
            difficulty = sample.get("difficulty")
            unique_id = sample.get("unique_id", str(sample_idx))

            prompt = problem + PROMPT_SUFFIX

            try:
                request_outputs = llm.generate([prompt], sampling_params)
                request_output = request_outputs[0]
                output = request_output.outputs[0]

                token_ids = [int(x) for x in output.token_ids]

                output_text_from_tokens = tokenizer.decode(
                    token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ) if token_ids else ""

                full_output = output_text_from_tokens

                output_logprobs = output.logprobs
                if output_logprobs is None:
                    raise RuntimeError(
                        "output.logprobs is None. Please make sure "
                        "SamplingParams(logprobs=1) is set."
                    )

                token_logprobs: List[Optional[float]] = []
                for token_id, logprob_dict in zip(token_ids, output_logprobs):
                    token_logprobs.append(
                        get_sampled_token_logprob(logprob_dict, token_id)
                    )

                prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
                if prompt_token_ids is None:
                    prompt_token_ids = tokenizer.encode(
                        prompt,
                        add_special_tokens=False,
                    )

                temp_record = {
                    "status": "ok",
                    "sample_idx": sample_idx,
                    "problem": problem,
                    "gold_answer": answer,
                    "difficulty": difficulty,
                    "unique_id": unique_id,
                    "full_output": full_output,
                    "prompt_token_ids": [int(x) for x in prompt_token_ids],
                    "response_token_ids": token_ids,
                    "token_logprobs": token_logprobs,
                    "finish_reason": getattr(output, "finish_reason", None),
                }

            except Exception as e:
                temp_record = {
                    "status": "error",
                    "sample_idx": sample_idx,
                    "problem": problem,
                    "gold_answer": answer,
                    "difficulty": difficulty,
                    "unique_id": unique_id,
                    "error": repr(e),
                }

            result_queue.put(("record", dp_rank, temp_record))

    except BaseException:
        result_queue.put(("worker_error", dp_rank, traceback.format_exc()))
    finally:
        if llm is not None:
            release_vllm(llm)
        result_queue.put(("worker_done", dp_rank, gpu_group))


def run_data_parallel_generation(
    args: argparse.Namespace,
    data: List[Dict[str, Any]],
    pending_generation_indices: List[int],
    gpu_groups: List[str],
    temp_generation_jsonl: str,
    temp_records_by_idx: Dict[int, Dict[str, Any]],
) -> None:
    """按 GPU 组启动多个 DP worker，并由主进程统一写生成断点。"""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=max(2, len(gpu_groups) * 2))

    shards: List[List[Tuple[int, Dict[str, Any]]]] = [
        [] for _ in gpu_groups
    ]
    for position, sample_idx in enumerate(pending_generation_indices):
        shards[position % len(gpu_groups)].append((sample_idx, data[sample_idx]))

    processes: List[Any] = []
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    try:
        for dp_rank, (gpu_group, assigned_samples) in enumerate(
            zip(gpu_groups, shards)
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
            process = ctx.Process(
                target=generation_dp_worker,
                args=(
                    dp_rank,
                    gpu_group,
                    args,
                    assigned_samples,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)
    finally:
        if original_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices

    worker_errors: List[str] = []
    finished_workers: set[int] = set()

    with tqdm(
        total=len(pending_generation_indices),
        desc="Generating",
    ) as progress_bar:
        while len(finished_workers) < len(processes):
            try:
                message_type, dp_rank, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue

            if message_type == "record":
                temp_record = payload
                append_jsonl(temp_generation_jsonl, temp_record)
                temp_records_by_idx[int(temp_record["sample_idx"])] = temp_record
                progress_bar.update(1)
            elif message_type == "worker_error":
                worker_errors.append(
                    f"DP worker {dp_rank} failed:\n{payload}"
                )
            elif message_type == "worker_done":
                finished_workers.add(int(dp_rank))

    for process in processes:
        process.join()

    abnormal_exit_workers = [
        dp_rank
        for dp_rank, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if abnormal_exit_workers:
        worker_errors.append(
            "DP workers exited abnormally: "
            f"{abnormal_exit_workers}"
        )

    if worker_errors:
        raise RuntimeError(
            "Data-parallel generation failed. Completed records remain in the "
            "temporary checkpoint and can be reused with --resume.\n"
            + "\n".join(worker_errors)
        )



def find_free_tcp_port() -> int:
    """为一个独立的 TP 进程组申请本机空闲端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_selected_temp_records(
    path: str,
    selected_indices: List[int],
    record_offsets=None,
) -> Dict[int, Dict[str, Any]]:
    """从生成阶段临时文件中读取指定 sample_idx 的最后一条记录。"""
    indexed = _GenerationRecordIndex(path, record_offsets)
    records = _GenerationRecordIndex(
        path, {idx: indexed.offsets[idx] for idx in selected_indices if idx in indexed},
    )

    missing = [idx for idx in selected_indices if idx not in records]
    if missing:
        raise RuntimeError(
            "Missing generation checkpoint records for entropy TP worker: "
            f"{missing[:20]}"
        )

    return records


def _put_phase_result(result_queue, args, message):
    slots = getattr(args, "_phase_result_slots", None)
    if slots is not None:
        slots.acquire()
    result_queue.put(message)


def entropy_tp_worker(
    dp_rank: int,
    tp_rank: int,
    gpu_group: str,
    args: argparse.Namespace,
    assigned_indices: List[int],
    temp_generation_jsonl: str,
    result_queue: Any,
) -> None:
    """第二阶段的一个 TP rank；同一 DP 组中的所有 rank 协同执行 forward。"""
    entropy_model = None
    distributed_initialized = False

    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(tp_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        distributed_initialized = True

        temp_records = read_selected_temp_records(
            path=temp_generation_jsonl,
            selected_indices=assigned_indices,
            record_offsets=getattr(args, "_generation_record_offsets", None),
        )

        need_entropy_model = any(
            temp_records[idx].get("status") == "ok"
            for idx in assigned_indices
        )
        if need_entropy_model:
            if args.tensor_parallel_size == 1:
                entropy_model = load_entropy_model_single_gpu(args)
            else:
                entropy_model = load_entropy_model_tensor_parallel(args)

        tokenizer = load_tokenizer_without_vllm(args) if tp_rank == 0 else None

        for sample_idx in assigned_indices:
            temp_record = temp_records[sample_idx]
            problem = temp_record["problem"]
            answer = temp_record.get("gold_answer")
            difficulty = temp_record.get("difficulty")

            if temp_record.get("status") != "ok":
                if tp_rank == 0:
                    summary_record, step_record = build_generation_error_records(
                        sample_idx=sample_idx,
                        problem=problem,
                        answer=answer,
                        difficulty=difficulty,
                        error=temp_record.get("error", "Unknown generation error"),
                    )
                    _put_phase_result(result_queue, args,
                        (
                            "record",
                            dp_rank,
                            (sample_idx, summary_record, step_record),
                        )
                    )
                dist.barrier()
                continue

            local_error: Optional[str] = None
            response_entropies: Optional[List[float]] = None

            try:
                if entropy_model is None:
                    raise RuntimeError("Entropy model was not loaded")

                token_ids = [int(x) for x in temp_record["response_token_ids"]]
                prompt_token_ids = [
                    int(x) for x in temp_record["prompt_token_ids"]
                ]

                response_entropies = compute_response_token_entropies_exact(
                    model=entropy_model,
                    prompt_token_ids=prompt_token_ids,
                    response_token_ids=token_ids,
                    chunk_size=args.entropy_chunk_size,
                )
            except Exception:
                local_error = traceback.format_exc()

            gathered_errors: List[Optional[str]] = [
                None for _ in range(args.tensor_parallel_size)
            ]
            dist.all_gather_object(gathered_errors, local_error)
            first_error = next(
                (error for error in gathered_errors if error is not None),
                None,
            )

            if first_error is not None:
                if tp_rank == 0:
                    summary_record, step_record = build_generation_error_records(
                        sample_idx=sample_idx,
                        problem=problem,
                        answer=answer,
                        difficulty=difficulty,
                        error=first_error,
                    )
                    _put_phase_result(result_queue, args,
                        (
                            "record",
                            dp_rank,
                            (sample_idx, summary_record, step_record),
                        )
                    )
                dist.barrier()
                continue

            if tp_rank == 0:
                try:
                    if tokenizer is None or response_entropies is None:
                        raise RuntimeError(
                            "Tokenizer or entropy result is unavailable on TP rank 0"
                        )

                    token_ids = [int(x) for x in temp_record["response_token_ids"]]
                    token_logprobs = temp_record["token_logprobs"]
                    full_output = temp_record["full_output"]

                    output_text_from_tokens, token_pieces = (
                        build_token_pieces_with_char_spans(
                            token_ids=token_ids,
                            tokenizer=tokenizer,
                        )
                    )

                    if output_text_from_tokens != full_output:
                        full_output = output_text_from_tokens

                    step_conf_list, step_records = compute_step_confidences(
                        text=full_output,
                        token_pieces=token_pieces,
                        token_logprobs=token_logprobs,
                        sep="\n\n",
                    )

                    boundary_token_positions, _ = _select_phase_boundaries(
                        response_entropies, token_pieces, full_output, args,
                    )

                    phase_ranges, phase_frontiers = build_phase_ranges_and_frontiers(
                        text=full_output,
                        token_pieces=token_pieces,
                        boundary_token_positions=boundary_token_positions,
                        sep="\n\n",
                    )

                    if len(phase_frontiers) != len(step_records):
                        raise RuntimeError(
                            "Phase frontier/step alignment failed: "
                            f"{len(phase_frontiers)} frontiers vs "
                            f"{len(step_records)} steps"
                        )

                    for record, phase_frontier in zip(
                        step_records,
                        phase_frontiers,
                    ):
                        record["phase_frontier"] = int(phase_frontier)

                    summary_record = {
                        "sample_idx": sample_idx,
                        "problem": problem,
                        "gold_answer": answer,
                        "difficulty": difficulty,
                        "full_output": full_output,
                        "step_avg_confidences": step_conf_list,
                        "num_steps": len(step_conf_list),
                        "num_generated_tokens": len(token_ids),
                        "finish_reason": temp_record.get("finish_reason"),
                        "phase_ranges": phase_ranges,
                    }

                    step_record = {
                        "sample_idx": sample_idx,
                        "problem": problem,
                        "gold_answer": answer,
                        "difficulty": difficulty,
                        "steps": step_records,
                    }

                except Exception as e:
                    summary_record, step_record = build_generation_error_records(
                        sample_idx=sample_idx,
                        problem=problem,
                        answer=answer,
                        difficulty=difficulty,
                        error=repr(e),
                    )

                _put_phase_result(result_queue, args,
                    (
                        "record",
                        dp_rank,
                        (sample_idx, summary_record, step_record),
                    )
                )

            dist.barrier()

    except BaseException:
        result_queue.put(
            (
                "worker_error",
                dp_rank,
                f"TP rank {tp_rank} on GPUs {gpu_group} failed:\n"
                + traceback.format_exc(),
            )
        )
    finally:
        if entropy_model is not None:
            del entropy_model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if distributed_initialized:
            try:
                import torch.distributed as dist

                dist.destroy_process_group()
            except Exception:
                pass

        result_queue.put(
            (
                "worker_done",
                dp_rank,
                (tp_rank, gpu_group),
            )
        )


def run_data_parallel_entropy_phase(
    args: argparse.Namespace,
    pending_phase_indices: List[int],
    gpu_groups: List[str],
    temp_generation_jsonl: str,
    completed_final_indices: set[int],
) -> None:
    """第二阶段按与生成阶段相同的 DP 组运行，并在组内使用 Transformers TP。"""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(
        maxsize=max(2, len(gpu_groups) * args.tensor_parallel_size * 2)
    )
    # At most two completed records per DP group can await ordered writing.
    # This bounds the reorder buffer even when another group has a long sample.
    result_slots = [ctx.BoundedSemaphore(2) for _ in gpu_groups]
    generation_index = _GenerationRecordIndex(temp_generation_jsonl)

    shards: List[List[int]] = [[] for _ in gpu_groups]
    for position, sample_idx in enumerate(pending_phase_indices):
        shards[position % len(gpu_groups)].append(sample_idx)

    processes: List[Tuple[int, int, Any]] = []
    original_environment = {
        key: os.environ.get(key)
        for key in [
            "CUDA_VISIBLE_DEVICES",
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
        ]
    }

    try:
        for dp_rank, (gpu_group, assigned_indices) in enumerate(
            zip(gpu_groups, shards)
        ):
            master_port = find_free_tcp_port()
            worker_args = argparse.Namespace(**vars(args))
            worker_args._generation_record_offsets = {
                idx: generation_index.offsets[idx] for idx in assigned_indices
            }
            worker_args._phase_result_slots = result_slots[dp_rank]

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
                        dp_rank,
                        tp_rank,
                        gpu_group,
                        worker_args,
                        assigned_indices,
                        temp_generation_jsonl,
                        result_queue,
                    ),
                )
                process.start()
                processes.append((dp_rank, tp_rank, process))
    finally:
        for key, value in original_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    worker_errors: List[str] = []
    finished_workers: set[Tuple[int, int]] = set()
    buffered_records = {}
    next_write_position = 0

    with tqdm(
        total=len(pending_phase_indices),
        desc="Computing entropy phases",
    ) as progress_bar:
        while len(finished_workers) < len(processes):
            try:
                message_type, dp_rank, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                abnormal_finished = [
                    (worker_dp_rank, worker_tp_rank)
                    for worker_dp_rank, worker_tp_rank, process in processes
                    if not process.is_alive()
                    and process.exitcode not in {None, 0}
                ]
                if abnormal_finished:
                    for _, _, process in processes:
                        if process.is_alive():
                            process.terminate()
                    break
                if all(not process.is_alive() for _, _, process in processes):
                    break
                continue

            if message_type == "record":
                sample_idx, summary_record, step_record = payload
                buffered_records[int(sample_idx)] = (dp_rank, summary_record, step_record)

                while next_write_position < len(pending_phase_indices):
                    next_sample_idx = pending_phase_indices[next_write_position]
                    if next_sample_idx not in buffered_records:
                        break

                    record_dp_rank, next_summary_record, next_step_record = buffered_records.pop(
                        next_sample_idx
                    )
                    append_jsonl(args.summary_jsonl, next_summary_record)
                    append_jsonl(args.step_jsonl, next_step_record)
                    completed_final_indices.add(next_sample_idx)
                    result_slots[record_dp_rank].release()
                    progress_bar.update(1)
                    next_write_position += 1

            elif message_type == "worker_error":
                worker_errors.append(str(payload))
                for _, _, process in processes:
                    if process.is_alive():
                        process.terminate()
                break
            elif message_type == "worker_done":
                tp_rank, _ = payload
                finished_workers.add((int(dp_rank), int(tp_rank)))

    for _, _, process in processes:
        process.join()

    abnormal_exit_workers = [
        (dp_rank, tp_rank)
        for dp_rank, tp_rank, process in processes
        if process.exitcode != 0
    ]
    if abnormal_exit_workers:
        worker_errors.append(
            "Entropy TP workers exited abnormally: "
            f"{abnormal_exit_workers}"
        )

    missing_results = [
        sample_idx
        for sample_idx in pending_phase_indices
        if sample_idx not in completed_final_indices
    ]
    if missing_results:
        worker_errors.append(
            "Entropy DP+TP stage did not return results for sample_idx: "
            f"{missing_results[:20]}"
        )

    if worker_errors:
        raise RuntimeError(
            "Data-parallel tensor-parallel entropy forward failed. Completed "
            "final records remain available for --resume.\n"
            + "\n".join(worker_errors)
        )

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--summary_jsonl", type=str, default="math500_conf_summary.jsonl")
    parser.add_argument("--step_jsonl", type=str, default="math500_conf_steps.jsonl")

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
            "指定每个 DP worker 使用的 GPU；分号分隔 DP 组，逗号分隔组内 TP GPU。"
            "例如 --gpus '0,1;2,3' 表示 2 个 DP worker，每个 worker 做 TP=2。"
        ),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--vllm_enforce_eager",
        action="store_true",
        help="禁用 vLLM CUDA Graph，强制 eager 执行。",
    )

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--num_input_parts",
        type=int,
        default=1,
        help="将输入数据按原顺序连续等分为 K 份，默认 1。",
    )
    parser.add_argument(
        "--input_part_id",
        type=int,
        default=0,
        help="只处理第 i 份，i 从 0 开始，默认 0。",
    )

    # 新增：目标 K 个 segment/phase。--K 与 --num_segments 完全等价。
    parser.add_argument(
        "--num_segments",
        "--K",
        dest="num_segments",
        type=int,
        default=8,
        help="目标 phase/segment 数 K，默认 8。实际 phase 数可能因多个边界落在同一 step 而小于 K。",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=1.2,
        help="高熵候选 token 阈值 tau（单位：nats），默认 1.2。",
    )
    add_phase_partition_arguments(parser)
    parser.add_argument(
        "--entropy_chunk_size",
        type=int,
        default=16,
        help="Transformers 精确熵前向的 token chunk 大小，越小越省峰值显存，默认 16。",
    )
    parser.add_argument(
        "--entropy_model_path",
        type=str,
        default=None,
        help="用于精确 entropy forward 的模型路径；默认与 --model_path 相同。",
    )
    parser.add_argument(
        "--entropy_device_map",
        type=str,
        default="auto",
        help="Transformers entropy 模型的 device_map，默认 auto；设为 none 可禁用自动分片。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果设置，则删除已有最终输出及生成阶段临时文件，并从头运行。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "断点续跑：复用 .entropy_phase_generation_tmp.jsonl 中已生成的完整输出，"
            "并跳过 summary_jsonl 与 step_jsonl 中已成对完成的样本。"
        ),
    )

    args = parser.parse_args()

    if args.phase_partition_algorithm == "adaptive":
        try:
            validate_phase_partition_options(args)
        except ValueError as error:
            parser.error(str(error))
    else:
        # Preserve the legacy validation and defaults in original mode.
        if args.num_segments < 1:
            parser.error("--num_segments/--K must be at least 1")
        if args.entropy_threshold < 0:
            parser.error("--entropy_threshold must be non-negative")
    if args.entropy_chunk_size < 1:
        parser.error("--entropy_chunk_size must be at least 1")
    if args.num_input_parts < 1:
        parser.error("--num_input_parts must be at least 1")
    if args.input_part_id < 0 or args.input_part_id >= args.num_input_parts:
        parser.error("--input_part_id must satisfy 0 <= i < --num_input_parts")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume cannot be used together")

    try:
        gpu_groups = parse_gpu_groups(
            args.gpus,
            args.tensor_parallel_size,
        )
    except ValueError as e:
        parser.error(str(e))

    entropy_gpu_groups = gpu_groups
    if not entropy_gpu_groups and args.tensor_parallel_size > 1:
        visible_gpu_ids = [
            gpu_id.strip()
            for gpu_id in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if gpu_id.strip()
        ]
        if visible_gpu_ids:
            if len(visible_gpu_ids) < args.tensor_parallel_size:
                parser.error(
                    "CUDA_VISIBLE_DEVICES contains fewer GPUs than "
                    "--tensor_parallel_size"
                )
            entropy_gpu_groups = [
                ",".join(visible_gpu_ids[: args.tensor_parallel_size])
            ]
        else:
            entropy_gpu_groups = [
                ",".join(str(i) for i in range(args.tensor_parallel_size))
            ]

    data = load_json_array(args.input_json)
    if args.limit is not None:
        data = data[: args.limit]

    num_samples = len(data)
    base_size, remainder = divmod(num_samples, args.num_input_parts)
    split_start = args.input_part_id * base_size + min(args.input_part_id, remainder)
    split_end = split_start + base_size + (
        1 if args.input_part_id < remainder else 0
    )
    data = data[split_start:split_end]

    # 生成阶段的中间文件。所有最终结果写完后才会删除。
    temp_generation_jsonl = args.summary_jsonl + ".entropy_phase_generation_tmp.jsonl"

    if args.overwrite:
        for path in [
            args.summary_jsonl,
            args.step_jsonl,
            temp_generation_jsonl,
        ]:
            if os.path.exists(path):
                os.remove(path)

    if args.resume:
        completed_final_indices = synchronize_final_outputs_for_resume(
            summary_path=args.summary_jsonl,
            step_path=args.step_jsonl,
            data=data,
        )
        temp_records_by_idx = compact_and_validate_temp_records(
            temp_path=temp_generation_jsonl,
            data=data,
        )

        print(
            "[resume] Found "
            f"{len(completed_final_indices)}/{len(data)} samples completed in both "
            "final JSONL files."
        )
        print(
            "[resume] Found "
            f"{len(temp_records_by_idx)} generated samples in temporary JSONL."
        )
    else:
        completed_final_indices: set[int] = set()
        temp_records_by_idx: Dict[int, Dict[str, Any]] = {}

        # 保持原先的非 resume 行为：每次重新生成临时文件。
        if os.path.exists(temp_generation_jsonl):
            os.remove(temp_generation_jsonl)
        temp_records_by_idx = _GenerationRecordIndex(temp_generation_jsonl, {})

    if args.resume and len(completed_final_indices) == len(data):
        if os.path.exists(temp_generation_jsonl):
            os.remove(temp_generation_jsonl)
        print("[resume] All samples are already complete. Nothing to do.")
        return

    pending_generation_indices = [
        sample_idx
        for sample_idx in range(len(data))
        if sample_idx not in completed_final_indices
        and sample_idx not in temp_records_by_idx
    ]

    llm = None
    tokenizer = None

    # 第一阶段：只为尚未生成且尚未完成最终输出的样本启动 vLLM。
    if pending_generation_indices:
        print(
            f"[resume] Need to generate {len(pending_generation_indices)} "
            f"remaining samples."
            if args.resume
            else f"Generating {len(pending_generation_indices)} samples."
        )

        if gpu_groups:
            run_data_parallel_generation(
                args=args,
                data=data,
                pending_generation_indices=pending_generation_indices,
                gpu_groups=gpu_groups,
                temp_generation_jsonl=temp_generation_jsonl,
                temp_records_by_idx=temp_records_by_idx,
            )
        else:
            from vllm import LLM

            llm = LLM(
                model=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                enforce_eager=args.vllm_enforce_eager,
                max_model_len=args.max_model_len,
            )

            tokenizer = llm.get_tokenizer()
            sampling_params = make_sampling_params(args)

            for sample_idx in tqdm(pending_generation_indices, desc="Generating"):
                sample = data[sample_idx]
                problem = sample["problem"]
                answer = sample.get("answer")
                difficulty = sample.get("difficulty")
                unique_id = sample.get("unique_id", str(sample_idx))

                prompt = problem + PROMPT_SUFFIX

                try:
                    request_outputs = llm.generate([prompt], sampling_params)
                    request_output = request_outputs[0]
                    output = request_output.outputs[0]

                    token_ids = [int(x) for x in output.token_ids]

                    output_text_from_tokens = tokenizer.decode(
                        token_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ) if token_ids else ""

                    # vLLM 的 output.text 通常和 decode(token_ids) 一致。
                    # 这里优先用 decode 后的文本，以保证 char span 对齐。
                    full_output = output_text_from_tokens

                    output_logprobs = output.logprobs
                    if output_logprobs is None:
                        raise RuntimeError(
                            "output.logprobs is None. Please make sure "
                            "SamplingParams(logprobs=1) is set."
                        )

                    token_logprobs: List[Optional[float]] = []
                    for token_id, logprob_dict in zip(token_ids, output_logprobs):
                        token_logprobs.append(
                            get_sampled_token_logprob(logprob_dict, token_id)
                        )

                    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
                    if prompt_token_ids is None:
                        # 兼容旧版 vLLM：尽量用同一 tokenizer 重建 prompt ids。
                        prompt_token_ids = tokenizer.encode(
                            prompt,
                            add_special_tokens=False,
                        )

                    temp_record = {
                        "status": "ok",
                        "sample_idx": sample_idx,
                        "problem": problem,
                        "gold_answer": answer,
                        "difficulty": difficulty,
                        "unique_id": unique_id,
                        "full_output": full_output,
                        "prompt_token_ids": [int(x) for x in prompt_token_ids],
                        "response_token_ids": token_ids,
                        "token_logprobs": token_logprobs,
                        "finish_reason": getattr(output, "finish_reason", None),
                    }

                except Exception as e:
                    temp_record = {
                        "status": "error",
                        "sample_idx": sample_idx,
                        "problem": problem,
                        "gold_answer": answer,
                        "difficulty": difficulty,
                        "unique_id": unique_id,
                        "error": repr(e),
                    }

                append_jsonl(temp_generation_jsonl, temp_record)
                temp_records_by_idx[sample_idx] = temp_record

            # 精确 full-vocabulary entropy 需要 Transformers forward。
            # 先释放 vLLM，避免两份模型同时占据显存。
            release_vllm(llm)
            llm = None
            gc.collect()

    elif args.resume:
        print("[resume] Generation stage is already complete; skip loading vLLM.")

    # 若此次完全跳过了 vLLM，仍需 tokenizer 来恢复 token 字符区间。
    if tokenizer is None and not entropy_gpu_groups:
        tokenizer = load_tokenizer_without_vllm(args)

    # 重新读取并整理 temp，确保刚刚追加的记录也只有一份且顺序稳定。
    temp_records_by_idx = compact_and_validate_temp_records(
        temp_path=temp_generation_jsonl,
        data=data,
    )

    pending_phase_indices = [
        sample_idx
        for sample_idx in range(len(data))
        if sample_idx not in completed_final_indices
    ]

    missing_temp_indices = [
        idx for idx in pending_phase_indices if idx not in temp_records_by_idx
    ]
    if missing_temp_indices:
        raise RuntimeError(
            "Missing generation checkpoint records for sample_idx: "
            f"{missing_temp_indices[:20]}. Run again with --resume; if the problem "
            "persists, use --overwrite to restart."
        )

    if entropy_gpu_groups and pending_phase_indices:
        run_data_parallel_entropy_phase(
            args=args,
            pending_phase_indices=pending_phase_indices,
            gpu_groups=entropy_gpu_groups,
            temp_generation_jsonl=temp_generation_jsonl,
            completed_final_indices=completed_final_indices,
        )
    else:
        need_entropy_model = any(
            temp_records_by_idx[idx].get("status") == "ok"
            for idx in pending_phase_indices
        )

        entropy_model = None
        if need_entropy_model:
            entropy_model = load_entropy_model(args)

        # 第二阶段：跳过已同时写入 summary/steps 的样本，其余样本逐条计算并落盘。
        for sample_idx in tqdm(
            pending_phase_indices,
            total=len(pending_phase_indices),
            desc="Computing entropy phases",
        ):
            temp_record = temp_records_by_idx[sample_idx]
            problem = temp_record["problem"]
            answer = temp_record.get("gold_answer")
            difficulty = temp_record.get("difficulty")

            if temp_record.get("status") != "ok":
                summary_record, step_record = build_generation_error_records(
                    sample_idx=sample_idx,
                    problem=problem,
                    answer=answer,
                    difficulty=difficulty,
                    error=temp_record.get("error", "Unknown generation error"),
                )
            else:
                try:
                    if entropy_model is None:
                        raise RuntimeError("Entropy model was not loaded")

                    token_ids = [int(x) for x in temp_record["response_token_ids"]]
                    prompt_token_ids = [
                        int(x) for x in temp_record["prompt_token_ids"]
                    ]
                    token_logprobs = temp_record["token_logprobs"]
                    full_output = temp_record["full_output"]

                    output_text_from_tokens, token_pieces = (
                        build_token_pieces_with_char_spans(
                            token_ids=token_ids,
                            tokenizer=tokenizer,
                        )
                    )

                    # 中间文件已保存 full_output；再次 decode 做一致性校验并沿用 decode 文本。
                    if output_text_from_tokens != full_output:
                        full_output = output_text_from_tokens

                    step_conf_list, step_records = compute_step_confidences(
                        text=full_output,
                        token_pieces=token_pieces,
                        token_logprobs=token_logprobs,
                        sep="\n\n",
                    )

                    response_entropies = compute_response_token_entropies_exact(
                        model=entropy_model,
                        prompt_token_ids=prompt_token_ids,
                        response_token_ids=token_ids,
                        chunk_size=args.entropy_chunk_size,
                    )

                    boundary_token_positions, _ = _select_phase_boundaries(
                        response_entropies, token_pieces, full_output, args,
                    )

                    phase_ranges, phase_frontiers = build_phase_ranges_and_frontiers(
                        text=full_output,
                        token_pieces=token_pieces,
                        boundary_token_positions=boundary_token_positions,
                        sep="\n\n",
                    )

                    if len(phase_frontiers) != len(step_records):
                        raise RuntimeError(
                            "Phase frontier/step alignment failed: "
                            f"{len(phase_frontiers)} frontiers vs "
                            f"{len(step_records)} steps"
                        )

                    # 仅按要求新增 phase_frontier，其余 step 元素字段保持不变。
                    for record, phase_frontier in zip(
                        step_records,
                        phase_frontiers,
                    ):
                        record["phase_frontier"] = int(phase_frontier)

                    summary_record = {
                        "sample_idx": sample_idx,
                        "problem": problem,
                        "gold_answer": answer,
                        "difficulty": difficulty,
                        "full_output": full_output,
                        "step_avg_confidences": step_conf_list,
                        "num_steps": len(step_conf_list),
                        "num_generated_tokens": len(token_ids),
                        "finish_reason": temp_record.get("finish_reason"),
                        "phase_ranges": phase_ranges,
                    }

                    step_record = {
                        "sample_idx": sample_idx,
                        "problem": problem,
                        "gold_answer": answer,
                        "difficulty": difficulty,
                        "steps": step_records,
                    }

                except Exception as e:
                    # 与原逻辑一致：单条样本失败时写错误记录，不中断整个数据集。
                    summary_record, step_record = build_generation_error_records(
                        sample_idx=sample_idx,
                        problem=problem,
                        answer=answer,
                        difficulty=difficulty,
                        error=repr(e),
                    )

            # 仍保持逐样本落盘。若恰好在两个 append 之间中断，下一次 --resume
            # 会由 synchronize_final_outputs_for_resume 丢弃单边记录并重算该样本。
            append_jsonl(args.summary_jsonl, summary_record)
            append_jsonl(args.step_jsonl, step_record)
            completed_final_indices.add(sample_idx)

        if entropy_model is not None:
            del entropy_model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # 仅当两个最终文件都完整覆盖当前 data 时，才删除生成阶段中间文件。
    final_completed_indices = synchronize_final_outputs_for_resume(
        summary_path=args.summary_jsonl,
        step_path=args.step_jsonl,
        data=data,
    )

    if len(final_completed_indices) == len(data):
        if os.path.exists(temp_generation_jsonl):
            os.remove(temp_generation_jsonl)
        print(f"Completed all {len(data)} samples; temporary checkpoint removed.")
    else:
        print(
            "Run stopped with incomplete outputs: "
            f"{len(final_completed_indices)}/{len(data)} completed. "
            "Keep the temporary checkpoint and rerun with --resume."
        )


if __name__ == "__main__":
    main()
