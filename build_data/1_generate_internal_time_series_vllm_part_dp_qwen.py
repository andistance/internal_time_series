#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用 vLLM 对 MATH-500 风格 JSON 数据集进行批量推理，
按 "\n\n" 切分 reasoning step，
并保存每个 step 的平均 token confidence。

并行层次：
- DP：启动多个独立 worker，每个 worker 处理不同数据 shard；
- TP：每个 worker 内将一个模型切分到多张 GPU；
- Batch：每个 worker 一次向 llm.generate() 提交多个 prompt。

支持：
1. Tensor Parallel, TP:
   通过 --tensor_parallel_size 控制。
   TP=1 表示不做张量并行。

2. Data Parallel, DP:
   通过 --data_parallel_size 控制。
   DP=1 表示不做数据并行。
   DP>1 时，主进程会自动启动多个 worker，
   每个 worker 处理数据集的一个 shard，
   最后自动合并结果。

3. Input Part Split:
   通过 --num_input_parts 和 --input_part_id 控制。
   先将输入 JSON 数据按原始顺序等分成 num_input_parts 个连续部分，
   再只推理 input_part_id 指定的那一部分。
   input_part_id 从 0 开始。

4. Batch Inference:
   通过 --batch_size 控制每个 DP worker 每次提交给 vLLM 的 prompt 数量。
   通过 --max_num_seqs 控制 vLLM 同时活跃的最大序列数；
   若不显式指定，则自动等于 batch_size。

5. Qwen3 Thinking / Non-thinking:
   通过 --enable_thinking true/false 控制。
   true 表示启用 thinking mode；
   false 表示启用 non-thinking mode。

6. Resume / 断点续跑：
   通过 --resume 启用。
   启动时读取已有最终 JSONL 和 DP shard JSONL，
   跳过 summary 与 step 两边都已经成功写入的 sample_idx。
   默认会重试带 error 的失败样本；可通过 --resume_skip_errors 跳过失败记录。

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
   每行包含完整输出、step confidence 列表等信息。

2. step_jsonl:
   每行包含每个 step 的文本、对应平均 confidence、
   以及该 step 第一个 token 在完整生成 token 序列中的下标。
"""

import argparse
import copy
import glob
import json
import math
import os
import subprocess
import sys
import traceback
from multiprocessing import Process, set_start_method
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


PROMPT_SUFFIX = (
    "\nLet's reason step by step. Enclose the reasoning process within "
    "<think>...</think>, then summarize it and present the final answer within \\boxed{}."
)

NON_THINKING_PROMPT_SUFFIX = (
    "\nSolve the problem clearly and concisely, then present the final answer "
    "within \\boxed{}."
)


def str2bool(v: Any) -> bool:
    """解析命令行中的 true/false 字符串。"""
    if isinstance(v, bool):
        return v

    if v is None:
        raise argparse.ArgumentTypeError("Boolean value expected.")

    v = str(v).strip().lower()

    if v in {"true", "t", "1", "yes", "y"}:
        return True

    if v in {"false", "f", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Boolean value expected, but got {v}. Use true or false."
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_jsonl_many(path: str, records: List[Dict[str, Any]]) -> None:
    """
    批量追加多行 JSONL，并在当前 batch 写完后 flush + fsync。

    与逐条打开文件相比，这样可显著减少 batch 推理时的磁盘 I/O 开销；
    若程序异常退出，最多只会损失尚未写入的当前 batch。
    """
    if not records:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def select_input_part(
    indexed_data: List[Tuple[int, Dict[str, Any]]],
    num_input_parts: int,
    input_part_id: int,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], int, int]:
    """
    将输入数据按原始顺序等分为 num_input_parts 个连续部分，
    返回 input_part_id 指定的那一部分。

    input_part_id 从 0 开始。

    例如 total=10, num_input_parts=3：
        part 0: indices [0, 1, 2, 3]
        part 1: indices [4, 5, 6]
        part 2: indices [7, 8, 9]
    """
    if num_input_parts < 1:
        raise ValueError("--num_input_parts must be >= 1")

    if not (0 <= input_part_id < num_input_parts):
        raise ValueError(
            f"--input_part_id must satisfy 0 <= input_part_id < num_input_parts, "
            f"but got input_part_id={input_part_id}, num_input_parts={num_input_parts}"
        )

    total = len(indexed_data)
    base_size = total // num_input_parts
    remainder = total % num_input_parts

    start = input_part_id * base_size + min(input_part_id, remainder)
    part_size = base_size + (1 if input_part_id < remainder else 0)
    end = start + part_size

    return indexed_data[start:end], start, end


def get_sampled_token_logprob(
    logprob_dict: Optional[Dict[Any, Any]],
    token_id: int,
) -> Optional[float]:
    """
    从 vLLM 每个生成位置返回的 logprobs 字典中取出实际生成 token 的 logprob。

    vLLM 中 logprob_dict 的 value 通常是 Logprob 对象，常见字段为 .logprob。
    为了兼容不同版本，也兼容 value 本身就是 float 的情况。
    """
    if logprob_dict is None:
        return None

    key = token_id

    if key not in logprob_dict and str(token_id) in logprob_dict:
        key = str(token_id)

    if key not in logprob_dict:
        return None

    value = logprob_dict[key]

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


def make_sampling_params(args: argparse.Namespace):
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

    if args.seed is not None:
        kwargs["seed"] = args.seed

    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)



def build_prompt(tokenizer, problem: str, enable_thinking: bool) -> str:
    """为单条题目构造 Qwen3 chat-template prompt。"""
    suffix = PROMPT_SUFFIX if enable_thinking else NON_THINKING_PROMPT_SUFFIX
    messages = [
        {
            "role": "user",
            "content": problem + suffix,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_error_records(
    sample_idx: int,
    sample: Dict[str, Any],
    error: Exception,
    prompt: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """为失败样本构造 summary / step 两类错误记录。"""
    problem = sample.get("problem")
    answer = sample.get("answer")
    difficulty = sample.get("difficulty")
    unique_id = sample.get("unique_id", str(sample_idx))

    summary_record = {
        "sample_idx": sample_idx,
        "unique_id": unique_id,
        "problem": problem,
        "gold_answer": answer,
        "difficulty": difficulty,
        "prompt": prompt,
        "error": repr(error),
        "full_output": None,
        "step_avg_confidences": [],
        "num_steps": 0,
        "num_generated_tokens": 0,
    }

    step_record = {
        "sample_idx": sample_idx,
        "unique_id": unique_id,
        "problem": problem,
        "gold_answer": answer,
        "difficulty": difficulty,
        "prompt": prompt,
        "error": repr(error),
        "steps": [],
        "full_output": None,
        "num_steps": 0,
        "num_generated_tokens": 0,
    }

    return summary_record, step_record


def build_success_records(
    sample_idx: int,
    sample: Dict[str, Any],
    prompt: str,
    request_output,
    tokenizer,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """解析单个 vLLM RequestOutput，并构造 summary / step 记录。"""
    problem = sample["problem"]
    answer = sample.get("answer")
    difficulty = sample.get("difficulty")
    unique_id = sample.get("unique_id", str(sample_idx))

    if not request_output.outputs:
        raise RuntimeError("vLLM returned an empty outputs list for this request.")

    output = request_output.outputs[0]
    token_ids = list(output.token_ids)

    output_text_from_tokens, token_pieces = build_token_pieces_with_char_spans(
        token_ids=token_ids,
        tokenizer=tokenizer,
    )

    # 优先使用 decode(token_ids) 的结果，保证字符 span 与 token 对齐。
    full_output = output_text_from_tokens

    output_logprobs = output.logprobs
    if output_logprobs is None:
        raise RuntimeError(
            "output.logprobs is None. "
            "Please make sure SamplingParams(logprobs=1) is set."
        )

    if len(output_logprobs) != len(token_ids):
        raise RuntimeError(
            f"The number of output logprob entries ({len(output_logprobs)}) "
            f"does not match token_ids ({len(token_ids)})."
        )

    token_logprobs: List[Optional[float]] = []
    for token_id, logprob_dict in zip(token_ids, output_logprobs):
        token_logprobs.append(
            get_sampled_token_logprob(logprob_dict, token_id)
        )

    step_conf_list, step_records = compute_step_confidences(
        text=full_output,
        token_pieces=token_pieces,
        token_logprobs=token_logprobs,
        sep="\n\n",
    )

    summary_record = {
        "sample_idx": sample_idx,
        "unique_id": unique_id,
        "problem": problem,
        "gold_answer": answer,
        "difficulty": difficulty,
        "prompt": prompt,
        "full_output": full_output,
        "step_avg_confidences": step_conf_list,
        "num_steps": len(step_conf_list),
        "num_generated_tokens": len(token_ids),
        "finish_reason": getattr(output, "finish_reason", None),
    }

    step_record = {
        "sample_idx": sample_idx,
        "unique_id": unique_id,
        "problem": problem,
        "gold_answer": answer,
        "difficulty": difficulty,
        "prompt": prompt,
        "steps": step_records,
        "full_output": full_output,
        "num_steps": len(step_records),
        "num_generated_tokens": len(token_ids),
    }

    return summary_record, step_record


def iter_batches(
    data: List[Tuple[int, Dict[str, Any]]],
    batch_size: int,
):
    """按 batch_size 顺序切分 worker 自己的数据 shard。"""
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    for start in range(0, len(data), batch_size):
        yield data[start : start + batch_size]


def parse_gpu_list(gpus: Optional[str]) -> Optional[List[str]]:
    """解析 --gpus 传入的 GPU 列表。"""
    if gpus is None:
        return None

    gpus = gpus.strip()
    if not gpus:
        return None

    return [x.strip() for x in gpus.split(",") if x.strip()]


def query_all_gpu_ids_by_nvidia_smi() -> List[str]:
    """在没有显式传入 --gpus 且没有 CUDA_VISIBLE_DEVICES 时，尝试用 nvidia-smi 获取 GPU id。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        gpu_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return gpu_ids
    except Exception:
        return []


def get_base_gpu_list(args: argparse.Namespace) -> List[str]:
    """
    获取主进程可用于分配的 GPU 列表。

    优先级：
    1. --gpus
    2. 当前环境变量 CUDA_VISIBLE_DEVICES
    3. nvidia-smi 查询到的所有 GPU index
    """
    gpu_list = parse_gpu_list(args.gpus)
    if gpu_list is not None:
        return gpu_list

    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_visible and env_visible not in {"-1", "none", "None"}:
        return [x.strip() for x in env_visible.split(",") if x.strip()]

    gpu_list = query_all_gpu_ids_by_nvidia_smi()
    if gpu_list:
        return gpu_list

    raise RuntimeError(
        "Cannot determine available GPUs. Please specify --gpus, e.g. --gpus 0,1."
    )


def build_gpu_groups(
    gpu_list: List[str],
    data_parallel_size: int,
    tensor_parallel_size: int,
) -> List[List[str]]:
    """
    根据 DP 和 TP，把 GPU 分成若干组。

    例如：
        gpu_list = ["0", "1", "2", "3"]
        DP = 2
        TP = 2

    得到：
        [["0", "1"], ["2", "3"]]
    """
    if data_parallel_size < 1:
        raise ValueError("--data_parallel_size must be >= 1")

    if tensor_parallel_size < 1:
        raise ValueError("--tensor_parallel_size must be >= 1")

    required_gpus = data_parallel_size * tensor_parallel_size

    if len(gpu_list) < required_gpus:
        raise ValueError(
            f"Not enough GPUs. Required DP x TP = "
            f"{data_parallel_size} x {tensor_parallel_size} = {required_gpus}, "
            f"but got GPU list: {gpu_list}"
        )

    used_gpus = gpu_list[:required_gpus]

    groups = []
    for dp_rank in range(data_parallel_size):
        start = dp_rank * tensor_parallel_size
        end = start + tensor_parallel_size
        groups.append(used_gpus[start:end])

    return groups


def shard_output_path(path: str, shard_id: int) -> str:
    """
    生成 shard 临时输出文件名。

    example:
        outputs/summary.jsonl
    ->
        outputs/summary.shard0.jsonl
    """
    directory = os.path.dirname(path)
    basename = os.path.basename(path)

    if "." in basename:
        stem, ext = os.path.splitext(basename)
        shard_name = f"{stem}.shard{shard_id}{ext}"
    else:
        shard_name = f"{basename}.shard{shard_id}"

    return os.path.join(directory, shard_name)


def remove_file_if_exists(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


def ensure_file_exists(path: str) -> None:
    """确保 JSONL 文件存在，便于空 shard 也能参与后续合并。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        pass


def list_existing_shard_paths(path: str) -> List[str]:
    """查找某个最终输出对应的所有历史 shard 文件。"""
    directory = os.path.dirname(path) or "."
    basename = os.path.basename(path)

    if "." in basename:
        stem, ext = os.path.splitext(basename)
        pattern = os.path.join(directory, f"{stem}.shard*{ext}")
    else:
        pattern = os.path.join(directory, f"{basename}.shard*")

    return sorted(glob.glob(pattern))


def unique_paths(paths: List[str]) -> List[str]:
    """按输入顺序去重文件路径。"""
    result: List[str] = []
    seen = set()
    for path in paths:
        if not path:
            continue
        normalized = os.path.abspath(path)
        if normalized not in seen:
            seen.add(normalized)
            result.append(path)
    return result


def record_has_error(record: Dict[str, Any]) -> bool:
    """判断输出记录是否是失败记录。"""
    return bool(record.get("error"))


def load_latest_jsonl_records(
    paths: List[str],
    key: str = "sample_idx",
) -> Dict[int, Dict[str, Any]]:
    """
    从若干 JSONL 中读取每个 sample_idx 的最新记录。

    - 同一文件中同一 sample_idx 出现多次时，后面的记录覆盖前面的记录；
    - 跨文件重复时，成功记录优先于 error 记录；
    - 容忍进程异常退出留下的空行或损坏行，并打印警告后跳过。
    """
    latest: Dict[int, Dict[str, Any]] = {}

    for path in unique_paths(paths):
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"[Resume] Warning: skip invalid JSONL line "
                        f"{path}:{line_no}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                if key not in record:
                    print(
                        f"[Resume] Warning: skip record without '{key}' at "
                        f"{path}:{line_no}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                try:
                    sample_idx = int(record[key])
                except (TypeError, ValueError):
                    print(
                        f"[Resume] Warning: invalid {key}={record.get(key)!r} at "
                        f"{path}:{line_no}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                old_record = latest.get(sample_idx)
                if old_record is None:
                    latest[sample_idx] = record
                elif record_has_error(old_record) and not record_has_error(record):
                    # 新成功记录覆盖旧 error。
                    latest[sample_idx] = record
                elif record_has_error(old_record) == record_has_error(record):
                    # 同为成功或同为失败时，后读到的记录视为更新。
                    latest[sample_idx] = record
                # 旧记录成功、新记录失败时保留旧成功记录。

    return latest


def collect_resume_completed_indices(
    summary_paths: List[str],
    step_paths: List[str],
    skip_error_records: bool,
) -> set:
    """
    只有 summary 和 step 两边都存在记录时，样本才可被视为已落盘。

    默认只把两边都成功的记录视为完成，因此上次 OOM 等导致的 error
    会在续跑时自动重试。skip_error_records=True 时，成对的 error 记录也跳过。
    """
    summary_records = load_latest_jsonl_records(summary_paths)
    step_records = load_latest_jsonl_records(step_paths)

    common_indices = set(summary_records).intersection(step_records)
    if skip_error_records:
        return common_indices

    return {
        idx
        for idx in common_indices
        if not record_has_error(summary_records[idx])
        and not record_has_error(step_records[idx])
    }


def rewrite_jsonl_deduplicated(
    input_paths: List[str],
    output_path: str,
    sort_key: str = "sample_idx",
) -> None:
    """
    合并并去重 JSONL，然后通过临时文件 + os.replace 原子覆盖最终文件。

    这会清理断点续跑过程中可能产生的旧 error / 新 success 重复记录。
    """
    records_by_idx = load_latest_jsonl_records(input_paths, key=sort_key)
    records = [records_by_idx[idx] for idx in sorted(records_by_idx)]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = f"{output_path}.tmp.{os.getpid()}"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def merge_jsonl_shards(
    shard_paths: List[str],
    output_path: str,
    sort_key: str = "sample_idx",
) -> None:
    """
    合并多个 shard JSONL 文件，并按 sample_idx 排序后追加写入最终输出文件。

    注意：
    - 如果主脚本指定了 --overwrite，最终输出文件已经在 main_process 中被删除。
    - 如果没有指定 --overwrite，则这里会追加到已有最终输出文件后面。
    """
    records: List[Dict[str, Any]] = []

    for path in shard_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shard file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    records.sort(key=lambda x: x.get(sort_key, -1))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_generation_worker(
    args: argparse.Namespace,
    shard_id: int,
    num_shards: int,
    gpu_ids: List[str],
    summary_jsonl_path: str,
    step_jsonl_path: str,
) -> None:
    """
    单个数据并行 worker。

    每个 worker：
    1. 通过 CUDA_VISIBLE_DEVICES 限定自己可见的 GPU；
    2. 根据 input part 和 DP shard 选出当前 worker 的样本；
    3. 若启用 --resume，读取最终文件和所有历史 shard，跳过已完成样本；
    4. 仅当仍有待处理样本时才加载 vLLM，并在实例内部使用 TP；
    5. 将待处理样本按 --batch_size 分批进行真正的 batch 推理；
    6. 每个 batch 完成后立即把 summary / step 结果写入磁盘。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    try:
        effective_max_num_seqs = (
            args.max_num_seqs
            if args.max_num_seqs is not None
            else args.batch_size
        )

        if args.batch_size < 1:
            raise ValueError("--batch_size must be >= 1")
        if effective_max_num_seqs < 1:
            raise ValueError("--max_num_seqs must be >= 1")

        print(
            f"[Worker {shard_id}] "
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, "
            f"TP={args.tensor_parallel_size}, "
            f"batch_size={args.batch_size}, "
            f"max_num_seqs={effective_max_num_seqs}",
            flush=True,
        )

        if effective_max_num_seqs < args.batch_size:
            print(
                f"[Worker {shard_id}] Warning: max_num_seqs "
                f"({effective_max_num_seqs}) < batch_size ({args.batch_size}). "
                "All prompts are still submitted together, but vLLM can keep at most "
                f"{effective_max_num_seqs} sequences active concurrently.",
                flush=True,
            )

        data = load_json_array(args.input_json)
        indexed_data = list(enumerate(data))

        # 保持原有 limit 逻辑：如果指定 limit，则先截取前 limit 条。
        if args.limit is not None:
            indexed_data = indexed_data[: args.limit]

        # 将输入数据等分成 num_input_parts 份，只推理 input_part_id 这一份。
        indexed_data, part_start, part_end = select_input_part(
            indexed_data=indexed_data,
            num_input_parts=args.num_input_parts,
            input_part_id=args.input_part_id,
        )

        print(
            f"[Worker {shard_id}] Input part "
            f"{args.input_part_id}/{args.num_input_parts}: "
            f"local range [{part_start}, {part_end}), "
            f"num_samples={len(indexed_data)}",
            flush=True,
        )

        # 在选中的 input part 内继续按原始 sample_idx 做 DP 分片。
        indexed_data = [
            (original_idx, sample)
            for original_idx, sample in indexed_data
            if original_idx % num_shards == shard_id
        ]

        shard_sample_count = len(indexed_data)
        print(
            f"[Worker {shard_id}] After DP shard filtering: "
            f"num_samples={shard_sample_count}",
            flush=True,
        )

        if args.resume:
            # 同时读取最终文件和所有历史 shard。这样即使上次 DP 运行中途崩溃、
            # 尚未来得及 merge，也可以直接从临时 shard 继续。
            summary_resume_paths = unique_paths(
                [args.summary_jsonl, summary_jsonl_path]
                + list_existing_shard_paths(args.summary_jsonl)
            )
            step_resume_paths = unique_paths(
                [args.step_jsonl, step_jsonl_path]
                + list_existing_shard_paths(args.step_jsonl)
            )

            completed_indices = collect_resume_completed_indices(
                summary_paths=summary_resume_paths,
                step_paths=step_resume_paths,
                skip_error_records=args.resume_skip_errors,
            )

            before_resume_filter = len(indexed_data)
            indexed_data = [
                (sample_idx, sample)
                for sample_idx, sample in indexed_data
                if sample_idx not in completed_indices
            ]
            skipped = before_resume_filter - len(indexed_data)

            print(
                f"[Worker {shard_id}] Resume scan: skipped={skipped}, "
                f"remaining={len(indexed_data)}, "
                f"retry_errors={not args.resume_skip_errors}",
                flush=True,
            )

        # 即便当前 shard 没有样本，也创建空文件，避免普通 DP merge 因文件不存在失败。
        ensure_file_exists(summary_jsonl_path)
        ensure_file_exists(step_jsonl_path)

        if not indexed_data:
            print(
                f"[Worker {shard_id}] No pending samples. Skip model loading.",
                flush=True,
            )
            return

        # 子进程内、且确认有待处理数据后再 import vLLM，避免主进程提前初始化 CUDA。
        from vllm import LLM

        llm_kwargs: Dict[str, Any] = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "max_model_len": args.max_model_len,
            "max_num_seqs": effective_max_num_seqs,
        }

        if args.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

        if args.enforce_eager or args.vllm_enforce_eager:
            llm_kwargs["enforce_eager"] = True

        llm = LLM(**llm_kwargs)

        tokenizer = llm.get_tokenizer()
        sampling_params = make_sampling_params(args)

        progress_desc = f"Worker {shard_id}/{num_shards}"
        progress_bar = tqdm(
            total=len(indexed_data),
            desc=progress_desc,
            position=shard_id,
            leave=True,
            unit="sample",
        )

        try:
            for batch_id, batch_data in enumerate(
                iter_batches(indexed_data, args.batch_size)
            ):
                summary_records: List[Dict[str, Any]] = []
                step_records: List[Dict[str, Any]] = []

                valid_metadata: List[Tuple[int, Dict[str, Any], str]] = []
                prompts: List[str] = []

                # prompt 构造失败只影响当前样本，不影响同一个 batch 的其他样本。
                for sample_idx, sample in batch_data:
                    try:
                        if "problem" not in sample:
                            raise KeyError(
                                f"Sample {sample_idx} does not contain required key 'problem'."
                            )

                        prompt = build_prompt(
                            tokenizer=tokenizer,
                            problem=sample["problem"],
                            enable_thinking=args.enable_thinking,
                        )
                        prompts.append(prompt)
                        valid_metadata.append((sample_idx, sample, prompt))
                    except Exception as e:
                        summary_record, step_record = build_error_records(
                            sample_idx=sample_idx,
                            sample=sample,
                            error=e,
                            prompt=None,
                        )
                        summary_records.append(summary_record)
                        step_records.append(step_record)

                if prompts:
                    try:
                        request_outputs = llm.generate(
                            prompts,
                            sampling_params,
                            use_tqdm=False,
                        )
                    except Exception as e:
                        # 整个 batch 的 generate 失败时，先落盘 error 记录再退出。
                        # 下次 --resume 默认会重新尝试这些 error 样本。
                        for sample_idx, sample, prompt in valid_metadata:
                            summary_record, step_record = build_error_records(
                                sample_idx=sample_idx,
                                sample=sample,
                                error=e,
                                prompt=prompt,
                            )
                            summary_records.append(summary_record)
                            step_records.append(step_record)

                        summary_records.sort(key=lambda x: x["sample_idx"])
                        step_records.sort(key=lambda x: x["sample_idx"])
                        append_jsonl_many(summary_jsonl_path, summary_records)
                        append_jsonl_many(step_jsonl_path, step_records)
                        progress_bar.update(len(batch_data))
                        raise

                    if len(request_outputs) != len(valid_metadata):
                        raise RuntimeError(
                            f"vLLM returned {len(request_outputs)} outputs for "
                            f"{len(valid_metadata)} prompts in batch {batch_id}."
                        )

                    # 单条后处理失败不会影响 batch 中其他已经生成成功的样本。
                    for request_output, metadata in zip(
                        request_outputs,
                        valid_metadata,
                    ):
                        sample_idx, sample, prompt = metadata
                        try:
                            summary_record, step_record = build_success_records(
                                sample_idx=sample_idx,
                                sample=sample,
                                prompt=prompt,
                                request_output=request_output,
                                tokenizer=tokenizer,
                            )
                        except Exception as e:
                            summary_record, step_record = build_error_records(
                                sample_idx=sample_idx,
                                sample=sample,
                                error=e,
                                prompt=prompt,
                            )

                        summary_records.append(summary_record)
                        step_records.append(step_record)

                # 每完成一个 batch 立刻写盘，这是断点续跑的持久化边界。
                summary_records.sort(key=lambda x: x["sample_idx"])
                step_records.sort(key=lambda x: x["sample_idx"])

                append_jsonl_many(summary_jsonl_path, summary_records)
                append_jsonl_many(step_jsonl_path, step_records)
                progress_bar.update(len(batch_data))
        finally:
            progress_bar.close()

    except Exception:
        print(f"[Worker {shard_id}] Fatal error:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise


def main_process(args: argparse.Namespace) -> None:
    """
    主进程逻辑：
    1. 根据 DP 和 TP 划分 GPU；
    2. 处理 overwrite / resume 所需的最终文件和 shard 文件；
    3. DP=1 时直接调用一个 worker；
    4. DP>1 时启动多个子进程；
    5. 全部完成后合并结果；resume 模式下会去重并原子重写最终 JSONL。
    """
    os.makedirs(os.path.dirname(args.summary_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.step_jsonl) or ".", exist_ok=True)

    gpu_list = get_base_gpu_list(args)
    gpu_groups = build_gpu_groups(
        gpu_list=gpu_list,
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    print(f"[Main] Available GPU list: {gpu_list}", flush=True)
    print(f"[Main] GPU groups for workers: {gpu_groups}", flush=True)
    print(
        f"[Main] DP={args.data_parallel_size}, TP={args.tensor_parallel_size}, "
        f"total used GPUs={args.data_parallel_size * args.tensor_parallel_size}",
        flush=True,
    )
    print(
        f"[Main] Input split: num_input_parts={args.num_input_parts}, "
        f"input_part_id={args.input_part_id}",
        flush=True,
    )
    print(
        f"[Main] Qwen3 enable_thinking={args.enable_thinking}",
        flush=True,
    )
    effective_max_num_seqs = (
        args.max_num_seqs
        if args.max_num_seqs is not None
        else args.batch_size
    )
    print(
        f"[Main] Batch inference: batch_size={args.batch_size}, "
        f"max_num_seqs={effective_max_num_seqs}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}",
        flush=True,
    )
    print(
        f"[Main] Resume={args.resume}, "
        f"retry_error_records={not args.resume_skip_errors}",
        flush=True,
    )

    # 当前 DP 配置将使用的 shard 路径。
    summary_shard_paths = [
        shard_output_path(args.summary_jsonl, shard_id)
        for shard_id in range(args.data_parallel_size)
    ]
    step_shard_paths = [
        shard_output_path(args.step_jsonl, shard_id)
        for shard_id in range(args.data_parallel_size)
    ]

    if args.overwrite:
        remove_file_if_exists(args.summary_jsonl)
        remove_file_if_exists(args.step_jsonl)
        # overwrite 应清理所有历史 shard，包括以前使用不同 DP 数量留下的 shard。
        for path in (
            list_existing_shard_paths(args.summary_jsonl)
            + list_existing_shard_paths(args.step_jsonl)
        ):
            remove_file_if_exists(path)
    elif not args.resume:
        # 普通非续跑模式保持旧行为：只清理本轮将使用的临时 shard。
        for path in summary_shard_paths + step_shard_paths:
            remove_file_if_exists(path)

    if args.data_parallel_size == 1:
        # DP=1 时直接写最终文件。
        run_generation_worker(
            args=args,
            shard_id=0,
            num_shards=1,
            gpu_ids=gpu_groups[0],
            summary_jsonl_path=args.summary_jsonl,
            step_jsonl_path=args.step_jsonl,
        )

        if args.resume:
            # retry error 时可能在同一文件中产生旧 error + 新 success，结束后做一次压缩去重。
            rewrite_jsonl_deduplicated(
                input_paths=[args.summary_jsonl],
                output_path=args.summary_jsonl,
                sort_key="sample_idx",
            )
            rewrite_jsonl_deduplicated(
                input_paths=[args.step_jsonl],
                output_path=args.step_jsonl,
                sort_key="sample_idx",
            )
            print("[Main] Resume outputs compacted and deduplicated.", flush=True)
        return

    processes: List[Process] = []

    for shard_id in range(args.data_parallel_size):
        worker_args = copy.deepcopy(args)

        p = Process(
            target=run_generation_worker,
            kwargs={
                "args": worker_args,
                "shard_id": shard_id,
                "num_shards": args.data_parallel_size,
                "gpu_ids": gpu_groups[shard_id],
                "summary_jsonl_path": summary_shard_paths[shard_id],
                "step_jsonl_path": step_shard_paths[shard_id],
            },
        )
        p.start()
        processes.append(p)

    exit_codes = []
    for p in processes:
        p.join()
        exit_codes.append(p.exitcode)

    failed = [
        (idx, code)
        for idx, code in enumerate(exit_codes)
        if code != 0
    ]

    if failed:
        raise RuntimeError(
            f"Some workers failed: {failed}. "
            "Final files and shard files are kept for --resume."
        )

    print("[Main] All workers finished. Merging shard files...", flush=True)

    if args.resume:
        # 包含所有历史 shard，支持上次不同 DP 数量或中途崩溃留下的结果。
        all_summary_shards = list_existing_shard_paths(args.summary_jsonl)
        all_step_shards = list_existing_shard_paths(args.step_jsonl)

        rewrite_jsonl_deduplicated(
            input_paths=[args.summary_jsonl] + all_summary_shards,
            output_path=args.summary_jsonl,
            sort_key="sample_idx",
        )
        rewrite_jsonl_deduplicated(
            input_paths=[args.step_jsonl] + all_step_shards,
            output_path=args.step_jsonl,
            sort_key="sample_idx",
        )
    else:
        merge_jsonl_shards(
            shard_paths=summary_shard_paths,
            output_path=args.summary_jsonl,
            sort_key="sample_idx",
        )
        merge_jsonl_shards(
            shard_paths=step_shard_paths,
            output_path=args.step_jsonl,
            sort_key="sample_idx",
        )

    print(f"[Main] Merged summary output: {args.summary_jsonl}", flush=True)
    print(f"[Main] Merged step output: {args.step_jsonl}", flush=True)

    if not args.keep_shard_files:
        # 删除所有历史 shard，最终结果已经安全进入最终 JSONL。
        for path in (
            list_existing_shard_paths(args.summary_jsonl)
            + list_existing_shard_paths(args.step_jsonl)
        ):
            remove_file_if_exists(path)
        print("[Main] Temporary shard files removed.", flush=True)
    else:
        print("[Main] Temporary shard files kept.", flush=True)


def parse_args() -> argparse.Namespace:
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

    parser.add_argument(
        "--enable_thinking",
        type=str2bool,
        default=True,
        help=(
            "Qwen3 thinking mode 开关。"
            "true 表示启用 thinking mode；false 表示启用 non-thinking mode。"
            "例如：--enable_thinking true 或 --enable_thinking false。"
        ),
    )

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "每个 DP worker 每次传给 llm.generate() 的 prompt 数量。"
            "默认 1 保持最省显存；实际批量推理可设为 2、4、8 等。"
        ),
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help=(
            "vLLM 同时活跃的最大序列数。若不指定，则自动设为 batch_size。"
            "显式设得小于 batch_size 时，整个 batch 仍会提交，但并发活跃序列受此值限制。"
        ),
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help=(
            "可选：限制 vLLM 每个调度迭代中的最大 token 数。"
            "默认 None，由 vLLM 自动决定。vLLM V1 开启 chunked prefill 时可尝试 2048 或 4096；"
            "若关闭 chunked prefill，该值通常不得小于 max_model_len。"
        ),
    )

    parser.add_argument(
        "--num_input_parts",
        type=int,
        default=1,
        help="将输入 JSON 按原始顺序等分成多少个连续部分。默认 1 表示不切分。",
    )
    parser.add_argument(
        "--input_part_id",
        type=int,
        default=0,
        help="本次只推理第几个输入部分，从 0 开始。默认 0。",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "指定本次运行使用哪些 GPU，例如 '0,1,2,3'。"
            "若不指定，则优先使用当前 CUDA_VISIBLE_DEVICES；"
            "若 CUDA_VISIBLE_DEVICES 也没有，则尝试通过 nvidia-smi 获取。"
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
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help=(
            "禁用 CUDA Graph、强制 eager 执行。可降低启动预热/图捕获显存，"
            "但通常会降低稳态推理速度；仅在启动仍 OOM 时尝试。"
        ),
    )

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "启用断点续跑。读取已有最终 JSONL 和 DP shard JSONL，"
            "跳过 summary 与 step 两边都已成功写入的 sample_idx。"
            "必须与上次运行使用相同的输入数据和 sample_idx 定义。"
        ),
    )
    parser.add_argument(
        "--resume_skip_errors",
        action="store_true",
        help=(
            "仅与 --resume 配合使用。默认续跑会重试带 error 的失败样本；"
            "设置本参数后，只要 summary 和 step 两边已有记录，即使是 error 也跳过。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果设置，则先删除已有最终输出文件。",
    )

    parser.add_argument(
        "--keep_shard_files",
        action="store_true",
        help="DP>1 时保留各 worker 的临时 shard 输出文件。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if args.resume_skip_errors and not args.resume:
        raise ValueError("--resume_skip_errors requires --resume")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.max_num_seqs is not None and args.max_num_seqs < 1:
        raise ValueError("--max_num_seqs must be >= 1")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        raise ValueError("--max_num_batched_tokens must be >= 1")
    if args.max_tokens > args.max_model_len:
        print(
            "[Main] Warning: max_tokens is greater than max_model_len. "
            "The actual available generation length also includes prompt tokens, "
            "so consider lowering --max_tokens.",
            file=sys.stderr,
            flush=True,
        )

    main_process(args)


if __name__ == "__main__":
    # vLLM / CUDA 多进程场景建议使用 spawn。
    set_start_method("spawn", force=True)
    main()
