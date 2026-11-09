#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用 vLLM 对 MATH-500 风格 JSON 数据集逐条推理，
按 "\n\n" 切分 reasoning step，
并保存每个 step 的平均 token confidence。

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
    2. 加载一个 vLLM 实例；
    3. 先根据 --num_input_parts / --input_part_id 选中输入数据的一部分；
    4. 再只处理 sample_idx % num_shards == shard_id 的样本；
    5. 每处理一条样本，就更新自己的两个 JSONL 文件。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    # 子进程内再 import vLLM，避免主进程提前初始化 CUDA。
    from vllm import LLM

    try:
        print(
            f"[Worker {shard_id}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, "
            f"TP={args.tensor_parallel_size}",
            flush=True,
        )

        data = load_json_array(args.input_json)
        indexed_data = list(enumerate(data))

        # 保持原有 limit 逻辑：如果指定 limit，则先截取前 limit 条。
        if args.limit is not None:
            indexed_data = indexed_data[: args.limit]

        # 新增逻辑：将输入数据等分成 num_input_parts 份，只推理 input_part_id 这一份。
        indexed_data, part_start, part_end = select_input_part(
            indexed_data=indexed_data,
            num_input_parts=args.num_input_parts,
            input_part_id=args.input_part_id,
        )

        print(
            f"[Worker {shard_id}] Input part {args.input_part_id}/{args.num_input_parts}: "
            f"local range [{part_start}, {part_end}), num_samples={len(indexed_data)}",
            flush=True,
        )

        # 原有 DP 逻辑：在选中的 input part 内继续按 sample_idx % num_shards 分片。
        indexed_data = [
            (original_idx, sample)
            for original_idx, sample in indexed_data
            if original_idx % num_shards == shard_id
        ]

        print(
            f"[Worker {shard_id}] After DP shard filtering: num_samples={len(indexed_data)}",
            flush=True,
        )

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

        progress_desc = f"Worker {shard_id}/{num_shards}"
        for local_idx, (sample_idx, sample) in enumerate(
            tqdm(
                indexed_data,
                desc=progress_desc,
                position=shard_id,
                leave=True,
            )
        ):
            problem = sample["problem"]
            answer = sample.get("answer")
            difficulty = sample.get("difficulty")
            unique_id = sample.get("unique_id", str(sample_idx))

            prompt = problem + PROMPT_SUFFIX

            try:
                request_outputs = llm.generate([prompt], sampling_params)
                output = request_outputs[0].outputs[0]

                token_ids = list(output.token_ids)

                output_text_from_tokens, token_pieces = build_token_pieces_with_char_spans(
                    token_ids=token_ids,
                    tokenizer=tokenizer,
                )

                # vLLM 的 output.text 通常和 decode(token_ids) 一致。
                # 这里优先用 decode 后的文本，以保证 char span 对齐。
                full_output = output_text_from_tokens

                output_logprobs = output.logprobs
                if output_logprobs is None:
                    raise RuntimeError(
                        "output.logprobs is None. "
                        "Please make sure SamplingParams(logprobs=1) is set."
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
                    "steps": step_records,
                    "full_output": full_output,
                    "num_steps": len(step_records),
                    "num_generated_tokens": len(token_ids),
                }

            except Exception as e:
                summary_record = {
                    "sample_idx": sample_idx,
                    "unique_id": unique_id,
                    "problem": problem,
                    "gold_answer": answer,
                    "difficulty": difficulty,
                    "error": repr(e),
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
                    "error": repr(e),
                    "steps": [],
                    "full_output": None,
                    "num_steps": 0,
                    "num_generated_tokens": 0,
                }

            # 每处理一条样本，就立刻更新两个 JSONL 文件。
            append_jsonl(summary_jsonl_path, summary_record)
            append_jsonl(step_jsonl_path, step_record)

    except Exception:
        print(f"[Worker {shard_id}] Fatal error:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise


def main_process(args: argparse.Namespace) -> None:
    """
    主进程逻辑：
    1. 根据 DP 和 TP 划分 GPU；
    2. DP=1 时直接调用一个 worker；
    3. DP>1 时启动多个子进程；
    4. 等全部 worker 完成后，自动合并 shard 结果。
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

    if args.overwrite:
        remove_file_if_exists(args.summary_jsonl)
        remove_file_if_exists(args.step_jsonl)

    if args.data_parallel_size == 1:
        # DP=1 时，不生成 shard 文件，直接写最终输出文件。
        run_generation_worker(
            args=args,
            shard_id=0,
            num_shards=1,
            gpu_ids=gpu_groups[0],
            summary_jsonl_path=args.summary_jsonl,
            step_jsonl_path=args.step_jsonl,
        )
        return

    # DP>1 时，每个 worker 写自己的临时 shard 文件。
    summary_shard_paths = [
        shard_output_path(args.summary_jsonl, shard_id)
        for shard_id in range(args.data_parallel_size)
    ]
    step_shard_paths = [
        shard_output_path(args.step_jsonl, shard_id)
        for shard_id in range(args.data_parallel_size)
    ]

    # shard 文件是本轮运行的临时文件，无论是否 overwrite，都先清掉。
    for path in summary_shard_paths + step_shard_paths:
        remove_file_if_exists(path)

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
        raise RuntimeError(f"Some workers failed: {failed}. Shard files are kept for debugging.")

    print("[Main] All workers finished. Merging shard files...", flush=True)

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
        for path in summary_shard_paths + step_shard_paths:
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

    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel_size", type=int, default=1)

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

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)

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
    main_process(args)


if __name__ == "__main__":
    # vLLM / CUDA 多进程场景建议使用 spawn。
    set_start_method("spawn", force=True)
    main()
