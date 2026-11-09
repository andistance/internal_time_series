#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-GPU / multi-process Monte Carlo prefix value labeling.

一键实现：
    4 张卡
    -> 两个独立 vLLM worker
    -> 每个 worker 用 2 张卡 tensor parallel
    -> 两个 worker 处理不同 sample shard
    -> 最后自动合并结果

默认：
    --gpu_groups "0,1;2,3"
    表示启动两个 worker：
        worker 0: CUDA_VISIBLE_DEVICES=0,1
        worker 1: CUDA_VISIBLE_DEVICES=2,3

每个 worker：
    只处理 source_line_idx % num_shards == shard_id 的样本。
    每个 worker 写自己的 steps_shard*.jsonl 和 details_shard*.jsonl。
    避免多个进程同时写同一个文件。

已有逻辑：
    使用 step 的 token 间距来跳过过近的 step。
    当前 step 的结束位置定义为：
        当前 step 的下一个 step 的 first_token_index
    如果当前 step 的结束位置与上一个真正计算 label 的 step 的结束位置距离小于
    --min_label_token_gap，则跳过当前 step，不计算 label，也不更新 last_labeled_end。

新增逻辑：
    如果一个样本中某个 step 的 "step" 文本已经包含 "</think>"，
    则该 step 和之后的所有 step 都跳过，不生成 step_prospective_correctness。

运行示例：
    python build_mc_prefix_value_labels_multigpu.py \
        --steps_path steps.jsonl \
        --output_steps_path steps_monte_carlo_labeled.jsonl \
        --output_details_path steps_monte_carlo_details_all.jsonl \
        --model_path /path/to/model \
        --gpu_groups "0,1;2,3" \
        --K 4 \
        --batch_size 8 \
        --save_every 50 \
        --max_tokens 1024 \
        --max_model_len 8192 \
        --min_label_token_gap 128 \
        --temperature 0.7 \
        --top_p 0.95 \
        --gpu_memory_utilization 0.90 \
        --dtype auto \
        --trust_remote_code

如果要重新从头跑：
    加 --restart

如果最后想直接覆盖原 steps.jsonl：
    加 --inplace
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


PROMPT_SUFFIX = (
    "\nLet's reason step by step. Enclose the reasoning process within "
    "<think>...</think>, then summarize it and present the final answer within \\boxed{}."
)


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

# =========================
# 原始判题函数
# =========================

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from deepscaler.rewards.math_utils.utils import grade_answer_verl


# =========================
# 通用文件读写
# =========================

def detect_file_format(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("["):
                return "json"
            return "jsonl"

    raise ValueError(f"Cannot detect file format for empty file: {path}")


def load_records(path: str) -> Tuple[List[Dict[str, Any]], str]:
    file_format = detect_file_format(path)

    if file_format == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} should be a JSON array.")
        return data, file_format

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL decode error in {path}, line {line_idx + 1}: {e}"
                ) from e

    return records, file_format


def save_records(path: str, records: List[Dict[str, Any]], file_format: str) -> None:
    tmp_path = path + ".tmp"

    if file_format == "json":
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    elif file_format == "jsonl":
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    os.replace(tmp_path, path)


def backup_file(path: str) -> str:
    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
    return backup_path


def make_detail_key(source_line_idx: int, sample_idx: Any, step_idx: int) -> str:
    return f"line={source_line_idx}__sample={sample_idx}__step={step_idx}"


def load_existing_details(details_path: str) -> Dict[str, Dict[str, Any]]:
    details: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(details_path):
        return details

    with open(details_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warning] Skip broken detail line {line_idx + 1}: {details_path}")
                continue

            source_line_idx = obj.get("source_line_idx")
            sample_idx = obj.get("sample_idx")
            step_idx = obj.get("step_idx")

            if source_line_idx is None or step_idx is None:
                continue

            key = make_detail_key(source_line_idx, sample_idx, step_idx)
            details[key] = obj

    return details


def save_details(details_path: str, details: Dict[str, Dict[str, Any]]) -> None:
    records = list(details.values())

    def sort_key(x: Dict[str, Any]) -> Tuple[int, int]:
        return (
            int(x.get("source_line_idx", 10**12)),
            int(x.get("step_idx", 10**12)),
        )

    records.sort(key=sort_key)

    tmp_path = details_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, details_path)


# =========================
# 判题逻辑
# =========================

def get_solution_after_think(full_output: Any) -> str:
    if full_output is None:
        return ""

    full_output = str(full_output)

    if "</think>" in full_output:
        return full_output.split("</think>", 1)[1].strip()

    return full_output.strip()


def judge_with_original_strategy(full_output: str, gold_answer: Any) -> Tuple[bool, str, Optional[str]]:
    solution_str = get_solution_after_think(full_output)

    try:
        is_correct = bool(grade_answer_verl(solution_str, gold_answer))
        return is_correct, solution_str, None
    except Exception as e:
        return False, solution_str, repr(e)


# =========================
# prompt 构造
# =========================

def build_prefix_text(steps: List[Dict[str, Any]], end_step_idx: int) -> str:
    prefix_parts: List[str] = []

    for i in range(end_step_idx + 1):
        step_text = str(steps[i].get("step", "")).strip()
        if step_text:
            prefix_parts.append(step_text)

    return "\n\n".join(prefix_parts)


def build_continuation_prompt(tokenizer, problem: str, prefix_text: str, enable_thinking) -> str:
    messages = [
        {
            "role": "user",
            "content": str(problem) + PROMPT_SUFFIX,
        }
    ]

    qwen3_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # 将已有 Qwen3-8B 推理前缀直接预填到 assistant 消息中，
    # 使 vLLM 从当前 step 之后继续生成。
    return qwen3_prompt + str(prefix_text)


# =========================
# think-end filtering
# =========================

def step_contains_think_end(step_obj: Dict[str, Any]) -> bool:
    """
    判断当前 step 文本中是否已经出现 </think>。

    一旦某个样本的某个 step 出现 </think>，
    该 step 以及之后的所有 step 都不再生成 step_prospective_correctness。
    """
    if not isinstance(step_obj, dict):
        return False

    step_text = step_obj.get("step", "")
    return "</think>" in str(step_text)


# =========================
# token-gap step filtering
# =========================

def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def get_step_end_token_index(steps: List[Dict[str, Any]], step_idx: int) -> Optional[int]:
    """
    当前 step 的结束位置定义为：
        下一个 step 的 first_token_index

    如果当前 step 是最后一个 step，没有下一个 step，则退化使用当前 step 的 first_token_index。
    这个 fallback 只是为了避免最后一个 step 永远无法比较；严格意义上最后一个 step
    的真实结束 token index 需要完整生成长度，但 steps.jsonl 里通常没有这个字段。
    """
    if not isinstance(steps, list):
        return None

    if step_idx + 1 < len(steps):
        next_step = steps[step_idx + 1]
        if isinstance(next_step, dict):
            next_first = safe_int(next_step.get("first_token_index"))
            if next_first is not None:
                return next_first

    if 0 <= step_idx < len(steps):
        cur_step = steps[step_idx]
        if isinstance(cur_step, dict):
            return safe_int(cur_step.get("first_token_index"))

    return None


def should_select_step_by_token_gap(
    current_end_token_index: Optional[int],
    last_labeled_end_token_index: Optional[int],
    min_label_token_gap: int,
) -> bool:
    """
    判断当前 step 是否应该计算 label。

    如果没有足够 token 位置信息，则保守选择计算。
    如果距离上一次计算 label 的 step 太近，则跳过。
    """
    if min_label_token_gap <= 0:
        return True

    if current_end_token_index is None:
        return True

    if last_labeled_end_token_index is None:
        return True

    return (current_end_token_index - last_labeled_end_token_index) >= min_label_token_gap


# =========================
# worker 内部逻辑
# =========================

def get_sample_idx(record: Dict[str, Any], fallback_idx: int) -> Any:
    if record.get("sample_idx") is not None:
        return record.get("sample_idx")
    if record.get("unique_id") is not None:
        return record.get("unique_id")
    return fallback_idx


def step_already_done(record: Dict[str, Any], step_idx: int, overwrite: bool) -> bool:
    if overwrite:
        return False

    steps = record.get("steps", [])
    if not isinstance(steps, list):
        return True

    if step_idx >= len(steps):
        return True

    step_obj = steps[step_idx]
    if not isinstance(step_obj, dict):
        return True

    return "step_prospective_correctness" in step_obj


def fill_steps_from_existing_details(
    steps_records: List[Dict[str, Any]],
    details: Dict[str, Dict[str, Any]],
    overwrite: bool,
) -> int:
    """
    如果 details 文件里已经有某个 step 的结果，但 steps.jsonl 里还没有，
    则先回填，支持中断后恢复。

    新增：
        如果扫描到某个 step 的文本包含 </think>，
        则该 step 和之后的 step 不再从 details 回填 label。
    """
    if overwrite:
        return 0

    filled = 0

    for source_line_idx, record in enumerate(steps_records):
        sample_idx = get_sample_idx(record, source_line_idx)
        steps = record.get("steps", [])

        if not isinstance(steps, list):
            continue

        for step_idx, step_obj in enumerate(steps):
            if not isinstance(step_obj, dict):
                continue

            if step_contains_think_end(step_obj):
                break

            # 仅 phase_frontier == 2 的 step 才允许回填/生成标签。
            if step_obj.get("phase_frontier") != 2:
                continue

            if "step_prospective_correctness" in step_obj:
                continue

            key = make_detail_key(source_line_idx, sample_idx, step_idx)
            detail = details.get(key)

            if detail is None:
                continue

            if "step_prospective_correctness" in detail:
                step_obj["step_prospective_correctness"] = detail[
                    "step_prospective_correctness"
                ]
                filled += 1

    return filled


def collect_pending_tasks(
    steps_records: List[Dict[str, Any]],
    overwrite: bool,
    num_shards: int,
    shard_id: int,
    min_label_token_gap: int,
) -> Tuple[List[Tuple[int, int]], int, int, int]:
    """
    收集所有待处理的 (source_line_idx, step_idx)。

    token-gap 过滤：
        对每个样本内部独立维护 last_labeled_end_token_index。
        如果当前 step 的 end_token_index 距离 last_labeled_end_token_index 太近，
        则跳过，不加入任务，也不更新 last_labeled_end_token_index。

    think-end 过滤：
        如果某个 step 的文本包含 </think>，
        则该 step 和之后的 step 全部跳过。

    返回：
        tasks
        skipped_by_gap_count
        already_done_selected_count
        skipped_by_think_end_count
    """
    tasks: List[Tuple[int, int]] = []
    skipped_by_gap_count = 0
    already_done_selected_count = 0
    skipped_by_think_end_count = 0

    for source_line_idx, record in enumerate(steps_records):
        # 样本级分片：同一个样本的所有 step 都由同一个 worker 处理
        if source_line_idx % num_shards != shard_id:
            continue

        steps = record.get("steps", [])
        if not isinstance(steps, list):
            continue

        last_labeled_end_token_index: Optional[int] = None

        for step_idx, step_obj in enumerate(steps):
            if not isinstance(step_obj, dict):
                continue

            if step_contains_think_end(step_obj):
                skipped_by_think_end_count += len(steps) - step_idx
                break

            # 只有 phase_frontier == 2 的 step 才需要计算
            # step_prospective_correctness；其他 step 不参与 token-gap 比较，
            # 也不会更新 last_labeled_end_token_index。
            if step_obj.get("phase_frontier") != 2:
                continue

            current_end_token_index = get_step_end_token_index(steps, step_idx)

            selected_by_gap = should_select_step_by_token_gap(
                current_end_token_index=current_end_token_index,
                last_labeled_end_token_index=last_labeled_end_token_index,
                min_label_token_gap=min_label_token_gap,
            )

            if not selected_by_gap:
                skipped_by_gap_count += 1
                # 关键：跳过当前 step 时，不更新 last_labeled_end_token_index
                continue

            # 走到这里，说明当前 step 是 token-gap 逻辑选中的“应计算 label 的 step”。
            # 如果它已经有 label，则不加入任务，但它仍然算作“最后一个已计算 label 的 step”。
            if step_already_done(record, step_idx, overwrite):
                already_done_selected_count += 1
                if current_end_token_index is not None:
                    last_labeled_end_token_index = current_end_token_index
                continue

            tasks.append((source_line_idx, step_idx))

            # 关键：只有真正被选中计算 label 的 step，才更新 last_labeled_end_token_index。
            # 虽然实际计算会在后面的 batch 里发生，但任务规划时先更新，保证后续 step 的选择正确。
            if current_end_token_index is not None:
                last_labeled_end_token_index = current_end_token_index

    return tasks, skipped_by_gap_count, already_done_selected_count, skipped_by_think_end_count


def build_generation_item(
    tokenizer,
    steps_records: List[Dict[str, Any]],
    source_line_idx: int,
    step_idx: int,
    enable_thinking: bool,
) -> Dict[str, Any]:
    record = steps_records[source_line_idx]
    sample_idx = get_sample_idx(record, source_line_idx)

    problem = record.get("problem")
    gold_answer = record.get("gold_answer")

    if gold_answer is None:
        gold_answer = record.get("answer")
    if gold_answer is None:
        gold_answer = record.get("ground_truth")

    if problem is None:
        raise ValueError(
            f"Missing problem at source_line_idx={source_line_idx}, sample_idx={sample_idx}"
        )

    if gold_answer is None:
        raise ValueError(
            f"Missing gold_answer/answer/ground_truth at "
            f"source_line_idx={source_line_idx}, sample_idx={sample_idx}"
        )

    steps = record.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError(
            f"Invalid steps at source_line_idx={source_line_idx}, sample_idx={sample_idx}"
        )

    if step_idx >= len(steps):
        raise ValueError(
            f"Invalid step_idx={step_idx} at "
            f"source_line_idx={source_line_idx}, sample_idx={sample_idx}"
        )

    prefix_text = build_prefix_text(steps, step_idx)
    prompt = build_continuation_prompt(tokenizer, problem, prefix_text, enable_thinking)
    end_token_index = get_step_end_token_index(steps, step_idx)

    return {
        "source_line_idx": source_line_idx,
        "sample_idx": sample_idx,
        "step_idx": step_idx,
        "problem": problem,
        "gold_answer": gold_answer,
        "steps": steps,
        "prefix_text": prefix_text,
        "prompt": prompt,
        "end_token_index": end_token_index,
    }


def make_sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    kwargs = {
        "n": args.K,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "skip_special_tokens": True,
    }

    if args.seed is not None:
        kwargs["seed"] = args.seed

    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


def infer_tp_size_from_cuda_visible_devices() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip():
        return 1
    return len([x for x in visible.split(",") if x.strip() != ""])


def load_vllm(args: argparse.Namespace):
    from vllm import LLM

    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None:
        tensor_parallel_size = infer_tp_size_from_cuda_visible_devices()

    kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.vllm_enforce_eager,
        "max_model_len": args.max_model_len,
    }

    try:
        return LLM(**kwargs)
    except TypeError as e:
        if "max_model_len" in str(e):
            print(
                "[Warning] Current vLLM does not support max_model_len. "
                "Retrying without max_model_len."
            )
            kwargs.pop("max_model_len", None)
            return LLM(**kwargs)
        raise


def process_batch(
    llm,
    tokenizer,
    sampling_params,
    steps_records: List[Dict[str, Any]],
    details: Dict[str, Dict[str, Any]],
    batch_tasks: List[Tuple[int, int]],
    args: argparse.Namespace,
) -> int:
    items: List[Dict[str, Any]] = []

    for source_line_idx, step_idx in batch_tasks:
        item = build_generation_item(
            tokenizer=tokenizer,
            steps_records=steps_records,
            source_line_idx=source_line_idx,
            step_idx=step_idx,
            enable_thinking=args.enable_thinking,
        )
        items.append(item)

    prompts = [item["prompt"] for item in items]

    request_outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )

    if len(request_outputs) != len(items):
        raise RuntimeError(
            f"vLLM returned {len(request_outputs)} outputs for {len(items)} prompts."
        )

    for item, request_output in zip(items, request_outputs):
        source_line_idx = item["source_line_idx"]
        sample_idx = item["sample_idx"]
        step_idx = item["step_idx"]
        gold_answer = item["gold_answer"]
        steps = item["steps"]
        prefix_text = item["prefix_text"]
        prompt = item["prompt"]
        end_token_index = item["end_token_index"]

        outputs = request_output.outputs

        continuation_records: List[Dict[str, Any]] = []
        correct_count = 0
        grade_error_count = 0

        for continuation_idx, output in enumerate(outputs):
            continuation_text = output.text
            full_output = prefix_text + continuation_text

            is_correct, solution_str, grade_error = judge_with_original_strategy(
                full_output=full_output,
                gold_answer=gold_answer,
            )

            if is_correct:
                correct_count += 1

            if grade_error is not None:
                grade_error_count += 1

            continuation_records.append(
                {
                    "continuation_idx": continuation_idx,
                    "continuation_text": continuation_text,
                    "full_output": full_output if args.save_full_output_in_details else None,
                    "solution_str": solution_str,
                    "is_correct": bool(is_correct),
                    "grade_error": grade_error,
                    "finish_reason": getattr(output, "finish_reason", None),
                }
            )

        K_actual = len(outputs)
        step_prospective_correctness = (
            float(correct_count) / float(K_actual) if K_actual > 0 else 0.0
        )

        steps_records[source_line_idx]["steps"][step_idx][
            "step_prospective_correctness"
        ] = step_prospective_correctness

        detail_key = make_detail_key(source_line_idx, sample_idx, step_idx)
        details[detail_key] = {
            "source_line_idx": source_line_idx,
            "sample_idx": sample_idx,
            "step_idx": step_idx,
            "num_steps": len(steps),
            "K": args.K,
            "K_actual": K_actual,
            "correct_count": correct_count,
            "grade_error_count": grade_error_count,
            "step_prospective_correctness": step_prospective_correctness,
            "step_end_token_index": end_token_index,
            "min_label_token_gap": args.min_label_token_gap,
            "problem": item["problem"],
            "gold_answer": gold_answer,
            "prefix_text": prefix_text if args.save_prefix_in_details else None,
            "prefix_step_indices": list(range(step_idx + 1)),
            "prefix_step_confidences": [
                steps[i].get("token_avg_confidence") for i in range(step_idx + 1)
            ],
            "prompt": prompt if args.save_prompt_in_details else None,
            "continuations": continuation_records,
        }

    return len(items)


def chunk_tasks(tasks: List[Tuple[int, int]], batch_size: int):
    for i in range(0, len(tasks), batch_size):
        yield tasks[i: i + batch_size]


def run_worker(args: argparse.Namespace) -> None:
    if args.K <= 0:
        raise ValueError("--K must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.save_every <= 0:
        raise ValueError("--save_every must be positive.")
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive.")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards.")
    if args.min_label_token_gap < 0:
        raise ValueError("--min_label_token_gap must be non-negative.")

    steps_path = args.steps_path
    details_path = args.details_path

    if not os.path.exists(steps_path):
        raise FileNotFoundError(f"steps_path not found: {steps_path}")

    if not args.no_backup:
        backup_path = backup_file(steps_path)
        print(f"[worker {args.shard_id}] Backup steps file to: {backup_path}")

    steps_records, steps_format = load_records(steps_path)
    print(f"[worker {args.shard_id}] Loaded steps records: {len(steps_records)}")

    details = load_existing_details(details_path)
    print(f"[worker {args.shard_id}] Loaded existing detail records: {len(details)}")

    filled = fill_steps_from_existing_details(
        steps_records=steps_records,
        details=details,
        overwrite=args.overwrite,
    )

    if filled > 0:
        print(f"[worker {args.shard_id}] Filled {filled} labels from details.")
        save_records(steps_path, steps_records, steps_format)

    (
        tasks,
        skipped_by_gap_count,
        already_done_selected_count,
        skipped_by_think_end_count,
    ) = collect_pending_tasks(
        steps_records=steps_records,
        overwrite=args.overwrite,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        min_label_token_gap=args.min_label_token_gap,
    )

    print(f"[worker {args.shard_id}] Pending steps: {len(tasks)}")
    print(f"[worker {args.shard_id}] Skipped by token gap: {skipped_by_gap_count}")
    print(f"[worker {args.shard_id}] Skipped by </think>: {skipped_by_think_end_count}")
    print(f"[worker {args.shard_id}] Already-done selected steps: {already_done_selected_count}")
    print(f"[worker {args.shard_id}] min_label_token_gap: {args.min_label_token_gap}")
    print(f"[worker {args.shard_id}] K: {args.K}")
    print(f"[worker {args.shard_id}] batch_size: {args.batch_size}")
    print(f"[worker {args.shard_id}] save_every: {args.save_every}")
    print(f"[worker {args.shard_id}] max_tokens: {args.max_tokens}")
    print(f"[worker {args.shard_id}] max_model_len: {args.max_model_len}")
    print(f"[worker {args.shard_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if len(tasks) == 0:
        print(f"[worker {args.shard_id}] Nothing to do.")
        return

    llm = load_vllm(args)
    tokenizer = llm.get_tokenizer()
    sampling_params = make_sampling_params(args)

    processed_since_save = 0
    total_processed = 0

    pbar = tqdm(
        total=len(tasks),
        desc=f"worker {args.shard_id}",
        unit="step",
        position=args.shard_id,
    )

    try:
        for batch_tasks in chunk_tasks(tasks, args.batch_size):
            first_line, first_step = batch_tasks[0]
            first_record = steps_records[first_line]
            first_sample = get_sample_idx(first_record, first_line)

            pbar.set_postfix(
                {
                    "line": first_line,
                    "sample": first_sample,
                    "step": first_step,
                    "bs": len(batch_tasks),
                }
            )

            processed = process_batch(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                steps_records=steps_records,
                details=details,
                batch_tasks=batch_tasks,
                args=args,
            )

            processed_since_save += processed
            total_processed += processed
            pbar.update(processed)

            if processed_since_save >= args.save_every:
                save_records(steps_path, steps_records, steps_format)
                save_details(details_path, details)
                processed_since_save = 0

        save_records(steps_path, steps_records, steps_format)
        save_details(details_path, details)

    except KeyboardInterrupt:
        print(f"\n[worker {args.shard_id}] Interrupted. Saving progress...")
        save_records(steps_path, steps_records, steps_format)
        save_details(details_path, details)
        raise

    finally:
        pbar.close()

    print(f"[worker {args.shard_id}] Done. Processed steps: {total_processed}")
    print(f"[worker {args.shard_id}] Updated steps file: {steps_path}")
    print(f"[worker {args.shard_id}] Updated details file: {details_path}")


# =========================
# launcher 逻辑
# =========================

def parse_gpu_groups(gpu_groups: str) -> List[str]:
    groups = []
    for group in gpu_groups.split(";"):
        group = group.strip()
        if group:
            groups.append(group)

    if not groups:
        raise ValueError("--gpu_groups cannot be empty.")

    return groups


def default_output_path(input_path: str) -> str:
    p = Path(input_path)
    if p.suffix:
        return str(p.with_name(p.stem + "_monte_carlo_labeled" + p.suffix))
    return str(p) + "_monte_carlo_labeled.jsonl"


def prepare_work_dir(args: argparse.Namespace, num_shards: int) -> Tuple[List[str], List[str]]:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.steps_path)
    suffix = input_path.suffix if input_path.suffix else ".jsonl"

    shard_steps_paths: List[str] = []
    shard_details_paths: List[str] = []

    for shard_id in range(num_shards):
        shard_steps = work_dir / f"{input_path.stem}_shard{shard_id}{suffix}"
        shard_details = work_dir / f"steps_monte_carlo_details_shard{shard_id}.jsonl"

        if args.restart:
            if shard_steps.exists():
                shard_steps.unlink()
            if shard_details.exists():
                shard_details.unlink()
            if Path(str(shard_steps) + ".bak").exists():
                Path(str(shard_steps) + ".bak").unlink()

        if not shard_steps.exists():
            shutil.copy2(args.steps_path, shard_steps)
            print(f"[launcher] Copied input to shard {shard_id}: {shard_steps}")
        else:
            print(f"[launcher] Reusing existing shard input: {shard_steps}")

        shard_steps_paths.append(str(shard_steps))
        shard_details_paths.append(str(shard_details))

    return shard_steps_paths, shard_details_paths


def build_worker_command(
    args: argparse.Namespace,
    shard_id: int,
    num_shards: int,
    shard_steps_path: str,
    shard_details_path: str,
    gpu_group: str,
) -> List[str]:
    tp_size = args.tensor_parallel_size
    if tp_size is None:
        tp_size = len([x for x in gpu_group.split(",") if x.strip()])

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "worker",

        "--steps_path", shard_steps_path,
        "--details_path", shard_details_path,
        "--model_path", args.model_path,

        "--K", str(args.K),
        "--batch_size", str(args.batch_size),
        "--save_every", str(args.save_every),

        "--max_tokens", str(args.max_tokens),
        "--max_model_len", str(args.max_model_len),
        "--min_label_token_gap", str(args.min_label_token_gap),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),

        "--tensor_parallel_size", str(tp_size),
        "--gpu_memory_utilization", str(args.gpu_memory_utilization),
        "--dtype", str(args.dtype),

        "--num_shards", str(num_shards),
        "--shard_id", str(shard_id),

        "--no_backup",
    ]

    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if args.vllm_enforce_eager:
        cmd.append("--vllm_enforce_eager")

    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed + shard_id)])

    if args.overwrite:
        cmd.append("--overwrite")

    if args.save_prefix_in_details:
        cmd.append("--save_prefix_in_details")
    if args.save_prompt_in_details:
        cmd.append("--save_prompt_in_details")
    if args.save_full_output_in_details:
        cmd.append("--save_full_output_in_details")

    return cmd


def merge_shard_steps(
    original_steps_path: str,
    shard_steps_paths: List[str],
    output_steps_path: str,
    inplace: bool,
) -> None:
    original_records, original_format = load_records(original_steps_path)
    shard_records_list = []

    for path in shard_steps_paths:
        records, fmt = load_records(path)
        if len(records) != len(original_records):
            raise ValueError(
                f"Shard file length mismatch: {path}, "
                f"got {len(records)}, expected {len(original_records)}"
            )
        shard_records_list.append(records)

    num_shards = len(shard_records_list)
    merged_records: List[Dict[str, Any]] = []

    for source_line_idx in range(len(original_records)):
        shard_id = source_line_idx % num_shards
        merged_records.append(shard_records_list[shard_id][source_line_idx])

    save_records(output_steps_path, merged_records, original_format)

    if inplace:
        backup_path = backup_file(original_steps_path)
        print(f"[launcher] Backup original steps to: {backup_path}")
        shutil.copy2(output_steps_path, original_steps_path)
        print(f"[launcher] Inplace updated original steps: {original_steps_path}")

    print(f"[launcher] Merged labeled steps saved to: {output_steps_path}")


def merge_shard_details(shard_details_paths: List[str], output_details_path: str) -> None:
    all_records: List[Dict[str, Any]] = []

    for path in shard_details_paths:
        if not os.path.exists(path):
            print(f"[launcher] Warning: details file not found, skip: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[launcher] Warning: broken detail line {line_idx + 1} in {path}")

    def sort_key(x: Dict[str, Any]) -> Tuple[int, int]:
        return (
            int(x.get("source_line_idx", 10**12)),
            int(x.get("step_idx", 10**12)),
        )

    all_records.sort(key=sort_key)

    tmp_path = output_details_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, output_details_path)
    print(f"[launcher] Merged details saved to: {output_details_path}")


def run_launcher(args: argparse.Namespace) -> None:
    gpu_groups = parse_gpu_groups(args.gpu_groups)
    num_shards = len(gpu_groups)

    print(f"[launcher] GPU groups: {gpu_groups}")
    print(f"[launcher] Number of shards/workers: {num_shards}")
    print(f"[launcher] min_label_token_gap: {args.min_label_token_gap}")

    if args.output_steps_path is None:
        args.output_steps_path = default_output_path(args.steps_path)

    if args.output_details_path is None:
        args.output_details_path = str(
            Path(args.output_steps_path).with_name("steps_monte_carlo_details_all.jsonl")
        )

    shard_steps_paths, shard_details_paths = prepare_work_dir(args, num_shards)

    processes: List[subprocess.Popen] = []

    try:
        for shard_id, gpu_group in enumerate(gpu_groups):
            cmd = build_worker_command(
                args=args,
                shard_id=shard_id,
                num_shards=num_shards,
                shard_steps_path=shard_steps_paths[shard_id],
                shard_details_path=shard_details_paths[shard_id],
                gpu_group=gpu_group,
            )

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_group

            log_path = Path(args.work_dir) / f"worker_shard{shard_id}.log"
            log_f = open(log_path, "w", encoding="utf-8")

            print(f"[launcher] Starting worker {shard_id} on GPUs {gpu_group}")
            print(f"[launcher] Log: {log_path}")

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc._log_file = log_f  # type: ignore[attr-defined]
            processes.append(proc)

        failed = False

        while True:
            all_done = True
            for shard_id, proc in enumerate(processes):
                ret = proc.poll()
                if ret is None:
                    all_done = False
                elif ret != 0:
                    failed = True

            if all_done:
                break

            time.sleep(10)

        for proc in processes:
            if hasattr(proc, "_log_file"):
                proc._log_file.close()  # type: ignore[attr-defined]

        return_codes = [proc.returncode for proc in processes]
        print(f"[launcher] Worker return codes: {return_codes}")

        if failed:
            raise RuntimeError(
                "At least one worker failed. "
                f"Please check logs in {args.work_dir}/worker_shard*.log"
            )

        merge_shard_steps(
            original_steps_path=args.steps_path,
            shard_steps_paths=shard_steps_paths,
            output_steps_path=args.output_steps_path,
            inplace=args.inplace,
        )

        merge_shard_details(
            shard_details_paths=shard_details_paths,
            output_details_path=args.output_details_path,
        )

        print("[launcher] All done.")

    except KeyboardInterrupt:
        print("\n[launcher] Interrupted. Terminating workers...")
        for proc in processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)

        time.sleep(5)

        for proc in processes:
            if proc.poll() is None:
                proc.terminate()

        for proc in processes:
            if hasattr(proc, "_log_file"):
                proc._log_file.close()  # type: ignore[attr-defined]

        print("[launcher] Workers terminated. Existing shard files can be reused for resume.")
        raise


# =========================
# 参数
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="launcher",
        choices=["launcher", "worker"],
        help="launcher: 一键启动多 worker；worker: 内部模式，一般不要手动用。",
    )

    parser.add_argument(
        "--steps_path",
        type=str,
        default="steps.jsonl",
        help="输入 steps.jsonl 或 steps.json。",
    )

    parser.add_argument(
        "--details_path",
        type=str,
        default="steps_monte_carlo_details.jsonl",
        help="worker 模式下的 details 输出路径。",
    )

    parser.add_argument(
        "--output_steps_path",
        type=str,
        default=None,
        help="launcher 合并后的 labeled steps 输出路径。",
    )

    parser.add_argument(
        "--output_details_path",
        type=str,
        default=None,
        help="launcher 合并后的 details 输出路径。",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="vLLM 模型路径或 HuggingFace 模型名。",
    )

    parser.add_argument(
        "--enable_thinking",
        type=parse_bool,
        default=True,
        help="是否启用 Qwen3 thinking 模式，可设置 true/false，默认 true。",
    )

    parser.add_argument(
        "--gpu_groups",
        type=str,
        default="0,1;2,3",
        help='launcher 模式使用。例如 "0,1;2,3" 表示启动两个 worker。',
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default="mc_parallel_work",
        help="launcher 存放 shard 文件和 log 的目录。",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="launcher 模式下删除已有 shard 文件，从头开始。",
    )

    parser.add_argument(
        "--inplace",
        action="store_true",
        help="launcher 合并完成后，用输出文件覆盖原 steps_path。",
    )

    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=50)

    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=10240)

    parser.add_argument(
        "--min_label_token_gap",
        type=int,
        default=128,
        help=(
            "相邻被计算 label 的 step 结束 token 位置的最小间隔。"
            "当前 step 的结束位置用下一个 step 的 first_token_index 表示。"
            "如果设为 0，则不启用 token-gap 跳过逻辑。"
        ),
    )

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="每个 worker 内部的 TP 大小。默认等于该 gpu_group 内 GPU 数。",
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

    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_backup", action="store_true")

    parser.add_argument("--save_prefix_in_details", action="store_true")
    parser.add_argument("--save_prompt_in_details", action="store_true")
    parser.add_argument("--save_full_output_in_details", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "worker":
        run_worker(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
