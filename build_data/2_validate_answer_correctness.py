#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from deepscaler.rewards.math_utils.utils import grade_answer_verl


def detect_file_format(path: str) -> str:
    """
    自动判断是 json array 还是 jsonl。
    返回:
        "json"  或  "jsonl"
    """
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return "jsonl"
    if lower.endswith(".json"):
        return "json"

    # 后缀不明确时，尝试读第一个非空字符
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
    """
    读取 json array 或 jsonl。
    返回:
        records, file_format
    """
    file_format = detect_file_format(path)

    if file_format == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} should be a JSON array.")
        return data, file_format

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL decode error in {path}, line {line_idx + 1}: {e}")

    return records, file_format


def save_records(path: str, records: List[Dict[str, Any]], file_format: str) -> None:
    """
    按原格式写回。
    """
    tmp_path = path + ".tmp"

    if file_format == "json":
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")
    elif file_format == "jsonl":
        with open(tmp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    os.replace(tmp_path, path)


def backup_file(path: str) -> str:
    """
    覆盖前备份原文件。
    """
    backup_path = path + ".bak"
    shutil.copy2(path, backup_path)
    return backup_path


def get_solution_after_think(full_output: Any) -> str:
    """
    取 full_output 中 </think> 之后的部分作为 solution_str。

    如果不存在 </think>，则退化为使用完整输出。
    """
    if full_output is None:
        return ""

    full_output = str(full_output)

    if "</think>" in full_output:
        return full_output.split("</think>", 1)[1].strip()

    return full_output.strip()


def get_ground_truth(record: Dict[str, Any]) -> Any:
    """
    兼容不同字段名。
    你前面的 summary.jsonl 里是 gold_answer。
    原始数据集里是 answer。
    """
    if "gold_answer" in record:
        return record.get("gold_answer")
    if "answer" in record:
        return record.get("answer")
    if "ground_truth" in record:
        return record.get("ground_truth")
    return None


def make_record_key(record: Dict[str, Any]) -> Tuple[str, Any]:
    """
    用于在 summary 和 steps 之间匹配样本。

    优先 unique_id，其次 sample_idx。
    """
    if record.get("unique_id") is not None:
        return ("unique_id", record.get("unique_id"))

    if record.get("sample_idx") is not None:
        return ("sample_idx", record.get("sample_idx"))

    raise ValueError(
        "Record has neither unique_id nor sample_idx, cannot match summary and steps."
    )


def build_steps_index(steps_records: List[Dict[str, Any]]) -> Dict[Tuple[str, Any], int]:
    """
    为 steps 文件建立索引。
    """
    index = {}

    for i, record in enumerate(steps_records):
        key = make_record_key(record)
        if key in index:
            raise ValueError(f"Duplicate key in steps file: {key}")
        index[key] = i

    return index


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary_path",
        type=str,
        required=True,
        help="summary.json 或 summary.jsonl 路径。",
    )

    parser.add_argument(
        "--steps_path",
        type=str,
        required=True,
        help="steps.json 或 steps.jsonl 路径。",
    )

    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="默认会生成 .bak 备份；加上该参数则不备份。",
    )

    args = parser.parse_args()

    summary_records, summary_format = load_records(args.summary_path)
    steps_records, steps_format = load_records(args.steps_path)

    print(f"Loaded summary records: {len(summary_records)} from {args.summary_path}")
    print(f"Loaded steps records: {len(steps_records)} from {args.steps_path}")

    if not args.no_backup:
        summary_backup = backup_file(args.summary_path)
        steps_backup = backup_file(args.steps_path)
        print(f"Backup summary to: {summary_backup}")
        print(f"Backup steps to: {steps_backup}")

    steps_index = build_steps_index(steps_records)

    num_correct = 0
    num_total = 0
    num_missing_steps_match = 0
    num_grade_error = 0

    for record in tqdm(summary_records, desc="Grading"):
        ground_truth = get_ground_truth(record)
        full_output = record.get("full_output")
        solution_str = get_solution_after_think(full_output)

        try:
            is_correct = bool(grade_answer_verl(solution_str, ground_truth))
        except Exception as e:
            is_correct = False
            num_grade_error += 1
            record["grade_error"] = repr(e)

        record["is_correct"] = is_correct

        key = make_record_key(record)
        if key in steps_index:
            steps_records[steps_index[key]]["is_correct"] = is_correct
        else:
            num_missing_steps_match += 1

        num_total += 1
        if is_correct:
            num_correct += 1

    save_records(args.summary_path, summary_records, summary_format)
    save_records(args.steps_path, steps_records, steps_format)

    acc = num_correct / num_total if num_total > 0 else 0.0

    print("Done.")
    print(f"Total: {num_total}")
    print(f"Correct: {num_correct}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Grade errors: {num_grade_error}")
    print(f"Missing steps matches: {num_missing_steps_match}")
    print(f"Updated summary file: {args.summary_path}")
    print(f"Updated steps file: {args.steps_path}")


if __name__ == "__main__":
    main()