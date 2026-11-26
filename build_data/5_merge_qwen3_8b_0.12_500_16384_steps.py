"""检查并按题目范围合并 Qwen3-8B 的七份 steps_labeled.jsonl（共 6048 条）。"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


SHARDS = (
    ("shard0_total80", 504),
    ("shard1_total80", 504),
    ("shard1_total40", 1008),
    ("shard2_total40", 1008),
    ("shard3_total40", 1008),
    ("shard4_total40", 1008),
    ("shard6_total40", 1008),
)
INPUT_NAME = "steps_labeled.jsonl"
OUTPUT_NAME = "steps_labeled_1-5040_6049-7056.jsonl"
DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B"
)


def records(path):
    """与通用合并脚本一致跳过空行，并检查非空行是否为有效 JSON。"""
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if line:
                try:
                    json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON: {exc}") from exc
                yield line


def validate_shards(input_root):
    errors = []
    for shard, expected in SHARDS:
        path = input_root / shard / INPUT_NAME
        if not path.is_file():
            errors.append(f"缺少文件: {path}")
            continue
        try:
            count = sum(1 for _ in records(path))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        print(f"{shard}/{INPUT_NAME}: {count} 条（预期 {expected} 条）")
        if count != expected:
            errors.append(f"{path}: 预期 {expected} 条，实际 {count} 条")
    if errors:
        raise ValueError("检查未通过，未写入合并文件：\n" + "\n".join(errors))
    print(f"检查通过：七份文件共 {sum(count for _, count in SHARDS)} 条。")


def merge_shards(input_root, output_file):
    input_root = Path(input_root)
    output_file = Path(output_file)
    for shard, _ in SHARDS:
        if output_file.resolve() == (input_root / shard / INPUT_NAME).resolve():
            raise ValueError("输出路径不能与输入文件相同")
    validate_shards(input_root)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=output_file.parent, prefix=output_file.name + ".", suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            total = 0
            for shard, expected in SHARDS:
                count = 0
                for line in records(input_root / shard / INPUT_NAME):
                    output.write(line + "\n")
                    count += 1
                if count != expected:
                    raise ValueError(f"{shard} 在合并时条数发生变化，未替换输出文件")
                total += count
        os.replace(temporary_path, output_file)
        print(f"合并完成：{total} 条 -> {output_file}")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_ROOT,
                        help="包含七个 shard 目录的 Qwen3-8B 数据根目录")
    parser.add_argument("--output", type=Path, help="默认写到 input_root 下的约定文件名")
    parser.add_argument("--check_only", action="store_true", help="仅检查文件及条数，不合并")
    args = parser.parse_args()
    try:
        if args.check_only:
            validate_shards(args.input_root)
        else:
            merge_shards(args.input_root, args.output or args.input_root / OUTPUT_NAME)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
