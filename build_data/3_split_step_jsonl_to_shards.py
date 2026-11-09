import argparse
import os
import json


def split_jsonl(input_file, output_dir, num_parts=10):
    """
    将jsonl文件平均拆分成多个jsonl文件

    Args:
        input_file: 输入jsonl路径
        output_dir: 输出文件夹
        num_parts: 拆分份数
    """

    os.makedirs(output_dir, exist_ok=True)

    # 读取所有数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = [line for line in f if line.strip()]

    total = len(data)
    print(f"总数据量: {total}")

    # 基础数量
    base = total // num_parts
    remainder = total % num_parts

    start = 0

    for i in range(num_parts):
        # 前 remainder 个文件多一个元素
        count = base + (1 if i < remainder else 0)

        end = start + count

        output_file = os.path.join(
            output_dir,
            f"part_{i+1:02d}.jsonl"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            for line in data[start:end]:
                f.write(line)

        print(
            f"{output_file}: {count} 条数据"
        )

        start = end


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_folder", type=str, required=True, help="输出文件夹路径")
    args = parser.parse_args()

    input_jsonl = args.input_jsonl
    output_folder = args.output_folder

    split_jsonl(
        input_jsonl,
        output_folder,
        num_parts=5
    )
