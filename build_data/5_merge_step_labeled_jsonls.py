import argparse
import os
from pathlib import Path
import re
import sys
import tempfile


def filename_key(path):
    # 数字按大小排序，例如 part_2.jsonl 排在 part_10.jsonl 前面。
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ], path.name


def merge_jsonl_parts(folder_path, output_file):
    """合并文件夹中的所有 JSONL 文件，无需使用固定文件名。"""
    folder = Path(folder_path)
    output_path = Path(output_file)
    files = sorted(
        (
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".jsonl"
            and path.resolve() != output_path.resolve()
        ),
        key=filename_key,
    )
    if not files:
        raise ValueError(f"没有找到可合并的 JSONL 文件: {folder}")

    print("合并顺序：")
    for path in files:
        print(path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=output_path.parent, prefix=output_path.name + ".", suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            count = 0
            for path in files:
                print(f"正在合并: {path.name}")
                with path.open("r", encoding="utf-8") as source:
                    for line in source:
                        line = line.strip()
                        if line:
                            output.write(line + "\n")
                            count += 1
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"\n合并完成，共 {count} 条。")
    print(f"输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="合并文件夹中的所有 JSONL 文件，文件名不限。")
    parser.add_argument("--folder", required=True, help="输入 JSONL 文件所在文件夹")
    parser.add_argument("--output", required=True, help="合并后的输出文件路径")
    args = parser.parse_args()
    try:
        merge_jsonl_parts(args.folder, args.output)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
