import argparse
import os
import re


def merge_jsonl_parts(folder_path, output_file):
    """
    合并 part_0x.jsonl 文件
    按 part 后数字升序排列
    """

    # 匹配文件名
    pattern = re.compile(r"part_(\d+)\.jsonl$")

    files = []

    # 获取所有符合条件的文件
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            part_num = int(match.group(1))
            files.append((part_num, filename))

    # 按 part 数字排序
    files.sort(key=lambda x: x[0])

    if not files:
        print("没有找到符合格式的 jsonl 文件")
        return

    print("合并顺序：")
    for num, filename in files:
        print(f"part_{num}: {filename}")

    # 写入大文件
    with open(output_file, "w", encoding="utf-8") as fout:

        for part_num, filename in files:
            filepath = os.path.join(folder_path, filename)

            print(f"正在合并: {filename}")

            with open(filepath, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()

                    # 跳过空行
                    if line:
                        fout.write(line + "\n")

    print("\n合并完成!")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="输入 JSONL 文件所在文件夹")
    parser.add_argument("--output", type=str, required=True, help="合并后的输出文件路径")
    args = parser.parse_args()

    # jsonl所在文件夹
    folder = args.folder

    # 输出文件
    output = args.output

    merge_jsonl_parts(folder, output)
