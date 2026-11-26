from pathlib import Path
import shutil

# 可以填写多个指定文件夹。
BASE_DIRS = [
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/aime_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/calendar_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/causal_judgement_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/gpqa_diamond_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/logical_deduction_five_objects_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/math_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/meeting_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/temporal_sequences_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/tracking_shuffled_objects_five_objects_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/trip_0.12_500_16384"),
    Path("internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/web_of_lies_0.12_500_16384"),
]

SOURCE_FOLDER = (
    "Qwen3-8B_deepscaler5000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_5000_r0.25_1e-6_b1"
)

TARGET_FOLDERS = [
    "Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4",

    #"Qwen2.5-Math-PRM-7B_phases",
    #"Qwen-2.5-7B-FoVer-PRM-2026_phases",
    #"VersaPRM_phases",
    #"RLHFlow-PRM-Deepseek-8B_phases",
    #"RLHFlow-PRM-Mistral-8B_phases",
    #"Math-Shepherd-PRM-7B_phases",
]

CACHE_LAYOUTS = [
    # NaturalPlan：calendar、meeting、trip
    (
        ".staging/.bon_generation.jsonl",
        #".staging/.bon_prepared.jsonl",
        ".bon_token_entropies.jsonl",
    ),
    # 独立评测脚本
    (
        ".generation_tmp.jsonl",
        ".token_entropies.jsonl",
        #".prepared_tmp.jsonl",
    ),
    # transformer_bon
    (
        ".bon_generation.jsonl",
        ".bon_token_entropies.jsonl",
        #".bon_prepared.jsonl",
    ),
]


def prepare_copy_jobs(base_dir):
    """检查源文件和目标文件，返回需要执行的复制任务。"""
    base_dir = Path(base_dir).expanduser().resolve(strict=True)
    source = base_dir / SOURCE_FOLDER
    if not source.is_dir():
        raise NotADirectoryError(f"源文件夹不存在：{source}")

    files = next(
        (
            layout
            for layout in CACHE_LAYOUTS
            if all((source / name).is_file() for name in layout)
        ),
        None,
    )
    if files is None:
        raise FileNotFoundError(
            f"源文件夹中未找到完整的三个缓存文件：{source}\n"
            f"支持的布局：{CACHE_LAYOUTS}"
        )

    jobs = []
    for folder in TARGET_FOLDERS:
        for name in files:
            destination = base_dir / folder / name
            if destination.exists():
                raise FileExistsError(f"目标文件已存在：{destination}")
            jobs.append((source / name, destination))

    return jobs


def main():
    # 所有指定文件夹检查通过后，再开始复制。
    jobs = []
    seen = set()

    for folder in BASE_DIRS:
        base_dir = Path(folder).expanduser().resolve(strict=True)
        if base_dir in seen:
            continue
        seen.add(base_dir)
        jobs.extend(prepare_copy_jobs(base_dir))

    for source, destination in jobs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"{source} -> {destination}")

    print(
        f"复制完成：{len(seen)} 个指定文件夹，"
        f"每个创建 {len(TARGET_FOLDERS)} 个 PRM 目录，"
        f"共复制 {len(jobs)} 个文件。"
    )


if __name__ == "__main__":
    main()