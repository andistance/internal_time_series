#!/usr/bin/env bash
# 实验：Qwen3-8B thinking，自适应 phase 配置 0.12_500，Monte Carlo 每终点续写 8 次。
# 样本范围：40315 条固定顺序题库中的第 1009～2016 条，共 1008 条；不截断输入。
# 原始题库分片：--num_input_parts 40 --input_part_id 1（分片编号从 0 开始）。
# 使用前修改本机模型路径和 GPU 分组；现有相对路径要求从 internal_time_series 的上一级目录启动。
# 用途：供人分发的命令清单；生成、判题和拆分只做一次，各台服务器只执行自己负责的 part 标注命令。
# 所有 part 标注完成并收集后再执行最后的合并命令；整份运行本文件会按列出的顺序执行。

# 第 1 步：为第 1009～2016 条题目生成推理轨迹、计算熵，并按 0.12_500 自适应划分 phase。
python internal_time_series/build_data/1_generate_internal_time_series_with_phases_vllm_qwen.py --input_json internal_time_series/deepscaler/data/train/deepscaler_random_shuffled.json --model_path /mnt/wangchenglong/models/Qwen3-8B --summary_jsonl internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/summary.jsonl --step_jsonl internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps.jsonl --max_tokens 16384 --max_model_len 16384 --tensor_parallel_size 1 --gpus "0;1;2;3;4;5" --gpu_memory_utilization 0.98 --num_input_parts 40 --input_part_id 1 --phase_partition_algorithm adaptive --high_entropy_ratio 0.12 --target_phase_tokens 500 --entropy_chunk_size 16

# 第 2 步：判定完整回答是否正确，将样本级 is_correct 写回 summary.jsonl 和 steps.jsonl。
python internal_time_series/build_data/2_validate_answer_correctness.py --summary_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/summary.jsonl --steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps.jsonl --no_backup

# 第 3 步：将 steps.jsonl 按完整样本拆成 5 个 part，供人工分配到不同服务器标注。
python internal_time_series/build_data/3_split_step_jsonl_to_shards.py --input_jsonl internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps.jsonl --output_folder internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_shards

# 标注 part_01：仅在负责本 part 的服务器执行，对选中 phase 终点各续写 8 次并保存标签与详情。
python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py --steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_shards/part_01.jsonl --output_steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_labeled_shards/part_01.jsonl --output_details_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/step_monte_carlo_details_shards/part_01.jsonl --model_path /mnt/wangchenglong/models/Qwen3-8B --tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5" --work_dir "internal_time_series/log/mc_parallel_work_shard1_total40_part_01" --K 8 --batch_size 8 --save_every 30 --max_tokens 16384 --max_model_len 16384 --gpu_memory_utilization 0.98

# 标注 part_02：仅在负责本 part 的服务器执行，对选中 phase 终点各续写 8 次并保存标签与详情。
python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py --steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_shards/part_02.jsonl --output_steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_labeled_shards/part_02.jsonl --output_details_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/step_monte_carlo_details_shards/part_02.jsonl --model_path /mnt/wangchenglong/models/Qwen3-8B --tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5" --work_dir "internal_time_series/log/mc_parallel_work_shard1_total40_part_02" --K 8 --batch_size 8 --save_every 30 --max_tokens 16384 --max_model_len 16384 --gpu_memory_utilization 0.98

# 标注 part_03：仅在负责本 part 的服务器执行，对选中 phase 终点各续写 8 次并保存标签与详情。
python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py --steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_shards/part_03.jsonl --output_steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_labeled_shards/part_03.jsonl --output_details_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/step_monte_carlo_details_shards/part_03.jsonl --model_path /mnt/wangchenglong/models/Qwen3-8B --tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5" --work_dir "internal_time_series/log/mc_parallel_work_shard1_total40_part_03" --K 8 --batch_size 8 --save_every 30 --max_tokens 16384 --max_model_len 16384 --gpu_memory_utilization 0.98

# 标注 part_04：仅在负责本 part 的服务器执行，对选中 phase 终点各续写 8 次并保存标签与详情。
python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py --steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_shards/part_04.jsonl --output_steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_labeled_shards/part_04.jsonl --output_details_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/step_monte_carlo_details_shards/part_04.jsonl --model_path /mnt/wangchenglong/models/Qwen3-8B --tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5" --work_dir "internal_time_series/log/mc_parallel_work_shard1_total40_part_04" --K 8 --batch_size 8 --save_every 30 --max_tokens 16384 --max_model_len 16384 --gpu_memory_utilization 0.98

# 标注 part_05：仅在负责本 part 的服务器执行，对选中 phase 终点各续写 8 次并保存标签与详情。
python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py --steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_shards/part_05.jsonl --output_steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_labeled_shards/part_05.jsonl --output_details_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/step_monte_carlo_details_shards/part_05.jsonl --model_path /mnt/wangchenglong/models/Qwen3-8B --tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5" --work_dir "internal_time_series/log/mc_parallel_work_shard1_total40_part_05" --K 8 --batch_size 8 --save_every 30 --max_tokens 16384 --max_model_len 16384 --gpu_memory_utilization 0.98

# 最后一步：收齐并确认 5 个 labeled part 完成后，按 part 编号合并为 steps_labeled.jsonl。
python internal_time_series/build_data/5_merge_step_labeled_jsonls.py --folder internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_labeled_shards --output internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard1_total40/steps_labeled.jsonl

