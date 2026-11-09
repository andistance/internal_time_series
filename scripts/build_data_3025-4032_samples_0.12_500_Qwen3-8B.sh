#!/usr/bin/env bash
# 实验：Qwen3-8B thinking，自适应 phase 配置 0.12_500，Monte Carlo 每终点续写 8 次。
# 样本范围：40315 条固定顺序题库中的第 3025～4032 条，共 1008 条；不截断输入。
# 原始题库分片：--num_input_parts 40 --input_part_id 3（分片编号从 0 开始）。
# 使用前修改本机模型路径和 GPU 分组；现有相对路径要求从 internal_time_series 的上一级目录启动。
# 用途：按顺序完成生成、判题和整份标注；使用 bash -e 启动可在命令失败时停止。

# 第 1 步：为第 3025～4032 条题目生成推理轨迹、计算熵，并按 0.12_500 自适应划分 phase。
python internal_time_series/build_data/1_generate_internal_time_series_with_phases_vllm_qwen.py --input_json internal_time_series/deepscaler/data/train/deepscaler_random_shuffled.json --model_path /mnt/wangchenglong/models/Qwen3-8B --summary_jsonl internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard3_total40/summary.jsonl --step_jsonl internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard3_total40/steps.jsonl --max_tokens 16384 --max_model_len 16384 --tensor_parallel_size 1 --gpus "0;1;2;3;4;5" --gpu_memory_utilization 0.98 --num_input_parts 40 --input_part_id 3 --phase_partition_algorithm adaptive --high_entropy_ratio 0.12 --target_phase_tokens 500 --entropy_chunk_size 16

# 第 2 步：判定完整回答是否正确，将样本级 is_correct 写回 summary.jsonl 和 steps.jsonl。
python internal_time_series/build_data/2_validate_answer_correctness.py --summary_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard3_total40/summary.jsonl --steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard3_total40/steps.jsonl --no_backup

# 第 3 步：对完整 steps.jsonl 的选中 phase 终点各续写 8 次，生成 labeled 文件与 Monte Carlo 详情。
python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py --steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard3_total40/steps.jsonl --output_steps_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard3_total40/steps_labeled.jsonl --output_details_path internal_time_series/data/train/deepscaler_phases_0.12_500_16384/Qwen3-8B/shard3_total40/steps_monte_carlo_details.jsonl --model_path /mnt/wangchenglong/models/Qwen3-8B --tensor_parallel_size 2 --gpu_groups "0,1;2,3;4,5" --work_dir "internal_time_series/log/mc_parallel_work_shard3_total40" --K 8 --batch_size 8 --save_every 30 --max_tokens 16384 --max_model_len 16384 --gpu_memory_utilization 0.98 --vllm_enforce_eager

