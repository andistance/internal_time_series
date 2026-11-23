# Qwen3.8-27B
# 分段配置：original，目标 phase 数 8，熵阈值 1.2；生成和标注的 max_tokens、max_model_len 均为 8192。
# 样本范围：40315 条固定顺序题库中的第 8065～9072 条，共 1008 条。
# 原始题库分片：--num_input_parts 40 --input_part_id 8（分片编号从 0 开始）。
# 数据与工作目录使用 deepscaler_phases_8_1.2_8192。

# 第 1 步：为第 8065～9072 条题目生成推理轨迹并按 original 算法划分 phase（num_segments=8，entropy_threshold=1.2）。
python internal_time_series/build_data/1_generate_internal_time_series_with_phases_vllm_qwen.py --input_json internal_time_series/deepscaler/data/train/deepscaler_random_shuffled.json --model_path /mnt/wangchenglong/models/Qwen3.8-27B --summary_jsonl internal_time_series/data/train/deepscaler_phases_8_1.2_8192/Qwen3.8-27B/shard8_total40/summary.jsonl --step_jsonl internal_time_series/data/train/deepscaler_phases_8_1.2_8192/Qwen3.8-27B/shard8_total40/steps.jsonl --max_tokens 8192 --max_model_len 8192 --tensor_parallel_size 1 --gpus "0;1;2;3;4;5" --gpu_memory_utilization 0.98 --num_input_parts 40 --input_part_id 8 --phase_partition_algorithm original --num_segments 8 --entropy_threshold 1.2 --entropy_chunk_size 16

# 第 2 步：判题并将样本级 is_correct 写回 summary.jsonl 和 steps.jsonl。
python internal_time_series/build_data/2_validate_answer_correctness.py --summary_path internal_time_series/data/train/deepscaler_phases_8_1.2_8192/Qwen3.8-27B/shard8_total40/summary.jsonl --steps_path internal_time_series/data/train/deepscaler_phases_8_1.2_8192/Qwen3.8-27B/shard8_total40/steps.jsonl --no_backup

# 第 3 步：整份 Monte Carlo 标注，每个选中终点续写 8 次，batch_size=32，输出标签和详情。
python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py --steps_path internal_time_series/data/train/deepscaler_phases_8_1.2_8192/Qwen3.8-27B/shard8_total40/steps.jsonl --output_steps_path internal_time_series/data/train/deepscaler_phases_8_1.2_8192/Qwen3.8-27B/shard8_total40/steps_labeled.jsonl --output_details_path internal_time_series/data/train/deepscaler_phases_8_1.2_8192/Qwen3.8-27B/shard8_total40/steps_monte_carlo_details.jsonl --model_path /mnt/wangchenglong/models/Qwen3.8-27B --tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5" --work_dir "internal_time_series/log/deepscaler_phases_8_1.2_8192/mc_parallel_work_Qwen3.8-27B_shard8_total40" --K 8 --batch_size 8 --save_every 30 --max_tokens 8192 --max_model_len 8192 --gpu_memory_utilization 0.98