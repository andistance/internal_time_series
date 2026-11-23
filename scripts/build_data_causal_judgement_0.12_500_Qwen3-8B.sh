# Qwen3-8B：causal_judgement 测试集
# 分段配置：adaptive，high_entropy_ratio=0.12，target_phase_tokens=500；生成和标注的 max_tokens、max_model_len 均为 16384。
# 输入：causal_judgement.json 的全部样本，不对输入数据分片。
# 数据与工作目录使用 causal_judgement_0.12_500_16384。

# 第 1 步：生成 causal_judgement 的推理轨迹、计算熵，并按 0.12_500 自适应划分 phase。
python internal_time_series/build_data/1_generate_internal_time_series_with_phases_vllm_qwen.py --input_json internal_time_series/deepscaler/data/test/causal_judgement.json --model_path /mnt/wangchenglong/models/Qwen3-8B --summary_jsonl internal_time_series/data/train/causal_judgement_0.12_500_16384/Qwen3-8B/summary.jsonl --step_jsonl internal_time_series/data/train/causal_judgement_0.12_500_16384/Qwen3-8B/steps.jsonl --max_tokens 16384 --max_model_len 16384 --tensor_parallel_size 1 --gpus "0;1;2;3;4;5" --gpu_memory_utilization 0.98 --phase_partition_algorithm adaptive --high_entropy_ratio 0.12 --target_phase_tokens 500 --entropy_chunk_size 16

# 第 2 步：判题并将样本级 is_correct 写回 summary.jsonl 和 steps.jsonl。
python internal_time_series/build_data/2_validate_answer_correctness.py --summary_path internal_time_series/data/train/causal_judgement_0.12_500_16384/Qwen3-8B/summary.jsonl --steps_path internal_time_series/data/train/causal_judgement_0.12_500_16384/Qwen3-8B/steps.jsonl --no_backup

# 第 3 步：整份 Monte Carlo 标注，每个选中终点续写 8 次，batch_size=8，输出标签和详情。
python internal_time_series/build_data/4_build_phase_prospective_correctness_label_via_monte_carlo_qwen.py --steps_path internal_time_series/data/train/causal_judgement_0.12_500_16384/Qwen3-8B/steps.jsonl --output_steps_path internal_time_series/data/train/causal_judgement_0.12_500_16384/Qwen3-8B/steps_labeled.jsonl --output_details_path internal_time_series/data/train/causal_judgement_0.12_500_16384/Qwen3-8B/steps_monte_carlo_details.jsonl --model_path /mnt/wangchenglong/models/Qwen3-8B --tensor_parallel_size 1 --gpu_groups "0;1;2;3;4;5" --work_dir "internal_time_series/log/causal_judgement_0.12_500_16384/mc_parallel_work_Qwen3-8B" --K 8 --batch_size 8 --save_every 30 --max_tokens 16384 --max_model_len 16384 --gpu_memory_utilization 0.98
