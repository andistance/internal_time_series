#!/usr/bin/env bash
set -euo pipefail

# Run from the parent directory of internal_time_series.
# Resolve paths relative to this file so data/output paths stay valid.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}/../../.."

LOG_DIR="internal_time_series/results/eval/logs/$(basename -- "${BASH_SOURCE[0]}" .sh)/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p -- "$LOG_DIR"
pids=()
task_names=()

run_on_gpu() {
  local task_name="$1"
  local gpu_id="$2"
  shift 2
  # The parent sees this physical GPU as cuda:0. The --gpus arguments below
  # also use physical IDs because Python workers reset CUDA_VISIBLE_DEVICES.
  CUDA_VISIBLE_DEVICES="$gpu_id" "$@" >"${LOG_DIR}/${task_name}.log" 2>&1 &
  pids+=("$!")
  task_names+=("$task_name")
  printf 'Started %s on GPU %s, log: %s/%s.log\n' \
    "$task_name" "$gpu_id" "$LOG_DIR" "$task_name"
}


# aime
run_on_gpu aime 0 python -u internal_time_series/scripts/3_best_of_n_via_phases_scores_qwen.py \
  --input_json internal_time_series/deepscaler/data/test/aime.json \
  --model_path /mnt/wangchenglong/models/Qwen3-8B \
  --checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/aime_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --transformer_type "causal" \
  --value_device "cuda:0" \
  --scoring_batch_size 128 \
  --N 8 \
  --batch_size 1 \
  --max_tokens 16384 \
  --max_model_len 16384 \
  --gpus "0" \
  --tensor_parallel_size 1 \
  --vllm_enforce_eager \
  --gpu_memory_utilization 0.98 \
  --phase_partition_algorithm adaptive \
  --high_entropy_ratio 0.12 \
  --target_phase_tokens 500 \
  --ns 1 2 4 8 \
  --resume


# math
run_on_gpu math 1 python -u internal_time_series/scripts/3_best_of_n_via_phases_scores_qwen.py \
  --input_json internal_time_series/deepscaler/data/test/math.json \
  --model_path /mnt/wangchenglong/models/Qwen3-8B \
  --checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/math_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --transformer_type "causal" \
  --value_device "cuda:0" \
  --scoring_batch_size 128 \
  --N 8 \
  --batch_size 8 \
  --max_tokens 16384 \
  --max_model_len 16384 \
  --tensor_parallel_size 1 \
  --gpus "1" \
  --gpu_memory_utilization 0.98 \
  --vllm_enforce_eager \
  --phase_partition_algorithm adaptive \
  --high_entropy_ratio 0.12 \
  --target_phase_tokens 500 \
  --ns 1 2 4 8 \
  --resume


# GPQA Diamond
run_on_gpu gpqa_diamond 2 python -u internal_time_series/GPQA/baselines/run_transformer_bon.py \
  --dataset_source local \
  --local_dataset_path internal_time_series/deepscaler/data/test/gpqa_diamond.json \
  --bon_model_path /mnt/wangchenglong/models/Qwen3-8B \
  --bon_checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --bon_training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --bon_output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/gpqa_diamond_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --bon_N 8 \
  --ns 1 2 4 8 \
  --bon_transformer_type "causal" \
  --bon_value_device "cuda:0" \
  --bon_scoring_batch_size 128 \
  --bon_candidate_batch_size 8 \
  --bon_max_tokens 16385 \
  --bon_max_model_len 16385 \
  --bon_tensor_parallel_size 1 \
  --bon_gpus "2" \
  --bon_vllm_enforce_eager \
  --bon_gpu_memory_utilization 0.98 \
  --bon_phase_partition_algorithm adaptive \
  --bon_high_entropy_ratio 0.12 \
  --bon_target_phase_tokens 500 \
  --bon_entropy_chunk_size 16 \
  --bon_resume


# calendar
run_on_gpu calendar 3 python -u internal_time_series/NaturalPlan/evaluate_calendar_scheduling.py \
  --input_json internal_time_series/deepscaler/data/test/calendar.json \
  --model_path /mnt/wangchenglong/models/Qwen3-8B \
  --checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/calendar_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --transformer_type "causal" \
  --value_device "cuda:0" \
  --scoring_batch_size 128 \
  --batch_size 8 \
  --max_tokens 16384 \
  --max_model_len 16384 \
  --gpus "3" \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.98 \
  --phase_partition_algorithm adaptive \
  --high_entropy_ratio 0.12 \
  --target_phase_tokens 500 \
  --BoN_N 8 \
  --ns 1 2 4 8 \
  --vllm_enforce_eager \
  --resume


# meeting
run_on_gpu meeting 4 python -u internal_time_series/NaturalPlan/evaluate_meeting_planning.py \
  --input_json internal_time_series/deepscaler/data/test/meeting.json \
  --model_path /mnt/wangchenglong/models/Qwen3-8B \
  --checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/meeting_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --transformer_type "causal" \
  --value_device "cuda:0" \
  --scoring_batch_size 128 \
  --batch_size 8 \
  --max_tokens 16384 \
  --max_model_len 16384 \
  --gpus "4" \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.98 \
  --phase_partition_algorithm adaptive \
  --high_entropy_ratio 0.12 \
  --target_phase_tokens 500 \
  --BoN_N 8 \
  --ns 1 2 4 8 \
  --vllm_enforce_eager \
  --resume


# trip
run_on_gpu trip 5 python -u internal_time_series/NaturalPlan/evaluate_trip_planning.py \
  --input_json internal_time_series/deepscaler/data/test/trip.json \
  --model_path /mnt/wangchenglong/models/Qwen3-8B \
  --checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/trip_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --transformer_type "causal" \
  --value_device "cuda:0" \
  --scoring_batch_size 128 \
  --batch_size 8 \
  --max_tokens 16384 \
  --max_model_len 16384 \
  --gpus "5" \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.98 \
  --phase_partition_algorithm adaptive \
  --high_entropy_ratio 0.12 \
  --target_phase_tokens 500 \
  --BoN_N 8 \
  --ns 1 2 4 8 \
  --vllm_enforce_eager \
  --resume


# logical_deduction_five_objects
run_on_gpu logical_deduction_five_objects 6 python -u internal_time_series/BBH/run_transformer_bon.py \
  --data_dir internal_time_series/deepscaler/data/test \
  --description_dir internal_time_series/BBH/cot_zeroshot \
  --tasks logical_deduction_five_objects \
  --bon_model_path /mnt/wangchenglong/models/Qwen3-8B \
  --bon_checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --bon_training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --bon_output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/logical_deduction_five_objects_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --bon_enable_thinking true \
  --bon_N 8 \
  --ns 1 2 4 8 \
  --bon_transformer_type "causal" \
  --bon_value_device "cuda:0" \
  --bon_scoring_batch_size 128 \
  --bon_candidate_batch_size 8 \
  --bon_max_tokens 16384 \
  --bon_max_model_len 16384 \
  --bon_tensor_parallel_size 1 \
  --bon_gpus "6" \
  --bon_vllm_enforce_eager \
  --bon_gpu_memory_utilization 0.98 \
  --bon_phase_partition_algorithm adaptive \
  --bon_high_entropy_ratio 0.12 \
  --bon_target_phase_tokens 500 \
  --bon_resume


# tracking_shuffled_objects_five_objects
run_on_gpu tracking_shuffled_objects_five_objects 7 python -u internal_time_series/BBH/run_transformer_bon.py \
  --data_dir internal_time_series/deepscaler/data/test \
  --description_dir internal_time_series/BBH/cot_zeroshot \
  --tasks tracking_shuffled_objects_five_objects \
  --bon_model_path /mnt/wangchenglong/models/Qwen3-8B \
  --bon_checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --bon_training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --bon_output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/tracking_shuffled_objects_five_objects_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --bon_enable_thinking true \
  --bon_N 8 \
  --ns 1 2 4 8 \
  --bon_transformer_type "causal" \
  --bon_value_device "cuda:0" \
  --bon_scoring_batch_size 128 \
  --bon_candidate_batch_size 8 \
  --bon_max_tokens 16384 \
  --bon_max_model_len 16384 \
  --bon_tensor_parallel_size 1 \
  --bon_gpus "7" \
  --bon_vllm_enforce_eager \
  --bon_gpu_memory_utilization 0.98 \
  --bon_phase_partition_algorithm adaptive \
  --bon_high_entropy_ratio 0.12 \
  --bon_target_phase_tokens 500 \
  --bon_resume


# causal_judgement
run_on_gpu causal_judgement 8 python -u internal_time_series/BBH/run_transformer_bon.py \
  --data_dir internal_time_series/deepscaler/data/test \
  --description_dir internal_time_series/BBH/cot_zeroshot \
  --tasks causal_judgement \
  --bon_model_path /mnt/wangchenglong/models/Qwen3-8B \
  --bon_checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --bon_training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --bon_output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/causal_judgement_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --bon_enable_thinking true \
  --bon_N 8 \
  --ns 1 2 4 8 \
  --bon_transformer_type "causal" \
  --bon_value_device "cuda:0" \
  --bon_scoring_batch_size 128 \
  --bon_candidate_batch_size 8 \
  --bon_max_tokens 16384 \
  --bon_max_model_len 16384 \
  --bon_tensor_parallel_size 1 \
  --bon_gpus "8" \
  --bon_vllm_enforce_eager \
  --bon_gpu_memory_utilization 0.98 \
  --bon_phase_partition_algorithm adaptive \
  --bon_high_entropy_ratio 0.12 \
  --bon_target_phase_tokens 500 \
  --bon_resume


# temporal_sequences
run_on_gpu temporal_sequences 9 python -u internal_time_series/BBH/run_transformer_bon.py \
  --data_dir internal_time_series/deepscaler/data/test \
  --description_dir internal_time_series/BBH/cot_zeroshot \
  --tasks temporal_sequences \
  --bon_model_path /mnt/wangchenglong/models/Qwen3-8B \
  --bon_checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --bon_training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --bon_output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/temporal_sequences_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --bon_N 8 \
  --ns 1 2 4 8 \
  --bon_transformer_type "causal" \
  --bon_value_device "cuda:0" \
  --bon_scoring_batch_size 128 \
  --bon_candidate_batch_size 8 \
  --bon_max_tokens 16384 \
  --bon_max_model_len 16384 \
  --bon_tensor_parallel_size 1 \
  --bon_gpus "9" \
  --bon_vllm_enforce_eager \
  --bon_gpu_memory_utilization 0.98 \
  --bon_phase_partition_algorithm adaptive \
  --bon_high_entropy_ratio 0.12 \
  --bon_target_phase_tokens 500 \
  --bon_resume


# web_of_lies
run_on_gpu web_of_lies 10 python -u internal_time_series/BBH/run_transformer_bon.py \
  --data_dir internal_time_series/deepscaler/data/test \
  --description_dir internal_time_series/BBH/cot_zeroshot \
  --tasks web_of_lies \
  --bon_model_path /mnt/wangchenglong/models/Qwen3-8B \
  --bon_checkpoint_path internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/best_model.pt \
  --bon_training_metrics_json internal_time_series/results/train/phase_adaptive_predict/0.12_500_16384/Qwen3-8B/Qwen3-8B_deepscaler6000_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4/metrics.json \
  --bon_output_dir internal_time_series/results/eval/best_of_n_prms/Qwen3-8B/web_of_lies_0.12_500_16384/Qwen3-8B_deepscaler6000_0.12_500_16384_transformer_prob_multi_dim_causal_hybrid_a0.8_p0.2_l8_h12_d384_dffn1536_r0.25_2e-6_b4 \
  --bon_enable_thinking true \
  --bon_N 8 \
  --ns 1 2 4 8 \
  --bon_transformer_type "causal" \
  --bon_value_device "cuda:0" \
  --bon_scoring_batch_size 128 \
  --bon_candidate_batch_size 8 \
  --bon_max_tokens 16384 \
  --bon_max_model_len 16384 \
  --bon_tensor_parallel_size 1 \
  --bon_gpus "10" \
  --bon_vllm_enforce_eager \
  --bon_gpu_memory_utilization 0.98 \
  --bon_phase_partition_algorithm adaptive \
  --bon_high_entropy_ratio 0.12 \
  --bon_target_phase_tokens 500 \
  --bon_resume

# All commands have been launched; wait for every task and report failures.
exit_status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf 'Completed %s\n' "${task_names[$i]}"
  else
    task_status=$?
    printf 'FAILED %s (exit %s), log: %s/%s.log\n' \
      "${task_names[$i]}" "$task_status" "$LOG_DIR" "${task_names[$i]}" >&2
    exit_status=1
  fi
done
exit "$exit_status"
