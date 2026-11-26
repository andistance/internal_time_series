# BIG-Bench-Hard Best-of-N

This directory is a BBH-specific adapter around `scripts/transformer_bon`.
It reads the official local BBH JSON files, loads the `description` from the
official `cot_zeroshot` YAML files, and generates the prompt:

```text
description

input

Please reason step by step, and put your final answer within \boxed{}.
```

The evaluator only scores the last `\boxed{...}`.  It then applies the BBH
task-type answer extraction rules (multiple choice, binary choice, or open
answer) before case-insensitive comparison.  No boxed answer is an incorrect
sample.

Transformer scoring:

```powershell
python BBH/run_transformer_bon.py `
  --data_dir C:\data\BIG-Bench-Hard\bbh `
  --description_dir C:\data\BIG-Bench-Hard\lm_eval\tasks\bbh\cot_zeroshot `
  --bon_model_path C:\models\generator `
  --bon_checkpoint_path C:\models\value_model.pt `
  --bon_output_dir results\bbh_transformer `
  --bon_N 32 --ns 1 2 4 8 16 32 `
  --bon_transformer_type causal --gpus '0,1;2,3'
```

PRM scoring uses the same command-line parameters and replaces the value-model
arguments:

```powershell
python BBH/run_prm_bon.py `
  --data_dir C:\data\BIG-Bench-Hard\bbh `
  --description_dir C:\data\BIG-Bench-Hard\lm_eval\tasks\bbh\cot_zeroshot `
  --bon_model_path C:\models\generator `
  --prm_model_path C:\models\prm `
  --bon_output_dir results\bbh_prm --bon_N 32 --ns 1 4 16 32
```

The output directory follows the other benchmark runners: rollout-level
`.bon_generation.jsonl`, `.bon_prepared.jsonl`, and (for PRM) `.bon_prm.jsonl`,
paired `summary.jsonl`/`steps.jsonl`, `eval_results_<model>.csv`, `metrics.json`,
`metrics.csv`, `details_<n>.csv`, and `metrics.png`.  `--bon_resume` resumes
individual rollouts, not only whole samples; `--bon_overwrite` starts over.
