"""Step boundaries for the existing PRM service; scoring protocols stay shared."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import require_success
from .prm import PRMBestOfNConfig, PRMBestOfNService, split_steps


@dataclass
class StepPRMBestOfNConfig(PRMBestOfNConfig):
    def legacy_args(self):
        args = super().legacy_args()
        args.prm_step_scoring = True
        return args


def build_step_prepared_record(legacy, generation, tokenizer, entropies, args):
    """Keep the original token/step metadata without selecting any phases."""
    require_success(generation)
    if generation.get("status") != "ok":
        raise ValueError("Candidate generation did not complete successfully")
    ids = [int(value) for value in generation["response_token_ids"]]
    from .phase_structure import compile_structure
    # Reuse the optimized decoder, interval mapping, saved steps and cropped
    # generation handling, without entropy sorting or phase selection/cache IO.
    source = dict(generation, response_token_ids=ids, _bon_step_scoring=True)
    tokens, facts, metadata = compile_structure(source, {"token_entropies": entropies}, tokenizer)
    text = generation.get("full_output") if metadata["decoded_output"] is None else metadata["decoded_output"]
    confidences = [value if valid else None for value, valid in
                   zip(facts["confidence"].tolist(), facts["has_confidence"].tolist())]
    steps = [dict(step=text[start:end], token_avg_confidence=confidence,
                  first_token_index=first if first >= 0 else None,
                  phase_frontier=2 if i == len(facts) - 1 else 0)
             for i, (start, end, confidence, first) in enumerate(zip(
                 facts["start"].tolist(), facts["end"].tolist(), confidences, facts["first_token"].tolist()))]
    record = {
        "status": "ok", "sample_idx": generation["sample_idx"],
        "sequence_idx": generation["sequence_idx"], "problem": generation["problem"],
        "gold_answer": generation.get("gold_answer"), "difficulty": generation.get("difficulty"),
        "unique_id": generation.get("unique_id"), "full_output": text,
        "step_avg_confidences": confidences, "num_steps": len(steps),
        "num_generated_tokens": len(ids), "finish_reason": generation.get("finish_reason"),
        "phase_ranges": [[i, i] for i in range(1, len(steps) + 1)],
        "boundary_token_positions": [step["first_token_index"] for step in steps[1:]
                                     if step.get("first_token_index") is not None],
        "num_entropy_threshold_candidates": 0, "steps": steps,
    }
    if not args.no_save_token_entropies:
        record["token_entropies"] = tokens["entropy"].tolist()
    return record


def run_step_preparation(args, tasks, legacy, on_record, on_complete):
    """Stream each rollout once; no phase workers, caches or boundary search."""
    if not tasks:
        return
    tokenizer = legacy.load_tokenizer_without_vllm(args)
    for generation, entropy in tasks:
        require_success(entropy)
        record = build_step_prepared_record(legacy, generation, tokenizer, entropy["token_entropies"], args)
        record["request_id"] = generation["request_id"]
        on_record(record)
        on_complete()


class StepPRMBestOfNService(PRMBestOfNService):
    """Reuse all backends, scheduling, persistence and selection unchanged."""

    def _score_task(self, legacy, request, prepared_records, args, on_rollout=None):
        records = []
        for original in prepared_records:
            require_success(original)
            count = len(split_steps(original.get("full_output"), self.config.prm_phase_separator))
            if len(original.get("steps", [])) != count:
                raise ValueError("Text step count does not match prepared step metadata")
            ranges = [[i, i] for i in range(1, count + 1)]
            if original.get("phase_ranges") == ranges and all(
                step.get("phase_frontier") == (2 if i == count - 1 else 0)
                for i, step in enumerate(original["steps"])
            ):
                records.append(original)
                continue
            record = dict(original)
            record["phase_ranges"] = ranges
            record["steps"] = [dict(step, phase_frontier=2 if i == count - 1 else 0)
                               for i, step in enumerate(original["steps"])]
            records.append(record)
        return (yield from super()._score_task(legacy, request, records, args, on_rollout))

    def _cumulative_prm_scores(self, scores):
        # Final selection aggregates still use the original exact reduction.
        values = np.asarray(scores, dtype=np.float64)
        products = np.exp(np.cumsum(np.log(np.clip(values, 1e-12, 1.0))))
        minima = np.minimum.accumulate(values)
        means = np.cumsum(values) / np.arange(1, len(values) + 1)
        return (dict(product_score=float(products[i]), min_score=float(minima[i]),
                     mean_score=float(means[i]), last_score=float(value))
                for i, value in enumerate(values))
