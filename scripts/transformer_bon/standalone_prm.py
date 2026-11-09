"""Shared scheduling for the six legacy PRM scripts; numerical bodies stay local."""

from __future__ import annotations

import importlib.util
import math
import multiprocessing as mp
import os
import queue
import sys
import traceback

import torch

from .errors import require_success
from .ordered_pipeline import background_iterator
from .prm_prefetch import prefetch_map
from .prm_recovery import (OOMScore, execution_device, is_cuda_oom,
                           model_load_kwargs, parameter_storage, recovery_load_policy,
                           recovery_memory_limits, release_cuda_cache, _fused_attention)


_PERSIST = ("args", "scores_by_key", "expected_counts", "pending_keys", "errors",
            "summary_records", "phase_ranges_by_key", "base_steps_record_for",
            "output_summary_by_key", "output_steps_by_key", "summary_output_path",
            "steps_output_path", "phase_checkpoint_path")


class InputLengthError(RuntimeError):
    """An over-length input, with its measured length for preflight reduction."""

    def __init__(self, message, token_length):
        super().__init__(message)
        self.token_length = int(token_length)


def check_errors(errors):
    if errors:
        key = next(iter(errors))
        raise RuntimeError(f"PRM scoring failed for {key}: {errors[key]}")


def resilient(scorer, texts, prepared=None, *, defer_oom=False):
    # This preserves the original bisection order and successful batch shapes.
    from .prm import _score_prm_batch_resilient

    return _score_prm_batch_resilient(scorer, texts, prepared, defer_oom=defer_oom)


def score_batch(scorer, batch, expand, prepared=None):
    values = resilient(scorer, [item[2] for item in batch], prepared, defer_oom=True)
    if len(values) != len(batch):
        raise RuntimeError("PRM returned an invalid batch result count")
    payload, failures = [], []
    for item, value in zip(batch, values):
        if isinstance(value, OOMScore):
            failures.append((item[0], value.message))
            continue
        rows = expand([item], [value])
        ends = item[2].get("phase_ends") if isinstance(item[2], dict) else None
        expected = len(ends) if item[1] == -1 and ends else max(1, len(rows))
        if len(rows) != expected or any(not math.isfinite(row[3]) for row in rows):
            raise RuntimeError(f"PRM returned incomplete or non-finite scores for {item[0]}")
        payload.extend(rows)
    return payload, failures


def make_scorer(ns, args, device):
    return ns["HFPRMScorer"](args.prm_model_path, device, args.dtype,
                             args.trust_remote_code, args.max_length,
                             args.positive_token, args.prm_system_prompt)


def batches(ns, args, jobs, device, stop, preflight=None):
    scorer = make_scorer(ns, args, device)
    try:
        try:
            ns["preflight_prm_input_lengths"](scorer, jobs)
        except InputLengthError as exc:
            if preflight is None:
                raise
            preflight(exc, stop)
            return
        if preflight is not None and not preflight(None, stop):
            return
        source = (jobs[start:start + args.scoring_batch_size]
                  for start in range(0, len(jobs), args.scoring_batch_size))
        # The inner prefetch scope owns its stop event. Closing it on normal
        # exhaustion must not cancel the outer iterator before it sends done.
        with prefetch_map(source, lambda batch: (batch, scorer.prepare_batch([j[2] for j in batch]))) as prepared:
            for batch, inputs in prepared:
                if stop.is_set():
                    return
                payload, failures = score_batch(scorer, batch, ns["expand_phase_batch_scores"], inputs)
                del inputs
                yield batch, payload, failures
    finally:
        scorer = None


def worker(ns, dp_rank, gpu_group, args, phase_jobs, result_queue, preflight_ready=None):
    def preflight(error, stop):
        payload = None if error is None else (error.token_length, str(error))
        result_queue.put(("preflight", dp_rank, payload))
        if error is not None:
            return False
        while not preflight_ready.wait(timeout=0.1):
            if stop.is_set():
                return False
        return not stop.is_set()

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
        jobs = sorted(phase_jobs, key=ns["estimate_prm_job_length"])
        device = ("auto" if len(gpu_group.split(",")) > 1 else
                  "cuda:0" if args.prm_device.lower().startswith("cuda") else args.prm_device)
        with background_iterator(lambda stop: batches(
                ns, args, jobs, device, stop,
                preflight if preflight_ready is not None else None)) as scored:
            for batch, payload, failures in scored:
                if payload:
                    result_queue.put(("batch", dp_rank, payload))
                for key, message in failures:
                    result_queue.put(("candidate_oom", dp_rank, (key, message)))
    except BaseException:
        result_queue.put(("worker_error", dp_rank, traceback.format_exc()))
    finally:
        result_queue.put(("worker_done", dp_rank, None))


def _persist(ns, state):
    check_errors(state["errors"])
    return ns["persist_phase_progress"](**{key: state[key] for key in _PERSIST})


def _accept(state, payload):
    for sample, sequence, phase, value in payload:
        if not math.isfinite(value):
            raise ValueError("PRM returned a non-finite score")
        state["scores_by_key"].setdefault((int(sample), int(sequence)), {})[int(phase)] = float(value)


def run_single(ns, state):
    args = state["args"]
    jobs = sorted(state["phase_jobs"], key=ns["estimate_prm_job_length"])
    failed = set()
    phases = ns["_PRM_BACKEND"] == "mistral"
    total = (sum(state["expected_counts"].get(job[0], 1) for job in jobs) if phases else
             (len(jobs) + args.scoring_batch_size - 1) // args.scoring_batch_size)
    with ns["tqdm"](total=total, desc="Scoring PRM phases", unit="phase" if phases else "it") as progress:
        with background_iterator(lambda stop: batches(ns, args, jobs, args.prm_device, stop)) as scored:
            for batch, payload, failures in scored:
                failed.update(key for key, _ in failures)
                _accept(state, payload)
                _persist(ns, state)
                progress.update(sum(state["expected_counts"].get(job[0], 1) for job in batch) if phases else 1)
    # The background generator and CPU prefetch thread have released the model.
    if failed:
        release_cuda_cache(torch)
        group = os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(
            str(device) for device in range(torch.cuda.device_count()))
        finish_recovery(ns, state, group, [j for j in jobs if j[0] in failed])


def finish_recovery(ns, state, group, jobs):
    if not jobs:
        return
    recovered = recover(ns, state["args"], group, jobs)
    if len(recovered) != len(jobs):
        raise RuntimeError("PRM recovery returned an invalid row count")
    for job, value in zip(jobs, recovered):
        payload = score_batch_values(ns, job, value)
        _accept(state, payload)
    _persist(ns, state)


def score_batch_values(ns, job, value):
    rows = ns["expand_phase_batch_scores"]([job], [value])
    ends = job[2].get("phase_ends") if isinstance(job[2], dict) else None
    expected = len(ends) if job[1] == -1 and ends else max(1, len(rows))
    if len(rows) != expected or any(not math.isfinite(row[3]) for row in rows):
        raise RuntimeError(f"Invalid PRM recovery scores for {job[0]}")
    return rows


def _load_script(path):
    spec = importlib.util.spec_from_file_location("_standalone_prm_recovery_source", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return vars(module)


def _recovery_score(scorer, text, backend):
    """Recovery only: apply the original head to small groups of marker states."""
    if backend == "qwen" and scorer.kind != "causal":
        return scorer.score_batch([text])[0]
    prepared = scorer.prepare_batch([text])
    ends = [int(end) for end in (text.get("phase_ends") or [])]
    if backend in ("fover", "versa"):
        inputs = prepared[0][0]
        encoded = {"input_ids": inputs}
        positions = (scorer._step_token_positions(inputs[0].numpy()).tolist()
                     if backend == "fover" else
                     inputs[0].eq(scorer.step_separator_id).nonzero().flatten().tolist())
    elif backend in ("deepseek", "mistral"):
        encoded, row_positions = prepared
        positions = row_positions[0]
    elif backend == "math":
        _, _, encoded = prepared
        positions = encoded["input_ids"][0].eq(scorer.step_tag_id).nonzero().flatten().tolist()
    else:
        _, encoded, _ = prepared
        positions = [encoded["input_ids"].shape[1] - 1]
    if not positions:
        raise RuntimeError("PRM recovery found no score positions")
    encoded = {key: value.to(scorer.device) for key, value in encoded.items()}
    root = scorer.model.get_base_model() if hasattr(scorer.model, "get_base_model") else scorer.model
    body = getattr(root, "model", None)
    head = scorer.model.get_output_embeddings()
    if body is None or head is None:
        raise RuntimeError("PRM recovery needs a transformer body and an output head")
    with torch.inference_mode():
        output = body(**encoded, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        selected = hidden[0].index_select(0, torch.as_tensor(positions, device=hidden.device))
        del hidden, output, encoded
        result = []
        # Even unusually many markers cannot create a marker-by-vocabulary blob.
        vocab = int(getattr(root.config, "vocab_size", 0) or 1)
        chunk = max(1, min(64, 16 * 1024**2 // (vocab * max(4, selected.element_size()))))
        for start in range(0, len(positions), chunk):
            logits = head(selected[start:start + chunk].to(execution_device(head)))
            if backend == "fover":
                negative = scorer.tokenizer.encode("incorrect", add_special_tokens=False)[0]
                positive = scorer.tokenizer.encode("correct", add_special_tokens=False)[0]
                scores = logits[:, [negative, positive]].softmax(-1)[:, 1]
            elif backend == "versa":
                scores = logits[:, list(scorer.candidate_tokens)].softmax(-1)[:, 1]
            elif backend == "math":
                scores = logits[:, list(scorer.candidate_tokens)].float().softmax(-1)[:, 0]
            elif backend in ("deepseek", "mistral"):
                scores = logits[:, [scorer.plus_id, scorer.minus_id]].softmax(-1)[:, 0]
            else:
                scores = logits.softmax(-1)[:, scorer.positive_id]
            result.extend(scores.float().cpu().tolist())
            del logits, scores
    if backend in ("deepseek", "mistral", "qwen"):
        return result if backend != "qwen" else result[0]
    if any(end < 1 or end > len(result) for end in ends):
        raise RuntimeError("PRM recovery phase end is outside the visible step scores")
    return [result[end - 1] for end in ends] if ends else result


def _recovery_worker(path, backend, args, group, jobs, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = group
    scorer = None
    try:
        ns = _load_script(path)
        limits = recovery_memory_limits(torch)
        protected = False
        for index, job in enumerate(jobs):
            for attempt in range(2):
                failed = False
                try:
                    if scorer is None:
                        with recovery_load_policy(limits):
                            scorer = make_scorer(ns, args, "auto")
                        placement = getattr(scorer.model, "hf_device_map", {})
                        if "disk" in placement.values():
                            raise RuntimeError("Disk offload is disabled for PRM recovery")
                        if not any(p.is_cuda for p in scorer.model.parameters()):
                            raise RuntimeError("No model layer fits on CUDA for PRM recovery")
                    from contextlib import nullcontext
                    with _fused_attention(torch) if protected else nullcontext():
                        value = _recovery_score(scorer, job[2], backend)
                    results.put(("score", index, value))
                    break
                except RuntimeError as exc:
                    if not is_cuda_oom(exc) or protected or attempt:
                        raise
                    failed = True
                if failed:
                    scorer = None
                    release_cuda_cache(torch)
                    protected = True
        results.put(("done", None, None))
    except BaseException:
        results.put(("error", None, traceback.format_exc()))


def recover(ns, args, group, jobs):
    context = mp.get_context("spawn")
    results = context.Queue(maxsize=2)
    process = context.Process(target=_recovery_worker,
                              args=(ns["__file__"], ns["_PRM_BACKEND"], args, group, jobs, results))
    values = {}
    try:
        print(f"[PRM OOM recovery] {len(jobs)} candidate(s), GPUs={group}, batch=1; bounded CPU offload.", flush=True)
        process.start()
        while True:
            try:
                kind, index, value = results.get(timeout=.2)
            except queue.Empty:
                if not process.is_alive():
                    raise RuntimeError(f"PRM recovery worker exited: {process.exitcode}")
                continue
            if kind == "error":
                raise RuntimeError("PRM OOM recovery failed:\n" + value)
            if kind == "done":
                if set(values) != set(range(len(jobs))):
                    raise RuntimeError("PRM recovery missed candidates")
                return [values[index] for index in range(len(jobs))]
            if kind != "score" or index in values or index not in range(len(jobs)):
                raise RuntimeError("Invalid PRM recovery message")
            values[index] = value
    finally:
        ns["reap_prm_processes"]([(0, process)])
        results.cancel_join_thread()
        results.close()
