"""Bounded, exception-only PRM recovery. Normal inference uses no load policy."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import queue
import traceback
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass


_LOAD_POLICY = ContextVar("prm_recovery_load_policy", default=None)
_GIB = 1024 ** 3


@dataclass(frozen=True)
class OOMScore:
    """Only strings cross the worker boundary, never a CUDA traceback."""
    message: str


def is_cuda_oom(exc):
    message = str(exc).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message


def release_cuda_cache(runtime):
    # Call only after leaving the exception scope holding failed forward frames.
    gc.collect()
    if runtime is not None and runtime.cuda.is_available():
        for device in range(runtime.cuda.device_count()):
            with runtime.cuda.device(device):
                runtime.cuda.empty_cache()


def model_load_kwargs(kwargs):
    policy = _LOAD_POLICY.get()
    return kwargs if policy is None else {**kwargs, **policy}


@contextmanager
def recovery_load_policy(limits):
    token = _LOAD_POLICY.set(dict(device_map="auto", max_memory=limits,
                                  low_cpu_mem_usage=True, offload_state_dict=False))
    try:
        yield
    finally:
        _LOAD_POLICY.reset(token)


def _hooks(module):
    pending = [getattr(module, "_hf_hook", None)]
    while pending:
        hook = pending.pop()
        if hook is not None:
            yield hook
            pending.extend(getattr(hook, "hooks", ()))


def parameter_storage(module, name="weight"):
    """Read CPU-offloaded head rows without materializing a full vocab head.

    Ordinary parameters are returned unchanged. Accelerate represents offloaded
    parameters by meta placeholders; their backing tensors live in weights_map.
    """
    owner, leaf = module, name
    if "." in name:
        prefix, leaf = name.rsplit(".", 1)
        owner = module.get_submodule(prefix)
    parameter = getattr(owner, leaf, None)
    if parameter is None or not getattr(parameter, "is_meta", False):
        return parameter
    for target, key in ((owner, leaf), (module, name)):
        for hook in _hooks(target):
            weights = getattr(hook, "weights_map", None)
            if weights is not None:
                try:
                    return weights[key]
                except KeyError:
                    pass
    raise RuntimeError(f"Cannot access offloaded PRM head parameter {name!r}")


def execution_device(module):
    parameter = next(module.parameters())
    if not parameter.is_meta:
        return parameter.device
    for child in module.modules():
        for hook in _hooks(child):
            device = getattr(hook, "execution_device", None)
            if device is not None:
                return device
    raise RuntimeError("Cannot resolve offloaded PRM head execution device")


def recovery_memory_limits(runtime):
    import psutil

    available = int(psutil.virtual_memory().available)
    # Leave at least half of available RAM (and 2 GiB) for loading buffers,
    # staging records and other applications. Never enable disk weight offload.
    host = min(32 * _GIB, available - max(2 * _GIB, available // 2))
    if host <= 0:
        raise MemoryError("Insufficient available RAM for bounded PRM OOM recovery")
    limits = {"cpu": host}
    for device in range(runtime.cuda.device_count()):
        free, _ = runtime.cuda.mem_get_info(device)
        limits[device] = min(3 * _GIB, int(free) // 4)
    if len(limits) == 1:
        raise RuntimeError("PRM OOM recovery requires a visible CUDA GPU")
    return limits


def _fused_attention(runtime):
    from .prm_attention import recovery_attention

    return recovery_attention(runtime)


def _recovery_worker(gpu_group, options, jobs, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
    scorer = None
    try:
        import torch
        from .prms.registry import get_prm_backend

        limits = recovery_memory_limits(torch)
        backend = get_prm_backend(options.pop("prm_name"))
        fused = False
        for key, text in jobs:
            # One ordinary-attention attempt with extra headroom, then at most
            # one bounded-attention attempt. A failed model is discarded: Accelerate
            # post-forward hooks may not have run when attention raised OOM.
            for attempt in range(2):
                failure = None
                try:
                    if scorer is None:
                        with recovery_load_policy(limits):
                            scorer = backend.create_scorer(device="auto", **options)
                        model = getattr(scorer, "model", None)
                        if model is not None:
                            placement = getattr(model, "hf_device_map", {})
                            devices = list(placement.values()) or [p.device for p in model.parameters()]
                            if any(str(device) == "disk" for device in devices):
                                raise RuntimeError("Disk offload is disabled for PRM recovery")
                            if not any(isinstance(device, int) or str(device).startswith("cuda")
                                       for device in devices):
                                raise RuntimeError("No model layer fits on GPU; refusing "
                                                   "unbounded CPU attention recovery")
                        del model
                    with _fused_attention(torch) if fused else nullcontext():
                        values = scorer.score_batch([text])
                    if len(values) != 1:
                        raise RuntimeError("PRM recovery returned an invalid row count")
                    results.put(("score", key, values[0]))
                    break
                except RuntimeError as exc:
                    if not is_cuda_oom(exc) or fused or attempt:
                        raise
                    failure = str(exc)
                if failure is not None:
                    scorer = None
                    release_cuda_cache(torch)
                    fused = True
                    print("[PRM OOM recovery] Extra headroom was insufficient; "
                          "retrying with Flash/Efficient SDPA and bounded attention fallback.", flush=True)
        results.put(("done", None, None))
    except Exception:
        results.put(("error", None, traceback.format_exc()))


def recover_jobs(config, gpu_group, jobs, *, stop=None):
    """One temporary replica over the configured GPU union, no output files."""
    from .ordered_pipeline import check_stopped
    from .prm import _close_queue, _reap_processes

    context = mp.get_context("spawn")
    results = context.Queue()
    options = dict(prm_name=config.prm_name, model_path=config.prm_model_path,
                   dtype=config.prm_dtype, max_length=config.prm_max_length,
                   positive_token=config.prm_positive_token,
                   system_prompt=config.prm_system_prompt,
                   trust_remote_code=config.prm_trust_remote_code)
    process = context.Process(target=_recovery_worker,
                              args=(gpu_group, options, jobs, results))
    expected = {key for key, _ in jobs}
    recovered = {}
    try:
        print(f"[PRM OOM recovery] {len(jobs)} candidate(s), GPU group {gpu_group}, "
              "batch=1; bounded CPU weight offload, no disk offload.", flush=True)
        check_stopped(stop)
        process.start()
        while True:
            check_stopped(stop)
            try:
                kind, key, value = results.get(timeout=0.2)
            except queue.Empty:
                if not process.is_alive() and process.exitcode is not None:
                    raise RuntimeError(f"PRM OOM recovery worker exited: {process.exitcode}")
                continue
            if kind == "error":
                raise RuntimeError("Bounded PRM OOM recovery failed. Model loading or "
                                   "remaining activations may still exceed available "
                                   "memory; see the underlying error.\n" + value)
            if kind == "done":
                if set(recovered) != expected:
                    raise RuntimeError("PRM OOM recovery missed candidates")
                return recovered
            if kind != "score" or key not in expected or key in recovered:
                raise RuntimeError("Invalid PRM OOM recovery message")
            recovered[key] = value
    finally:
        _reap_processes([process])
        _close_queue(results)
