"""Spawn-safe data/tensor-parallel helpers for Transformer Best-of-N.

The public service keeps benchmark requests dataset-independent.  This module
contains only scheduling and worker lifecycle code so children created with the
``spawn`` multiprocessing context can import their targets without capturing a
live vLLM or PyTorch object from the parent.
"""

from __future__ import annotations
from .errors import raise_record_error

import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue
import socket
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .entropy_cache import build_token_entropy_record

from .models import BestOfNRequest


@dataclass(frozen=True)
class GenerationJob:
    """One legacy-compatible, indivisible candidate generation batch."""

    request: BestOfNRequest
    start: int
    stop: int
    indices: Optional[Tuple[int, ...]] = None

    @property
    def sequence_indices(self) -> Sequence[int]:
        if self.indices is not None:
            return self.indices
        return range(self.start, self.stop)


def parse_gpu_groups(
    value: Optional[str], tensor_parallel_size: int
) -> Tuple[str, ...]:
    """Parse ``0,1;2,3`` into non-overlapping TP groups."""

    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be at least 1")
    if value is None or not value.strip():
        return ()
    groups: List[str] = []
    used: set[str] = set()
    for raw_group in value.split(";"):
        ids = [item.strip() for item in raw_group.split(",") if item.strip()]
        if not ids:
            raise ValueError("gpu_groups contains an empty GPU group")
        if len(ids) != tensor_parallel_size:
            raise ValueError(
                f"GPU group {raw_group!r} contains {len(ids)} GPUs, but "
                f"tensor_parallel_size={tensor_parallel_size}"
            )
        duplicates = used.intersection(ids)
        if duplicates:
            raise ValueError(
                "The same GPU cannot be assigned to multiple DP workers: "
                + ", ".join(sorted(duplicates))
            )
        used.update(ids)
        groups.append(",".join(ids))
    return tuple(groups)


def build_generation_jobs(
    requests: Sequence[BestOfNRequest], n: int, candidate_batch_size: int
) -> List[GenerationJob]:
    """Build the exact candidate-batch boundaries used by the legacy script."""

    jobs: List[GenerationJob] = []
    for request in requests:
        for start in range(0, n, candidate_batch_size):
            jobs.append(
                GenerationJob(
                    request=request,
                    start=start,
                    stop=min(n, start + candidate_batch_size),
                )
            )
    return jobs


def build_missing_generation_jobs(
    requests: Sequence[BestOfNRequest],
    n: int,
    candidate_batch_size: int,
    completed_indices: Mapping[str, Sequence[int]],
) -> List[GenerationJob]:
    """Build jobs only for missing rollout indices.

    A completely missing legacy batch keeps its original boundaries and seed.
    When only part of a batch is missing, each missing index becomes an
    independent job so already persisted rollouts are never regenerated and
    holes before a persisted higher index are still filled.
    """

    jobs: List[GenerationJob] = []
    for request in requests:
        completed = {int(value) for value in completed_indices.get(request.request_id, ())}
        for start in range(0, n, candidate_batch_size):
            stop = min(n, start + candidate_batch_size)
            missing = tuple(index for index in range(start, stop) if index not in completed)
            if not missing:
                continue
            if len(missing) == stop - start:
                jobs.append(GenerationJob(request=request, start=start, stop=stop))
            else:
                jobs.extend(
                    GenerationJob(
                        request=request,
                        start=index,
                        stop=index + 1,
                        indices=(index,),
                    )
                    for index in missing
                )
    return jobs


def find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def lpt_shards(
    weighted_items: Iterable[Tuple[int, Any]], shard_count: int
) -> List[List[Any]]:
    """Greedy longest-processing-time sharding with deterministic ties."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    shards: List[List[Any]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    ordered = sorted(
        enumerate(weighted_items),
        key=lambda pair: (-int(pair[1][0]), pair[0]),
    )
    for _, (weight, item) in ordered:
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[target].append(item)
        loads[target] += max(0, int(weight))
    return shards


def _restore_environment(previous: Mapping[str, Optional[str]]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def effective_vllm_cache_mode(config: Any, gpu_groups: Sequence[str]) -> str:
    mode = str(config.vllm_cache_mode)
    if mode == "auto":
        return "isolated" if len(gpu_groups) > 1 else "shared"
    return mode


def generation_cache_environment(
    config: Any,
    worker_index: int,
    gpu_groups: Sequence[str],
) -> Tuple[Dict[str, str], str]:
    """Return a stable cache namespace for one independent vLLM DP worker."""

    configured_root = config.vllm_cache_root or os.environ.get("VLLM_CACHE_ROOT")
    base_root = Path(configured_root).expanduser() if configured_root else (
        Path.home() / ".cache" / "vllm"
    )
    mode = effective_vllm_cache_mode(config, gpu_groups)
    if mode == "isolated":
        fingerprint_payload = {
            "model_path": str(Path(config.model_path).expanduser().resolve()),
            "tensor_parallel_size": config.tensor_parallel_size,
            "dtype": config.dtype,
            "max_model_len": config.max_model_len,
            "trust_remote_code": config.trust_remote_code,
            "enforce_eager": config.vllm_enforce_eager,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        cache_root = (
            base_root
            / "transformer_bon"
            / fingerprint
            / f"dp_{worker_index}"
        )
        environment = {
            "VLLM_CACHE_ROOT": str(cache_root),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
        }
    else:
        cache_root = base_root
        environment = {"VLLM_CACHE_ROOT": str(cache_root)}
    return environment, str(cache_root)


def _generation_worker(
    worker_index: int,
    gpu_group: str,
    config: Any,
    worker_environment: Mapping[str, str],
    cache_root: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    llm = None
    try:
        # vLLM reads these values while importing its runtime.  They must be set
        # before loading the legacy module and resolving its lazy vLLM API.
        os.environ.update(worker_environment)
        from .engine import BestOfNService
        from .legacy import load_legacy_module

        legacy = load_legacy_module(config.legacy_script_path)
        args = config.legacy_args()
        service = BestOfNService(config)
        llm = service._create_generation_llm(legacy, args)
        tokenizer = llm.get_tokenizer()
        result_queue.put(
            (
                "worker_ready",
                worker_index,
                {"gpu_group": gpu_group, "cache_root": cache_root},
            )
        )
        prompt_cache: Dict[str, str] = {}
        max_active = config.generation_max_active_jobs or {
            "throughput": 8,
            "balanced": 4,
            "memory": 1,
        }[config.performance_profile]
        while True:
            job = task_queue.get()
            if job is None:
                break
            jobs = [job]
            stop_after_batch = False
            while len(jobs) < max_active:
                try:
                    next_job = task_queue.get_nowait()
                except queue.Empty:
                    break
                if next_job is None:
                    stop_after_batch = True
                    break
                jobs.append(next_job)
            try:
                records = service._generate_jobs_with_llm(
                    legacy,
                    llm,
                    tokenizer,
                    jobs,
                    args,
                    prompt_cache,
                )
                result_queue.put(
                    (
                        "records",
                        worker_index,
                        {"records": records, "job_count": len(jobs)},
                    )
                )
            except BaseException:
                raise
            if stop_after_batch:
                break
    except BaseException:
        result_queue.put(("worker_error", worker_index, traceback.format_exc()))
    finally:
        if llm is not None:
            try:
                from .legacy import load_legacy_module

                load_legacy_module(config.legacy_script_path).release_vllm(llm)
            except Exception:
                pass
        result_queue.put(("worker_done", worker_index, gpu_group))


def run_parallel_generation(
    config: Any,
    jobs: Sequence[GenerationJob],
    gpu_groups: Sequence[str],
    on_records: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run generation jobs on a dynamic DP work queue.

    ``on_records`` runs in the parent process as soon as one worker returns a
    group of responses.  Callers use it to durably stage responses without
    waiting for the other DP workers or for the full generation phase.
    """

    if not jobs:
        return {}
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(8, len(gpu_groups) * 4))
    result_queue = ctx.Queue(maxsize=max(8, len(gpu_groups) * 4))
    processes: List[Any] = []
    worker_cache_roots: Dict[int, str] = {}
    previous = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        for worker_index, gpu_group in enumerate(gpu_groups):
            worker_environment, cache_root = generation_cache_environment(
                config, worker_index, gpu_groups
            )
            for path_string in worker_environment.values():
                Path(path_string).mkdir(parents=True, exist_ok=True)
            worker_cache_roots[worker_index] = cache_root
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
            process = ctx.Process(
                target=_generation_worker,
                args=(
                    worker_index,
                    gpu_group,
                    config,
                    worker_environment,
                    cache_root,
                    task_queue,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)
    finally:
        _restore_environment(previous)

    def terminate_workers() -> None:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10.0)

    # Do not queue expensive requests until every vLLM engine has completed
    # model loading, compilation and KV-cache profiling successfully.
    ready: set[int] = set()
    startup_errors: List[str] = []
    startup_deadline = time.monotonic() + float(
        config.generation_worker_startup_timeout
    )
    while len(ready) < len(processes) and not startup_errors:
        remaining = startup_deadline - time.monotonic()
        if remaining <= 0:
            waiting = sorted(set(range(len(processes))).difference(ready))
            startup_errors.append(
                "Generation worker startup timed out after "
                f"{config.generation_worker_startup_timeout:.1f}s; waiting for "
                f"workers {waiting}. Cache roots: {worker_cache_roots}"
            )
            break
        try:
            message_type, worker_index, payload = result_queue.get(
                timeout=min(1.0, remaining)
            )
        except queue.Empty:
            dead = [
                index
                for index, process in enumerate(processes)
                if index not in ready and not process.is_alive()
            ]
            if dead:
                startup_errors.append(
                    "Generation workers exited before becoming ready: "
                    f"{dead}. Cache roots: {worker_cache_roots}"
                )
            continue
        if message_type == "worker_ready":
            ready.add(int(worker_index))
            print(
                "[BoN] generation worker "
                f"{worker_index} ready on GPU group {payload['gpu_group']} "
                f"(cache={payload['cache_root']})",
                flush=True,
            )
        elif message_type == "worker_error":
            startup_errors.append(
                f"Generation worker {worker_index} failed during startup "
                f"(cache={worker_cache_roots.get(int(worker_index))}):\n{payload}"
            )
        elif message_type == "worker_done":
            startup_errors.append(
                f"Generation worker {worker_index} exited before becoming ready"
            )
        else:
            startup_errors.append(
                f"Unexpected generation message during startup: {message_type}"
            )
    if startup_errors:
        terminate_workers()
        raise RuntimeError("DP generation failed.\n" + "\n".join(startup_errors))

    producer_error: List[str] = []

    def produce() -> None:
        try:
            for job in jobs:
                task_queue.put(job)
            for _ in processes:
                task_queue.put(None)
        except BaseException:
            producer_error.append(traceback.format_exc())

    producer = threading.Thread(
        target=produce,
        name="bon-generation-producer",
        daemon=True,
    )
    producer.start()

    results: Dict[str, List[Dict[str, Any]]] = {}
    completed_jobs = 0
    finished: set[int] = set()
    errors: List[str] = []
    while len(finished) < len(processes):
        try:
            message_type, worker_index, payload = result_queue.get(timeout=1.0)
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                break
            continue
        if message_type == "records":
            if isinstance(payload, Mapping):
                records = list(payload.get("records", []))
                returned_job_count = int(payload.get("job_count", 0))
            else:
                # Compatibility with workers started from an older module copy.
                records = list(payload)
                returned_job_count = max(
                    1,
                    len(
                        {
                            (
                                str(record.get("request_id")),
                                int(record.get("sequence_idx", -1))
                                // max(1, config.candidate_batch_size),
                            )
                            for record in records
                        }
                    ),
                )
            if on_records is not None:
                try:
                    on_records(records)
                except BaseException:
                    errors.append(
                        "Generation result persistence failed:\n"
                        + traceback.format_exc()
                    )
                    break
            for record in records:
                request_id = str(record.get("request_id", ""))
                results.setdefault(request_id, []).append(record)
            completed_jobs += returned_job_count
        elif message_type == "worker_error":
            errors.append(
                f"Generation worker {worker_index} failed "
                f"(cache={worker_cache_roots.get(int(worker_index))}):\n{payload}"
            )
            break
        elif message_type == "worker_done":
            finished.add(int(worker_index))

    if errors:
        terminate_workers()
    else:
        for process in processes:
            process.join()
    producer.join(timeout=5.0)
    if producer.is_alive():
        errors.append("Generation producer did not finish after workers exited")
    if producer_error:
        errors.append("Generation producer failed:\n" + producer_error[0])
    abnormal = [index for index, p in enumerate(processes) if p.exitcode != 0]
    if abnormal:
        errors.append(f"Generation workers exited abnormally: {abnormal}")
    if completed_jobs != len(jobs):
        errors.append(
            f"Generation returned {completed_jobs}/{len(jobs)} candidate batches"
        )
    if errors:
        raise RuntimeError("DP generation failed.\n" + "\n".join(errors))
    for records in results.values():
        records.sort(key=lambda record: int(record["sequence_idx"]))
    return results


def _entropy_tp_worker(
    dp_rank: int,
    tp_rank: int,
    gpu_group: str,
    args: SimpleNamespace,
    records: Sequence[Dict[str, Any]],
    result_queue: Any,
) -> None:
    model = None
    distributed = False
    try:
        import torch
        import torch.distributed as dist

        from .legacy import load_legacy_module

        legacy = load_legacy_module(getattr(args, "legacy_script_path", None))
        torch.cuda.set_device(tp_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        distributed = True
        if any(record.get("status") == "ok" for record in records):
            model = (
                legacy.load_entropy_model_single_gpu(args)
                if args.tensor_parallel_size == 1
                else legacy.load_entropy_model_tensor_parallel(args)
            )
        cache_only = getattr(args, "_entropy_cache_only", False)
        tokenizer = (
            legacy.load_tokenizer_without_vllm(args)
            if tp_rank == 0 and not cache_only else None
        )
        for record in records:
            if hasattr(record, "read"):
                record = record.read()
            if record.get("status") != "ok":
                raise_record_error(record, record.get("error", "Generation failed"))
            local_error = None
            entropies = None
            try:
                entropies = legacy.compute_response_token_entropies_exact(
                    model=model,
                    prompt_token_ids=record["prompt_token_ids"],
                    response_token_ids=record["response_token_ids"],
                    chunk_size=args.entropy_chunk_size,
                )
            except Exception:
                local_error = traceback.format_exc()
            gathered: List[Optional[str]] = [None] * args.tensor_parallel_size
            dist.all_gather_object(gathered, local_error)
            first_error = next((item for item in gathered if item), None)
            if first_error:
                raise_record_error(record, first_error)
            if tp_rank == 0:
                try:
                    prepared = build_token_entropy_record(
                        record, entropies, args, first_error
                    ) if cache_only else (
                        raise_record_error(record, first_error)
                        if first_error
                        else legacy.build_prepared_record(
                            record, tokenizer, entropies, args
                        )
                    )
                except Exception:
                    prepared = build_token_entropy_record(
                        record, None, args, traceback.format_exc()
                    ) if cache_only else raise_record_error(
                        record, traceback.format_exc()
                    )
                prepared["request_id"] = record.get("request_id")
                result_queue.put(("record", dp_rank, prepared))
            dist.barrier()
    except BaseException:
        result_queue.put(
            (
                "worker_error",
                dp_rank,
                f"TP rank {tp_rank} on {gpu_group} failed:\n{traceback.format_exc()}",
            )
        )
    finally:
        if model is not None:
            del model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        if distributed:
            try:
                import torch.distributed as dist

                dist.destroy_process_group()
            except Exception:
                pass
        result_queue.put(("worker_done", dp_rank, tp_rank))


def run_parallel_entropy(
    args: SimpleNamespace,
    records: Sequence[Dict[str, Any]],
    gpu_groups: Sequence[str],
    on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
    cache_only: bool = False,
    collect_results: bool = True,
) -> List[Dict[str, Any]]:
    """Run exact legacy entropy with DP across groups and TP inside a group.

    Prepared candidates are exposed to ``on_record`` immediately on arrival at
    the parent process, allowing candidate-granular interruption recovery.
    """

    if not records:
        return []
    args = SimpleNamespace(**{**vars(args), "_entropy_cache_only": cache_only})
    weighted = [
        (reference.weight, reference)
        for reference in (records.reference(index) for index in range(len(records)))
    ] if hasattr(records, "reference") else [
        (
            len(record.get("prompt_token_ids", []))
            + len(record.get("response_token_ids", [])),
            record,
        )
        for record in records
    ]
    shards = lpt_shards(weighted, len(gpu_groups))
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(
        maxsize=max(8, len(gpu_groups) * args.tensor_parallel_size * 2)
    )
    processes: List[Tuple[int, int, Any]] = []
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for dp_rank, (gpu_group, shard) in enumerate(zip(gpu_groups, shards)):
            port = find_free_tcp_port()
            for tp_rank in range(args.tensor_parallel_size):
                os.environ.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu_group,
                        "MASTER_ADDR": "127.0.0.1",
                        "MASTER_PORT": str(port),
                        "RANK": str(tp_rank),
                        "WORLD_SIZE": str(args.tensor_parallel_size),
                        "LOCAL_RANK": str(tp_rank),
                    }
                )
                process = ctx.Process(
                    target=_entropy_tp_worker,
                    args=(
                        dp_rank,
                        tp_rank,
                        gpu_group,
                        args,
                        shard,
                        result_queue,
                    ),
                )
                process.start()
                processes.append((dp_rank, tp_rank, process))
    finally:
        _restore_environment(previous)

    prepared: List[Dict[str, Any]] = []
    finished: set[Tuple[int, int]] = set()
    errors: List[str] = []
    persistence_failed = False
    completed_count = 0
    while len(finished) < len(processes):
        try:
            message_type, dp_rank, payload = result_queue.get(timeout=1.0)
        except queue.Empty:
            if all(not process.is_alive() for _, _, process in processes):
                break
            continue
        if message_type == "record":
            completed_count += 1
            if collect_results:
                prepared.append(payload)
            if on_record is not None and not persistence_failed:
                try:
                    on_record(payload)
                except BaseException:
                    persistence_failed = True
                    errors.append(
                        "Entropy result persistence failed:\n"
                        + traceback.format_exc()
                    )
        elif message_type == "worker_error":
            errors.append(str(payload))
        elif message_type == "worker_done":
            finished.add((int(dp_rank), int(payload)))
    for _, _, process in processes:
        process.join()
    abnormal = [
        (dp_rank, tp_rank)
        for dp_rank, tp_rank, process in processes
        if process.exitcode != 0
    ]
    if abnormal:
        errors.append(f"Entropy workers exited abnormally: {abnormal}")
    if completed_count != len(records):
        errors.append(f"Entropy returned {completed_count}/{len(records)} candidates")
    if errors:
        raise RuntimeError("DP+TP entropy failed.\n" + "\n".join(errors))
    prepared.sort(
        key=lambda record: (
            int(record.get("sample_idx", 0)),
            int(record.get("sequence_idx", 0)),
        )
    )
    return prepared
