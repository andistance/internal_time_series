"""One independent phase-partition worker per configured GPU; no TP or model."""

from __future__ import annotations
from .errors import raise_record_error

import multiprocessing as mp
import os
import queue
import threading
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .entropy_cache import entropy_record_matches


_SPAWN_LOCK = threading.Lock()
_RANK_ENV = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")


def phase_gpu_ids(gpu_groups: Sequence[str]) -> Tuple[str, ...]:
    """Flatten TP groups in their configured order, using each GPU once."""
    result: List[str] = []
    for group in gpu_groups:
        for value in group.split(","):
            gpu = value.strip()
            if gpu and gpu not in result:
                result.append(gpu)
    return tuple(result)


def _phase_gpu_worker(
    worker_index: int,
    gpu: str,
    args: Any,
    tasks: Sequence[Tuple[int, Dict[str, Any], Dict[str, Any]]],
    legacy_script_path: Optional[str],
    result_queue: Any,
) -> None:
    torch = None
    try:
        # The parent also sets this before spawn, so direct script imports in
        # the child cannot initialize CUDA against a former TP group's devices.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        for name in _RANK_ENV:
            os.environ.pop(name, None)
        if not tasks:
            return
        import torch

        from .legacy import load_legacy_module

        if not torch.cuda.is_available():
            raise RuntimeError(f"Phase DP worker cannot access configured GPU {gpu}")
        torch.cuda.set_device(0)
        worker_args = SimpleNamespace(**{**vars(args), "_phase_device": "cuda:0"})
        legacy = load_legacy_module(legacy_script_path)
        tokenizer = legacy.load_tokenizer_without_vllm(worker_args)
        with torch.inference_mode():
            for index, generation, entropy in tasks:
                if entropy.get("status") != "ok":
                    prepared = raise_record_error(
                        generation, entropy.get("error", "Entropy failed"),
                    )
                else:
                    if not entropy_record_matches(entropy, generation, worker_args):
                        raise ValueError(f"Token/entropy cache mismatch in phase worker: {index}")
                    try:
                        prepared = legacy.build_prepared_record(
                            generation, tokenizer, entropy["token_entropies"], worker_args,
                        )
                    except Exception:
                        prepared = raise_record_error(generation, traceback.format_exc())
                if "request_id" in generation:
                    prepared["request_id"] = generation["request_id"]
                result_queue.put(("record", worker_index, (index, prepared)))
    except BaseException:
        result_queue.put(("worker_error", worker_index, traceback.format_exc()))
    finally:
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        result_queue.put(("worker_done", worker_index, None))


def _run_legacy_parallel_phases(
    args: Any,
    tasks: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    gpu_groups: Sequence[str],
    *,
    legacy_script_path: Optional[str] = None,
    on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Run complete rollouts on independent GPUs and stream unchanged records.

    Only phase workers see singleton CUDA_VISIBLE_DEVICES. The caller's args,
    GPU groups, TP size and environment are restored/unchanged for other stages.
    No NCCL process group, reasoning model, entropy model or forward is involved.
    """
    if not tasks:
        return []
    devices = phase_gpu_ids(gpu_groups)
    if not devices:
        raise ValueError("GPU phase DP requires nonempty gpu_groups")
    for index, (generation, entropy) in enumerate(tasks):
        if entropy.get("status") == "ok" and not entropy_record_matches(entropy, generation, args):
            raise ValueError(f"Token/entropy cache mismatch before phase DP: {index}")
    # Reuse the existing load-balancing rule, independently of TP grouping.
    from .parallel import lpt_shards

    shards = lpt_shards(
        ((len(generation.get("response_token_ids", [])), (index, generation, entropy))
         for index, (generation, entropy) in enumerate(tasks)),
        len(devices),
    )
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=max(8, len(devices) * 2))
    processes = []
    results: Dict[int, Dict[str, Any]] = {}
    next_to_persist = 0
    finished = set()
    errors = []
    try:
        with _SPAWN_LOCK:
            names = ("CUDA_VISIBLE_DEVICES",) + _RANK_ENV
            previous = {name: os.environ.get(name) for name in names}
            try:
                for worker_index, (gpu, shard) in enumerate(zip(devices, shards)):
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
                    for name in _RANK_ENV:
                        os.environ.pop(name, None)
                    process = ctx.Process(
                        target=_phase_gpu_worker,
                        args=(worker_index, gpu, args, shard, legacy_script_path, result_queue),
                    )
                    process.start()
                    processes.append(process)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        while len(finished) < len(processes):
            try:
                kind, worker_index, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                if all(not process.is_alive() for process in processes):
                    break
                continue
            if kind == "record":
                index, record = payload
                if index in results or not 0 <= index < len(tasks):
                    raise RuntimeError(f"Phase DP returned an invalid or duplicate rollout index: {index}")
                results[index] = record
                # GPU completion order can differ from rollout order. Preserve
                # the former sequential phase file/scoring order at the writer.
                while next_to_persist in results:
                    if on_record is not None:
                        on_record(results[next_to_persist])
                    next_to_persist += 1
            elif kind == "worker_error":
                errors.append(f"GPU {devices[worker_index]} phase worker failed:\n{payload}")
                break
            elif kind == "worker_done":
                finished.add(worker_index)
        for process in processes:
            if errors and process.is_alive():
                process.terminate()
            process.join()
        abnormal = [index for index, process in enumerate(processes) if process.exitcode != 0]
        if abnormal:
            errors.append(f"Phase DP workers exited abnormally: {abnormal}")
        if len(results) != len(tasks):
            errors.append(f"Phase DP returned {len(results)}/{len(tasks)} rollouts")
        if errors:
            raise RuntimeError("GPU phase DP failed; saved entropy remains reusable.\n" + "\n".join(errors))
        return [results[index] for index in range(len(tasks))]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        result_queue.close()
        result_queue.join_thread()


def run_parallel_phases(
    args: Any,
    tasks: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    gpu_groups: Sequence[str],
    *,
    legacy_script_path: Optional[str] = None,
    on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_complete: Optional[Callable[[], None]] = None,
    cache_dir: Optional[str] = None,
    collect_results: bool = True,
    validated: bool = False,
) -> List[Dict[str, Any]]:
    """Stream bounded work to lightweight, persistent single-GPU subprocesses.

    Warm tasks contain mmap offsets, not trajectories. Cold tasks are sent only
    as capacity becomes available; there is no whole-dataset pickle or parent
    validation barrier. Completion progress is independent of ordered output.
    """
    if not tasks:
        return []
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import tempfile
    import math
    from multiprocessing.connection import Listener
    from .phase_structure import StructureCache, build_error, build_prepared, structure_key, tokenizer_identity

    # Explicit third-party legacy runtimes retain their own preparation rules.
    if legacy_script_path and Path(legacy_script_path).name not in {
        "3_best_of_n_via_phases_scores.py", "3_best_of_n_via_phases_scores_qwen.py",
        "3_best_of_n_via_phases_scores_qwen_sampled.py",
    }:
        def persist(record):
            if on_complete:
                on_complete()
            if on_record:
                on_record(record)
        return _run_legacy_parallel_phases(args, tasks, gpu_groups,
                                          legacy_script_path=legacy_script_path, on_record=persist)
    devices = phase_gpu_ids(gpu_groups)
    if not devices:
        raise ValueError("GPU phase DP requires nonempty gpu_groups")
    temporary = tempfile.TemporaryDirectory(prefix="bon-phase-") if cache_dir is None else None
    directory = cache_dir or temporary.name
    cache = StructureCache(directory)
    directory = str(cache.root)
    tokenizer_key = tokenizer_identity(args)
    authkey = os.urandom(32)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    # AF_INET Listener exposes a socket on both Linux and Windows. A timeout
    # lets startup failures surface instead of hanging at accept indefinitely.
    listener._listener._socket.settimeout(1.0)
    events = queue.Queue(maxsize=max(8, len(devices) * 2))
    dispatch_lock = threading.Lock()
    next_task = 0
    stopping = threading.Event()
    processes, connections, threads = [], [], []
    root = str(Path(__file__).resolve().parents[2])
    batch_limit = min(32, max(1, math.ceil(len(tasks) / (len(devices) * 4))))

    def emit(event):
        while not stopping.is_set():
            try:
                events.put(event, timeout=0.2)
                return
            except queue.Full:
                pass

    def next_batch():
        nonlocal next_task
        batch, largest = [], 0
        while len(batch) < batch_limit:
            with dispatch_lock:
                if next_task >= len(tasks):
                    break
                index = next_task
                next_task += 1
            generation, entropy = tasks.metadata(index) if hasattr(tasks, "metadata") else tasks[index]
            if entropy.get("status") != "ok":
                raise_record_error(generation, entropy.get("error", "Entropy failed"))
            if not validated and not entropy_record_matches(entropy, generation, args):
                raise ValueError(f"Token/entropy cache mismatch before phase DP: {index}")
            key = structure_key(generation, entropy, tokenizer_key)
            descriptor = cache.entries.get(key)
            if descriptor is None:
                # NumPy's pickle representation is compact and avoids sending
                # duplicated prompt IDs and unrelated benchmark metadata.
                import numpy as np
                payload = {"response_token_ids": np.asarray(generation["response_token_ids"], dtype=np.int64),
                           "token_logprobs": generation["token_logprobs"], "full_output": generation.get("full_output")}
                if "_bon_source_structure" in generation:
                    payload["_bon_source_structure"] = generation["_bon_source_structure"]
                if hasattr(tasks, "metadata"):
                    entropy = tasks[index][1]
                values = {"token_entropies": np.asarray(entropy["token_entropies"], dtype=np.float64)}
                batch.append((index, None, key, payload, values))
                break  # cold migration reports progress one rollout at a time
            batch.append((index, descriptor, None, None, None))
            size = descriptor["num_available"] if getattr(args, "phase_partition_algorithm", "original") == "adaptive" else descriptor["num_valid"]
            largest = max(largest, size)
            if largest * (len(batch) + 1) > 131072:
                break
        return batch

    def drive(connection, worker):
        try:
            connection.send((vars(args), directory))
            kind, payload = connection.recv()
            if kind != "ready":
                raise RuntimeError(payload)
            while not stopping.is_set():
                batch = next_batch()
                if not batch:
                    break
                connection.send(batch)
                kind, payload = connection.recv()
                if kind != "results":
                    raise RuntimeError(payload)
                emit(("results", payload))
            connection.send(None)
        except BaseException:
            emit(("error", f"GPU {devices[worker]} phase worker failed:\n{traceback.format_exc()}"))
        finally:
            emit(("done", worker))

    received, pending, output, cursor, finished = set(), {}, [], 0, 0
    try:
        for index, gpu in enumerate(devices):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["_BON_PHASE_AUTH"] = authkey.hex()
            environment["PYTHONPATH"] = root + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
            for name in _RANK_ENV:
                environment.pop(name, None)
            # -m avoids spawn re-executing a heavyweight benchmark __main__.
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "scripts.transformer_bon.phase_worker", str(listener.address[1]), str(index)],
                cwd=os.getcwd(), env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ))
        while len(connections) < len(processes):
            try:
                connection = listener.accept()
            except socket.timeout:
                failed = [index for index, process in enumerate(processes) if process.poll() is not None]
                if failed:
                    raise RuntimeError(f"Phase DP subprocess exited during startup: {failed}")
                continue
            connections.append(connection)
            worker = connection.recv()
            thread = threading.Thread(target=drive, args=(connection, worker), daemon=True)
            thread.start()
            threads.append(thread)
        while finished < len(processes):
            try:
                kind, payload = events.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "error":
                raise RuntimeError("GPU phase DP failed; saved entropy remains reusable.\n" + payload)
            if kind == "done":
                finished += 1
                continue
            for item in payload:
                index = item[0]
                if index in received or not 0 <= index < len(tasks):
                    raise RuntimeError(f"Phase DP returned an invalid/duplicate rollout: {index}")
                received.add(index)
                pending[index] = item
                if item[1] is not None:
                    cache.entries[item[1]["key"]] = item[1]
                if on_complete is not None:
                    on_complete()
            while cursor in pending:
                index, descriptor, boundaries, count, error = pending.pop(cursor)
                generation = tasks.generations[index] if hasattr(tasks, "generations") else tasks[index][0]
                record = build_error(generation, error) if error else build_prepared(generation, descriptor, boundaries, count, args)
                if "request_id" in generation:
                    record["request_id"] = generation["request_id"]
                if on_record is not None:
                    on_record(record)
                if collect_results:
                    output.append(record)
                cursor += 1
        if cursor != len(tasks):
            raise RuntimeError(f"Phase DP returned {cursor}/{len(tasks)} rollouts")
        for process in processes:
            process.wait(timeout=30)
            if process.returncode:
                raise RuntimeError(f"Phase DP subprocess exited with code {process.returncode}")
        return output
    finally:
        stopping.set()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for connection in connections:
            connection.close()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        listener.close()
        for thread in threads:
            thread.join(timeout=1)
        if temporary is not None:
            temporary.cleanup()


def run_serial_phases(args, tasks, cache_dir, runtime, on_record, on_complete=None, *, include_request_id=True):
    """The same reusable structure cache for CPU-only configurations."""
    from .phase_structure import (StructureCache, build_error, build_prepared, compile_structure,
                                  select_cached, structure_key, tokenizer_identity)
    from pathlib import Path
    custom = getattr(runtime, "__file__", None)
    if custom and Path(custom).name not in {
        "3_best_of_n_via_phases_scores.py", "3_best_of_n_via_phases_scores_qwen.py",
        "3_best_of_n_via_phases_scores_qwen_sampled.py",
    }:
        tokenizer = runtime.load_tokenizer_without_vllm(args) if tasks else None
        for generation, entropy in tasks:
            if entropy.get("status") != "ok":
                prepared = raise_record_error(generation, entropy.get("error", "Entropy failed"))
            else:
                try:
                    prepared = runtime.build_prepared_record(generation, tokenizer, entropy["token_entropies"], args)
                except Exception:
                    prepared = raise_record_error(generation, traceback.format_exc())
            if include_request_id and "request_id" in generation:
                prepared["request_id"] = generation["request_id"]
            if on_complete:
                on_complete()
            on_record(prepared)
        return
    cache = StructureCache(cache_dir)
    tokenizer_key, tokenizer = tokenizer_identity(args), None
    for index in range(len(tasks)):
        generation, entropy = tasks.metadata(index) if hasattr(tasks, "metadata") else tasks[index]
        if entropy.get("status") != "ok":
            prepared = build_error(generation, entropy.get("error", "Entropy failed"))
        else:
            try:
                key = structure_key(generation, entropy, tokenizer_key)
                descriptor = cache.entries.get(key)
                if descriptor is None:
                    if hasattr(tasks, "metadata"):
                        entropy = tasks[index][1]
                    if tokenizer is None and "_bon_source_structure" not in generation:
                        tokenizer = runtime.load_tokenizer_without_vllm(args)
                    descriptor = cache.save(key, *compile_structure(generation, entropy, tokenizer))
                boundaries, count = select_cached(descriptor, args)
                prepared = build_prepared(generation, descriptor, boundaries, count, args)
            except Exception:
                prepared = build_error(generation, traceback.format_exc())
        if include_request_id and "request_id" in generation:
            prepared["request_id"] = generation["request_id"]
        if on_complete is not None:
            on_complete()
        on_record(prepared)
