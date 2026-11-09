from __future__ import annotations
from .errors import raise_record_error, require_success

import copy
import gc
import inspect
import json
import math
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .legacy import load_legacy_module
from .entropy_cache import (
    append_entropy_record,
    build_token_entropy_record,
    entropy_record_matches,
    repair_prepared_tail,
    read_entropy_records,
)
from .phases import validate_phase_partition_options
from .phase_parallel import run_parallel_phases, run_serial_phases
from .prepared_cache import load_benchmark_prepared, prepared_paths
from .prepared_index import PreparedIndex
from .record_index import NEWLINE
from .progress import DatasetProgress
from .models import BestOfNCandidate, BestOfNRequest, BestOfNResult
from .parallel import (
    build_missing_generation_jobs,
    parse_gpu_groups,
    run_parallel_entropy,
    run_parallel_generation,
)
from .templates import (
    is_chat_template_registered,
    prompt_add_special_tokens,
    render_prompt,
)


@dataclass
class BestOfNConfig:
    model_path: str
    checkpoint_path: str
    n: int
    output_aggregation: str = "product"
    chat_template: str = "qwen3"
    enable_thinking: Optional[bool] = None
    training_metrics_json: Optional[str] = None
    transformer_type: str = "auto"
    fold_index: Optional[int] = None
    value_device: str = "auto"
    scoring_batch_size: int = 8
    candidate_batch_size: int = 8
    max_tokens: int = 8192
    max_model_len: int = 10240
    temperature: float = 0.7
    top_p: float = 0.95
    logprobs: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    trust_remote_code: bool = False
    num_segments: int = 8
    entropy_threshold: float = 1.2
    entropy_chunk_size: int = 16
    entropy_model_path: Optional[str] = None
    entropy_device_map: str = "auto"
    save_token_entropies: bool = True
    seed: Optional[int] = 42
    gpu_groups: Optional[str] = None
    performance_profile: Literal["throughput", "balanced", "memory"] = "throughput"
    reproducibility_mode: Literal["legacy_strict"] = "legacy_strict"
    model_transition: Literal["auto", "reload", "cpu_swap"] = "auto"
    value_residency: Literal["auto", "gpu", "cpu"] = "auto"
    generation_max_active_jobs: Optional[int] = None
    staging_flush_records: int = 32
    legacy_script_path: Optional[str] = None
    vllm_cache_mode: Literal["auto", "isolated", "shared"] = "auto"
    vllm_cache_root: Optional[str] = None
    vllm_enforce_eager: bool = False
    generation_worker_startup_timeout: float = 600.0
    phase_partition_algorithm: str = "original"
    high_entropy_ratio: Optional[float] = None
    target_phase_tokens: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if self.output_aggregation not in {"product", "min", "mean", "last"}:
            raise ValueError("Unsupported aggregation")
        if not is_chat_template_registered(self.chat_template):
            raise ValueError(f"Unsupported chat template: {self.chat_template}")
        if self.chat_template == "qwen3" and self.enable_thinking is None:
            raise ValueError("qwen3 requires enable_thinking")
        if self.chat_template != "qwen3" and self.enable_thinking is not None:
            raise ValueError("enable_thinking is only valid for qwen3")
        if self.logprobs < 1:
            raise ValueError("logprobs must be at least 1")
        if self.candidate_batch_size < 1 or self.scoring_batch_size < 1:
            raise ValueError("candidate/scoring batch sizes must be at least 1")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")
        validate_phase_partition_options(self)
        if self.entropy_chunk_size < 1:
            raise ValueError("entropy chunk size must be at least 1")
        if self.performance_profile not in {"throughput", "balanced", "memory"}:
            raise ValueError("Unsupported performance_profile")
        if self.reproducibility_mode != "legacy_strict":
            raise ValueError("Only legacy_strict reproducibility is supported")
        if self.model_transition not in {"auto", "reload", "cpu_swap"}:
            raise ValueError("Unsupported model_transition")
        if self.value_residency not in {"auto", "gpu", "cpu"}:
            raise ValueError("Unsupported value_residency")
        if self.generation_max_active_jobs is not None and self.generation_max_active_jobs < 1:
            raise ValueError("generation_max_active_jobs must be at least 1")
        if self.staging_flush_records < 1:
            raise ValueError("staging_flush_records must be at least 1")
        if self.vllm_cache_mode not in {"auto", "isolated", "shared"}:
            raise ValueError("Unsupported vllm_cache_mode")
        if self.generation_worker_startup_timeout <= 0:
            raise ValueError("generation_worker_startup_timeout must be positive")
        parse_gpu_groups(self.gpu_groups, self.tensor_parallel_size)

    @property
    def parsed_gpu_groups(self) -> Tuple[str, ...]:
        return parse_gpu_groups(self.gpu_groups, self.tensor_parallel_size)

    def legacy_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            model_path=self.model_path,
            checkpoint_path=self.checkpoint_path,
            training_metrics_json=self.training_metrics_json,
            transformer_type=self.transformer_type,
            fold_index=self.fold_index,
            value_device=self.value_device,
            scoring_batch_size=self.scoring_batch_size,
            batch_size=self.candidate_batch_size,
            max_tokens=self.max_tokens,
            max_model_len=self.max_model_len,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=self.logprobs,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            num_segments=self.num_segments,
            entropy_threshold=self.entropy_threshold,
            phase_partition_algorithm=self.phase_partition_algorithm,
            high_entropy_ratio=self.high_entropy_ratio,
            target_phase_tokens=self.target_phase_tokens,
            entropy_chunk_size=self.entropy_chunk_size,
            entropy_model_path=self.entropy_model_path,
            entropy_device_map=self.entropy_device_map,
            no_save_token_entropies=not self.save_token_entropies,
            seed=self.seed,
            N=self.n,
            project_root=None,
            legacy_script_path=self.legacy_script_path,
        )


def _link_fields(request: BestOfNRequest, sequence_idx: int) -> Dict[str, Any]:
    fields: Dict[str, Any] = {
        "benchmark": request.metadata.get("benchmark"),
        "sample_id": request.sample_id,
        "request_id": request.request_id,
        "candidate_key": f"{request.request_id}:c{sequence_idx}",
        "metadata": copy.deepcopy(request.metadata),
    }
    for name in (
        "problem_id",
        "generation_idx",
        "attempt_idx",
        "attempt_number",
    ):
        if name in request.metadata:
            fields[name] = request.metadata[name]
    return fields


def _sample_index(request: BestOfNRequest) -> int:
    value = request.metadata.get(
        "sample_idx", request.metadata.get("problem_id", 0)
    )
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _iter_stage_entries(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid staged JSONL at {path}:{line_number}") from error
            if not isinstance(entry, dict) or "request_id" not in entry:
                raise ValueError(f"Invalid staged entry at {path}:{line_number}")
            require_success(entry)
            yield entry


def _repair_truncated_stage_tail(path: Path) -> None:
    """Drop only an incomplete final JSONL record left by interruption."""

    if not path.exists():
        return
    with path.open("rb+") as file:
        last_good_offset = 0
        line_number = 0
        while True:
            raw_line = file.readline()
            if not raw_line:
                return
            line_number += 1
            if not raw_line.strip():
                last_good_offset = file.tell()
                continue
            try:
                json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if file.read().strip():
                    raise ValueError(
                        f"Corrupt staged JSONL before EOF at {path}:{line_number}"
                    ) from error
                file.seek(last_good_offset)
                file.truncate()
                file.flush()
                os.fsync(file.fileno())
                return
            last_good_offset = file.tell()


def _stage_index(
    path: Path,
    expected_records: Optional[int] = None,
) -> Dict[str, bool]:
    return {
        str(entry["request_id"]): True
        for entry in _iter_stage_request_entries(path)
        if expected_records is None
        or len(entry.get("records", [])) == expected_records
    }


def _stage_records_index(path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Index every durably written rollout by request and sequence number."""

    from .generation_prefix import stage_records_index
    return stage_records_index(path)


def _iter_stage_request_entries(path: Path) -> Iterator[Dict[str, Any]]:
    """Read legacy request bundles and candidate-granular stage lines."""

    _repair_truncated_stage_tail(path)
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for entry in _iter_stage_entries(path):
        request_id = str(entry["request_id"])
        if request_id not in grouped:
            order.append(request_id)
            grouped[request_id] = {
                "records": {},
            }
        if "record" in entry:
            record = entry["record"]
            grouped[request_id]["records"][
                int(entry.get("sequence_idx", record["sequence_idx"]))
            ] = record
        else:
            for record in entry.get("records", []):
                grouped[request_id]["records"][int(record["sequence_idx"])] = record
    for request_id in order:
        item = grouped[request_id]
        yield {
            "request_id": request_id,
            "records": [item["records"][index] for index in sorted(item["records"])],
        }


def _append_stage_entry(path: Path, entry: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _append_stage_records(
    path: Path,
    request_id: str,
    records: Sequence[Dict[str, Any]],
    flush_records: int = 32,
) -> None:
    """Persist candidates separately with bounded durability batches."""

    if flush_records < 1:
        raise ValueError("flush_records must be at least 1")
    with path.open("a", encoding="utf-8") as file:
        records_since_flush = 0
        for record in records:
            file.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "sequence_idx": int(record["sequence_idx"]),
                        "record": record,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            records_since_flush += 1
            if records_since_flush >= flush_records:
                file.flush()
                os.fsync(file.fileno())
                records_since_flush = 0
        if records_since_flush:
            file.flush()
            os.fsync(file.fileno())


class _IncrementalStageWriter:
    """Append candidate stage records and fsync every fixed response count."""

    def __init__(
        self,
        path: Path,
        flush_records: int,
    ) -> None:
        if flush_records < 1:
            raise ValueError("flush_records must be at least 1")
        self.path = path
        self.flush_records = flush_records
        self._file: Any = None
        self._records_since_flush = 0

    def __enter__(self) -> "_IncrementalStageWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("ab")
        return self

    def write_record(self, record: Dict[str, Any]) -> int:
        require_success(record)
        if self._file is None:
            raise RuntimeError("Incremental stage writer is not open")
        request_id = str(record.get("request_id", ""))
        offset = self._file.tell()
        self._file.write(
            json.dumps(
                {
                    "request_id": request_id,
                    "sequence_idx": int(record["sequence_idx"]),
                    "record": record,
                },
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            + NEWLINE
        )
        self._records_since_flush += 1
        if self._records_since_flush >= self.flush_records:
            self.flush()
        return offset

    def write_records(self, records: Sequence[Dict[str, Any]]) -> None:
        for record in records:
            self.write_record(record)

    def flush(self) -> None:
        if self._file is None or self._records_since_flush == 0:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._records_since_flush = 0

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self.flush()
        finally:
            self._file.close()
            self._file = None

    def __exit__(self, exc_type: Any, exc: Any, traceback_obj: Any) -> None:
        self.close()


class BestOfNService:
    """Thread-safe online Best-of-N generation and trajectory scoring service."""

    SCORE_KEYS = {
        "product": "product_score",
        "min": "min_score",
        "mean": "mean_score",
        "last": "last_score",
    }

    def __init__(self, config: BestOfNConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._value_components: Optional[Tuple[Any, Any, Any]] = None
        self._value_target_device: Any = None
        self._value_transformer_type: Optional[str] = None

    def _llm_kwargs(self, args: SimpleNamespace) -> Dict[str, Any]:
        """Build vLLM arguments without changing legacy defaults."""

        kwargs: Dict[str, Any] = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "max_model_len": args.max_model_len,
        }
        if self.config.vllm_enforce_eager:
            kwargs["enforce_eager"] = True
        return kwargs

    def _create_generation_llm(self, legacy: Any, args: SimpleNamespace) -> Any:
        """Resolve vLLM lazily, after the generation environment is installed."""

        load_api = getattr(legacy, "load_vllm_api", None)
        if callable(load_api):
            llm_cls, _ = load_api()
        else:
            # Preserve compatibility with older/custom runtime scripts.
            llm_cls = legacy.LLM
        return llm_cls(**self._llm_kwargs(args))

    def _generation_record(
        self,
        legacy: Any,
        request_output: Any,
        output: Any,
        tokenizer: Any,
        prompt: str,
        request: BestOfNRequest,
        sequence_idx: int,
    ) -> Dict[str, Any]:
        token_ids = [int(token_id) for token_id in output.token_ids]
        full_output, _ = legacy.build_token_pieces_with_char_spans(token_ids, tokenizer)
        output_logprobs = output.logprobs
        if output_logprobs is None or len(output_logprobs) != len(token_ids):
            raise RuntimeError("vLLM did not return one logprob mapping per token")
        token_logprobs = [
            legacy.get_sampled_token_logprob(logprob_dict, token_id)
            for token_id, logprob_dict in zip(token_ids, output_logprobs)
        ]
        prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
        if prompt_token_ids is None:
            prompt_token_ids = tokenizer.encode(
                prompt,
                add_special_tokens=prompt_add_special_tokens(
                    self.config.chat_template
                ),
            )
        return {
            "status": "ok",
            "request_id": request.request_id,
            "sample_idx": _sample_index(request),
            "sequence_idx": sequence_idx,
            "problem": request.task_text,
            "gold_answer": request.gold_answer,
            "difficulty": request.metadata.get("difficulty"),
            "unique_id": request.sample_id,
            "full_output": full_output,
            "prompt_token_ids": [int(value) for value in prompt_token_ids],
            "response_token_ids": token_ids,
            "token_logprobs": token_logprobs,
            "finish_reason": getattr(output, "finish_reason", None),
        }

    def _generate(
        self, legacy: Any, request: BestOfNRequest, args: SimpleNamespace
    ) -> List[Dict[str, Any]]:
        llm = self._create_generation_llm(legacy, args)
        records: List[Dict[str, Any]] = []
        try:
            tokenizer = llm.get_tokenizer()
            records = self._generate_with_llm(
                legacy,
                llm,
                tokenizer,
                request,
                args,
            )
        finally:
            legacy.release_vllm(llm)
        return records

    def _generate_with_llm(
        self,
        legacy: Any,
        llm: Any,
        tokenizer: Any,
        request: BestOfNRequest,
        args: SimpleNamespace,
        on_records: Optional[Callable[[Sequence[Dict[str, Any]]], None]] = None,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for start in range(0, self.config.n, self.config.candidate_batch_size):
            batch_records = self._generate_job_with_llm(
                legacy,
                llm,
                tokenizer,
                request,
                args,
                start,
                min(self.config.n, start + self.config.candidate_batch_size),
            )
            records.extend(batch_records)
            if on_records is not None:
                on_records(batch_records)
        return records

    def _generate_job_with_llm(
        self,
        legacy: Any,
        llm: Any,
        tokenizer: Any,
        request: BestOfNRequest,
        args: SimpleNamespace,
        start: int,
        stop: int,
    ) -> List[Dict[str, Any]]:
        """Generate one indivisible legacy candidate batch."""

        from .parallel import GenerationJob

        return self._generate_jobs_with_llm(
            legacy,
            llm,
            tokenizer,
            [GenerationJob(request=request, start=start, stop=stop)],
            args,
        )

    def _generate_jobs_with_llm(
        self,
        legacy: Any,
        llm: Any,
        tokenizer: Any,
        jobs: Sequence[Any],
        args: SimpleNamespace,
        prompt_cache: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Submit preserved candidate batches together when vLLM supports it."""

        cache = prompt_cache if prompt_cache is not None else {}
        prompts: List[str] = []
        params_list: List[Any] = []
        metadata: List[Tuple[Any, List[int], str]] = []
        for job in jobs:
            request = job.request
            prompt = cache.get(request.request_id)
            if prompt is None:
                prompt = render_prompt(
                    request,
                    tokenizer,
                    self.config.chat_template,
                    enable_thinking=self.config.enable_thinking,
                )
                cache[request.request_id] = prompt
            sequence_indices = list(job.sequence_indices)
            seed = legacy.generation_seed(
                self.config.seed,
                _sample_index(request),
                sequence_indices,
                self.config.n,
            )
            prompts.append(prompt)
            params_list.append(
                legacy.make_sampling_params(
                    args, n=len(sequence_indices), seed=seed
                )
            )
            metadata.append((request, sequence_indices, prompt))

        try:
            try:
                request_outputs = llm.generate(prompts, params_list)
            except (TypeError, ValueError):
                if len(jobs) == 1:
                    request_outputs = llm.generate(prompts, params_list[0])
                else:
                    records: List[Dict[str, Any]] = []
                    for job in jobs:
                        records.extend(
                            self._generate_jobs_with_llm(
                                legacy,
                                llm,
                                tokenizer,
                                [job],
                                args,
                                cache,
                            )
                        )
                    return records
            if len(request_outputs) != len(jobs):
                raise RuntimeError("vLLM returned an unexpected request count")
            records = []
            for request_output, (request, sequence_indices, prompt) in zip(
                request_outputs, metadata
            ):
                outputs = list(request_output.outputs)
                if len(outputs) != len(sequence_indices):
                    raise RuntimeError(
                        f"Expected {len(sequence_indices)} candidates, "
                        f"received {len(outputs)}"
                    )
                for sequence_idx, output in zip(sequence_indices, outputs):
                    try:
                        records.append(
                            self._generation_record(
                                legacy,
                                request_output,
                                output,
                                tokenizer,
                                prompt,
                                request,
                                sequence_idx,
                            )
                        )
                    except Exception:
                        raise
            return records
        except Exception:
            raise

    @staticmethod
    def _generation_error_record(
        request: BestOfNRequest, sequence_idx: int, error: str
    ) -> Dict[str, Any]:
        raise_record_error({"request_id": request.request_id, "sequence_idx": sequence_idx}, error)

    def _prepare(
        self,
        legacy: Any,
        generation_records: Sequence[Dict[str, Any]],
        args: SimpleNamespace,
    ) -> List[Dict[str, Any]]:
        tokenizer = legacy.load_tokenizer_without_vllm(args)
        needs_model = any(record.get("status") == "ok" for record in generation_records)
        entropy_model = legacy.load_entropy_model(args) if needs_model else None
        try:
            prepared = self._prepare_with_entropy_model(
                legacy,
                generation_records,
                args,
                tokenizer,
                entropy_model,
            )
        finally:
            if entropy_model is not None:
                del entropy_model
            gc.collect()
            if legacy.torch.cuda.is_available():
                legacy.torch.cuda.empty_cache()
        return prepared

    @staticmethod
    def _prepare_with_entropy_model(
        legacy: Any,
        generation_records: Sequence[Dict[str, Any]],
        args: SimpleNamespace,
        tokenizer: Any,
        entropy_model: Any,
        on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for record in generation_records:
            if record.get("status") != "ok":
                prepared_record = raise_record_error(
                    record, record.get("error", "Generation failed")
                )
            else:
                try:
                    if entropy_model is None:
                        raise RuntimeError("Entropy model was not loaded")
                    entropies = legacy.compute_response_token_entropies_exact(
                        model=entropy_model,
                        prompt_token_ids=record["prompt_token_ids"],
                        response_token_ids=record["response_token_ids"],
                        chunk_size=args.entropy_chunk_size,
                    )
                    if getattr(args, "prm_step_scoring", False):
                        from .step_prm import build_step_prepared_record
                        prepared_record = build_step_prepared_record(legacy, record, tokenizer, entropies, args)
                    else:
                        prepared_record = legacy.build_prepared_record(
                            record, tokenizer, entropies, args
                        )
                except Exception:
                    prepared_record = raise_record_error(
                        record, traceback.format_exc()
                    )
            require_success(prepared_record)
            prepared_record["request_id"] = record.get("request_id")
            prepared.append(prepared_record)
            if on_record is not None:
                on_record(prepared_record)
        return prepared

    def _get_value_components(
        self, legacy: Any, args: SimpleNamespace
    ) -> Tuple[Any, Any, Any]:
        if self._value_components is None:
            model, normalizer, device, metadata = legacy.load_value_components(args)
            self._value_target_device = device
            transformer_type = None
            if isinstance(metadata, Mapping):
                transformer_type = metadata.get("transformer_type")
            if transformer_type is None:
                transformer_type = getattr(args, "transformer_type", None)
            if transformer_type in {None, "auto"}:
                transformer_type = "bidirectional"
            self._value_transformer_type = str(transformer_type)
            if self.config.value_residency == "cpu":
                model = model.to("cpu")
                device = legacy.torch.device("cpu")
            self._value_components = (model, normalizer, device)
        elif (
            self.config.value_residency != "cpu"
            and self._value_target_device is not None
            and str(self._value_components[2]) != str(self._value_target_device)
        ):
            model, normalizer, _ = self._value_components
            model = model.to(self._value_target_device)
            self._value_components = (model, normalizer, self._value_target_device)
        return self._value_components

    def _park_value_model_if_needed(self) -> None:
        if self._value_components is None:
            return
        should_park = self.config.value_residency == "cpu" or (
            self.config.value_residency == "auto"
            and bool(self.config.parsed_gpu_groups)
        )
        if not should_park or str(self._value_components[2]) == "cpu":
            return
        model, normalizer, _ = self._value_components
        model = model.to("cpu")
        try:
            legacy = load_legacy_module(self.config.legacy_script_path)
            cpu = legacy.torch.device("cpu")
            if legacy.torch.cuda.is_available():
                legacy.torch.cuda.empty_cache()
        except Exception:
            cpu = "cpu"
        self._value_components = (model, normalizer, cpu)

    def _score(
        self,
        legacy: Any,
        request: BestOfNRequest,
        prepared_records: Sequence[Dict[str, Any]],
        args: SimpleNamespace,
        on_rollout: Optional[Callable[[], None]] = None,
    ) -> List[BestOfNCandidate]:
        model, normalizer, device = self._get_value_components(legacy, args)
        candidates_by_index: Dict[int, BestOfNCandidate] = {}
        valid_items: List[Tuple[Dict[str, Any], Any]] = []


        for prepared in prepared_records:
            sequence_idx = int(prepared["sequence_idx"])
            try:
                if prepared.get("status") != "ok":
                    raise ValueError(prepared.get("error", "Candidate preparation failed"))
                features = normalizer.transform_one(
                    legacy.build_value_features(
                        prepared,
                        transformer_type=self._value_transformer_type
                        or "bidirectional",
                    )
                )
                if not prepared.get("phase_ranges"):
                    raise ValueError("Candidate has no phase ranges")
                valid_items.append((prepared, features))
            except Exception:
                raise

        for batch_start in range(0, len(valid_items), args.scoring_batch_size):
            batch_items = valid_items[
                batch_start : batch_start + args.scoring_batch_size
            ]
            arrays = [features for _, features in batch_items]
            max_len = max(array.shape[0] for array in arrays)
            feature_dim = arrays[0].shape[1]
            x_tensor = legacy.torch.zeros(
                len(arrays), max_len, feature_dim, dtype=legacy.torch.float32
            )
            attention_mask = legacy.torch.zeros(
                len(arrays), max_len, dtype=legacy.torch.bool
            )
            lengths: List[int] = []
            for batch_index, array in enumerate(arrays):
                length = int(array.shape[0])
                lengths.append(length)
                x_tensor[batch_index, :length] = legacy.torch.from_numpy(array)
                attention_mask[batch_index, :length] = True

            try:
                with legacy.torch.no_grad():
                    predictions = (
                        model(
                            x_tensor.to(device),
                            attention_mask=attention_mask.to(device),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
            except Exception:
                raise

            for batch_index, ((prepared, _), length) in enumerate(
                zip(batch_items, lengths)
            ):
                sequence_idx = int(prepared["sequence_idx"])
                links = _link_fields(request, sequence_idx)
                try:
                    step_predictions = [
                        float(value)
                        for value in predictions[batch_index, :length]
                    ]
                    steps = copy.deepcopy(prepared["steps"])
                    for step, prediction in zip(steps, step_predictions):
                        step["trajectory_model_step_score"] = prediction
                    phase_scores, final_scores = legacy.attach_phase_scores_to_steps(
                        steps,
                        prepared["phase_ranges"],
                        step_predictions,
                    )
                    common = {
                        "sample_idx": _sample_index(request),
                        "sequence_idx": sequence_idx,
                        "problem": request.task_text,
                        "gold_answer": request.gold_answer,
                        "difficulty": request.metadata.get("difficulty"),
                        "unique_id": request.sample_id,
                        "is_correct": None,
                        "valid_for_selection": True,
                        **links,
                    }
                    summary = {
                        **common,
                        "full_output": prepared.get("full_output"),
                        "step_avg_confidences": prepared["step_avg_confidences"],
                        "num_steps": prepared["num_steps"],
                        "num_generated_tokens": prepared["num_generated_tokens"],
                        "finish_reason": prepared.get("finish_reason"),
                        "phase_ranges": prepared["phase_ranges"],
                        "phase_scores": phase_scores,
                        **final_scores,
                    }
                    if "token_entropies" in prepared:
                        summary["token_entropies"] = prepared["token_entropies"]
                    score = float(
                        final_scores[
                            self.SCORE_KEYS[self.config.output_aggregation]
                        ]
                    )
                    candidates_by_index[sequence_idx] = BestOfNCandidate(
                        sequence_idx=sequence_idx,
                        candidate_key=links["candidate_key"],
                        text=str(prepared.get("full_output") or ""),
                        score=score,
                        summary_record=summary,
                        steps_record={**common, "steps": steps},
                    )
                except Exception:
                    raise
                finally:
                    if on_rollout is not None:
                        on_rollout()

        return [
            candidates_by_index[int(prepared["sequence_idx"])]
            for prepared in prepared_records
        ]

    def _result_from_candidates(
        self,
        request: BestOfNRequest,
        candidates: List[BestOfNCandidate],
        *,
        require_valid_selection: bool,
    ) -> BestOfNResult:
        for candidate in candidates:
            require_success(candidate.summary_record)
            require_success(candidate.steps_record)
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                0 if math.isfinite(candidate.score) else 1,
                -candidate.score if math.isfinite(candidate.score) else 0.0,
                candidate.sequence_idx,
            ),
        )
        selectable = [
            candidate
            for candidate in ranked
            if math.isfinite(candidate.score)
            and candidate.summary_record.get("valid_for_selection")
        ]
        if selectable:
            selected = selectable[0]
        elif require_valid_selection:
            raise RuntimeError(
                f"No valid Best-of-N candidate for request {request.request_id}"
            )
        elif ranked:
            selected = ranked[0]
        else:
            raise RuntimeError(f"No candidates for request {request.request_id}")
        return BestOfNResult(
            request=request,
            aggregation=self.config.output_aggregation,
            selected_index=selected.sequence_idx,
            selected_candidate_key=selected.candidate_key,
            selected_text=selected.text,
            selected_score=selected.score,
            candidates=candidates,
        )

    def _batch_manifest(self) -> Dict[str, Any]:
        config = self.config
        return {
            "version": 1,
            "model_path": str(Path(config.model_path).expanduser()),
            "checkpoint_path": str(Path(config.checkpoint_path).expanduser()),
            "training_metrics_json": config.training_metrics_json,
            "n": config.n,
            "output_aggregation": config.output_aggregation,
            "chat_template": config.chat_template,
            "enable_thinking": config.enable_thinking,
            "transformer_type": config.transformer_type,
            "fold_index": config.fold_index,
            "value_device": config.value_device,
            "scoring_batch_size": config.scoring_batch_size,
            "candidate_batch_size": config.candidate_batch_size,
            "max_tokens": config.max_tokens,
            "max_model_len": config.max_model_len,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "logprobs": config.logprobs,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "num_segments": config.num_segments,
            "entropy_threshold": config.entropy_threshold,
            "entropy_chunk_size": config.entropy_chunk_size,
            "entropy_model_path": config.entropy_model_path,
            "entropy_device_map": config.entropy_device_map,
            "save_token_entropies": config.save_token_entropies,
            "seed": config.seed,
            "gpu_groups": list(config.parsed_gpu_groups),
            "performance_profile": config.performance_profile,
            "reproducibility_mode": config.reproducibility_mode,
            "model_transition": config.model_transition,
            "value_residency": config.value_residency,
            "generation_max_active_jobs": config.generation_max_active_jobs,
            "staging_flush_records": config.staging_flush_records,
            "legacy_script_path": (
                str(Path(config.legacy_script_path).expanduser().resolve())
                if config.legacy_script_path
                else None
            ),
            "vllm_cache_mode": config.vllm_cache_mode,
            "vllm_cache_root": (
                str(Path(config.vllm_cache_root).expanduser().resolve())
                if config.vllm_cache_root
                else None
            ),
            "vllm_enforce_eager": config.vllm_enforce_eager,
            "generation_worker_startup_timeout": (
                config.generation_worker_startup_timeout
            ),
        }

    def _run_cached_phase_stages(
        self,
        legacy: Any,
        args: SimpleNamespace,
        generation_index: Dict[str, Dict[int, Dict[str, Any]]],
        prepared_path: Path,
        request_list: Sequence[BestOfNRequest],
        skipped: Set[str],
        on_stage_start: Optional[Callable[[str, int], None]],
        on_rollout_complete: Optional[Callable[[str], None]],
        prepared_index: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
    ) -> None:
        """Complete the rollout entropy checkpoint before any phase partition."""
        if prepared_index is None:
            prepared_index = load_benchmark_prepared(prepared_path, request_list, self.config.n,
                                                      _indexed=True, max_tokens=self.config.max_tokens)
        missing_keys = [
            (request.request_id, sequence_idx)
            for request in request_list if request.request_id not in skipped
            for sequence_idx in range(self.config.n)
            if sequence_idx not in prepared_index.get(request.request_id, {})
        ]
        if not missing_keys:
            print("[resume] Prepared phase records complete; skip entropy/phase partition and continue scoring.")
            if on_stage_start is not None:
                on_stage_start("entropy", 0)
                on_stage_start("phase", 0)
            return
        cache_dir = prepared_path.parent
        # NaturalPlan keeps disposable phase/generation files in .staging.
        # Keep the expensive entropy checkpoint in its stable output directory.
        if cache_dir.name == ".staging":
            cache_dir = cache_dir.parent
        cache_path = cache_dir / ".bon_token_entropies.jsonl"

        def cache_key(record: Dict[str, Any]) -> Tuple[str, int]:
            return str(record["request_id"]), int(record["sequence_idx"])

        from .entropy_index import EntropyIndex, PhaseTasks
        cached = EntropyIndex(cache_path, cache_key)
        from .generation_prefix import GenerationRecords, SelectedGenerations
        for request_id, sequence_idx in missing_keys:
            by_index = generation_index.get(request_id, {})
            if sequence_idx not in by_index:
                raise RuntimeError(f"Missing generated rollout: {request_id}:c{sequence_idx}")
        records = GenerationRecords(generation_index, missing_keys, args, legacy,
                                    cache_dir / ".phase_structure_cache",
                                    getattr(prepared_index, "originals", None))
        pending = SelectedGenerations(records, [i for i in range(len(records))
            if not cached.matches(cache_key(records[i]), records[i], args)])
        if on_stage_start is not None:
            on_stage_start("entropy", len(pending))
        progress = (
            DatasetProgress(len(pending), description="Exact token entropy", unit="rollout")
            if on_rollout_complete is None else None
        )

        def persist_entropy(record: Dict[str, Any]) -> None:
            append_entropy_record(str(cache_path), record)
            if on_rollout_complete is not None:
                on_rollout_complete("entropy")
            if progress is not None:
                progress.update()

        try:
            if pending and self.config.parsed_gpu_groups:
                run_parallel_entropy(
                    args, pending, self.config.parsed_gpu_groups,
                    on_record=persist_entropy, cache_only=True, collect_results=False,
                )
            elif pending:
                model = (
                    legacy.load_entropy_model(args)
                    if any(record.get("status") == "ok" for record in pending) else None
                )
                try:
                    for record in pending:
                        if record.get("status") != "ok":
                            entropy = build_token_entropy_record(
                                record, None, args, record.get("error", "Generation failed")
                            )
                        else:
                            try:
                                values = legacy.compute_response_token_entropies_exact(
                                    model=model,
                                    prompt_token_ids=record["prompt_token_ids"],
                                    response_token_ids=record["response_token_ids"],
                                    chunk_size=args.entropy_chunk_size,
                                )
                                entropy = build_token_entropy_record(record, values, args)
                            except Exception:
                                entropy = build_token_entropy_record(record, None, args, traceback.format_exc())
                        persist_entropy(entropy)
                finally:
                    if model is not None:
                        del model
                    gc.collect()
                    if legacy.torch.cuda.is_available():
                        legacy.torch.cuda.empty_cache()
        finally:
            if progress is not None:
                progress.close()

        # Reopen the durable file after all workers have exited. Phase selection
        # never consumes transient GPU results and cannot trigger a forward.
        cached.refresh(appended=True)
        missing = [cache_key(record) for record in records if cache_key(record) not in cached]
        if missing:
            raise RuntimeError(f"Token entropy cache incomplete: {missing[:20]}")
        if on_stage_start is not None:
            on_stage_start("phase", len(records))
        progress = (
            DatasetProgress(len(records), description="Phase partition from cached entropy", unit="rollout")
            if on_rollout_complete is None else None
        )
        try:
            # Repair a phase-file tail interrupted by the previous process
            # before appending fresh records in the unchanged prepared schema.
            repair_prepared_tail(str(prepared_path))
            with _IncrementalStageWriter(prepared_path, self.config.staging_flush_records) as writer:
                def persist_phase(prepared: Dict[str, Any]) -> None:
                    offset = writer.write_record(prepared)
                    if isinstance(prepared_index, PreparedIndex):
                        prepared_index.remember(prepared, prepared_path, offset)
                    else:
                        prepared_index.setdefault(str(prepared["request_id"]), {})[int(prepared["sequence_idx"])] = prepared

                def complete_phase() -> None:
                    if on_rollout_complete is not None:
                        on_rollout_complete("phase")
                    if progress is not None:
                        progress.update()

                if getattr(args, "prm_step_scoring", False):
                    from .step_prm import run_step_preparation
                    run_step_preparation(args, PhaseTasks(records, cached, cache_key), legacy,
                                         persist_phase, complete_phase)
                elif self.config.parsed_gpu_groups:
                    run_parallel_phases(
                        args,
                        PhaseTasks(records, cached, cache_key),
                        self.config.parsed_gpu_groups,
                        legacy_script_path=self.config.legacy_script_path,
                        on_record=persist_phase,
                        on_complete=complete_phase,
                        cache_dir=str(cache_dir / ".phase_structure_cache"),
                        collect_results=False,
                        validated=True,
                    )
                else:
                    run_serial_phases(
                        args, PhaseTasks(records, cached, cache_key),
                        str(cache_dir / ".phase_structure_cache"), legacy,
                        persist_phase, complete_phase,
                    )
            if isinstance(prepared_index, PreparedIndex) and any(
                location.path == prepared_path for location in getattr(prepared_index, "originals", {}).values()
            ):
                from .generation_prefix import compact_stage
                compact_stage(prepared_path, prepared_index)
        finally:
            if progress is not None:
                progress.close()

    def _initialize_batch_staging(
        self,
        staging_dir: str | Path,
        *,
        resume: bool,
        overwrite: bool,
    ) -> Tuple[Path, Path]:
        directory = Path(staging_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stale_manifest_path = directory / ".bon_batch_manifest.json"
        generation_path = directory / ".bon_generation.jsonl"
        prepared_path = directory / ".bon_prepared.jsonl"
        # Generation and entropy stages are independently resumable.  Do not
        # persist or validate a run-wide manifest: users may resume a staged
        # directory after changing scoring/runtime parameters, while the
        # candidate records themselves remain the source of truth.
        stage_paths = (generation_path, prepared_path)

        if resume and overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")
        if overwrite:
            overwrite_paths = list(stage_paths)
            if self._prepare_with_entropy_model is BestOfNService._prepare_with_entropy_model:
                overwrite_paths.extend(prepared_paths(prepared_path))
            for path in dict.fromkeys(overwrite_paths):
                if path.exists():
                    path.unlink()
            # Remove a manifest left by an older version when the caller has
            # explicitly requested a destructive overwrite. New runs never
            # create this file.
            if stale_manifest_path.exists():
                stale_manifest_path.unlink()
        elif not resume and any(path.exists() for path in stage_paths) and not (
            self._prepare_with_entropy_model is BestOfNService._prepare_with_entropy_model
            and any(path.exists() for path in prepared_paths(prepared_path))
        ):
            raise FileExistsError(
                f"BoN staging data already exists in {directory}; use resume or overwrite"
            )

        return generation_path, prepared_path

    def _score_prepared_entries(
        self, legacy, entries, request_by_id, args, on_rollout_complete=None,
    ):
        """Score in input order; PRM subclasses may overlap bounded lookahead."""
        for entry in entries:
            request = request_by_id[str(entry["request_id"])]
            score_kwargs: Dict[str, Any] = {}
            if on_rollout_complete is not None:
                try:
                    parameters = inspect.signature(self._score).parameters
                    if "on_rollout" in parameters:
                        score_kwargs["on_rollout"] = lambda: on_rollout_complete("scoring")
                except (TypeError, ValueError):
                    score_kwargs["on_rollout"] = lambda: on_rollout_complete("scoring")
            yield request, self._score(legacy, request, entry["records"], args, **score_kwargs)

    def generate_and_select_many(
        self,
        requests: Sequence[BestOfNRequest],
        *,
        staging_dir: str | Path,
        resume: bool = False,
        overwrite: bool = False,
        skip_request_ids: Optional[Set[str]] = None,
        on_stage_start: Optional[Callable[[str, int], None]] = None,
        on_rollout_complete: Optional[Callable[[str], None]] = None,
    ) -> Iterator[BestOfNResult]:
        """Run a staged offline batch while loading each large model once.

        Generation responses and entropy preparation are persisted as workers
        return them, with durability controlled by ``staging_flush_records``.
        Scored requests are then yielded one at a time so benchmark result stores
        can durably write summary/steps records. The iterator omits requests in
        ``skip_request_ids``, allowing the final store to resume independently.

        ``on_stage_start`` receives ``(stage, total_rollouts)`` for the stages
        ``generation``, ``entropy``, ``phase`` and ``scoring``. ``on_rollout_complete`` is
        called once for each rollout completed in one of those stages.
        """

        request_list = list(requests)
        request_by_id = {request.request_id: request for request in request_list}
        if len(request_by_id) != len(request_list):
            raise ValueError("Best-of-N batch request_id values must be unique")
        skipped = set(skip_request_ids or set())
        unknown_skips = skipped.difference(request_by_id)
        if unknown_skips:
            raise ValueError(
                "Unknown skipped request ids: " + ", ".join(sorted(unknown_skips))
            )

        def run() -> Iterator[BestOfNResult]:
            with self._lock:
                legacy = load_legacy_module(self.config.legacy_script_path)
                args = self.config.legacy_args()
                generation_path, prepared_path = self._initialize_batch_staging(
                    staging_dir,
                    resume=resume,
                    overwrite=overwrite,
                )
                generated_by_request = {}
                standard_preparation = self._prepare_with_entropy_model is BestOfNService._prepare_with_entropy_model
                prepared_records_index = (
                    load_benchmark_prepared(prepared_path, request_list, self.config.n,
                                            _indexed=True, max_tokens=self.config.max_tokens)
                    if standard_preparation and not overwrite else PreparedIndex() if standard_preparation else {}
                )
                fully_prepared = all(
                    sequence_idx in prepared_records_index.get(request.request_id, {})
                    for request in request_list if request.request_id not in skipped
                    for sequence_idx in range(self.config.n)
                )
                generation_records_index = {} if standard_preparation and fully_prepared else _stage_records_index(generation_path)
                completed_generation_indices = {
                    request_id: {
                        sequence_idx
                        for sequence_idx in records_by_index
                        if 0 <= sequence_idx < self.config.n
                    }
                    for request_id, records_by_index in generation_records_index.items()
                }
                if standard_preparation:
                    for request_id, slots in prepared_records_index.items():
                        completed_generation_indices.setdefault(request_id, set()).update(slots)
                missing_generation_by_request = {
                    request.request_id: [
                        sequence_idx
                        for sequence_idx in range(self.config.n)
                        if sequence_idx
                        not in completed_generation_indices.get(
                            request.request_id, set()
                        )
                    ]
                    for request in request_list
                    if request.request_id not in skipped
                }
                pending_generation = [
                    request
                    for request in request_list
                    if missing_generation_by_request.get(request.request_id)
                ]
                if standard_preparation:
                    for request in pending_generation:
                        if any((request.request_id, sequence) in getattr(prepared_records_index, "originals", {})
                               for sequence in missing_generation_by_request[request.request_id]):
                            raise ValueError("Truncating saved steps requires the original .bon_generation.jsonl; refusing regeneration")
                if on_stage_start is not None:
                    on_stage_start(
                        "generation",
                        sum(
                            len(indices)
                            for indices in missing_generation_by_request.values()
                        ),
                    )
                if pending_generation:
                    gpu_groups = self.config.parsed_gpu_groups
                    jobs = build_missing_generation_jobs(
                        pending_generation,
                        self.config.n,
                        self.config.candidate_batch_size,
                        completed_generation_indices,
                    )
                    with _IncrementalStageWriter(
                        generation_path,
                        self.config.staging_flush_records,
                    ) as stage_writer:
                        def persist_generation_records(
                            records: Sequence[Dict[str, Any]],
                        ) -> None:
                            for record in records:
                                stage_writer.write_record(record)
                                if on_rollout_complete is not None:
                                    on_rollout_complete("generation")

                        if gpu_groups:
                            generated_by_request = run_parallel_generation(
                                self.config,
                                jobs,
                                gpu_groups,
                                on_records=persist_generation_records,
                            )
                        else:
                            generated_by_request: Dict[
                                str, List[Dict[str, Any]]
                            ] = {}
                            llm = self._create_generation_llm(legacy, args)
                            try:
                                tokenizer = llm.get_tokenizer()
                                prompt_cache: Dict[str, str] = {}
                                supports_incremental = "on_records" in (
                                    inspect.signature(
                                        self._generate_with_llm
                                    ).parameters
                                )
                                for request in pending_generation:
                                    missing_indices = missing_generation_by_request[
                                        request.request_id
                                    ]
                                    if len(missing_indices) == self.config.n:
                                        if supports_incremental:
                                            records = self._generate_with_llm(
                                                legacy,
                                                llm,
                                                tokenizer,
                                                request,
                                                args,
                                                on_records=persist_generation_records,
                                            )
                                        else:
                                            records = self._generate_with_llm(
                                                legacy,
                                                llm,
                                                tokenizer,
                                                request,
                                                args,
                                            )
                                            for record in records:
                                                record.setdefault(
                                                    "request_id", request.request_id
                                                )
                                            persist_generation_records(records)
                                    else:
                                        records = []
                                        request_jobs = [
                                            job
                                            for job in jobs
                                            if job.request.request_id
                                            == request.request_id
                                        ]
                                        for job in request_jobs:
                                            job_records = self._generate_jobs_with_llm(
                                                legacy,
                                                llm,
                                                tokenizer,
                                                [job],
                                                args,
                                                prompt_cache,
                                            )
                                            persist_generation_records(job_records)
                                            records.extend(job_records)
                                    generated_by_request[request.request_id] = records
                            finally:
                                legacy.release_vllm(llm)
                    for request in pending_generation:
                        records = generated_by_request.get(request.request_id, [])
                        expected_indices = set(
                            missing_generation_by_request[request.request_id]
                        )
                        returned_indices = {
                            int(record["sequence_idx"]) for record in records
                        }
                        if returned_indices != expected_indices:
                            raise RuntimeError(
                                "Generation did not return exactly the missing "
                                f"candidates for {request.request_id}; "
                                f"expected={sorted(expected_indices)}, "
                                f"returned={sorted(returned_indices)}"
                            )
                if pending_generation:
                    generation_records_index = _stage_records_index(generation_path)

                # Preserve custom preparation hooks (e.g. the separate multi-signal
                # service). The standard Transformer and all PRMs share the new
                # independent entropy checkpoint followed by cached phase partition.
                if standard_preparation:
                    self._run_cached_phase_stages(
                        legacy, args, generation_records_index, prepared_path,
                        request_list, skipped, on_stage_start, on_rollout_complete,
                        prepared_index=prepared_records_index,
                    )
                    # Neither generation payloads nor their duplicate loaded
                    # staging records are used by scoring. Prepared slots hold
                    # locations, so completed preparation does not retain them.
                    generation_records_index.clear()
                    generated_by_request.clear()
                else:
                    prepared_records_index = _stage_records_index(prepared_path)
                    completed_preparation_indices = {
                        request_id: {
                            sequence_idx
                            for sequence_idx in records_by_index
                            if 0 <= sequence_idx < self.config.n
                        }
                        for request_id, records_by_index in prepared_records_index.items()
                    }
                    missing_preparation_by_request = {
                        request.request_id: [
                            sequence_idx
                            for sequence_idx in range(self.config.n)
                            if sequence_idx
                            not in completed_preparation_indices.get(
                                request.request_id, set()
                            )
                        ]
                        for request in request_list
                        if request.request_id not in skipped
                    }
                    pending_preparation = {
                        request_id
                        for request_id, missing_indices in missing_preparation_by_request.items()
                        if missing_indices
                    }
                    if on_stage_start is not None:
                        on_stage_start(
                            "entropy",
                            sum(
                                len(indices)
                                for indices in missing_preparation_by_request.values()
                            ),
                        )
                    if pending_preparation:
                        generation_records_for_preparation: Dict[
                            str, List[Dict[str, Any]]
                        ] = {}
                        for request_id in pending_preparation:
                            records_by_index = generation_records_index.get(
                                request_id, {}
                            )
                            missing_generation_records = [
                                sequence_idx
                                for sequence_idx in missing_preparation_by_request[
                                    request_id
                                ]
                                if sequence_idx not in records_by_index
                            ]
                            if missing_generation_records:
                                raise RuntimeError(
                                    "Missing generated candidates required for entropy "
                                    f"for {request_id}: {missing_generation_records}"
                                )
                            generation_records_for_preparation[request_id] = [
                                records_by_index[sequence_idx]
                                for sequence_idx in missing_preparation_by_request[
                                    request_id
                                ]
                            ]
                        gpu_groups = self.config.parsed_gpu_groups
                        prepared_by_request: Dict[str, List[Dict[str, Any]]] = {}
                        with _IncrementalStageWriter(
                            prepared_path,
                            self.config.staging_flush_records,
                        ) as stage_writer:
                            def persist_prepared_record(record: Dict[str, Any]) -> None:
                                stage_writer.write_record(record)
                                if on_rollout_complete is not None:
                                    on_rollout_complete("entropy")
    
                            if gpu_groups:
                                all_records = [
                                    record
                                    for request_id in pending_preparation
                                    for record in generation_records_for_preparation.get(
                                        request_id, []
                                    )
                                ]
                                prepared_all = run_parallel_entropy(
                                    args,
                                    all_records,
                                    gpu_groups,
                                    on_record=persist_prepared_record,
                                )
                                for record in prepared_all:
                                    prepared_by_request.setdefault(
                                        str(record.get("request_id")), []
                                    ).append(record)
                            else:
                                tokenizer = legacy.load_tokenizer_without_vllm(args)
                                needs_entropy_model = any(
                                    record.get("status") == "ok"
                                    for records in generation_records_for_preparation.values()
                                    for record in records
                                )
                                entropy_model = (
                                    legacy.load_entropy_model(args)
                                    if needs_entropy_model
                                    else None
                                )
                                try:
                                    supports_incremental = "on_record" in (
                                        inspect.signature(
                                            self._prepare_with_entropy_model
                                        ).parameters
                                    )
                                    for request_id in pending_preparation:
                                        generation_records = (
                                            generation_records_for_preparation.get(
                                                request_id
                                            )
                                        )
                                        if generation_records is None:
                                            raise RuntimeError(
                                                "Missing generated batch request: "
                                                f"{request_id}"
                                            )
                                        if supports_incremental:
                                            prepared_records = self._prepare_with_entropy_model(
                                                legacy,
                                                generation_records,
                                                args,
                                                tokenizer,
                                                entropy_model,
                                                on_record=persist_prepared_record,
                                            )
                                        else:
                                            prepared_records = self._prepare_with_entropy_model(
                                                legacy,
                                                generation_records,
                                                args,
                                                tokenizer,
                                                entropy_model,
                                            )
                                            for prepared_record in prepared_records:
                                                persist_prepared_record(prepared_record)
                                        prepared_by_request[request_id] = (
                                            prepared_records
                                        )
                                finally:
                                    if entropy_model is not None:
                                        del entropy_model
                                    gc.collect()
                                    if legacy.torch.cuda.is_available():
                                        legacy.torch.cuda.empty_cache()
                        for request_id in pending_preparation:
                            prepared_records = prepared_by_request.get(request_id, [])
                            expected_indices = set(
                                missing_preparation_by_request[request_id]
                            )
                            returned_indices = {
                                int(record["sequence_idx"])
                                for record in prepared_records
                            }
                            if returned_indices != expected_indices:
                                raise RuntimeError(
                                    "Entropy did not return exactly the missing "
                                    f"candidates for {request_id}; "
                                    f"expected={sorted(expected_indices)}, "
                                    f"returned={sorted(returned_indices)}"
                                )

                yielded: Set[str] = set()
                if on_stage_start is not None:
                    scoring_total = (len(request_list) - len(skipped)) * self.config.n
                    scoring_total_hook = getattr(self, "scoring_rollout_total", None)
                    if scoring_total_hook is not None:
                        try:
                            scoring_total = int(scoring_total_hook(request_list, skipped))
                        except TypeError:
                            scoring_total = int(scoring_total_hook(request_list, skipped, staging_dir))
                    on_stage_start(
                        "scoring",
                        scoring_total,
                    )
                def scoring_entries():
                    prepared_entries = (
                        ({"request_id": request_id, "records": slots.read_all(range(self.config.n))}
                         for request_id, slots in prepared_records_index.items()
                         if request_id not in skipped)
                        if standard_preparation else _iter_stage_request_entries(prepared_path)
                    )
                    return (
                        entry for entry in prepared_entries
                        if str(entry["request_id"]) not in skipped
                        and str(entry["request_id"]) in request_by_id
                    )
                preflight = getattr(self, "_preflight_prm_inputs", None)
                if preflight is not None:
                    preflight(legacy, scoring_entries(), request_by_id, args)
                scored = self._score_prepared_entries(
                    legacy, scoring_entries(), request_by_id, args, on_rollout_complete,
                )
                try:
                    for request, candidates in scored:
                        request_id = request.request_id
                        yielded.add(request_id)
                        yield self._result_from_candidates(
                            request,
                            candidates,
                            require_valid_selection=False,
                        )
                finally:
                    scored.close()
                missing_results = set(request_by_id).difference(skipped, yielded)
                if missing_results:
                    raise RuntimeError(
                        "Missing prepared batch requests: "
                        + ", ".join(sorted(missing_results))
                    )
                self._park_value_model_if_needed()

        return run()

    def generate_and_select(self, request: BestOfNRequest) -> BestOfNResult:
        with self._lock:
            legacy = load_legacy_module(self.config.legacy_script_path)
            args = self.config.legacy_args()
            generation_records = self._generate(legacy, request, args)
            prepared_records = self._prepare(legacy, generation_records, args)
            preflight = getattr(self, "_preflight_prm_inputs", None)
            if preflight is not None:
                preflight(legacy, iter([{"request_id": request.request_id, "records": prepared_records}]),
                          {request.request_id: request}, args)
            candidates = self._score(legacy, request, prepared_records, args)
            result = self._result_from_candidates(
                request,
                candidates,
                require_valid_selection=True,
            )
            self._park_value_model_if_needed()
            return result

    def close(self) -> None:
        with self._lock:
            if self._value_components is not None:
                model, _, _ = self._value_components
                del model
                self._value_components = None
                gc.collect()
                try:
                    legacy = load_legacy_module(self.config.legacy_script_path)
                    if legacy.torch.cuda.is_available():
                        legacy.torch.cuda.empty_cache()
                except Exception:
                    pass

    def __enter__(self) -> "BestOfNService":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
