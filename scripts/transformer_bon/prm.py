from __future__ import annotations

import copy
import gc
import math
import multiprocessing as mp
import os
import queue
import threading
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch

try:
    import torch as torch_runtime
except ModuleNotFoundError:  # Keep non-PRM users of transformer_bon lightweight.
    torch_runtime = None  # type: ignore[assignment]

from .engine import (
    BestOfNConfig,
    BestOfNService,
    _link_fields,
    _sample_index,
)
from .models import BestOfNCandidate, BestOfNRequest
from .parallel import lpt_shards, parse_gpu_groups
from .prms.registry import DEFAULT_PRM_NAME, get_prm_backend, normalize_prm_name
from .prm_checkpoint import PRMCheckpoint
from .prm_prefetch import prefetch_map, capture_inference_context
from .ordered_pipeline import OrderedAppender, background_iterator, check_stopped
from .prm_recovery import (OOMScore, execution_device, is_cuda_oom, model_load_kwargs,
                           parameter_storage, recover_jobs, release_cuda_cache)
from .errors import require_success


def _reap_processes(
    processes: Sequence[Any],
    *,
    graceful_timeout: float = 2.0,
    terminate_timeout: float = 5.0,
) -> None:
    """Bound worker cleanup so a PRM failure cannot leave the parent hung."""

    for process in processes:
        try:
            process.join(timeout=graceful_timeout)
        except (AssertionError, OSError):
            continue
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (AssertionError, OSError):
            continue
    for process in processes:
        try:
            process.join(timeout=terminate_timeout)
        except (AssertionError, OSError):
            continue
    # ``kill`` is available on supported Python versions and handles a worker
    # stuck in CUDA/IPC cleanup after terminate().  Keep a bounded fallback for
    # older multiprocessing implementations.
    for process in processes:
        try:
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
        except (AssertionError, OSError):
            continue
    for process in processes:
        try:
            if process.is_alive():
                process.join(timeout=1.0)
        except (AssertionError, OSError):
            continue


def _close_queue(queue_: Any) -> None:
    """Close a multiprocessing queue without waiting on a broken feeder."""

    try:
        queue_.cancel_join_thread()
    except (AttributeError, OSError):
        pass
    try:
        queue_.close()
    except (AttributeError, OSError):
        pass


def split_steps(full_output: Any, separator: str = "\n\n") -> List[str]:
    """Split a generated response exactly like the standalone PRM evaluator."""

    text = "" if full_output is None else str(full_output)
    return [part.strip() for part in text.split(separator) if part.strip()]


def phase_prefixes(
    record: Dict[str, Any],
    separator: str = "\n\n",
    build_prefixes: bool = True,
) -> Tuple[List[str], List[List[int]]]:
    """Return phase boundaries and optionally materialize cumulative prefixes."""

    parts = split_steps(record.get("full_output", ""), separator)
    ranges = record.get("phase_ranges") or []
    if not parts or not ranges:
        raise ValueError("record has no full_output steps or phase_ranges")
    prefixes: List[str] = []
    normalized: List[List[int]] = []
    for phase_range in ranges:
        if not isinstance(phase_range, (list, tuple)) or len(phase_range) != 2:
            raise ValueError(f"Invalid phase range: {phase_range!r}")
        start, end = int(phase_range[0]), int(phase_range[1])
        if not (1 <= start <= end <= len(parts)):
            raise ValueError(
                f"Phase range {phase_range} outside 1..{len(parts)}"
            )
        if build_prefixes:
            prefixes.append(separator.join(parts[:end]))
        normalized.append([start, end])
    return prefixes, normalized


def build_prm_text(
    problem: Any,
    prefix: str,
    marker: str = "<extra_0>",
    separator: str = "\n\n",
    messages: Optional[Sequence[Dict[str, str]]] = None,
    raw_prompt: Optional[str] = None,
    prm_name: Optional[str] = DEFAULT_PRM_NAME,
    phase_ends: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build the selected backend's structured input.

    The default remains the legacy Qwen2.5-Math-PRM format. Causal process
    backends such as FoVer and VersaPRM receive their own structured
    ``solution_steps`` representation, so each backend owns its format logic.
    """

    return build_prm_input(
        prm_name,
        problem,
        prefix,
        marker,
        separator,
        messages,
        raw_prompt,
        phase_ends,
    )


def build_prm_input(
    prm_name: Optional[str],
    problem: Any,
    prefix: str,
    marker: str = "<extra_0>",
    separator: str = "\n\n",
    messages: Optional[Sequence[Dict[str, str]]] = None,
    raw_prompt: Optional[str] = None,
    phase_ends: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    from .prms.registry import get_prm_backend

    backend = get_prm_backend(prm_name)
    return backend.build_input(
        problem=problem,
        prefix=prefix,
        marker=marker,
        separator=separator,
        messages=messages,
        raw_prompt=raw_prompt,
        phase_ends=phase_ends,
    )


def _prm_input_length(value: Any) -> int:
    """Approximate serialized input size for load-balancing PRM jobs."""

    if isinstance(value, dict):
        messages = value.get("messages") or []
        message_size = sum(
            len(str(message.get("content", "")))
            for message in messages
            if isinstance(message, dict)
        )
        return max(
            1,
            len(str(value.get("problem", "")))
            + len(str(value.get("assistant", "")))
            + len(str(value.get("raw_prompt", "") or ""))
            + message_size,
        )
    return max(1, len(str(value)))


def _backend_input_length(prm_name: Optional[str], value: Any) -> int:
    from .prms.registry import get_prm_backend

    return get_prm_backend(prm_name).input_length(value)


def _score_prm_batch_resilient(
    scorer: "HFPRMScorer", texts: Sequence[Any], prepared: Any = None,
    *, defer_oom: bool = False,
) -> List[Any]:
    """Retry a CUDA-OOM batch as progressively smaller micro-batches."""

    pending: List[List[Any]] = [list(texts)]
    values: List[Any] = []
    while pending:
        current = pending.pop(0)
        failed = None
        try:
            if prepared is not None:
                batch_prepared, prepared = prepared, None
                try:
                    result = scorer.score_prepared_batch(batch_prepared)
                finally:
                    del batch_prepared
            else:
                result = scorer.score_batch(current)
            values.extend(result)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message:
                raise
            if len(current) <= 1:
                if not defer_oom or not is_cuda_oom(exc):
                    raise
                failed = OOMScore(str(exc))
        else:
            continue
        # The traceback must release failed CUDA tensors before cache cleanup.
        release_cuda_cache(torch_runtime)
        if failed is not None:
            values.append(failed)
        else:
            midpoint = max(1, len(current) // 2)
            pending.insert(0, current[midpoint:])
            pending.insert(0, current[:midpoint])
    return values


def _expand_phase_scores(
    sequence_idx: int, phase_idx: int, value: Any
) -> List[Tuple[int, int, float]]:
    """Map one scorer result back to phase indices.

    New backends return all requested phase-end scores from one complete
    solution.  The scalar branch keeps compatibility with older/custom
    backends that still return one score per job.
    """

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [
            (int(sequence_idx), int(index), float(score))
            for index, score in enumerate(value)
        ]
    if int(phase_idx) < 0:
        raise RuntimeError(
            "PRM backend returned a scalar for a full-solution phase job"
        )
    return [(int(sequence_idx), int(phase_idx), float(value))]


def _expected_phase_keys(
    jobs: Sequence[Tuple[int, int, Any]],
) -> set[Tuple[int, int]]:
    expected: set[Tuple[int, int]] = set()
    for sequence_idx, phase_idx, value in jobs:
        if int(phase_idx) >= 0:
            expected.add((int(sequence_idx), int(phase_idx)))
            continue
        phase_ends = value.get("phase_ends") if isinstance(value, dict) else None
        if not phase_ends:
            raise RuntimeError(
                "Full-solution PRM job is missing phase_ends metadata"
            )
        expected.update(
            (int(sequence_idx), index) for index, _ in enumerate(phase_ends)
        )
    return expected


def aggregate_phase_scores(scores: Sequence[float]) -> Dict[str, float]:
    if not scores:
        raise ValueError("No phase scores")
    values = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite phase score")
    clipped = np.clip(values, 1e-12, 1.0)
    return {
        "product_score": float(np.exp(np.log(clipped).sum())),
        "min_score": float(values.min()),
        "mean_score": float(values.mean()),
        "last_score": float(values[-1]),
    }


class HFPRMScorer:
    """Adapter for common HF sequence/token classification and causal PRMs.

    Model loading and score extraction intentionally mirror
    ``4_best_of_n_via_phases_prm_rewards_on_generated_summary.py`` so scores
    produced by the integrated GPQA pipeline remain comparable to that script.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        max_length: int = 4096,
        positive_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if torch_runtime is None:
            raise ModuleNotFoundError(
                "Hugging Face PRM scoring requires PyTorch to be installed"
            )
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        self.max_length = int(max_length)
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else "Please reason step by step, and put your final answer within \\boxed{}."
        )
        # Load the config first so a Qwen PRM can opt into its custom remote
        # model class even when an older benchmark runner did not expose a
        # --prm_trust_remote_code flag.
        try:
            self.config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
        except Exception:
            if trust_remote_code:
                raise
            self.config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True
            )
            trust_remote_code = True
        architecture_names = [
            str(name)
            for name in (getattr(self.config, "architectures", []) or [])
        ]
        architectures = " ".join(architecture_names).lower()
        is_qwen_process_reward = any(
            name.lower() == "qwen2forprocessrewardmodel"
            for name in architecture_names
        )
        if is_qwen_process_reward:
            trust_remote_code = True
        configured_limit = getattr(self.config, "max_position_embeddings", None)
        if configured_limit and self.max_length > int(configured_limit):
            print(
                f"[WARN] PRM max_length={self.max_length} exceeds the checkpoint's "
                f"max_position_embeddings={configured_limit}; long inputs may be unsupported."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )
        # With left padding the final non-pad token is at the final absolute
        # sequence position.  The scorer never uses attention_mask.sum()-1 for
        # left-padded batches.
        self.tokenizer.padding_side = "left"
        # Preserve the final <extra_0> marker if a long prompt must be cut.
        self.tokenizer.truncation_side = "left"
        # Qwen2.5-Math-PRM's custom Qwen2RMConfig may not define
        # pad_token_id, but its model constructor accesses that attribute.
        # Populate it from the tokenizer and reuse this patched config during
        # model loading.
        if getattr(self.config, "pad_token_id", None) is None:
            self.config.pad_token_id = self.tokenizer.pad_token_id
        classes = []
        if "tokenclassification" in architectures:
            classes.append(AutoModelForTokenClassification)
        if "sequenceclassification" in architectures:
            classes.append(AutoModelForSequenceClassification)
        if "causallm" in architectures or "forcausallm" in architectures:
            classes.append(AutoModelForCausalLM)
        classes += [
            AutoModelForTokenClassification,
            AutoModelForSequenceClassification,
            AutoModelForCausalLM,
            AutoModel,
        ]

        self.model = None
        errors: List[str] = []
        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch_runtime, dtype)
        if device == "auto" and torch_runtime.cuda.is_available():
            kwargs["device_map"] = "auto"
        kwargs = model_load_kwargs(kwargs)

        if is_qwen_process_reward:
            try:
                self.model = AutoModel.from_pretrained(
                    model_path, config=self.config, **kwargs
                )
                self.model_class = self.model.__class__.__name__
            except Exception as exc:
                errors.append(f"Qwen2ForProcessRewardModel: {exc}")
        else:
            seen = set()
            for cls in classes:
                if cls in seen:
                    continue
                seen.add(cls)
                try:
                    self.model = cls.from_pretrained(model_path, **kwargs)
                    self.model_class = cls.__name__
                    break
                except Exception as exc:
                    errors.append(f"{cls.__name__}: {exc}")
        if self.model is None:
            raise RuntimeError(
                "Could not load PRM with supported HF classes:\n"
                + "\n".join(errors)
            )
        self.model.eval()
        self.device = self._resolve_device(device)
        if "device_map" not in kwargs and self.device.type != "cpu":
            self.model.to(self.device)
        self.kind = self._infer_kind()
        self.step_separator_id: Optional[int] = None
        if self.kind == "process_reward":
            separator_ids = self.tokenizer.encode(
                "<extra_0>", add_special_tokens=False
            )
            if len(separator_ids) != 1:
                raise RuntimeError(
                    "Qwen process PRM requires <extra_0> to encode as exactly "
                    f"one token; got token ids {separator_ids!r}."
                )
            self.step_separator_id = int(separator_ids[0])
        self._verify_reward_head()
        self.positive_id = self._resolve_positive_id(positive_token)
        print(
            f"[INFO] PRM model: {model_path} "
            f"({self.model_class}, kind={self.kind}, device={self.device})"
        )

    def _resolve_device(self, requested: str) -> torch.device:
        if requested != "auto":
            return torch_runtime.device(requested)
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for value in device_map.values():
                if isinstance(value, str) and value not in {"cpu", "disk"}:
                    return torch_runtime.device(value)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch_runtime.device(
                "cuda" if torch_runtime.cuda.is_available() else "cpu"
            )

    def _infer_kind(self) -> str:
        name = self.model_class.lower()
        if "qwen2forprocessrewardmodel" in name:
            return "process_reward"
        if "tokenclassification" in name:
            return "token"
        if "sequenceclassification" in name:
            return "sequence"
        if "causallm" in name:
            return "causal"
        return "base"

    def _verify_reward_head(self) -> None:
        if self.kind not in {"token", "sequence", "process_reward"}:
            return
        head = getattr(self.model, "score", None)
        if head is None:
            head = getattr(self.model, "classifier", None)
        weight = getattr(head, "weight", None) if head is not None else None
        if weight is None and isinstance(head, torch_runtime.nn.Sequential):
            linear_layers = [
                layer
                for layer in head
                if getattr(layer, "weight", None) is not None
            ]
            if linear_layers:
                weight = linear_layers[-1].weight
        if weight is None:
            raise RuntimeError(
                f"PRM reward head is missing on {self.model_class}; "
                "the checkpoint is incomplete."
            )
        if weight.is_meta:
            name = next(name for name, value in head.named_parameters() if value is weight)
            weight = parameter_storage(head, name)
        weight = weight.detach()
        if not torch_runtime.isfinite(weight).all():
            raise RuntimeError(
                "PRM reward head contains non-finite values; "
                "the checkpoint is corrupted."
            )
        std = float(weight.float().std())
        if std <= 1e-6:
            raise RuntimeError(
                "PRM reward head looks uninitialized (near-zero variance). "
                "Re-download the full checkpoint including the "
                "score.weight/score.bias tensors."
            )

    def _resolve_positive_id(self, token: Optional[str]) -> Optional[int]:
        if self.kind not in {"causal", "base"}:
            return None
        if token:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    "--prm_positive_token must encode to exactly one token"
                )
            return int(ids[0])
        for candidate in ("ки", "Ġки", "✓", "correct", " positive"):
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        raise ValueError(
            "Causal PRM requires --prm_positive_token "
            "(for example 'ки' for Math-Shepherd-style PRMs)"
        )

    @staticmethod
    def _binary_probability(
        logits: torch.Tensor, positive_index: int = 1
    ) -> torch.Tensor:
        if logits.shape[-1] == 1:
            return torch_runtime.sigmoid(logits[..., 0])
        return torch_runtime.softmax(logits, dim=-1)[..., positive_index]

    def _render_input(self, value: Any) -> str:
        if not isinstance(value, dict):
            return str(value)
        problem = "" if value.get("problem") is None else str(value.get("problem"))
        assistant = "" if value.get("assistant") is None else str(value.get("assistant"))
        messages = value.get("messages")
        if self.kind != "process_reward":
            return problem + "\n\n" + assistant
        raw_prompt = value.get("raw_prompt")
        if raw_prompt is not None and not messages:
            return str(raw_prompt) + "\n\n" + assistant
        if isinstance(messages, list) and messages:
            rendered_messages = [dict(message) for message in messages]
            if self.system_prompt and not any(
                message.get("role") == "system" for message in rendered_messages
            ):
                rendered_messages.insert(
                    0, {"role": "system", "content": self.system_prompt}
                )
        else:
            rendered_messages = []
            if self.system_prompt:
                rendered_messages.append(
                    {"role": "system", "content": self.system_prompt}
                )
            rendered_messages.append({"role": "user", "content": problem})
        rendered_messages.append({"role": "assistant", "content": assistant})
        try:
            return self.tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "The Qwen PRM tokenizer does not provide a usable chat template."
            ) from exc

    def _last_nonpad_positions(self, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.tokenizer.padding_side == "left":
            return torch_runtime.full(
                (attention_mask.shape[0],),
                attention_mask.shape[1] - 1,
                dtype=torch_runtime.long,
                device=attention_mask.device,
            )
        return attention_mask.sum(dim=1) - 1

    def score_batch(self, texts: Sequence[Any]) -> List[Any]:
        if torch_runtime is None:
            raise ModuleNotFoundError(
                "Hugging Face PRM scoring requires PyTorch to be installed"
            )
        with torch_runtime.inference_mode():
            return self._score_batch(texts)

    def _score_process_reward_markers(
        self,
        encoded: Dict[str, torch.Tensor],
        marker_mask: torch.Tensor,
        phase_ends_batch: Sequence[Optional[List[int]]],
        marker_positions_batch=None,
    ) -> List[Any]:
        """Score Qwen process markers without materializing sequence logits."""

        backbone = getattr(self.model, "model", None)
        head = getattr(self.model, "score", None)
        if head is None:
            head = getattr(self.model, "classifier", None)
        if backbone is None or head is None:
            raise RuntimeError(
                "Qwen process PRM must expose a transformer body and reward head "
                "to compute marker-only logits"
            )
        try:
            body_output = backbone(
                **encoded,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        except TypeError:
            try:
                body_output = backbone(**encoded, use_cache=False)
            except TypeError:
                body_output = backbone(**encoded)
        hidden_states = getattr(body_output, "last_hidden_state", None)
        if hidden_states is None and isinstance(body_output, (tuple, list)):
            hidden_states = body_output[0]
        if hidden_states is None or hidden_states.ndim != 3:
            raise RuntimeError(
                "Qwen process PRM transformer body returned no "
                "[batch, seq, hidden] states"
            )

        selected_rows: List[torch.Tensor] = []
        row_sizes: List[int] = []
        for batch_index in range(hidden_states.shape[0]):
            marker_positions = (marker_positions_batch[batch_index]
                                if marker_positions_batch is not None else
                                marker_mask[batch_index].nonzero(as_tuple=False).flatten())
            row_hidden = hidden_states[batch_index].index_select(
                0, marker_positions.to(hidden_states.device)
            )
            selected_rows.append(row_hidden)
            row_sizes.append(int(row_hidden.shape[0]))
        # ``torch`` is imported only under TYPE_CHECKING above; the runtime
        # module is intentionally named ``torch_runtime`` so non-PRM users can
        # import this package without requiring PyTorch.  This branch is used
        # by Qwen process-reward models and must therefore use the runtime
        # alias rather than the type-only name.
        selected_hidden = torch_runtime.cat(selected_rows, dim=0)
        del body_output, hidden_states, selected_rows

        # The process-reward head has only the reward classes (normally two),
        # so applying it after marker selection avoids a sequence-wide output.
        head_device = execution_device(head)
        selected_hidden = selected_hidden.to(head_device)
        selected_logits = head(selected_hidden)
        if selected_logits.ndim != 2:
            raise RuntimeError(
                "Qwen process PRM reward head did not return [markers, classes] logits"
            )

        all_row_scores = []
        offset = 0
        for batch_index, size in enumerate(row_sizes):
            row_logits = selected_logits[offset : offset + size]
            offset += size
            row_probs = self._binary_probability(
                row_logits,
                self._positive_class_index(row_logits.shape[-1]),
            )
            requested = phase_ends_batch[batch_index]
            requested_indices = (
                [int(row_probs.shape[0]) - 1]
                if requested is None
                else [int(end) - 1 for end in requested]
            )
            if any(
                index < 0 or index >= int(row_probs.shape[0])
                for index in requested_indices
            ):
                raise RuntimeError(
                    "Qwen PRM phase end exceeds the number of visible step "
                    f"markers in row {batch_index}; increase prm_max_length or "
                    "inspect truncation."
                )
            # Selection/copy only: retain the old per-row probability operation
            # while transferring all requested phase values in one operation.
            all_row_scores.append(row_probs[requested_indices].detach().float())
        del selected_hidden, selected_logits
        sizes = [row.numel() for row in all_row_scores]
        flat = torch_runtime.cat(all_row_scores).cpu().tolist()
        rows, offset = [], 0
        for size in sizes:
            rows.append(flat[offset:offset + size])
            offset += size
        return [
            row_scores if phase_ends_batch[index] is not None else row_scores[0]
            for index, row_scores in enumerate(rows)
        ]

    def prepare_batch(self, texts: Sequence[Any]):
        """CPU-only rendering/tokenization; safe to prefetch without CUDA work."""
        rendered = [self._render_input(value) for value in texts]
        encoded = self.tokenizer(
            rendered,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        visible_lengths = encoded["attention_mask"].sum(dim=1)
        too_long = (visible_lengths > self.max_length).nonzero(
            as_tuple=False
        ).flatten().tolist()
        if too_long:
            lengths = [int(visible_lengths[index]) for index in too_long]
            raise RuntimeError(
                "PRM input exceeds --prm_max_length without silent truncation; "
                f"rows={too_long}, token_lengths={lengths}, "
                f"limit={self.max_length}"
            )
        phase_ends_batch = [None if not isinstance(value, dict) or value.get("phase_ends") is None
                            else [int(end) for end in value["phase_ends"]] for value in texts]
        marker_positions = None
        if self.kind == "process_reward":
            if self.step_separator_id is None:
                raise RuntimeError("Qwen PRM separator token was not initialized")
            mask = encoded["input_ids"].eq(self.step_separator_id)
            if not mask.any(dim=1).all():
                missing = (~mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
                raise RuntimeError("<extra_0> was truncated or missing from PRM input rows "
                                   f"{missing}; increase prm_max_length or inspect truncation.")
            marker_positions = [row.nonzero(as_tuple=False).flatten() for row in mask]
        return encoded, phase_ends_batch, marker_positions

    def _score_batch(self, texts: Sequence[Any]) -> List[Any]:
        return self.score_prepared_batch(self.prepare_batch(texts))

    def score_prepared_batch(self, prepared):
        with torch_runtime.inference_mode():
            return self._score_prepared_batch(prepared)

    def _score_prepared_batch(self, prepared):
        encoded, phase_ends_batch, marker_positions_batch = prepared
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if self.kind == "process_reward" and self.tokenizer.padding_side == "left":
            # The released Qwen2 PRM remote code defaults to sequential
            # position_ids (0..seq_len-1) when none are supplied.  With left
            # padding that shifts the real tokens to different RoPE positions
            # depending on the other rows in the batch.  Recompute positions
            # from the attention mask so every unpadded sequence starts at 0,
            # while padded positions use the same harmless filler value as
            # Qwen's generation helper.
            position_ids = encoded["attention_mask"].long().cumsum(-1) - 1
            position_ids.masked_fill_(encoded["attention_mask"] == 0, 1)
            encoded["position_ids"] = position_ids
        selected = None
        marker_mask = None
        positions = None
        row = None
        # Kept for compatibility with the shared cleanup path below.  The
        # Qwen process-reward branch returns earlier after marker-only scoring.
        probs = None

        if self.kind == "process_reward":
            if self.step_separator_id is None:
                raise RuntimeError("Qwen PRM separator token was not initialized")
            scores = self._score_process_reward_markers(
                encoded, None, phase_ends_batch, marker_positions_batch
            )
            del encoded, marker_mask
            return scores

        try:
            output = self.model(
                **encoded,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        except TypeError:
            # Keep compatibility with custom sequence/token PRMs whose forward
            # signature does not expose all generation-related keyword args.
            try:
                output = self.model(**encoded, use_cache=False)
            except TypeError:
                output = self.model(**encoded)
        logits = getattr(output, "logits", None)
        if logits is None:
            logits = getattr(output, "scores", None)
        if logits is None:
            raise RuntimeError("PRM output has neither logits nor scores")
        if self.kind == "sequence":
            probs = self._binary_probability(
                logits, self._positive_class_index(logits.shape[-1])
            )
        elif self.kind == "token":
            positions = self._last_nonpad_positions(
                encoded["attention_mask"]
            ).to(logits.device)
            row = torch_runtime.arange(logits.shape[0], device=logits.device)
            if logits.ndim == 2:
                probs = torch_runtime.sigmoid(logits[row, positions])
            else:
                selected = logits[row, positions]
                probs = self._binary_probability(
                    selected,
                    self._positive_class_index(selected.shape[-1]),
                )
        else:
            positions = self._last_nonpad_positions(
                encoded["attention_mask"]
            ).to(logits.device)
            row = torch_runtime.arange(logits.shape[0], device=logits.device)
            selected = logits[row, positions]
            probs = torch_runtime.softmax(selected, dim=-1)[:, self.positive_id]
        scores = [
            float(value)
            for value in probs.detach().float().cpu().tolist()
        ]
        # Release all batch-local GPU references before the next batch.  The
        # allocator may otherwise retain large reserved blocks and look like a
        # steadily growing leak in long PRM evaluations.
        del output, logits, probs, encoded
        del selected, marker_mask, positions, row
        return scores

    def _positive_class_index(self, count: int) -> int:
        labels = getattr(self.config, "id2label", {}) or {}
        for index, label in labels.items():
            if any(
                word in str(label).lower()
                for word in ("correct", "positive", "true", "yes")
            ):
                return int(index)
        return 1 if count > 1 else 0


def _prm_dp_tp_worker(
    worker_index: int,
    gpu_group: str,
    jobs: Sequence[Tuple[int, int, Any]],
    model_path: str,
    prm_name: str,
    prm_device: str,
    prm_dtype: str,
    max_length: int,
    positive_token: Optional[str],
    system_prompt: Optional[str],
    trust_remote_code: bool,
    batch_size: int,
    result_queue: Any,
) -> None:
    """Score one shard in a fresh process.

    A group with multiple visible GPUs is one TP worker: Hugging Face
    ``device_map=auto`` shards the PRM across that group.  Multiple groups run
    concurrently, providing DP across complete PRM replicas/TP groups.
    """

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
        if prm_device == "cpu":
            scorer_device = "cpu"
        elif len([item for item in gpu_group.split(",") if item.strip()]) > 1:
            scorer_device = "auto"
        elif prm_device == "auto":
            scorer_device = "cuda:0" if (torch_runtime is not None and torch_runtime.cuda.is_available()) else "cpu"
        else:
            scorer_device = prm_device
        scorer = get_prm_backend(prm_name).create_scorer(
            model_path=model_path,
            device=scorer_device,
            dtype=prm_dtype,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
            positive_token=positive_token,
            system_prompt=system_prompt,
        )
        jobs = sorted(jobs, key=lambda item: _backend_input_length(prm_name, item[2]))
        for start in range(0, len(jobs), max(1, int(batch_size))):
            batch = list(jobs[start : start + max(1, int(batch_size))])
            values = _score_prm_batch_resilient(scorer, [item[2] for item in batch])
            if len(values) != len(batch):
                raise RuntimeError(
                    f"PRM returned {len(values)} scores for a batch of {len(batch)}"
                )
            result_queue.put(
                (
                    "scores",
                    worker_index,
                    [
                        expanded
                        for item, value in zip(batch, values)
                        for expanded in _expand_phase_scores(item[0], item[1], value)
                    ],
                )
            )
        result_queue.put(("done", worker_index, None))
    except Exception:
        result_queue.put(("error", worker_index, traceback.format_exc()))


def _score_phase_jobs_dp_tp(
    jobs: Sequence[Tuple[int, int, Any]],
    config: "PRMBestOfNConfig",
    batch_size: int,
) -> Dict[Tuple[int, int], float]:
    """Score phase jobs with data parallel workers and optional TP groups."""

    groups = parse_gpu_groups(config.prm_gpus, config.prm_tensor_parallel_size)
    if not groups:
        raise ValueError("DP+TP scoring requires --prm_gpus")
    if not jobs:
        return {}
    # Do not launch empty workers when there are fewer jobs than groups.
    shard_count = min(len(groups), len(jobs))
    weighted = [(_backend_input_length(config.prm_name, text), (seq, phase, text)) for seq, phase, text in jobs]
    shards = lpt_shards(weighted, shard_count)
    active_groups = groups[:shard_count]
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_prm_dp_tp_worker,
            args=(
                index,
                active_groups[index],
                shard,
                config.prm_model_path,
                config.prm_name,
                config.prm_device,
                config.prm_dtype,
                config.prm_max_length,
                config.prm_positive_token,
                config.prm_system_prompt,
                config.prm_trust_remote_code,
                batch_size,
                result_queue,
            ),
        )
        for index, shard in enumerate(shards)
    ]
    scores: Dict[Tuple[int, int], float] = {}
    finished = 0
    errors: List[str] = []
    try:
        for process in processes:
            process.start()
        while finished < len(processes):
            try:
                kind, _, payload = result_queue.get(timeout=1.0)
            except queue.Empty:
                dead = [
                    (index, process.exitcode)
                    for index, process in enumerate(processes)
                    if not process.is_alive() and process.exitcode is not None
                ]
                if dead:
                    errors.append(f"PRM DP+TP workers exited without a result: {dead}")
                    _reap_processes(processes)
                    break
                continue
            if kind == "scores":
                for sequence_idx, phase_idx, value in payload:
                    if not math.isfinite(float(value)):
                        raise ValueError("PRM returned a non-finite score")
                    scores[(int(sequence_idx), int(phase_idx))] = float(value)
            elif kind == "done":
                finished += 1
            elif kind == "error":
                errors.append(str(payload))
                _reap_processes(processes)
                break
        if errors:
            raise RuntimeError("PRM DP+TP worker failed:\n" + "\n".join(errors))
    finally:
        _reap_processes(processes)
        _close_queue(result_queue)
    expected = _expected_phase_keys(jobs)
    if set(scores) != expected:
        missing = sorted(expected.difference(scores))
        raise RuntimeError(f"PRM DP+TP scoring missed phase jobs: {missing[:10]}")
    return scores


def _prm_persistent_worker(
    worker_index: int,
    gpu_group: str,
    model_path: str,
    prm_name: str,
    prm_device: str,
    prm_dtype: str,
    max_length: int,
    positive_token: Optional[str],
    system_prompt: Optional[str],
    trust_remote_code: bool,
    batch_size: int,
    input_queue: Any,
    result_queue: Any,
) -> None:
    """Keep one PRM replica alive and score successive request shards."""
    ticket = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
        group_size = len([item for item in gpu_group.split(",") if item.strip()])
        if prm_device == "cpu":
            scorer_device = "cpu"
        elif group_size > 1:
            scorer_device = "auto"
        elif prm_device == "auto":
            scorer_device = "cuda:0" if (torch_runtime is not None and torch_runtime.cuda.is_available()) else "cpu"
        else:
            scorer_device = prm_device
        scorer = get_prm_backend(prm_name).create_scorer(
            model_path=model_path,
            device=scorer_device,
            dtype=prm_dtype,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
            positive_token=positive_token,
            system_prompt=system_prompt,
        )
        stop = threading.Event()

        def batches():
            while not stop.is_set():
                try:
                    packet = input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if packet is None:
                    return
                ticket, jobs = packet
                jobs = sorted(jobs, key=lambda item: _backend_input_length(prm_name, item[2]))
                for start in range(0, len(jobs), max(1, int(batch_size))):
                    yield ticket, list(jobs[start : start + max(1, int(batch_size))])
                yield ticket, None

        def prepare(packet):
            ticket, batch = packet
            prepared = None
            try:
                if batch is not None and hasattr(scorer, "prepare_batch"):
                    prepared = scorer.prepare_batch([item[2] for item in batch])
            except Exception as exc:
                # Deliver preparation failures with their own request ticket,
                # after all earlier batches/done messages from this worker.
                prepared = exc
            return ticket, batch, prepared

        with prefetch_map(batches(), prepare, stop=stop) as prepared_batches:
            for ticket, batch, prepared in prepared_batches:
                if isinstance(prepared, Exception):
                    raise prepared
                if batch is None:
                    result_queue.put(("done", worker_index, ticket))
                    continue
                values = _score_prm_batch_resilient(scorer, [item[2] for item in batch], prepared,
                                                   defer_oom=True)
                del prepared
                if len(values) != len(batch):
                    raise RuntimeError(f"PRM returned {len(values)} scores for a batch of {len(batch)}")
                failed = [(item[0], item[1], value.message) for item, value in zip(batch, values)
                          if isinstance(value, OOMScore)]
                if failed:
                    result_queue.put(("oom", worker_index, (ticket, failed)))
                result_queue.put(
                    (
                        "scores",
                        worker_index,
                        (ticket, [
                            expanded
                            for item, value in zip(batch, values)
                            if not isinstance(value, OOMScore)
                            for expanded in _expand_phase_scores(item[0], item[1], value)
                        ]),
                    )
                )
    except Exception:
        result_queue.put(("error", worker_index, (ticket, traceback.format_exc())))


class _PRMDPWorkerPool:
    """Persistent DP workers; each worker auto-shards its PRM over its TP group."""

    def __init__(self, config: "PRMBestOfNConfig", batch_size: int) -> None:
        self.config = config
        self.batch_size = batch_size
        self.groups = parse_gpu_groups(config.prm_gpus, config.prm_tensor_parallel_size)
        self._aborted = False
        self._next_ticket = 0
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._start_workers()

    def _start_workers(self):
        config, batch_size = self.config, self.batch_size
        context = mp.get_context("spawn")
        self.result_queue = context.Queue()
        self.input_queues = [context.Queue() for _ in self.groups]
        self.processes = []
        try:
            for index, group in enumerate(self.groups):
                process = context.Process(
                    target=_prm_persistent_worker,
                    args=(index, group, config.prm_model_path, config.prm_name, config.prm_device, config.prm_dtype,
                          config.prm_max_length, config.prm_positive_token, config.prm_system_prompt,
                          config.prm_trust_remote_code,
                          batch_size, self.input_queues[index], self.result_queue),
                )
                process.start()
                self.processes.append(process)
        except BaseException:
            _reap_processes(self.processes)
            _close_queue(self.result_queue)
            for queue_ in self.input_queues:
                _close_queue(queue_)
            self._aborted = True
            raise

    def score(self, jobs: Sequence[Tuple[int, int, Any]]) -> Dict[Tuple[int, int], float]:
        if not jobs:
            return {}
        return self.result(self.submit(jobs))

    def submit(self, jobs: Sequence[Tuple[int, int, Any]]) -> int:
        """Keep the old per-request LPT assignment and exact batch composition."""
        if self._aborted:
            raise RuntimeError("PRM DP+TP worker pool has already been aborted")
        if len(self._pending) >= 4:
            raise RuntimeError("PRM lookahead exceeded its four-request bound")
        ticket = self._next_ticket
        self._next_ticket += 1
        self._pending[ticket] = {"scores": {}, "done": set(), "error": None,
                                 "expected": _expected_phase_keys(jobs), "oom": {},
                                 "jobs": {(int(seq), int(phase)): text for seq, phase, text in jobs}}
        shards = lpt_shards(
            [(_backend_input_length(self.config.prm_name, text), (seq, phase, text)) for seq, phase, text in jobs],
            len(self.groups),
        )
        for queue_, shard in zip(self.input_queues, shards):
            shard.sort(key=lambda item: _backend_input_length(self.config.prm_name, item[2]))
            queue_.put((ticket, shard))
        return ticket

    def result(self, ticket: int) -> Dict[Tuple[int, int], float]:
        if self._aborted:
            raise RuntimeError("PRM DP+TP worker pool has already been aborted")
        target = self._pending[ticket]
        errors: List[str] = []
        if target["error"] is not None:
            errors.append(target["error"])
            self.abort()
        while len(target["done"]) < len(self.groups):
            check_stopped(getattr(self, "stop", None))
            if errors:
                break
            try:
                kind, worker, payload = self.result_queue.get(timeout=0.1 if getattr(self, "stop", None) is not None else 1.0)
            except queue.Empty:
                dead = [
                    (index, process.exitcode)
                    for index, process in enumerate(self.processes)
                    if index not in target["done"] and not process.is_alive() and process.exitcode is not None
                ]
                if dead:
                    errors.append(f"PRM persistent workers exited without a result: {dead}")
                    self.abort()
                    break
                continue
            if kind == "scores":
                result_ticket, rows = payload
                for sequence_idx, phase_idx, value in rows:
                    if not math.isfinite(float(value)):
                        message = "PRM returned a non-finite score"
                        if result_ticket == ticket:
                            errors.append(message)
                            self.abort()
                        else:
                            self._pending[result_ticket]["error"] = message
                        break
                    self._pending[result_ticket]["scores"][(int(sequence_idx), int(phase_idx))] = float(value)
            elif kind == "done":
                self._pending[payload]["done"].add(worker)
            elif kind == "oom":
                failed_ticket, rows = payload
                for seq, phase, message in rows:
                    self._pending[failed_ticket]["oom"][(int(seq), int(phase))] = message
            elif kind == "error":
                failed_ticket, message = payload
                if failed_ticket is None or failed_ticket == ticket:
                    errors.append(str(message))
                    self.abort()
                    break
                self._pending[failed_ticket]["error"] = str(message)
            if errors:
                break
        if errors:
            raise RuntimeError("PRM DP+TP worker failed:\n" + "\n".join(errors))
        if target["oom"]:
            self._recover_pending(ticket)
        scores = target["scores"]
        if set(scores) != target["expected"]:
            self.abort()
            raise RuntimeError("PRM DP+TP scoring missed phase jobs")
        del self._pending[ticket]
        return scores

    def _recover_pending(self, ticket):
        """Drain the existing four-ticket window before borrowing every GPU.

        Ordinary batches keep their original shards/shapes and completed scores
        are never discarded. No later requests are submitted during this call.
        """
        try:
            while any(len(state["done"]) < len(self.groups) for state in self._pending.values()):
                check_stopped(getattr(self, "stop", None))
                if any(state["error"] is not None for state in self._pending.values()):
                    raise RuntimeError("A PRM worker failed while draining OOM recovery tasks")
                try:
                    kind, worker, payload = self.result_queue.get(timeout=0.2)
                except queue.Empty:
                    if any(not process.is_alive() for process in self.processes):
                        raise RuntimeError("PRM worker exited while draining OOM recovery tasks")
                    continue
                if kind == "done":
                    self._pending[payload]["done"].add(worker)
                elif kind == "scores":
                    result_ticket, rows = payload
                    for seq, phase, value in rows:
                        if not math.isfinite(float(value)):
                            raise RuntimeError("PRM returned a non-finite score")
                        self._pending[result_ticket]["scores"][(int(seq), int(phase))] = float(value)
                elif kind == "oom":
                    result_ticket, rows = payload
                    for seq, phase, message in rows:
                        self._pending[result_ticket]["oom"][(int(seq), int(phase))] = message
                elif kind == "error":
                    raise RuntimeError(str(payload))
                else:
                    raise RuntimeError(f"Invalid PRM drain message: {kind}")
            if any(state["error"] is not None for state in self._pending.values()):
                raise RuntimeError("A PRM worker failed before OOM recovery")
            jobs = [((tick, seq, phase), state["jobs"][(seq, phase)])
                    for tick, state in self._pending.items() for seq, phase in state["oom"]]
            self.close()
            group = ",".join(gpu for shard in self.groups for gpu in shard.split(","))
            recovered = recover_jobs(self.config, group, jobs, stop=getattr(self, "stop", None))
            if set(recovered) != {key for key, _ in jobs}:
                raise RuntimeError("PRM recovery missed candidates")
            for (tick, seq, phase), value in recovered.items():
                rows = _expand_phase_scores(seq, phase, value)
                expected = _expected_phase_keys([(seq, phase, self._pending[tick]["jobs"][(seq, phase)])])
                if {(s, p) for s, p, _ in rows} != expected:
                    raise RuntimeError("PRM recovery missed phase scores")
                for s, p, score in rows:
                    if not math.isfinite(float(score)):
                        raise RuntimeError("PRM recovery returned a non-finite score")
                    self._pending[tick]["scores"][(s, p)] = float(score)
            for state in self._pending.values():
                state["oom"].clear()
            check_stopped(getattr(self, "stop", None))
            self._start_workers()
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Immediately stop all persistent workers after a scoring failure."""

        if self._aborted:
            return
        self._aborted = True
        self._pending.clear()
        _reap_processes(self.processes)
        for queue_ in self.input_queues:
            _close_queue(queue_)
        _close_queue(self.result_queue)

    def close(self) -> None:
        if self._aborted:
            return
        for queue_ in self.input_queues:
            try:
                queue_.put(None, timeout=2.0)
            except (queue.Full, OSError, ValueError):
                pass
        _reap_processes(self.processes)
        _close_queue(self.result_queue)
        for queue_ in self.input_queues:
            _close_queue(queue_)


class _PRMLocalWorkerPool:
    """One inference thread, retaining the existing single-scorer device map."""

    def __init__(self, service, args):
        self.service, self.args = service, args
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prm-inference")
        self._pending = set()

    def submit(self, jobs):
        context = capture_inference_context(torch_runtime, self.service._prm_scorer)

        def infer():
            with context():
                return self.service._run_prm_jobs(jobs, self.args)

        future = self.executor.submit(infer)
        self._pending.add(future)
        return future

    def result(self, future):
        while True:
            check_stopped(self.service._prm_pipeline_stop)
            try:
                result = future.result(timeout=0.1)
                break
            except FutureTimeout:
                continue
        self._pending.remove(future)
        return result

    def abort(self):
        for future in self._pending:
            future.cancel()
        self.close()
        self._pending.clear()

    def close(self):
        for future in self._pending:
            future.cancel()
        self.executor.shutdown(wait=True)


@dataclass
class PRMBestOfNConfig(BestOfNConfig):
    """Best-of-N generation/phase config with a Hugging Face PRM scorer."""

    prm_model_path: str = ""
    # Canonical registry name; defaults to the original Qwen math PRM.
    prm_name: str = DEFAULT_PRM_NAME
    # PRM transformer activations are much larger than the value model's
    # activations; keep the PRM-specific default conservative.
    scoring_batch_size: int = 2
    prm_device: str = "auto"
    prm_dtype: str = "auto"
    # Qwen2.5-Math-PRM-7B advertises max_position_embeddings=4096.  A
    # 40960-token default causes quadratic attention memory and is unsafe for
    # the reusable scorer; callers can still opt into a larger value.
    prm_max_length: int = 4096
    prm_positive_token: Optional[str] = None
    prm_phase_marker: str = "<extra_0>"
    prm_phase_separator: str = "\n\n"
    prm_system_prompt: Optional[str] = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    prm_trust_remote_code: bool = True
    prm_gpus: Optional[str] = None
    prm_tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.prm_model_path:
            raise ValueError("prm_model_path must not be empty")
        self.prm_name = normalize_prm_name(self.prm_name)
        if self.prm_max_length < 1:
            raise ValueError("prm_max_length must be at least 1")
        if not self.prm_phase_separator:
            raise ValueError("prm_phase_separator must not be empty")
        if self.prm_tensor_parallel_size < 1:
            raise ValueError("prm_tensor_parallel_size must be at least 1")
        if self.prm_gpus:
            parse_gpu_groups(self.prm_gpus, self.prm_tensor_parallel_size)


class PRMBestOfNService(BestOfNService):
    """Existing three-stage BoN service with only stage-three scoring replaced."""

    config: PRMBestOfNConfig

    def __init__(self, config: PRMBestOfNConfig) -> None:
        super().__init__(config)
        self._prm_scorer: Optional[HFPRMScorer] = None
        self._prm_dp_pool: Optional[_PRMDPWorkerPool] = None
        self._prm_checkpoint_path: Optional[Any] = None
        self._prm_checkpoint: Optional[PRMCheckpoint] = None
        self._prm_pipeline_stop = None
        self._prm_async_writer = None
        self._prm_pending_writes = []
        self._prm_callback_events = []

    def generate_and_select_many(self, requests: Sequence[BestOfNRequest], **kwargs: Any):
        """Configure the PRM rollout checkpoint before the shared three stages."""
        from pathlib import Path

        staging_dir = kwargs.get("staging_dir")
        if self._prm_checkpoint is not None:
            self._prm_checkpoint.close()
            self._prm_checkpoint = None
        if staging_dir is not None:
            checkpoint = Path(staging_dir).expanduser().resolve() / ".bon_prm.jsonl"
            self._prm_checkpoint_path = checkpoint
            if kwargs.get("overwrite") and checkpoint.exists():
                checkpoint.unlink()
            elif not kwargs.get("resume") and checkpoint.exists():
                raise FileExistsError(
                    f"PRM staging data already exists in {checkpoint.parent}; use resume or overwrite"
                )
        return super().generate_and_select_many(requests, **kwargs)

    def _load_prm_checkpoint(self):
        path = self._prm_checkpoint_path
        if path is None:
            return {}
        if self._prm_checkpoint is None or self._prm_checkpoint.path != path:
            if self._prm_checkpoint is not None:
                self._prm_checkpoint.close()
            self._prm_checkpoint = PRMCheckpoint(path, self.config.prm_name)
        return self._prm_checkpoint

    def _persist_prm_candidate(
        self, request_id: str, sequence_idx: int, candidate: BestOfNCandidate
    ) -> None:
        path = self._prm_checkpoint_path
        if path is None:
            return
        payload = {
            "complete": True,
            "prm_name": self.config.prm_name,
            "request_id": request_id,
            "sequence_idx": int(sequence_idx),
            "candidate": {
                "sequence_idx": int(candidate.sequence_idx),
                "candidate_key": candidate.candidate_key,
                "text": candidate.text,
                "score": (float(candidate.score) if math.isfinite(candidate.score) else None),
                "summary_record": candidate.summary_record,
                "steps_record": candidate.steps_record,
            },
        }
        checkpoint = self._load_prm_checkpoint()
        if self._prm_async_writer is None:
            checkpoint.append(payload)
        else:
            key = (str(request_id), int(sequence_idx))
            self._prm_pending_writes.append(self._prm_async_writer.submit(checkpoint.encode(payload), key))

    def scoring_rollout_total(
        self, requests: Sequence[BestOfNRequest], skipped: set[str]
    ) -> int:
        checkpoint = self._load_prm_checkpoint()
        return sum(
            1
            for request in requests
            if request.request_id not in skipped
            for sequence_idx in range(self.config.n)
            if (request.request_id, sequence_idx) not in checkpoint
            or getattr(checkpoint, "lengths", {}).get((request.request_id, sequence_idx), 0) > self.config.max_tokens
        )

    @staticmethod
    def _candidate_from_checkpoint(payload: Dict[str, Any]) -> BestOfNCandidate:
        require_success(payload)
        return BestOfNCandidate(
            sequence_idx=int(payload["sequence_idx"]),
            candidate_key=str(payload["candidate_key"]),
            text=str(payload.get("text") or ""),
            score=(float(payload["score"]) if payload.get("score") is not None else float("-inf")),
            summary_record=dict(payload["summary_record"]),
            steps_record=dict(payload["steps_record"]),
        )

    def _get_prm_scorer(self) -> HFPRMScorer:
        if self._prm_scorer is None:
            self._prm_scorer = get_prm_backend(self.config.prm_name).create_scorer(
                model_path=self.config.prm_model_path,
                device=self.config.prm_device,
                dtype=self.config.prm_dtype,
                trust_remote_code=self.config.prm_trust_remote_code,
                max_length=self.config.prm_max_length,
                positive_token=self.config.prm_positive_token,
                system_prompt=self.config.prm_system_prompt,
            )
        return self._prm_scorer

    def _get_prm_dp_pool(self) -> _PRMDPWorkerPool:
        if self._prm_dp_pool is None:
            self._prm_dp_pool = _PRMDPWorkerPool(self.config, self.config.scoring_batch_size)
        self._prm_dp_pool.stop = self._prm_pipeline_stop
        return self._prm_dp_pool

    def _preflight_prm_jobs(self, requests_and_jobs):
        from .prm_lengths import create_length_counter

        counter = None
        longest = None
        for request, jobs in requests_and_jobs:
            for sequence_idx, _, value in jobs:
                try:
                    if counter is None:
                        counter = create_length_counter(self.config)
                    length = counter(value)
                except Exception as exc:
                    raise RuntimeError(
                        f"PRM input preflight failed for request_id={request.request_id}, "
                        f"sequence_idx={sequence_idx}: {exc}"
                    ) from exc
                if length > self.config.prm_max_length and (longest is None or length > longest[0]):
                    longest = (length, request.request_id, sequence_idx)
        if longest is not None:
            length, request_id, sequence_idx = longest
            raise RuntimeError(
                "PRM input exceeds --prm_max_length without silent truncation; "
                f"max_token_length={length}, limit={self.config.prm_max_length}, "
                f"request_id={request_id}, sequence_idx={sequence_idx}"
            )

    def _preflight_prm_inputs(self, legacy, entries, request_by_id, args):
        if getattr(self._score, "__func__", None) is not PRMBestOfNService._score:
            return

        def jobs():
            for entry in entries:
                request = request_by_id[str(entry["request_id"])]
                task = self._score_task(legacy, request, entry["records"], args)
                try:
                    yield request, next(task)
                finally:
                    task.close()

        self._preflight_prm_jobs(jobs())

    def _run_prm_jobs(self, phase_jobs, args):
        if not phase_jobs:
            return {}
        try:
            if self.config.prm_gpus:
                return self._get_prm_dp_pool().score(phase_jobs)
            scorer = self._get_prm_scorer()
            scores = {}
            failed_jobs = []
            phase_jobs = sorted(phase_jobs, key=lambda item: _backend_input_length(self.config.prm_name, item[2]))
            batches = [phase_jobs[start : start + args.scoring_batch_size]
                       for start in range(0, len(phase_jobs), args.scoring_batch_size)]

            def prepare(batch):
                prepared = (scorer.prepare_batch([item[2] for item in batch])
                            if hasattr(scorer, "prepare_batch") else None)
                return batch, prepared

            with prefetch_map(batches, prepare) as ready:
                for batch, prepared in ready:
                    check_stopped(self._prm_pipeline_stop)
                    values = _score_prm_batch_resilient(scorer, [item[2] for item in batch], prepared,
                                                       defer_oom=self.config.prm_device != "cpu")
                    del prepared
                    if len(values) != len(batch):
                        raise RuntimeError(f"PRM returned {len(values)} scores for a batch of {len(batch)}")
                    for item, value in zip(batch, values):
                        if isinstance(value, OOMScore):
                            failed_jobs.append(((item[0], item[1]), item[2]))
                            continue
                        for seq, phase, numeric in _expand_phase_scores(item[0], item[1], value):
                            if not math.isfinite(numeric):
                                raise ValueError("PRM returned a non-finite score")
                            scores[(seq, phase)] = numeric
            if failed_jobs:
                self._prm_scorer = scorer = None
                release_cuda_cache(torch_runtime)
                group = os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(
                    str(device) for device in range(torch_runtime.cuda.device_count()))
                recovered = recover_jobs(self.config, group, failed_jobs, stop=self._prm_pipeline_stop)
                for (seq, phase), value in recovered.items():
                    rows = _expand_phase_scores(seq, phase, value)
                    text = next(text for key, text in failed_jobs if key == (seq, phase))
                    if {(s, p) for s, p, _ in rows} != _expected_phase_keys([(seq, phase, text)]):
                        raise RuntimeError("PRM recovery missed phase scores")
                    for s, p, numeric in rows:
                        if not math.isfinite(numeric):
                            raise RuntimeError("PRM recovery returned a non-finite score")
                        scores[(s, p)] = numeric
            return scores
        except Exception as exc:
            raise RuntimeError("PRM scoring failed; aborting Best-of-N evaluation.\n" + traceback.format_exc()) from exc

    @staticmethod
    def _finish_score_task(task, scores):
        try:
            task.send(scores)
        except StopIteration as result:
            return result.value
        raise RuntimeError("PRM scoring task yielded more than once")

    def _score(self, legacy, request, prepared_records, args, on_rollout=None):
        task = self._score_task(legacy, request, prepared_records, args, on_rollout)
        try:
            jobs = next(task)
            return self._finish_score_task(task, self._run_prm_jobs(jobs, args))
        finally:
            task.close()

    def _complete_prm_rollout(self, callback):
        if self._prm_async_writer is None:
            callback("scoring")
        else:
            self._prm_callback_events.append((self._prm_pending_writes[-1], callback))

    def _wait_prm_writes(self):
        futures, self._prm_pending_writes = self._prm_pending_writes, []
        events, self._prm_callback_events = self._prm_callback_events, []
        by_future = {}
        for future, callback in events:
            by_future.setdefault(future, []).append(callback)
        for future in futures:
            future.result()
            for callback in by_future.get(future, ()):
                callback("scoring")

    def _score_prepared_entries(self, legacy, entries, request_by_id, args, on_rollout_complete=None):
        if getattr(self._score, "__func__", None) is not PRMBestOfNService._score:
            yield from super()._score_prepared_entries(legacy, entries, request_by_id, args, on_rollout_complete)
            return

        # Initialize the shared read index before the producer can read it.
        checkpoint = self._load_prm_checkpoint()
        if self._prm_checkpoint_path is not None:
            self._prm_async_writer = OrderedAppender(checkpoint.append_encoded)
        context = capture_inference_context(torch_runtime, self._prm_scorer)
        handed, handed_lock = {}, threading.Lock()

        def source(stop):
            self._prm_pipeline_stop = stop
            core = self._score_prepared_entries_core(
                legacy, entries, request_by_id, args, on_rollout_complete, _defer=True)
            try:
                with context():
                    for request, task, scores in core:
                        with handed_lock:
                            handed[id(task)] = task
                        yield request, task, scores
            finally:
                core.close()
                self._prm_pipeline_stop = None

        try:
            with background_iterator(source) as ready:
                for request, task, scores in ready:
                    with handed_lock:
                        handed.pop(id(task))
                    try:
                        candidates = self._finish_score_task(task, scores)
                        self._wait_prm_writes()
                    finally:
                        task.close()
                    yield request, candidates
        finally:
            for task in handed.values():
                task.close()
            try:
                if self._prm_async_writer is not None:
                    self._prm_async_writer.close()
            finally:
                self._prm_async_writer = None
                self._prm_pending_writes.clear()
                self._prm_callback_events.clear()
                if self._prm_checkpoint is not None:
                    self._prm_checkpoint.close()

    def _score_prepared_entries_core(self, legacy, entries, request_by_id, args, on_rollout_complete=None, *, _defer=False):
        # Respect existing custom/subclass scorers.
        if getattr(self._score, "__func__", None) is not PRMBestOfNService._score:
            try:
                yield from super()._score_prepared_entries(
                    legacy, entries, request_by_id, args, on_rollout_complete,
                )
            finally:
                if self._prm_checkpoint is not None:
                    self._prm_checkpoint.close()
            return

        pending = deque()
        entries = iter(entries)
        exhausted = False
        pending_bytes = 0
        pool = None
        local_pool = None
        callback = (lambda: self._complete_prm_rollout(on_rollout_complete)) if on_rollout_complete else None

        def fill():
            nonlocal exhausted, pending_bytes, pool, local_pool
            # Both limits matter: very long solutions must not multiply RAM/IPC
            # usage by the number of questions. One existing request is atomic.
            while not exhausted and len(pending) < 4 and pending_bytes < 16 * 1024 * 1024:
                entry = next(entries, None)
                if entry is None:
                    exhausted = True
                    break
                request = request_by_id[str(entry["request_id"])]
                task = self._score_task(legacy, request, entry["records"], args, callback)
                try:
                    jobs = next(task)
                except Exception as exc:
                    # A malformed future request must not suppress successful
                    # results preceding it in the iterator.
                    task.close()
                    pending.append((request, None, None, 0, exc))
                    exhausted = True
                    break
                try:
                    size = (sum(_backend_input_length(self.config.prm_name, job[2]) for job in jobs)
                            + sum(len(str(row.get("full_output") or "")) for row in entry["records"])) * 4
                    if jobs:
                        if pool is None:
                            if self.config.prm_gpus:
                                pool = self._get_prm_dp_pool()
                            else:
                                # Load on the caller's device before starting the
                                # one inference thread. No second model replica.
                                self._get_prm_scorer()
                                local_pool = pool = _PRMLocalWorkerPool(self, args)
                        ticket = pool.submit(jobs)
                    else:
                        ticket = None
                except BaseException:
                    task.close()
                    raise
                pending.append((request, task, ticket, size, None))
                pending_bytes += size

        current = None
        try:
            fill()
            while pending:
                request, current, ticket, size, error = pending.popleft()
                if error is not None:
                    raise error
                pending_bytes -= size
                scores = pool.result(ticket) if ticket is not None else {}
                # Replenish before JSON serialization, fsync or benchmark work.
                fill()
                if _defer:
                    task, current = current, None
                    yield request, task, scores
                else:
                    candidates = self._finish_score_task(current, scores)
                    current.close()
                    current = None
                    yield request, candidates
        except Exception as exc:
            raise RuntimeError("PRM scoring failed; aborting Best-of-N evaluation.\n" + traceback.format_exc()) from exc
        finally:
            if current is not None:
                current.close()
            if pool is not None and pool._pending:
                pool.abort()
                self._prm_dp_pool = None
            if pool is self._prm_dp_pool and getattr(pool, "_aborted", False):
                self._prm_dp_pool = None
            for _, task, _, _, _ in pending:
                if task is not None:
                    task.close()
            if not _defer and self._prm_checkpoint is not None:
                self._prm_checkpoint.close()
            if local_pool is not None:
                local_pool.close()

    def _cumulative_prm_scores(self, scores):
        return (aggregate_phase_scores(scores[:index + 1]) for index in range(len(scores)))

    def _score_task(
        self,
        legacy: Any,
        request: BestOfNRequest,
        prepared_records: Sequence[Dict[str, Any]],
        args: SimpleNamespace,
        on_rollout: Optional[Callable[[], None]] = None,
    ):
        del legacy
        candidates_by_index: Dict[int, BestOfNCandidate] = {}
        valid_records: Dict[int, Dict[str, Any]] = {}
        phase_ranges_by_index: Dict[int, List[List[int]]] = {}
        scores_by_index: Dict[int, Dict[int, float]] = {}
        phase_jobs: List[Tuple[int, int, Any]] = []
        checkpoint = self._load_prm_checkpoint()


        for prepared in prepared_records:
            sequence_idx = int(prepared["sequence_idx"])
            cached = checkpoint.get((request.request_id, sequence_idx))
            if cached is not None:
                summary = cached.get("summary_record", {})
                if (summary.get("num_generated_tokens") != prepared.get("num_generated_tokens")
                        or summary.get("full_output") != prepared.get("full_output")
                        or summary.get("phase_ranges") != prepared.get("phase_ranges")):
                    cached = None
            if cached is not None:
                require_success(cached)
                try:
                    candidates_by_index[sequence_idx] = self._candidate_from_checkpoint(cached)
                    continue
                except Exception:
                    # Ignore malformed/stale checkpoint entries and recompute.
                    pass
            try:
                if prepared.get("status") != "ok":
                    raise ValueError(
                        prepared.get("error", "Candidate preparation failed")
                    )
                _, normalized_ranges = phase_prefixes(
                    prepared,
                    self.config.prm_phase_separator,
                    build_prefixes=False,
                )
                valid_records[sequence_idx] = prepared
                phase_ranges_by_index[sequence_idx] = normalized_ranges
                scores_by_index[sequence_idx] = {}
                full_solution = self.config.prm_phase_separator.join(
                    split_steps(
                        prepared.get("full_output", ""),
                        self.config.prm_phase_separator,
                    )
                )
                phase_jobs.append(
                    (
                        sequence_idx,
                        -1,
                        build_prm_input(
                            self.config.prm_name,
                            prepared.get("problem", request.task_text),
                            full_solution,
                            self.config.prm_phase_marker,
                            self.config.prm_phase_separator,
                            request.messages,
                            request.raw_prompt,
                            [end for _, end in normalized_ranges],
                        ),
                    )
                )
            except Exception:
                raise

        phase_scores = yield phase_jobs

        for (sequence_idx, phase_index), value in phase_scores.items():
            scores_by_index[sequence_idx][phase_index] = value

        for sequence_idx, prepared in valid_records.items():
            try:
                phase_ranges = phase_ranges_by_index[sequence_idx]
                score_map = scores_by_index[sequence_idx]
                phase_scores = [
                    score_map[phase_index]
                    for phase_index in range(len(phase_ranges))
                ]
                final_scores = aggregate_phase_scores(phase_scores)
                steps = copy.deepcopy(prepared["steps"])
                cumulative_scores = self._cumulative_prm_scores(phase_scores)
                for phase_index, phase_range in enumerate(phase_ranges):
                    start_step, end_step = phase_range
                    if not (1 <= start_step <= end_step <= len(steps)):
                        raise ValueError(
                            f"Phase range {phase_range} outside "
                            f"1..{len(steps)}"
                        )
                    score = float(phase_scores[phase_index])
                    cumulative = next(cumulative_scores)
                    steps[end_step - 1].update(
                        {
                            "phase_index": phase_index,
                            "trajectory_model_step_score": score,
                            "phase_quality_score": score,
                            "phase_product_score": cumulative[
                                "product_score"
                            ],
                            "phase_min_score": cumulative["min_score"],
                            "phase_mean_score": cumulative["mean_score"],
                            "phase_last_score": cumulative["last_score"],
                        }
                    )
                links = _link_fields(request, sequence_idx)
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
                    "step_avg_confidences": prepared[
                        "step_avg_confidences"
                    ],
                    "num_steps": prepared["num_steps"],
                    "num_generated_tokens": prepared[
                        "num_generated_tokens"
                    ],
                    "finish_reason": prepared.get("finish_reason"),
                    "phase_ranges": phase_ranges,
                    "phase_scores": phase_scores,
                    **final_scores,
                }
                if "token_entropies" in prepared:
                    summary["token_entropies"] = prepared["token_entropies"]
                candidate = BestOfNCandidate(
                    sequence_idx=sequence_idx,
                    candidate_key=links["candidate_key"],
                    text=str(prepared.get("full_output") or ""),
                    score=float(
                        final_scores[
                            self.SCORE_KEYS[self.config.output_aggregation]
                        ]
                    ),
                    summary_record=summary,
                    steps_record={**common, "steps": steps},
                )
                candidates_by_index[sequence_idx] = candidate
                self._persist_prm_candidate(request.request_id, sequence_idx, candidate)
            except Exception:
                raise
            finally:
                if on_rollout is not None:
                    on_rollout()

        return [
            candidates_by_index[int(prepared["sequence_idx"])]
            for prepared in prepared_records
        ]

    def close(self) -> None:
        with self._lock:
            if self._prm_checkpoint is not None:
                self._prm_checkpoint.close()
                self._prm_checkpoint = None
            if self._prm_dp_pool is not None:
                pool = self._prm_dp_pool
                self._prm_dp_pool = None
                pool.close()
            if self._prm_scorer is not None:
                scorer = self._prm_scorer
                self._prm_scorer = None
                del scorer
                gc.collect()
                if torch_runtime is not None and torch_runtime.cuda.is_available():
                    torch_runtime.cuda.empty_cache()
