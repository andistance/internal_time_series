"""RLHFlow/Llama3.1-8B-PRM-Mistral-Data backend.

RLHFlow's Mistral-data PRM is a causal language model rather than a token or
sequence-classification model.  Its process-reward protocol represents every
solution step as a user turn followed by an assistant ``+`` label.  The model
card's evaluator reads the logits at ``-3`` (the position predicting that
label) and normalizes the ``+`` and ``-`` logits with a two-way softmax.

This module keeps that protocol isolated from the generic/Qwen scorer while
returning the same ``List[List[float]]`` phase-probability interface as every
other ``transformer_bon`` PRM backend.
"""

from __future__ import annotations
from ..prm_recovery import model_load_kwargs, parameter_storage

from ..prm_prefetch import candidate_token_ids

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:  # Keep registry imports light for generation-only users.
    torch = None  # type: ignore[assignment]


MODEL_ID = "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"


def _split_steps(prefix: Any, separator: str) -> List[str]:
    text = "" if prefix is None else str(prefix)
    return [part.strip() for part in text.split(separator) if part.strip()]


def _prompt_text(
    problem: Any,
    messages: Optional[Sequence[Dict[str, str]]],
    raw_prompt: Optional[str],
) -> str:
    """Resolve the benchmark prompt without dropping adapter-provided context."""

    if raw_prompt is not None:
        return str(raw_prompt)
    if messages:
        return "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") != "assistant"
        )
    return "" if problem is None else str(problem)


def build_input(
    problem: Any,
    prefix: str,
    marker: str = "",
    separator: str = "\n\n",
    messages: Optional[Sequence[Dict[str, str]]] = None,
    raw_prompt: Optional[str] = None,
    phase_ends: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build the structured input consumed by the RLHFlow scorer.

    ``prefix`` is the complete cumulative rollout.  The caller supplies
    ``phase_ends`` as one-based step indices, so the scorer can run one model
    forward pass over the full conversation and return exactly one score per
    requested phase.  ``marker`` is intentionally ignored: unlike Qwen's PRM,
    RLHFlow uses chat turns and fixed ``+``/``-`` candidate tokens.
    """

    del marker
    steps = _split_steps(prefix, separator)
    if not steps:
        raise ValueError("RLHFlow PRM prefix is empty")
    normalized_ends = (
        [int(end) for end in phase_ends] if phase_ends is not None else None
    )
    if normalized_ends is not None and any(
        end < 1 or end > len(steps) for end in normalized_ends
    ):
        raise ValueError(
            f"RLHFlow phase_ends {normalized_ends!r} outside 1..{len(steps)}"
        )
    return {
        "problem": _prompt_text(problem, messages, raw_prompt),
        "solution_steps": steps,
        "phase_ends": normalized_ends,
    }


def input_length(value: Any) -> int:
    if not isinstance(value, dict):
        return max(1, len(str(value)))
    steps = value.get("solution_steps")
    if isinstance(steps, list):
        return max(
            1,
            len(str(value.get("problem", "")))
            + sum(len(str(step)) for step in steps),
        )
    return max(1, len(str(value)))


class RLHFlowMistralPRMScorer:
    """Score RLHFlow Mistral-data PRM conversations with ``P(+)``."""

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
        del positive_token, system_prompt  # The RLHFlow protocol fixes +/-.
        if torch is None:
            raise ModuleNotFoundError(
                "RLHFlow PRM scoring requires PyTorch to be installed"
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )
        # Right padding preserves the absolute ``-3``-relative positions from
        # the official single-example evaluator when rows are batched.
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        self.plus_id = self._single_token_id("+")
        self.minus_id = self._single_token_id("-")
        if self.plus_id == self.minus_id:
            raise RuntimeError(
                "RLHFlow PRM '+' and '-' must map to distinct tokenizer tokens"
            )

        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            try:
                kwargs["torch_dtype"] = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"Unknown torch dtype: {dtype!r}") from exc
        if device == "auto" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        kwargs = model_load_kwargs(kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        # The official reference sets the padding id to EOS for this
        # Llama-derived checkpoint.  Keep both tokenizer and model config in
        # sync so generation-compatible model code also accepts padded batches.
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = self._resolve_device(device)
        if "device_map" not in kwargs and self.device.type != "cpu":
            self.model.to(self.device)
        print(
            f"[INFO] RLHFlow PRM model: {model_path} "
            f"({self.model.__class__.__name__}, device={self.device}, "
            f"candidate_tokens=({self.plus_id}, {self.minus_id}))"
        )

    def _single_token_id(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"RLHFlow PRM requires {token!r} to encode as one token; "
                f"got {ids!r}"
            )
        return int(ids[0])

    def _resolve_device(self, requested: str):
        if requested != "auto":
            return torch.device(requested)
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for value in device_map.values():
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str) and value not in {"cpu", "disk"}:
                    return torch.device(value)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _chat_token_ids(self, messages: Sequence[Dict[str, str]]) -> List[int]:
        """Tokenize a complete conversation using the checkpoint chat template."""

        try:
            encoded = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )
        except TypeError:
            # Older Transformers releases infer tokenization from
            # ``return_tensors`` but reject the explicit ``tokenize`` kwarg.
            encoded = self.tokenizer.apply_chat_template(
                list(messages),
                add_generation_prompt=False,
                return_tensors="pt",
            )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if not isinstance(encoded, torch.Tensor):
            encoded = torch.as_tensor(encoded, dtype=torch.long)
        if encoded.ndim == 2:
            if encoded.shape[0] != 1:
                raise RuntimeError(
                    "RLHFlow chat template returned more than one sequence"
                )
            encoded = encoded[0]
        if encoded.ndim != 1:
            raise RuntimeError(
                "RLHFlow chat template must return a one-dimensional token sequence"
            )
        return [int(token) for token in encoded.detach().cpu().tolist()]

    def _prepare_row(self, value: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        problem = "" if value.get("problem") is None else str(value.get("problem"))
        raw_steps = value.get("solution_steps")
        if raw_steps is None:
            raw_steps = value.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("RLHFlow PRM input has no solution_steps")
        steps = [str(step).strip() for step in raw_steps]
        if any(not step for step in steps):
            raise ValueError("RLHFlow PRM input contains an empty solution step")

        phase_ends = value.get("phase_ends")
        if phase_ends is None:
            phase_ends = list(range(1, len(steps) + 1))
        phase_ends = [int(end) for end in phase_ends]
        if not phase_ends or any(end < 1 or end > len(steps) for end in phase_ends):
            raise ValueError(
                f"RLHFlow phase_ends {phase_ends!r} outside 1..{len(steps)}"
            )

        conversation: List[Dict[str, str]] = []
        prediction_positions: Dict[int, int] = {}
        requested_ends = set(phase_ends)
        full_ids: Optional[List[int]] = None
        for step_index, step in enumerate(steps):
            # This is the exact official construction: the first user turn
            # contains the problem and first step; later turns contain only a
            # new step, each followed by the dummy assistant '+' label.
            user_text = f"{problem} {step}" if step_index == 0 else step
            conversation.append({"role": "user", "content": user_text})
            conversation.append({"role": "assistant", "content": "+"})
            if step_index + 1 not in requested_ends and step_index + 1 != len(steps):
                continue
            prefix_ids = self._chat_token_ids(conversation)
            # RLHFlow's reference uses logits[:, -3, candidate_tokens].  In
            # the Llama 3.1 template, '+' is the penultimate token and -3 is
            # therefore the position whose logits predict '+'.
            if len(prefix_ids) < 3 or prefix_ids[-2] != self.plus_id:
                raise RuntimeError(
                    "Unexpected RLHFlow chat-template layout: assistant '+' "
                    f"is not at token -2 (last ids={prefix_ids[-8:]!r})"
                )
            prediction_positions[step_index + 1] = len(prefix_ids) - 3
            full_ids = prefix_ids

        assert full_ids is not None
        selected_positions = [prediction_positions[end] for end in phase_ends]
        return full_ids, selected_positions

    def _marker_pair_logits(
        self, encoded: Dict[str, torch.Tensor], positions: Sequence[Sequence[int]]
    ) -> List[torch.Tensor]:
        """Compute only ``+``/``-`` logits at each row's marker positions."""

        backbone = getattr(self.model, "model", None)
        output_head = self.model.get_output_embeddings()
        if backbone is None or output_head is None:
            raise RuntimeError(
                "RLHFlow PRM model must expose a transformer body and output "
                "head to compute marker-only logits"
            )
        try:
            body_output = backbone(
                **encoded,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            body_output = backbone(**encoded, use_cache=False)
        hidden_states = getattr(body_output, "last_hidden_state", None)
        if hidden_states is None and isinstance(body_output, (tuple, list)):
            hidden_states = body_output[0]
        if hidden_states is None or hidden_states.ndim != 3:
            raise RuntimeError(
                "RLHFlow transformer body returned no [batch, seq, hidden] states"
            )

        selected_rows: List[torch.Tensor] = []
        row_sizes: List[int] = []
        for batch_index, row_positions in enumerate(positions):
            position_tensor = torch.as_tensor(
                row_positions, dtype=torch.long, device=hidden_states.device
            )
            row_hidden = hidden_states[batch_index].index_select(0, position_tensor)
            selected_rows.append(row_hidden)
            row_sizes.append(int(row_hidden.shape[0]))
        selected_hidden = torch.cat(selected_rows, dim=0)
        del body_output, hidden_states, selected_rows

        ids = (self.plus_id, self.minus_id)
        candidate_ids = candidate_token_ids(self, ids, selected_hidden.device, torch)
        try:
            head_weight = parameter_storage(output_head)
            head_device = head_weight.device
            selected_hidden = selected_hidden.to(head_device)
            candidate_ids = candidate_token_ids(self, ids, head_device, torch)
            selected_weight = head_weight.index_select(0, candidate_ids)
            head_bias = parameter_storage(output_head, "bias")
            selected_bias = (
                head_bias.index_select(0, candidate_ids)
                if head_bias is not None
                else None
            )
            pair_logits = torch.nn.functional.linear(
                selected_hidden, selected_weight, selected_bias
            )
        except Exception:
            selected_hidden = selected_hidden.to(next(output_head.parameters()).device)
            selected_logits = output_head(selected_hidden)
            pair_logits = selected_logits.index_select(
                1, candidate_ids.to(selected_logits.device)
            )
            del selected_logits
        del selected_hidden, candidate_ids
        if "selected_weight" in locals():
            del selected_weight

        rows: List[torch.Tensor] = []
        offset = 0
        for size in row_sizes:
            rows.append(pair_logits[offset : offset + size])
            offset += size
        return rows

    def prepare_batch(self, texts):
        rows: List[List[int]] = []
        positions: List[List[int]] = []
        for value in texts:
            if not isinstance(value, dict):
                raise TypeError("RLHFlow PRM input must be a structured dict")
            row_ids, row_positions = self._prepare_row(value)
            if len(row_ids) > self.max_length:
                raise RuntimeError(
                    "RLHFlow PRM input exceeds --prm_max_length without silent "
                    f"truncation: token_length={len(row_ids)}, "
                    f"limit={self.max_length}"
                )
            rows.append(row_ids)
            positions.append(row_positions)

        encoded = self.tokenizer.pad(
            [
                {"input_ids": row, "attention_mask": [1] * len(row)}
                for row in rows
            ],
            padding=True,
            return_tensors="pt",
        )
        return encoded, positions

    def score_batch(self, texts: Sequence[Any]) -> List[List[float]]:
        if not texts:
            return []
        return self.score_prepared_batch(self.prepare_batch(texts))

    def score_prepared_batch(self, prepared):
        with torch.inference_mode():
            return self._score_prepared_batch(prepared)

    def _score_prepared_batch(self, prepared):
        encoded, positions = prepared
        encoded = {
            key: tensor.to(self.device) for key, tensor in encoded.items()
        }
        pair_logits_rows = self._marker_pair_logits(encoded, positions)
        scores: List[List[float]] = []
        for row_pair_logits in pair_logits_rows:
            row_scores = torch.softmax(row_pair_logits.float(), dim=-1)[:, 0]
            scores.append(row_scores.detach())

        del pair_logits_rows, encoded
        flat = torch.cat(scores).cpu().tolist()
        result, offset = [], 0
        for row in scores:
            size = row.numel()
            result.append(flat[offset:offset + size])
            offset += size
        return result


def scorer_class():
    return RLHFlowMistralPRMScorer


def create_scorer(*args: Any, **kwargs: Any) -> RLHFlowMistralPRMScorer:
    return RLHFlowMistralPRMScorer(*args, **kwargs)
