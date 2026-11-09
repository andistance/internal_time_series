"""FoVer PRM backend for ``ryokamoi/Qwen-2.5-7B-FoVer-PRM-2026``.

FoVer is a causal language model used as a process reward model.  Its official
format is a conversation containing the problem/instructions and one user
message per solution step, each followed by an assistant ``correct`` label.
The score for a step is the softmax probability of ``correct`` versus
``incorrect`` at the token immediately preceding that dummy label.
"""

from __future__ import annotations
from ..prm_recovery import model_load_kwargs, parameter_storage

from ..prm_prefetch import PreparedRowsScorer, candidate_token_ids

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - importing the registry stays light
    torch = None  # type: ignore[assignment]


def _split_steps(prefix: Any, separator: str) -> List[str]:
    text = "" if prefix is None else str(prefix)
    return [part.strip() for part in text.split(separator) if part.strip()]


def build_input(
    problem: Any,
    prefix: str,
    marker: str = "",
    separator: str = "\n\n",
    messages: Optional[Sequence[Dict[str, str]]] = None,
    raw_prompt: Optional[str] = None,
    phase_ends: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build the exact FoVer input consumed by its official preprocessing."""

    del marker, messages, raw_prompt
    steps = _split_steps(prefix, separator)
    if not steps:
        raise ValueError("PRM prefix is empty")
    return {
        "problem": "" if problem is None else str(problem),
        "solution_steps": steps,
        "phase_ends": [int(end) for end in phase_ends] if phase_ends is not None else None,
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


class FoVerPRMScorer(PreparedRowsScorer):
    """Hugging Face scorer implementing the FoVer model-card protocol."""

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
        del positive_token, system_prompt
        if torch is None:
            raise ModuleNotFoundError("FoVer PRM scoring requires PyTorch")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        if device == "auto" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        kwargs = model_load_kwargs(kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        self.device = self._resolve_device(device)
        if "device_map" not in kwargs and self.device.type != "cpu":
            self.model.to(self.device)
        print(
            f"[INFO] FoVer PRM model: {model_path} "
            f"({self.model.__class__.__name__}, device={self.device})"
        )

    def _resolve_device(self, requested: str):
        if requested != "auto":
            return torch.device(requested)
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for value in device_map.values():
                if isinstance(value, str) and value not in {"cpu", "disk"}:
                    return torch.device(value)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _one_token_id(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"FoVer tokenizer must encode {text!r} as one token; got {ids!r}"
            )
        return int(ids[0])

    def _render_conversation(self, value: Dict[str, Any]) -> List[Dict[str, str]]:
        problem = "" if value.get("problem") is None else str(value.get("problem"))
        raw_steps = value.get("solution_steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("FoVer PRM input has no solution_steps")
        steps = [str(step) for step in raw_steps]
        first = (
            "** Problem **\n"
            f"{problem}\n\n"
            "** Task **\n"
            "Your task is to evaluate the accuracy of each step in the provided solution to the above question. "
            "For each step, respond with \"correct\" if the reasoning is logically valid and mathematically sound, "
            "or if the step is a general statement or transition that does not contain reasoning. "
            "Respond with \"incorrect\" if the step includes any errors or flawed logic.\n\n"
            "** Sotluion **\n"
            f"{steps[0]}"
        )
        conversation: List[Dict[str, str]] = [
            {"role": "user", "content": first},
            {"role": "assistant", "content": "correct"},
        ]
        for step in steps[1:]:
            conversation.extend(
                [
                    {"role": "user", "content": step},
                    {"role": "assistant", "content": "correct"},
                ]
            )
        return conversation

    def _label_positions(self, tokenized_prompt: np.ndarray) -> np.ndarray:
        """Locate logits immediately before every dummy ``correct`` label."""

        ids = np.asarray(tokenized_prompt)
        model_type = str(getattr(self.model.config, "model_type", "")).lower()
        is_llama = "llama" in model_type or "llama" in str(
            getattr(self.tokenizer, "name_or_path", "")
        ).lower()
        masks: List[np.ndarray] = []

        def shifted_mask(token_id: int, offset: int) -> np.ndarray:
            mask = np.zeros_like(ids, dtype=bool)
            positions = np.where(ids == token_id)[0] + offset
            positions = positions[(positions >= 0) & (positions < len(mask))]
            mask[positions] = True
            return mask

        if is_llama:
            newline_id = self._one_token_id("\n\n")
            masks.append(ids == newline_id)
            masks.append(shifted_mask(self._one_token_id("assistant"), 2))
            masks.append(shifted_mask(self._one_token_id("<|end_header_id|>"), 1))
        else:
            newline_id = self._one_token_id("\n")
            masks.append(ids == newline_id)
            masks.append(shifted_mask(self._one_token_id("assistant"), 1))
            masks.append(shifted_mask(self._one_token_id("<|im_start|>"), 2))
        positions = np.where(np.logical_and.reduce(masks))[0]
        if len(positions) == 0:
            raise RuntimeError(
                "FoVer postprocessing found no dummy-label positions; "
                "check the tokenizer chat template and --prm_max_length"
            )
        return positions

    def _marker_pair_logits(
        self,
        inputs: torch.Tensor,
        tokenized_prompt: np.ndarray,
        *,
        positions: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Return only ``incorrect``/``correct`` logits at FoVer markers.

        Calling ``AutoModelForCausalLM`` normally materializes logits for every
        token in the sequence.  For a long rollout this creates a tensor with
        shape ``[1, sequence_length, vocab_size]`` and can consume tens of GB.
        FoVer only needs two vocabulary entries at its marker positions, so we
        run the transformer body first, select those hidden states, and apply
        the output head only to the selected rows.
        """
        if positions is None:
            positions = self._label_positions(tokenized_prompt)
        position_tensor = torch.as_tensor(positions, dtype=torch.long)
        candidate_pair = getattr(self, "_candidate_token_ids", None)
        if candidate_pair is None:
            candidate_pair = (self._one_token_id("incorrect"), self._one_token_id("correct"))
        incorrect, correct = candidate_pair
        ids = (incorrect, correct)
        candidate_ids = candidate_token_ids(self, ids, "cpu", torch)

        backbone = getattr(self.model, "model", None)
        output_head = self.model.get_output_embeddings()
        if backbone is None or output_head is None:
            raise RuntimeError(
                "FoVer model must expose a transformer body and output head "
                "to compute marker-only logits"
            )

        try:
            body_output = backbone(
                input_ids=inputs,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            body_output = backbone(inputs, use_cache=False)

        hidden_states = getattr(body_output, "last_hidden_state", None)
        if hidden_states is None and isinstance(body_output, (tuple, list)):
            hidden_states = body_output[0]
        if hidden_states is None or hidden_states.ndim != 3:
            raise RuntimeError(
                "FoVer transformer body returned no [batch, seq, hidden] states"
            )

        position_tensor = position_tensor.to(hidden_states.device)
        selected_hidden = hidden_states[0].index_select(0, position_tensor)
        del body_output, hidden_states, position_tensor

        # The ordinary Qwen2 lm_head is a Linear layer.  Indexing its two rows
        # avoids materializing even a marker-by-vocabulary logits tensor.  Keep
        # a fallback for compatible custom/quantized heads whose weights do not
        # support direct row selection.
        try:
            head_weight = parameter_storage(output_head)
            head_device = head_weight.device
            candidate_ids = candidate_token_ids(self, ids, head_device, torch)
            selected_hidden = selected_hidden.to(head_device)
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

        del selected_hidden
        if "selected_weight" in locals():
            del selected_weight
        return pair_logits

    def _prepare_one(self, value: Any):
        # The inference thread must not call the tokenizer while the producer
        # is encoding another row (fast tokenizers mutate padding state).
        if not hasattr(self, "_candidate_token_ids"):
            self._candidate_token_ids = (self._one_token_id("incorrect"), self._one_token_id("correct"))
        if not isinstance(value, dict):
            raise TypeError("FoVer PRM input must be a dict")
        conversation = self._render_conversation(value)
        try:
            inputs = self.tokenizer.apply_chat_template(
                conversation,
                return_tensors="pt",
                add_generation_prompt=False,
            )
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(
                conversation, return_tensors="pt", add_generation_prompt=False
            )
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise RuntimeError("FoVer chat template must return a [1, seq_len] tensor")
        if inputs.shape[-1] > self.max_length:
            raise RuntimeError(
                "FoVer input exceeds --prm_max_length without silent truncation; "
                f"token_length={int(inputs.shape[-1])}, limit={self.max_length}"
            )
        # Keep token/marker discovery entirely on the CPU.
        return inputs, self._label_positions(inputs[0].numpy()), value.get("phase_ends")

    def _score_prepared_tensor(self, prepared):
        inputs, positions, phase_ends = prepared
        pair_logits = self._marker_pair_logits(inputs.to(self.device), None, positions=positions)

        scores = torch.softmax(pair_logits.float(), dim=1)[:, 1]
        del pair_logits
        if scores.numel() == 0:
            raise RuntimeError("FoVer PRM returned no step scores")
        if phase_ends is None:
            selected_scores = scores
        else:
            indices = [int(end) - 1 for end in phase_ends]
            if any(index < 0 or index >= len(scores) for index in indices):
                raise RuntimeError(
                    "FoVer phase end exceeds the number of visible step scores; "
                    "ensure the complete FoVer conversation and all phase steps "
                    "fit within --prm_max_length"
                )
            selected_scores = scores[indices]
        return selected_scores.detach()



def scorer_class():
    return FoVerPRMScorer


def create_scorer(*args: Any, **kwargs: Any) -> FoVerPRMScorer:
    return FoVerPRMScorer(*args, **kwargs)
