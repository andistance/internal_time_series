"""RLHFlow/Llama3.1-8B-PRM-Deepseek-Data backend.

RLHFlow's model is a generative process reward model rather than a sequence
classification head.  Its documented evaluation protocol builds a multi-turn
conversation with one ``user`` message for each solution step and a dummy
``assistant`` message containing ``+`` after every step.  The phase score is
the softmax probability of ``+`` versus ``-`` at the logits position that
predicts that dummy ``+`` token.

The backend returns the same ``[0, 1]`` phase probabilities as the other
registered PRMs.  The shared evaluator therefore keeps identical aggregation,
selection, checkpoint, and output semantics across all PRM choices.
"""

from __future__ import annotations
from ..prm_recovery import model_load_kwargs, parameter_storage

from ..prm_prefetch import PreparedRowsScorer, candidate_token_ids

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:  # Keep registry imports light for non-PRM users.
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
    """Build the backend-neutral RLHFlow input.

    ``marker``, ``messages`` and ``raw_prompt`` are accepted because all PRM
    backends share this builder signature.  RLHFlow's official protocol uses
    the task text as the first user message, so the backend intentionally does
    not inject the generator's chat history or the Qwen ``<extra_0>`` marker.
    """

    del marker, messages, raw_prompt
    steps = _split_steps(prefix, separator)
    if not steps:
        raise ValueError("PRM prefix is empty")
    return {
        "problem": "" if problem is None else str(problem),
        "steps": steps,
        "phase_ends": [int(end) for end in phase_ends]
        if phase_ends is not None
        else None,
    }


def input_length(value: Any) -> int:
    if not isinstance(value, dict):
        return max(1, len(str(value)))
    steps = value.get("steps") or []
    return max(
        1,
        len(str(value.get("problem", "")))
        + sum(len(str(step)) for step in steps),
    )


class RLHFlowDeepseekPRMScorer(PreparedRowsScorer):
    """Scorer implementing RLHFlow's official ``+``/``-`` protocol."""

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
            raise ModuleNotFoundError("RLHFlow PRM scoring requires PyTorch")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # Match the reference implementation. Right padding leaves each
        # unpadded prefix's absolute logits position unchanged in a batch.
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )

        self.plus_id = self._single_token_id("+")
        self.minus_id = self._single_token_id("-")
        if self.plus_id == self.minus_id:
            raise RuntimeError("RLHFlow '+' and '-' tokens must be distinct")

        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            try:
                kwargs["torch_dtype"] = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"Unknown torch dtype: {dtype!r}") from exc
        if device == "auto" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        kwargs = model_load_kwargs(kwargs)

        # This is the exact model family documented by RLHFlow: a causal LM,
        # not AutoModelForSequenceClassification or AutoModelForTokenClassification.
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        self.device = self._resolve_device(device)
        if "device_map" not in kwargs and self.device.type != "cpu":
            self.model.to(self.device)
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", None)
        if vocab_size is not None and max(self.plus_id, self.minus_id) >= int(vocab_size):
            raise RuntimeError(
                f"RLHFlow candidate token ids ({self.plus_id}, {self.minus_id}) "
                f"exceed model vocab size {vocab_size}"
            )
        print(
            f"[INFO] RLHFlow PRM model: {model_path} "
            f"({self.model.__class__.__name__}, device={self.device}, "
            f"candidate_tokens=({self.plus_id}, {self.minus_id}))"
        )

    def _single_token_id(self, token: str) -> int:
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"RLHFlow tokenizer must encode {token!r} as one token; got {ids!r}"
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
        encoded = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=False
        )
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().reshape(-1).tolist()
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(value) for value in encoded]

    def _prepare_row(self, value: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        problem = "" if value.get("problem") is None else str(value.get("problem"))
        steps = value.get("steps") or []
        raw_phase_ends = value.get("phase_ends")
        phase_ends = (
            [int(index) for index in raw_phase_ends]
            if raw_phase_ends is not None
            else list(range(1, len(steps) + 1))
        )
        if not steps or not phase_ends:
            raise ValueError("RLHFlow PRM input must contain non-empty steps")
        if any(index < 1 or index > len(steps) for index in phase_ends):
            raise ValueError(f"phase_ends {phase_ends!r} outside 1..{len(steps)}")

        messages: List[Dict[str, str]] = []
        score_positions: Dict[int, int] = {}
        requested_ends = set(phase_ends)
        full_ids: Optional[List[int]] = None
        for step_index, raw_step in enumerate(steps):
            step = str(raw_step).strip()
            if not step:
                raise ValueError(f"RLHFlow PRM step {step_index + 1} is empty")
            user_text = f"{problem} {step}" if step_index == 0 else step
            messages.append({"content": user_text, "role": "user"})
            messages.append({"content": "+", "role": "assistant"})
            if step_index + 1 not in requested_ends and step_index + 1 != len(steps):
                continue
            prefix_ids = self._chat_token_ids(messages)
            # The official scorer reads logits[:, -3, candidate_tokens].  In a
            # Llama chat template the final two tokens are '+' and <|eot_id|>.
            if len(prefix_ids) < 3 or prefix_ids[-2] != self.plus_id:
                raise RuntimeError(
                    "Unexpected RLHFlow Llama chat-template layout: '+' is not "
                    f"at -2 (last ids={prefix_ids[-6:]!r})"
                )
            score_positions[step_index + 1] = len(prefix_ids) - 3
            full_ids = prefix_ids

        assert full_ids is not None
        return full_ids, [score_positions[index] for index in phase_ends]

    def _marker_pair_logits(
        self, input_ids: torch.Tensor, positions: Sequence[int]
    ) -> torch.Tensor:
        """Compute only ``+``/``-`` logits at the requested marker positions."""

        backbone = getattr(self.model, "model", None)
        output_head = self.model.get_output_embeddings()
        if backbone is None or output_head is None:
            raise RuntimeError(
                "RLHFlow PRM model must expose a transformer body and output "
                "head to compute marker-only logits"
            )
        try:
            body_output = backbone(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
        except TypeError:
            body_output = backbone(input_ids, use_cache=False)
        hidden_states = getattr(body_output, "last_hidden_state", None)
        if hidden_states is None and isinstance(body_output, (tuple, list)):
            hidden_states = body_output[0]
        if hidden_states is None or hidden_states.ndim != 3:
            raise RuntimeError(
                "RLHFlow transformer body returned no [batch, seq, hidden] states"
            )
        position_tensor = torch.as_tensor(
            positions, dtype=torch.long, device=hidden_states.device
        )
        selected_hidden = hidden_states[0].index_select(0, position_tensor)
        del body_output, hidden_states, position_tensor

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
        return pair_logits

    def _prepare_one(self, value: Any):
        if not isinstance(value, dict):
            raise TypeError("RLHFlow PRM input must be a dict")
        input_ids, positions = self._prepare_row(value)
        if len(input_ids) > self.max_length:
            raise RuntimeError(
                "RLHFlow input exceeds --prm_max_length without silent truncation; "
                f"token_length={len(input_ids)}, limit={self.max_length}"
            )

        return input_ids, positions

    def _score_prepared_tensor(self, prepared):
        input_ids, positions = prepared
        input_tensor = torch.as_tensor(
            [input_ids], dtype=torch.long, device=self.device
        )
        pair_logits = self._marker_pair_logits(input_tensor, positions)
        # Candidate order is '+', '-', so column 0 is P('+').
        scores = torch.softmax(pair_logits.float(), dim=-1)[:, 0]
        result = scores.detach()
        del pair_logits, input_tensor
        return result



def scorer_class():
    return RLHFlowDeepseekPRMScorer


def create_scorer(*args: Any, **kwargs: Any) -> RLHFlowDeepseekPRMScorer:
    return RLHFlowDeepseekPRMScorer(*args, **kwargs)
