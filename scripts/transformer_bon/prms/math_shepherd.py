"""Math-Shepherd Mistral-7B process-reward backend.

This backend follows the inference procedure documented for
``peiyi9979/math-shepherd-mistral-7b-prm``: append the ``ки`` step tag after
every solution step, run ``AutoModelForCausalLM``, and convert the logits of
the ``+``/``-`` candidate tokens at each tag position into a probability for
``+``. The returned values therefore have the same ``[0, 1]`` phase-score
semantics as the other registered PRMs.
"""

from __future__ import annotations
from ..prm_recovery import model_load_kwargs, parameter_storage

from ..prm_prefetch import PreparedRowsScorer, candidate_token_ids

from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep registry imports light for non-PRM users.
    torch = None  # type: ignore[assignment]


STEP_TAG = "ки"


def _split_steps(prefix: Any, separator: str) -> List[str]:
    text = "" if prefix is None else str(prefix)
    return [part.strip() for part in text.split(separator) if part.strip()]


def _prompt_text(
    problem: Any,
    messages: Optional[Sequence[Dict[str, str]]],
    raw_prompt: Optional[str],
) -> str:
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
    """Build the backend-neutral Math-Shepherd input.

    ``marker`` is intentionally ignored: the shared Best-of-N config defaults
    to Qwen's ``<extra_0>`` marker, but Math-Shepherd has a fixed ``ки`` tag.
    """

    del marker
    steps = _split_steps(prefix, separator)
    if not steps:
        raise ValueError("PRM prefix is empty")
    assistant = f" {STEP_TAG}{separator}".join(steps) + f" {STEP_TAG}"
    return {
        "problem": _prompt_text(problem, messages, raw_prompt),
        "assistant": assistant,
        "phase_ends": [int(end) for end in phase_ends]
        if phase_ends is not None
        else None,
    }


def input_length(value: Any) -> int:
    if not isinstance(value, dict):
        return max(1, len(str(value)))
    return max(
        1,
        len(str(value.get("problem", "")))
        + len(str(value.get("assistant", ""))),
    )


class MathShepherdPRMScorer(PreparedRowsScorer):
    """Scorer implementing the official Math-Shepherd model-card protocol."""

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
        # Kept for the shared PRM constructor signature. Math-Shepherd uses
        # fixed ``+``/``-`` candidates and a plain-text prompt, so generic
        # positive-token/system-prompt options do not alter its protocol.
        del positive_token, system_prompt
        if torch is None:
            raise ModuleNotFoundError("Math-Shepherd PRM scoring requires PyTorch")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        # Match the model card. Most Mistral tokenizers prepend BOS, hence the
        # [1:] in the published snippet; the fallback handles no-BOS configs.
        candidate_tokens = self.tokenizer.encode("+ -")[1:]
        if len(candidate_tokens) != 2:
            candidate_tokens = self.tokenizer.encode(
                "+ -", add_special_tokens=False
            )
        if len(candidate_tokens) != 2:
            raise RuntimeError(
                "Math-Shepherd tokenizer must encode '+ -' as two tokens; "
                f"got {candidate_tokens!r}."
            )
        self.candidate_tokens = tuple(int(token) for token in candidate_tokens)
        step_tag_tokens = self.tokenizer.encode(STEP_TAG)
        if not step_tag_tokens:
            raise RuntimeError("Math-Shepherd tokenizer cannot encode the 'ки' step tag")
        self.step_tag_id = int(step_tag_tokens[-1])

        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        if device == "auto" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        kwargs = model_load_kwargs(kwargs)
        # from_pretrained handles both safetensors and the official PyTorch
        # .bin / pytorch_model.bin.index.json checkpoint layout.
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        self.device = self._resolve_device(device)
        if "device_map" not in kwargs and self.device.type != "cpu":
            self.model.to(self.device)
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", None)
        if vocab_size is not None and max(self.candidate_tokens) >= int(vocab_size):
            raise RuntimeError(
                f"Math-Shepherd candidate token ids {self.candidate_tokens} "
                f"exceed vocab size {vocab_size}"
            )
        print(
            f"[INFO] Math-Shepherd PRM model: {model_path} "
            f"({self.model.__class__.__name__}, device={self.device}, "
            f"candidate_tokens={self.candidate_tokens}, "
            f"step_tag_id={self.step_tag_id})"
        )

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
            embedding = self.model.get_input_embeddings()
            return embedding.weight.device
        except (AttributeError, RuntimeError):
            pass
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _marker_pair_logits(
        self, input_ids: torch.Tensor, marker_positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute only ``+``/``-`` logits at the ``ки`` marker positions."""

        backbone = getattr(self.model, "model", None)
        output_head = self.model.get_output_embeddings()
        if backbone is None or output_head is None:
            raise RuntimeError(
                "Math-Shepherd model must expose a transformer body and output "
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
                "Math-Shepherd transformer body returned no "
                "[batch, seq, hidden] states"
            )
        selected_hidden = hidden_states[0].index_select(
            0, marker_positions.to(hidden_states.device)
        )
        del body_output, hidden_states

        ids = self.candidate_tokens
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
            raise TypeError("Math-Shepherd PRM input must be a dict")
        problem = "" if value.get("problem") is None else str(value.get("problem"))
        assistant = "" if value.get("assistant") is None else str(value.get("assistant"))
        if not assistant:
            raise ValueError("Math-Shepherd PRM input has no assistant solution")

        # This is the exact model-card construction: question + space + tagged
        # solution. Encode without silent truncation so phase indices cannot be
        # associated with the wrong visible step.
        input_text = problem + " " + assistant
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError("Math-Shepherd tokenizer must return [1, seq_len] input_ids")
        if input_ids.shape[1] > self.max_length:
            raise RuntimeError(
                f"Math-Shepherd input has {input_ids.shape[1]} tokens, exceeding "
                f"--prm_max_length={self.max_length}; increase the limit."
            )
        marker_positions = input_ids[0].eq(self.step_tag_id).nonzero(
            as_tuple=False
        ).flatten()
        if marker_positions.numel() == 0:
            raise RuntimeError(
                "Math-Shepherd step tag 'ки' was not found; check the tokenizer "
                "and tagged input construction"
            )
        return input_ids, marker_positions, value.get("phase_ends")

    def _score_prepared_tensor(self, prepared):
        input_ids, marker_positions, phase_ends = prepared
        input_ids = input_ids.to(self.device)
        pair_logits = self._marker_pair_logits(input_ids, marker_positions)
        # candidate_tokens[0] is '+', so this is P('+ | step prefix) in [0, 1].
        all_step_scores = torch.softmax(pair_logits.float(), dim=-1)[:, 0]
        all_step_scores = all_step_scores.detach()

        if phase_ends is None:
            selected_scores = all_step_scores
        else:
            indices = [int(end) - 1 for end in phase_ends]
            if any(index < 0 or index >= len(all_step_scores) for index in indices):
                raise RuntimeError(
                    "Math-Shepherd phase end exceeds the number of visible step "
                    "scores; increase --prm_max_length"
                )
            selected_scores = all_step_scores[indices]

        del marker_positions, pair_logits, input_ids
        return selected_scores.detach()



def scorer_class():
    return MathShepherdPRMScorer


def create_scorer(*args: Any, **kwargs: Any) -> MathShepherdPRMScorer:
    return MathShepherdPRMScorer(*args, **kwargs)
