"""UW-Madison-Lee-Lab/VersaPRM backend.

VersaPRM is a LoRA/PEFT adapter over ``UW-Madison-Lee-Lab/Llama-PRM800K``.
Its model-card protocol is different from Qwen2.5-Math-PRM: the question and
solution steps are serialized as plain text, and the probability of candidate
token 10 versus 12 is read at token id 23535, the four-newline step marker.
"""

from __future__ import annotations
from ..prm_recovery import model_load_kwargs, parameter_storage

from ..prm_prefetch import PreparedRowsScorer, candidate_token_ids

from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
except ModuleNotFoundError:  # Keep package import light for transformer-only runs.
    torch = None  # type: ignore[assignment]


BASE_MODEL_FALLBACK = "UW-Madison-Lee-Lab/Llama-PRM800K"
CANDIDATE_TOKEN_IDS = (12, 10)
STEP_SEPARATOR_TOKEN_ID = 23535
STEP_SEPARATOR_TEXT = " \n\n\n\n"


def encode_solution_steps(tokenizer, problem, steps, max_length):
    """Record inserted boundaries, excluding lookalike markers inside user text.

    Preserve the original whole-text encoding when its boundary markers can be
    located unambiguously. Otherwise encode each boundary separately so a BPE
    whitespace merge cannot swallow the marker or merge it with the next step.
    No text is stripped, no scores are invented, and no input is truncated.
    """
    if not steps:
        raise ValueError('VersaPRM input has no solution_steps')
    chunks = [str(problem) + ' \n\n']
    boundaries = []
    cursor = len(chunks[0])
    for step in steps:
        step = str(step)
        chunks.extend((step, STEP_SEPARATOR_TEXT))
        cursor += len(step)
        boundaries.append((cursor, cursor + len(STEP_SEPARATOR_TEXT)))
        cursor += len(STEP_SEPARATOR_TEXT)
    text = ''.join(chunks)
    ids = list(tokenizer.encode(text))
    if len(ids) > max_length:
        raise RuntimeError(
            'VersaPRM input exceeds configured maximum length without silent truncation; '
            f'token_length={len(ids)}, limit={max_length}')

    positions = None
    try:
        encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        if list(encoded['input_ids']) == ids and len(offsets) == len(ids):
            located = []
            for left, right in boundaries:
                matches = [i for i, (token, (start, end)) in enumerate(zip(ids, offsets))
                           if token == STEP_SEPARATOR_TOKEN_ID and left <= start < end <= right]
                if len(matches) != 1:
                    break
                located.append(matches[0])
            if len(located) == len(steps):
                positions = located
    except (TypeError, KeyError, ValueError, AttributeError, NotImplementedError):
        # Slow/custom tokenizers may not expose character offsets.
        pass

    if positions is None:
        separator_ids = list(tokenizer.encode(STEP_SEPARATOR_TEXT, add_special_tokens=False))
        marker_offsets = [i for i, token in enumerate(separator_ids) if token == STEP_SEPARATOR_TOKEN_ID]
        if len(marker_offsets) != 1:
            raise RuntimeError(
                'VersaPRM tokenizer is incompatible with the expected step separator: '
                f'expected token {STEP_SEPARATOR_TOKEN_ID} exactly once, got {separator_ids}. '
                'Check that the tokenizer belongs to the VersaPRM checkpoint.')
        # BOS/special tokens appear only in the first chunk, matching encode(text).
        ids = list(tokenizer.encode(chunks[0] + str(steps[0])))
        positions = []
        for index in range(len(steps)):
            positions.append(len(ids) + marker_offsets[0])
            ids.extend(separator_ids)
            if index + 1 < len(steps):
                ids.extend(tokenizer.encode(str(steps[index + 1]), add_special_tokens=False))
        if len(ids) > max_length:
            raise RuntimeError(
                'VersaPRM boundary-preserving input exceeds configured maximum length '
                f'without silent truncation; token_length={len(ids)}, limit={max_length}')
    if len(positions) != len(steps) or any(ids[i] != STEP_SEPARATOR_TOKEN_ID for i in positions):
        raise RuntimeError('VersaPRM failed to align one scoring position per solution step')
    return ids, positions


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
    """Build the backend-neutral representation of a VersaPRM prefix.

    VersaPRM consumes a plain question string, not a chat template. When the
    benchmark supplies an exact raw prompt or semantic messages, preserve that
    prompt context so GPQA choices, ICPC history, and code-generation
    instructions are not silently dropped.
    """

    del marker
    steps = _split_steps(prefix, separator)
    if not steps:
        raise ValueError("PRM prefix is empty")
    if raw_prompt is not None:
        problem_text = str(raw_prompt)
    elif messages:
        problem_text = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") != "assistant"
        )
    else:
        problem_text = "" if problem is None else str(problem)
    return {
        "problem": problem_text,
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


class VersaPRMScorer(PreparedRowsScorer):
    """PEFT-backed causal-LM scorer implementing the official Versa protocol."""

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
            raise ModuleNotFoundError("VersaPRM scoring requires PyTorch")
        try:
            from peft import PeftConfig, PeftModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VersaPRM is a PEFT adapter and requires the `peft` package"
            ) from exc
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        # Match the model-card tokenizer setup. Scores are computed one row at
        # a time, so left padding never changes the marker position.
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        try:
            peft_config = PeftConfig.from_pretrained(model_path)
        except Exception as exc:
            raise RuntimeError(
                "VersaPRM must point to a PEFT adapter containing adapter_config.json"
            ) from exc
        base_model_path = (
            getattr(peft_config, "base_model_name_or_path", None)
            or BASE_MODEL_FALLBACK
        )

        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        if device == "auto" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        kwargs = model_load_kwargs(kwargs)

        # The VersaPRM repository stores adapter weights. Load its declared
        # base model first, then attach the adapter exactly as documented.
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        self.device = self._resolve_device(device)
        if "device_map" not in kwargs and self.device.type != "cpu":
            self.model.to(self.device)
        print(
            f"[INFO] VersaPRM model: {model_path} (base={base_model_path}, "
            f"device={self.device})"
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

    def _marker_pair_logits(
        self, input_ids: torch.Tensor, marker_positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute only candidate logits at Versa's step-marker positions."""

        output_head = self.model.get_output_embeddings()
        base_model = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        backbone = getattr(base_model, "model", None)
        if backbone is None or output_head is None:
            raise RuntimeError(
                "VersaPRM model must expose a transformer body and output head "
                "to compute marker-only logits"
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
                "VersaPRM transformer body returned no [batch, seq, hidden] states"
            )
        selected_hidden = hidden_states[0].index_select(
            0, marker_positions.to(hidden_states.device)
        )
        del body_output, hidden_states

        ids = CANDIDATE_TOKEN_IDS
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
            raise TypeError("VersaPRM input must be a dict")
        problem = "" if value.get("problem") is None else str(value.get("problem"))
        raw_steps = value.get("solution_steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("VersaPRM input has no solution_steps")
        steps = [str(step) for step in raw_steps]

        phase_ends = value.get("phase_ends")
        if phase_ends is not None and any(int(end) < 1 or int(end) > len(steps) for end in phase_ends):
            raise ValueError(f'VersaPRM phase_ends {phase_ends!r} outside 1..{len(steps)}')
        ids, positions = encode_solution_steps(self.tokenizer, problem, steps, self.max_length)
        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
            device="cpu",
        )
        marker_positions = torch.tensor(positions, dtype=torch.long, device="cpu")
        return input_ids, marker_positions, phase_ends

    def _score_prepared_tensor(self, prepared):
        input_ids, marker_positions, phase_ends = prepared
        input_ids = input_ids.to(self.device)
        pair_logits = self._marker_pair_logits(input_ids, marker_positions)
        step_scores = torch.softmax(pair_logits.float(), dim=-1)[:, 1]
        if step_scores.numel() == 0:
            raise RuntimeError("VersaPRM returned no step scores")
        all_step_scores = step_scores.detach().float()
        if phase_ends is None:
            selected_scores = all_step_scores
        else:
            indices = [int(end) - 1 for end in phase_ends]
            if any(index < 0 or index >= len(all_step_scores) for index in indices):
                raise RuntimeError(
                    "VersaPRM phase end exceeds aligned step scores; "
                    f"phase_ends={phase_ends!r}, num_step_scores={len(all_step_scores)}"
                )
            selected_scores = all_step_scores[indices]
        del marker_positions, pair_logits, step_scores, input_ids
        return selected_scores.detach()



def scorer_class():
    return VersaPRMScorer


def create_scorer(*args: Any, **kwargs: Any) -> VersaPRMScorer:
    return VersaPRMScorer(*args, **kwargs)
