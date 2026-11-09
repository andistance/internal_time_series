from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from .models import BestOfNRequest


Renderer = Callable[[BestOfNRequest, Any, bool | None], str]


@dataclass(frozen=True)
class ChatTemplateSpec:
    renderer: Renderer
    fallback_add_special_tokens: bool = False


_REGISTRY: Dict[str, ChatTemplateSpec] = {}


def register_chat_template(
    name: str,
    renderer: Renderer,
    *,
    fallback_add_special_tokens: bool = False,
) -> None:
    if not name:
        raise ValueError("Chat template name must not be empty")
    _REGISTRY[name] = ChatTemplateSpec(
        renderer=renderer,
        fallback_add_special_tokens=fallback_add_special_tokens,
    )


def is_chat_template_registered(name: str) -> bool:
    return name in _REGISTRY


def _qwen3_renderer(
    request: BestOfNRequest, tokenizer: Any, enable_thinking: bool | None
) -> str:
    if request.messages is None:
        raise ValueError("qwen3 template requires request.messages")
    if enable_thinking is None:
        raise ValueError("qwen3 template requires enable_thinking to be set")
    try:
        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(enable_thinking),
        )
    except TypeError as error:
        raise RuntimeError(
            "The Qwen tokenizer/chat template does not support enable_thinking"
        ) from error
    return prompt


def _deepseek_renderer(
    request: BestOfNRequest, tokenizer: Any, enable_thinking: bool | None
) -> str:
    if request.messages is None:
        raise ValueError("deepseek_r1_distill_qwen requires request.messages")
    if enable_thinking is not None:
        raise ValueError("DeepSeek-R1-Distill-Qwen does not accept enable_thinking")
    return tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _raw_renderer(
    request: BestOfNRequest, tokenizer: Any, enable_thinking: bool | None
) -> str:
    del tokenizer
    if request.raw_prompt is None:
        raise ValueError("raw template requires request.raw_prompt")
    if enable_thinking is not None:
        raise ValueError("raw template does not accept enable_thinking")
    return request.raw_prompt


register_chat_template("qwen3", _qwen3_renderer)
register_chat_template("deepseek_r1_distill_qwen", _deepseek_renderer)
register_chat_template("raw", _raw_renderer, fallback_add_special_tokens=True)
CHAT_TEMPLATES = tuple(_REGISTRY)


def render_prompt(
    request: BestOfNRequest,
    tokenizer: Any,
    template_name: str,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Render a normalized request without embedding benchmark prompt logic."""

    try:
        spec = _REGISTRY[template_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown chat template {template_name!r}; registered: {sorted(_REGISTRY)}"
        ) from error
    prompt = spec.renderer(request, tokenizer, enable_thinking)
    if not isinstance(prompt, str):
        raise TypeError("A chat-template renderer must return str")
    return prompt


def prompt_add_special_tokens(template_name: str) -> bool:
    """Whether fallback tokenizer.encode should inject model special tokens."""

    try:
        return _REGISTRY[template_name].fallback_add_special_tokens
    except KeyError as error:
        raise ValueError(f"Unknown chat template: {template_name}") from error
