"""Bounded length-only tokenization for the six standalone phase evaluators."""

from __future__ import annotations

from .standalone_prm import InputLengthError


def _native_tokenizer(tokenizer):
    from transformers import PreTrainedTokenizerBase

    # Custom chat-template implementations may use different encoding options.
    return (isinstance(tokenizer, PreTrainedTokenizerBase) and
            type(tokenizer).apply_chat_template is PreTrainedTokenizerBase.apply_chat_template)


def _render(scorer, value, backend):
    tokenizer = scorer.tokenizer
    if backend == 'qwen':
        return scorer._render_input(value), True
    if not isinstance(value, dict):
        raise TypeError('PRM input must be a structured rollout record')
    problem = '' if value.get('problem') is None else str(value['problem'])
    if backend == 'math':
        assistant = '' if value.get('assistant') is None else str(value['assistant'])
        if not assistant:
            raise ValueError('Math-Shepherd PRM input has an empty assistant solution')
        return problem + ' ' + assistant, True
    if backend == 'versa':
        steps = value.get('solution_steps')
        if not isinstance(steps, list) or not steps:
            raise ValueError('VersaPRM input has no solution_steps')
        return problem + ' \n\n' + ' \n\n\n\n'.join(str(step) for step in steps) + ' \n\n\n\n', True
    if backend == 'fover':
        steps = value.get('solution_steps')
        if not isinstance(steps, list) or not steps:
            raise ValueError('FoVer PRM input has no solution_steps')
        conversation = scorer._fover_input_format(problem, [str(step) for step in steps])
        # _tokenize_conversation uses apply_chat_template's default kwargs.
        return tokenizer.apply_chat_template(conversation, tokenize=False), False
    steps = value.get('steps') or []
    ends = [int(end) for end in (value.get('phase_ends') or [])]
    if not steps or not ends or any(end < 1 or end > len(steps) for end in ends):
        raise ValueError('RLHFlow PRM input must contain valid steps and phase_ends')
    conversation = []
    for index, raw_step in enumerate(steps):
        step = str(raw_step).strip()
        if not step:
            raise ValueError('RLHFlow PRM step is empty')
        conversation.extend([{'content': f'{problem} {step}' if index == 0 else step, 'role': 'user'},
                             {'content': '+', 'role': 'assistant'}])
    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False), False


def _lengths(scorer, batch, backend):
    rendered = [_render(scorer, job[2], backend) for job in batch]
    encoded = scorer.tokenizer([row[0] for row in rendered], add_special_tokens=rendered[0][1],
                               padding=False, truncation=False, return_attention_mask=False,
                               return_token_type_ids=False, return_length=True)
    if backend in ('deepseek', 'mistral'):
        if any(len(ids) < 3 or ids[-2] != scorer.plus_id for ids in encoded['input_ids']):
            raise ValueError('Unexpected RLHFlow chat-template layout')
    return [int(length) for length in encoded['length']]


def preflight(scorer, jobs, backend):
    """Preserve ordered errors and exact longest-row diagnostics without padding."""
    native = hasattr(scorer, 'tokenizer') and _native_tokenizer(scorer.tokenizer)
    if native and backend == 'fover':
        import inspect
        option = inspect.signature(scorer.tokenizer.apply_chat_template).parameters.get('return_dict')
        # New Transformers APIs can return a mapping by default; the original
        # FoVer validator requires a tensor. Preserve that compatibility error.
        if option is not None and option.default is True:
            native = False
    longest = None

    def reference(batch):
        nonlocal longest
        for key, _, value in batch:
            try:
                scorer.validate_input_lengths([value])
            except InputLengthError as exc:
                if longest is None or exc.token_length > longest[0]:
                    longest = (exc.token_length, key, value, str(exc))
            except Exception as exc:
                raise RuntimeError(f'PRM input preflight failed for {key}: {exc}') from exc

    def check(batch):
        nonlocal longest
        if not native:
            reference(batch)
            return
        try:
            lengths = _lengths(scorer, batch, backend)
            if len(lengths) != len(batch):
                raise ValueError('Invalid tokenizer length count')
        except Exception:
            # Preserve the reference exception text and first failing key.
            reference(batch)
            return
        for (key, _, value), length in zip(batch, lengths):
            if length > scorer.max_length and (longest is None or length > longest[0]):
                longest = (length, key, value, None)

    batch, size = [], 0
    for job in jobs:
        # Bound both row count and input characters. One long row is atomic.
        value_size = len(str(job[2]))
        if batch and (len(batch) >= 64 or size + value_size > 256 * 1024):
            check(batch)
            batch, size = [], 0
        batch.append(job)
        size += value_size
    if batch:
        check(batch)
    if longest is not None:
        length, key, value, message = longest
        if message is None:
            try:
                scorer.validate_input_lengths([value])
            except InputLengthError as exc:
                message = str(exc)
            except Exception as exc:
                raise RuntimeError(f'PRM input preflight failed for {key}: {exc}') from exc
            else:
                raise RuntimeError('PRM tokenizer length preflight disagrees with reference validation')
        raise InputLengthError(
            f'PRM input preflight failed for {key}: {message}; max_token_length={length}', length)
