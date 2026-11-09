#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Teacher-forcing multi-signal extraction for already generated Qwen3 CoTs.

The script never samples new text.  It rebuilds the original Qwen3 prompt,
feeds the saved response tokens through Transformers, and writes token-level
logit/probability/entropy/</think>-attention signals plus step-level files.
"""
import argparse, copy, gc, json, math, multiprocessing as mp, os, tempfile, time, types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The workload contains highly variable sequence lengths.  Expandable CUDA
# segments substantially reduce allocator fragmentation between samples.
# This is set before worker processes import torch.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Query rows are computed a chunk at a time instead of materialising a full
# [batch, heads, query_length, key_length] attention tensor.  A small fixed
# chunk keeps the peak proportional to sequence length rather than its square.
ATTENTION_QUERY_CHUNK_SIZE = 64
LOGIT_CHUNK_SIZE = 32
PROMPT_SUFFIX = "\nLet's reason step by step. Enclose the reasoning process within <think>...</think>, then summarize it and present the final answer within \\boxed{}."

def parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).lower().strip()
    if s in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'invalid boolean: {x}')

def build_prompt(problem, tokenizer, enable_thinking):
    messages = [{'role': 'user', 'content': problem + PROMPT_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError as e:
        raise RuntimeError('Qwen3 tokenizer/chat template must support enable_thinking') from e

def read_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f'bad JSON at {path}:{n}') from e
    return out

def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def dtype(name, torch):
    m = {'auto': 'auto', 'float16': torch.float16, 'half': torch.float16, 'fp16': torch.float16, 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16, 'float32': torch.float32, 'fp32': torch.float32, 'float': torch.float32}
    if str(name).lower() not in m:
        raise ValueError('unsupported dtype')
    return m[str(name).lower()]

def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM
    kw = dict(trust_remote_code=args.trust_remote_code, torch_dtype=dtype(args.dtype, torch), low_cpu_mem_usage=True, attn_implementation='eager')
    if args.tp_size > 1:
        try:
            kw['tp_plan'] = 'auto'
            return AutoModelForCausalLM.from_pretrained(args.model_path, **kw).eval()
        except Exception:
            kw.pop('attn_implementation', None)
        kw.pop('tp_plan', None)
        kw['device_map'] = 'auto'
    else:
        kw['device_map'] = {'': 0} if torch.cuda.is_available() else None
    try:
        m = AutoModelForCausalLM.from_pretrained(args.model_path, **kw).eval()
    except TypeError:
        kw.pop('attn_implementation', None)
        m = AutoModelForCausalLM.from_pretrained(args.model_path, **kw).eval()
    return m

def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device

def _backbone_and_lm_head(model):
    """Return the transformer body and output projection without using CausalLM.forward.

    Calling the CausalLM wrapper normally creates logits for every input
    position.  We only need the positions that predict the saved response, so
    the body and lm_head are split and the projection is applied in chunks.
    """
    candidates = []
    for name in ('model', 'transformer', 'base_model'):
        candidate = getattr(model, name, None)
        if candidate is not None and candidate is not model and hasattr(candidate, 'forward'):
            candidates.append(candidate)
    backbone = candidates[0] if candidates else None
    head = None
    try:
        head = model.get_output_embeddings()
    except Exception:
        head = None
    if head is None:
        head = getattr(model, 'lm_head', None)
    if backbone is None or head is None:
        raise RuntimeError('model must expose a transformer backbone and output embeddings')
    return backbone, head

def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None

def _install_chunked_attention(model, think_pos, attention_rows):
    """Patch Qwen3 self-attention to retain only the requested query row.

    The old forward hook ran *after* eager attention had already allocated the
    complete attention matrix.  Qwen3 exposes the projections and rotary
    embedding directly, so we can reproduce its eager attention in query
    chunks, capture the exact requested row, and return the same attention
    output without retaining the matrix.  The returned list is empty when the
    installed model does not expose the Qwen3 attention pieces; callers then
    use the compatibility hook from ``extract_one``.
    """
    import importlib
    import torch

    patched = []
    for name, module in model.named_modules():
        if not name.endswith('self_attn'):
            continue
        required = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm')
        if not all(hasattr(module, x) for x in required) or not hasattr(module, 'head_dim'):
            continue
        try:
            source = importlib.import_module(module.__class__.__module__)
            apply_rope = getattr(source, 'apply_rotary_pos_emb')
            repeat_kv = getattr(source, 'repeat_kv')
        except Exception:
            continue
        original = module.forward

        def chunked_forward(
            self,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            past_key_value=None,
            _original=original,
            _apply_rope=apply_rope,
            _repeat_kv=repeat_kv,
            **kwargs,
        ):
            # The normal call in this script always supplies position
            # embeddings and disables caching.  Fall back to the original
            # implementation for an incompatible model invocation.
            if position_embeddings is None or past_key_values is not None or past_key_value is not None:
                return _original(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    past_key_value=past_key_value,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = _apply_rope(query_states, key_states, cos, sin)

            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)
            q_len = query_states.shape[-2]
            k_len = key_states.shape[-2]
            # Preallocate the compact attention output.  This avoids retaining
            # all per-chunk outputs in a Python list before concatenation.
            attn_output = torch.empty_like(query_states)
            scale = getattr(self, 'scaling', self.head_dim ** -0.5)
            for q_start in range(0, q_len, ATTENTION_QUERY_CHUNK_SIZE):
                q_end = min(q_start + ATTENTION_QUERY_CHUNK_SIZE, q_len)
                q = query_states[:, :, q_start:q_end, :]
                weights = torch.matmul(q, key_states.transpose(2, 3)) * scale
                if attention_mask is not None:
                    mask = attention_mask
                    if torch.is_tensor(mask) and mask.ndim >= 3:
                        # Standard Qwen3 masks are [B, 1, Q, K].  Handle a
                        # broadcasted single-query mask as well.
                        if mask.shape[-2] == q_len:
                            mask = mask[..., q_start:q_end, :k_len]
                        elif mask.shape[-2] == 1:
                            mask = mask[..., :, :k_len]
                        else:
                            mask = mask[..., :q_end - q_start, :k_len]
                    weights = weights + mask
                weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                if think_pos is not None and q_start <= think_pos < q_end:
                    row = weights[0, :, think_pos - q_start, :].detach().to('cpu')
                    attention_rows.append(row)
                attn_output[:, :, q_start:q_end, :] = torch.matmul(weights, value_states)
                del q, weights
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            return self.o_proj(attn_output), None

        module.forward = types.MethodType(chunked_forward, module)
        patched.append((module, original))
    return patched

def _restore_chunked_attention(patched):
    for module, original in patched:
        module.forward = original

def _forward_signals_in_chunks(model, inp, prompt_len, response_ids, args, attention_rows, use_compat_hooks):
    """Run the transformer once, then project only response-predicting states."""
    import torch

    backbone, lm_head = _backbone_and_lm_head(model)
    call_kwargs = dict(
        output_attentions=bool(use_compat_hooks),
        use_cache=False,
        return_dict=True,
    )
    base_out = backbone(inp, **call_kwargs)
    hidden = getattr(base_out, 'last_hidden_state', None)
    if hidden is None and isinstance(base_out, dict):
        hidden = base_out['last_hidden_state']
    if hidden is None:
        raise RuntimeError('transformer backbone did not return last_hidden_state')

    first = prompt_len - 1
    last = first + len(response_ids)
    head_device = _module_device(lm_head) or hidden.device
    token_logits, token_probs, token_entropies = [], [], []
    # Applying lm_head to a small block is the main protection against the
    # multi-gigabyte [sequence_length, vocab_size] logits allocation.
    for start in range(first, last, LOGIT_CHUNK_SIZE):
        stop = min(start + LOGIT_CHUNK_SIZE, last)
        h = hidden[:, start:stop, :]
        if h.device != head_device:
            h = h.to(head_device)
        chunk_logits = lm_head(h)[0]
        lgs, prs, ens = sampling_signals_from_logits(
            chunk_logits,
            response_ids[start - first:stop - first],
            args.temperature,
            args.top_p,
            args.top_k,
        )
        token_logits.extend(lgs)
        token_probs.extend(prs)
        token_entropies.extend(ens)
        del h, chunk_logits, lgs, prs, ens
    del hidden, base_out
    return token_logits, token_probs, token_entropies

def step_spans(text, sep='\n\n'):
    spans = []
    cur = 0
    for part in text.split(sep):
        rs, re = (cur, cur + len(part))
        s = rs + len(part) - len(part.lstrip())
        e = re - len(part) + len(part.rstrip())
        if s < e:
            spans.append((s, e))
        cur = re + len(sep)
    return spans

def token_pieces(ids, tok):
    pieces = []
    prev = ''
    plen = 0
    for i in range(len(ids)):
        t = tok.decode(ids[:i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        p = t[len(prev):]
        pieces.append((i, p, plen, plen + len(p)))
        prev = t
        plen += len(p)
    return (prev, pieces)

def sampling_signals_from_logits(logits, target_ids, temperature, top_p, top_k):
    """Score existing generated tokens under temperature/top-p sampling.

    This is teacher forcing: the sampling step selects the already generated
    token ID, and does not draw a replacement token. For each position, sort
    temperature-scaled logits, retain the smallest nucleus whose original
    softmax mass reaches top_p, then subtract logsumexp over that nucleus.
    Entropy is computed from the resulting log-probabilities; top_k > 0
    truncates the entropy sum without renormalizing those top-k probabilities.
    """
    import torch
    token_logits, token_probs, token_entropies = [], [], []
    for row, target_id in zip(logits, target_ids):
        sorted_logits, sorted_ids = torch.sort(row.float() / temperature, descending=True)
        original_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = original_probs.cumsum(dim=-1)
        cutoff = torch.searchsorted(cumulative, cumulative.new_tensor(top_p), right=False)
        keep_count = min(int(cutoff.item()) + 1, sorted_logits.numel())
        kept_logits = sorted_logits[:keep_count]
        kept_ids = sorted_ids[:keep_count]
        normalized_logits = kept_logits - torch.logsumexp(kept_logits, dim=-1)
        kept_probs = torch.softmax(normalized_logits, dim=-1)
        selected = (kept_ids == int(target_id)).nonzero(as_tuple=False)
        if selected.numel():
            target_index = int(selected[0, 0])
            token_logits.append(float(normalized_logits[target_index]))
            token_probs.append(float(kept_probs[target_index]))
        else:
            # The saved token is outside the nucleus: its filtered probability
            # is zero, and its normalized logit is undefined (-inf).
            token_logits.append(None)
            token_probs.append(0.0)
        entropy_count = min(top_k, keep_count) if top_k else keep_count
        entropy_logprobs = torch.log_softmax(normalized_logits, dim=-1)[:entropy_count]
        entropy = -(kept_probs[:entropy_count] * entropy_logprobs).sum()
        token_entropies.append(float(entropy))
    return token_logits, token_probs, token_entropies

def extract_one(rec, step_rec, tok, model, args):
    import torch
    problem = rec.get('problem', step_rec.get('problem', ''))
    text = rec.get('full_output')
    if text is None:
        text = '\n\n'.join((s['step'] for s in step_rec.get('steps', [])))
    prompt = build_prompt(problem, tok, args.enable_thinking)
    pids = tok(prompt, add_special_tokens=False).input_ids
    st_ids = tok.encode('<think>', add_special_tokens=False)
    en_ids = tok.encode('</think>', add_special_tokens=False)

    # Some upstream generations stop exactly at max_tokens before producing
    # the closing </think> marker.  This script does not sample new tokens;
    # append the marker to the saved response so the forced final token is
    # processed by the same teacher-forcing, attention, pruning, and step
    # aggregation logic as a normally generated marker.  The updated text is
    # stored in token_details as well, because the final-output pass reads it
    # back from that file.
    raw_rids = tok(text, add_special_tokens=False).input_ids
    has_end = bool(en_ids) and any(
        raw_rids[i:i + len(en_ids)] == en_ids
        for i in range(len(raw_rids) - len(en_ids) + 1)
    ) if en_ids else True
    if en_ids and not has_end:
        text = text + '</think>'

    rids = tok(text, add_special_tokens=False).input_ids
    if not rids:
        raise RuntimeError('empty response')
    dev = input_device(model)
    allids = pids + rids
    inp = torch.tensor([allids], dtype=torch.long, device=dev)
    start = next((i for i in range(len(rids) - len(st_ids) + 1) if rids[i:i + len(st_ids)] == st_ids), 0) if st_ids else 0
    end0 = next((i for i in range(start, len(rids) - len(en_ids) + 1) if rids[i:i + len(en_ids)] == en_ids), len(rids) - len(en_ids)) if en_ids else len(rids) - 1
    end = min(len(rids) - 1, end0 + len(en_ids) - 1)
    think_pos = None
    if en_ids:
        for i in range(len(pids) + start, len(allids) - len(en_ids) + 1):
            if allids[i:i + len(en_ids)] == en_ids:
                think_pos = i
                break
    # Capture only the </think> query row from each self-attention module.
    # Qwen3 modules use a chunked implementation that never materialises the
    # full attention matrix.  A forward-hook fallback is retained for older or
    # otherwise incompatible model implementations.
    attention_rows = []
    hooks = []
    def capture_think_row(_module, _inputs, output):
        if think_pos is None or not isinstance(output, (tuple, list)) or len(output) < 2:
            return output
        weights = output[1]
        if weights is None or not torch.is_tensor(weights) or weights.ndim != 4:
            return output
        if think_pos >= weights.shape[-2]:
            return output
        row = weights[0, :, think_pos, :].detach().to('cpu')
        attention_rows.append(row)
        if isinstance(output, tuple):
            return (output[0], None, *output[2:])
        updated = list(output)
        updated[1] = None
        return tuple(updated)
    patched = _install_chunked_attention(model, think_pos, attention_rows)
    patched_ids = {id(module) for module, _ in patched}
    for name, module in model.named_modules():
        if name.endswith('self_attn') and id(module) not in patched_ids:
            hooks.append(module.register_forward_hook(capture_think_row))
    try:
        with torch.inference_mode():
            token_logits, token_probs, token_entropies = _forward_signals_in_chunks(
                model,
                inp,
                len(pids),
                rids,
                args,
                attention_rows,
                use_compat_hooks=bool(hooks),
            )
    finally:
        for hook in hooks:
            hook.remove()
        _restore_chunked_attention(patched)
    token_logits = token_logits[start:end + 1]
    token_probs = token_probs[start:end + 1]
    token_entropies = token_entropies[start:end + 1]
    att_score = [None] * (end - start + 1)
    if attention_rows:
        # Each captured row is [num_heads, sequence_length]. Average heads and
        # then layers exactly as before, but only for the requested query row.
        row_means = [row.float().mean(dim=0) for row in attention_rows]
        v = torch.stack(row_means).mean(dim=0)
        for j in range(start, end + 1):
            q = len(pids) + j
            if q < v.numel():
                att_score[j - start] = float(v[q])
    # Do not let the input or temporary CUDA allocations survive until the
    # next (possibly much longer) sample.  empty_cache only releases unused
    # blocks; model parameters remain resident.
    del inp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    details = copy.deepcopy(rec)
    details['full_output'] = text
    details['token_logits'] = token_logits
    details['token_probs'] = token_probs
    details['token_entropies'] = [float(x) for x in token_entropies]
    details['token_attentioni_scores'] = att_score
    return (details, token_logits, token_probs, token_entropies, att_score, text, start, end)

def aggregate(vals, mode):
    xs = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    if mode == 'mean':
        return sum(xs) / len(xs)
    if mode == 'geo':
        return math.exp(sum((math.log(max(x, 1e-30)) for x in xs)) / len(xs))

def format_percentage_label(value):
    """Make a stable filename label without a decimal point when unnecessary."""
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace('.', 'p')

def make_step_output(base, step_rec, text, pieces, logits, probs, ents, att, prune_percent=0.0):
    out = copy.deepcopy(step_rec)
    spans = step_spans(text)
    steps = out.get('steps', [])
    buckets = [[] for _ in steps]
    for j, (_, _, s, e) in enumerate(pieces):
        if e <= s:
            continue
        ovs = [max(0, min(e, b) - max(s, a)) for a, b in spans]
        if ovs and max(ovs) > 0:
            step_index = ovs.index(max(ovs))
            # A synthetic closing marker used only by the forward pass can
            # create an extra text span.  Never assign such a span outside
            # the original step list.
            if 0 <= step_index < len(buckets):
                buckets[step_index].append(j)

    # Percentage-based pruning is performed globally over the valid LRM token
    # positions in this sample.  The lowest-attention ceil(N*p/100) positions
    # are removed; no absolute attention-score cutoff is used.
    valid_attention_indices = [j for j, x in enumerate(att) if x is not None and math.isfinite(float(x))]
    remove_count = int(math.ceil(len(valid_attention_indices) * float(prune_percent) / 100.0))
    remove_count = min(max(remove_count, 0), len(valid_attention_indices))
    ranked = sorted(valid_attention_indices, key=lambda j: (float(att[j]), j))
    pruned_indices = set(ranked[remove_count:])

    def build(kind, pruned):
        z = copy.deepcopy(out)
        for i, st in enumerate(z.get('steps', [])):
            js = buckets[i]
            if pruned:
                js = [j for j in js if j in pruned_indices]
            lv = [logits[j] for j in js]
            pv = [probs[j] for j in js]
            ev = [ents[j] for j in js]
            if kind == 'logit':
                val = aggregate(lv, 'mean') if lv else None
            elif kind == 'prob':
                val = aggregate(pv, 'geo') if pv else None
            else:
                val = aggregate(ev, 'mean')
            st['token_avg_confidence'] = val
        return z
    return [build(k, False) for k in ('logit', 'prob', 'entropy')] + [build(k, True) for k in ('logit', 'prob', 'entropy')]

def worker_main(group, args_dict, records, steps, out_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = group
    args = argparse.Namespace(**args_dict)
    args.tp_size = len(group.split(','))
    # Keep the result file open throughout the worker lifetime and flush after
    # every sample.  The parent process can therefore update the progress bar
    # and checkpoint token_details.jsonl as soon as each sample finishes.
    with open(out_path, 'w', encoding='utf-8') as f:
        from transformers import AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        model = load_model(args)
        for i, (r, s) in enumerate(zip(records, steps)):
            try:
                d, *rest = extract_one(r, s, tok, model, args)
                x = (i, d, rest)
            except Exception as e:
                x = (i, None, repr(e))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
            f.flush()

def main():
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser()
    p.add_argument('--summary_jsonl', required=True)
    p.add_argument('--step_jsonl', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--output_dir', default='.')
    p.add_argument('--enable_thinking', type=parse_bool, default=True)
    p.add_argument('--trust_remote_code', action='store_true')
    p.add_argument('--dtype', default='auto')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=0, help='top-k terms used only for entropy approximation (0=full entropy)')
    p.add_argument('--attention_threshold', type=str, default='0',
                   help='comma-separated percentages of lowest-attention LRM tokens to remove, e.g. 0,10,20 (each in [0,100]); retained parameter name for compatibility')
    p.add_argument('--gpus', default=None, help="DP groups separated by ';', TP GPUs within a group separated by ','")
    p.add_argument('--tensor_parallel_size', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok_out = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=a.trust_remote_code)

    if not a.temperature > 0.0 or not math.isfinite(a.temperature):
        p.error('--temperature must be a finite value greater than 0')
    if not 0.0 < a.top_p <= 1.0 or not math.isfinite(a.top_p):
        p.error('--top_p must be in (0, 1]')
    if a.top_k < 0:
        p.error('--top_k must be nonnegative')
    os.makedirs(a.output_dir, exist_ok=True)
    try:
        attention_thresholds = [float(x.strip()) for x in a.attention_threshold.split(',') if x.strip()]
    except ValueError as e:
        p.error('--attention_threshold must be a comma-separated list of percentages')
    if not attention_thresholds:
        p.error('--attention_threshold must contain at least one percentage')
    if any(not 0.0 <= x <= 100.0 for x in attention_thresholds):
        p.error('each --attention_threshold percentage must be in [0, 100]')
    # Preserve user order while avoiding duplicate output files.
    attention_thresholds = list(dict.fromkeys(attention_thresholds))
    summary = read_jsonl(a.summary_jsonl)
    steps = read_jsonl(a.step_jsonl)
    bys = {int(x.get('sample_idx', i)): x for i, x in enumerate(summary)}
    byt = {int(x.get('sample_idx', i)): x for i, x in enumerate(steps)}
    ids = sorted(set(bys) & set(byt))
    ids = ids[:a.limit] if a.limit else ids
    state = Path(a.output_dir) / 'token_details.jsonl'
    done = {}
    if a.resume and state.exists():
        for x in read_jsonl(state):
            # Old token_details files contain raw logits and unfiltered
            # probabilities.  They cannot be reused for the new definition.
            for logit, prob in zip(x.get('token_logits', []), x.get('token_probs', [])):
                if logit is not None and prob is not None and 0.0 < prob <= 1.0:
                    if abs(float(logit) - math.log(float(prob))) > 1e-3:
                        raise ValueError(
                            'Existing token_details.jsonl uses the old signal definition; '
                            'use a fresh --output_dir instead of --resume.'
                        )
            done[int(x['sample_idx'])] = x
    todo = [i for i in ids if i not in done]
    if a.gpus:
        groups = [g.strip() for g in a.gpus.split(';') if g.strip()]
    elif a.tensor_parallel_size > 1:
        groups = [','.join((str(i) for i in range(a.tensor_parallel_size)))]
    else:
        groups = ['0']
    tempdir = tempfile.mkdtemp(prefix='forward_')
    chunks = [todo[i::len(groups)] for i in range(len(groups))]
    procs = []
    ad = vars(a).copy()
    for wi, (g, ch) in enumerate(zip(groups, chunks)):
        if not ch:
            continue
        path = str(Path(tempdir) / f'w{wi}.jsonl')
        rr = [bys[i] for i in ch]
        ss = [byt[i] for i in ch]
        q = mp.Process(target=worker_main, args=(g, ad, rr, ss, path))
        q.start()
        procs.append((q, path, ch))
    total = len(ids)
    completed = len(done)
    from tqdm import tqdm
    pbar = tqdm(total=total, initial=completed, desc='Forward samples', unit='sample')
    # Workers stream one JSON record per completed sample.  Polling the files
    # lets the main process reflect completion in real time even when workers
    # are running in parallel and samples have very different lengths.
    # Track byte offsets and incomplete trailing lines instead of repeatedly
    # parsing the whole file.  This also avoids decoding a large JSON record
    # while a worker is still flushing it.
    read_offsets = {path: 0 for _, path, _ in procs}
    read_buffers = {path: b'' for _, path, _ in procs}
    while True:
        received_this_round = False
        for q, path, ch in procs:
            try:
                with open(path, 'rb') as f:
                    f.seek(read_offsets[path])
                    chunk = f.read()
                    read_offsets[path] = f.tell()
            except FileNotFoundError:
                chunk = b''
            if chunk:
                read_buffers[path] += chunk
            raw_parts = read_buffers[path].split(b'\n')
            read_buffers[path] = raw_parts.pop()
            if raw_parts:
                received_this_round = True
            for raw_line in raw_parts:
                if not raw_line.strip():
                    continue
                idx, d, rest = json.loads(raw_line.decode('utf-8'))
                sid = ch[idx]
                if d is None:
                    print(f'[{sid}] failed: {rest}')
                else:
                    done[sid] = dict(d, sample_idx=sid)
                    completed += 1
                # A result (including a failed sample) means that one sample
                # has finished running, so advance the main progress bar once.
                pbar.update(1)
                # Checkpoint after every completed sample so interruption or
                # OOM recovery can resume without losing prior results.
                write_jsonl(state, [done[i] for i in ids if i in done])
        all_stopped = all(not q.is_alive() for q, _, _ in procs)
        if all_stopped and not received_this_round:
            break
        if not received_this_round:
            time.sleep(0.1)
    for q, _, _ in procs:
        q.join()
    pbar.close()
    write_jsonl(state, [done[i] for i in ids if i in done])
    # The first three files are the unchanged, unpruned outputs.  Each
    # requested percentage gets its own three files with a percentage suffix.
    out_specs = [(x, None) for x in ('logit', 'prob', 'entropy')]
    for pct in attention_thresholds:
        pct_label = format_percentage_label(pct)
        out_specs.extend((x, pct_label) for x in ('logit', 'prob', 'entropy'))
    outs = []
    for metric, pct_label in out_specs:
        if pct_label is None:
            name = f'steps_labeled_{metric}.jsonl'
        else:
            name = f'steps_labeled_{metric}_pruned_{pct_label}pct.jsonl'
        outs.append(Path(a.output_dir) / name)
    files = [[] for _ in outs]
    # tokenizer loaded before post-processing
    final_ids = [sid for sid in ids if sid in done]
    for sid in tqdm(final_ids, desc='Final outputs', unit='sample'):
        d = done[sid]
        s = byt[sid]
        # The forward pass may append a synthetic </think> solely to obtain
        # its attention row.  Final files must use the original text and
        # original token positions, never that synthetic suffix.
        original_text = s.get('full_output') or '\n\n'.join((x['step'] for x in s.get('steps', [])))
        text = original_text
        tok = tok_out
        rid = tok(text, add_special_tokens=False).input_ids
        _, pc = token_pieces(rid, tok)
        st_ids = tok.encode('<think>', add_special_tokens=False)
        en_ids = tok.encode('</think>', add_special_tokens=False)
        st = next((j for j in range(len(rid) - len(st_ids) + 1) if rid[j:j + len(st_ids)] == st_ids), 0) if st_ids else 0
        e0 = next((j for j in range(st, len(rid) - len(en_ids) + 1) if rid[j:j + len(en_ids)] == en_ids), len(rid) - len(en_ids)) if en_ids else len(rid) - 1
        en = min(len(rid) - 1, e0 + len(en_ids) - 1)
        # d's arrays may include one or more synthetic </think> tokens from
        # extract_one().  Truncate them to the original response token span.
        token_count = max(0, en - st + 1)
        final_logits = d['token_logits'][:token_count]
        final_probs = d['token_probs'][:token_count]
        final_entropies = d['token_entropies'][:token_count]
        final_attention = d['token_attentioni_scores'][:token_count]
        final_pieces = pc[st:en + 1]
        vals = make_step_output(d, s, text, final_pieces, final_logits, final_probs, final_entropies, final_attention, 0.0)
        for j, v in enumerate(vals[:3]):
            files[j].append(v)
        for threshold_index, pct in enumerate(attention_thresholds):
            pruned_vals = make_step_output(d, s, text, final_pieces, final_logits, final_probs, final_entropies, final_attention, pct)[3:]
            offset = 3 + threshold_index * 3
            for metric_index, v in enumerate(pruned_vals):
                files[offset + metric_index].append(v)
    for path, recs in zip(outs, files):
        write_jsonl(path, recs)
    print(f'\nFinished {completed}/{total}; outputs in {a.output_dir}')
if __name__ == '__main__':
    main()