"""Attention workspace fallback used only inside the isolated OOM worker."""

from __future__ import annotations

from contextlib import contextmanager


def _kernel_unavailable(message):
    message = message.lower()
    return any(value in message for value in (
        "no available kernel", "no viable backend", "no suitable kernel",
        "no kernel found to launch", "not compiled with flash attention",
        "not compiled with memory efficient attention"))


class _RecoverySDPA:
    def __init__(self, runtime, original):
        self.runtime, self.original = runtime, original
        # Cache decisions, never tensors. Recovery has a bounded request window.
        self.modes = {}
        self.reported = set()

    def _kernel(self, name):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(getattr(SDPBackend, name))

    def _report(self, mode, query, key):
        if mode not in self.reported:
            self.reported.add(mode)
            print(f"[PRM OOM recovery] attention={mode}, "
                  f"q={tuple(query.shape)}, k={tuple(key.shape)}, "
                  f"dtype={query.dtype}, device={query.device}", flush=True)

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, *, scale=None, enable_gqa=False):
        from .prm_recovery import is_cuda_oom, release_cuda_cache

        if dropout_p:
            raise ValueError("PRM recovery attention requires inference dropout_p=0")
        kwargs = dict(attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
        if scale is not None:
            kwargs["scale"] = scale
        if enable_gqa:
            kwargs["enable_gqa"] = True
        signature = (tuple(query.shape), tuple(key.shape), tuple(value.shape),
                     query.dtype, query.device, query.stride(), key.stride(), value.stride(),
                     None if attn_mask is None else
                     (tuple(attn_mask.shape), attn_mask.dtype, attn_mask.stride()),
                     is_causal, enable_gqa)
        if len(self.modes) >= 64 and signature not in self.modes:
            self.modes.clear()
        mode = self.modes.get(signature)
        if query.is_cuda:
            # A list of fused kernels does not enforce Flash priority. Try each
            # backend separately; cuDNN's potentially large workspace is excluded.
            for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION"):
                if mode is not None and mode != name:
                    continue
                failure = None
                try:
                    with self._kernel(name):
                        result = self.original(query, key, value, **kwargs)
                    self.modes[signature] = name
                    self._report(name, query, key)
                    return result
                except RuntimeError as exc:
                    if not is_cuda_oom(exc) and not _kernel_unavailable(str(exc)):
                        raise
                    failure = str(exc)
                # Failed native-forward frames must die before cache cleanup.
                if is_cuda_oom(failure):
                    release_cuda_cache(self.runtime)
                mode = None
        self.modes[signature] = "blocked"
        self._report("blocked", query, key)
        budget = 512 * 1024**2
        if query.is_cuda:
            budget = min(budget, int(self.runtime.cuda.mem_get_info(query.device)[0]) // 4)
        for attempt in range(8):
            failure = None
            try:
                return self._blocked(query, key, value, attn_mask, is_causal,
                                     scale, enable_gqa, budget)
            except RuntimeError as exc:
                if not is_cuda_oom(exc) or attempt == 7:
                    raise
                failure = str(exc)
            if failure is not None:
                release_cuda_cache(self.runtime)
                budget //= 2
        raise AssertionError("unreachable")

    def _blocked(self, query, key, value, mask, causal, scale, gqa, budget):
        """Slice Q/heads, retaining all visible K/V and global causal positions.

        The ordinary math SDPA supplies its stable softmax and fully-masked-row
        handling. Large score matrices and GQA expansion stay within each tile.
        The budget is conservative scratch sizing, not a total CUDA/RSS limit.
        """
        torch = self.runtime
        if any(t.ndim != 4 for t in (query, key, value)):
            raise ValueError("Bounded PRM attention requires [batch, heads, seq, dim] tensors")
        batch, heads, length, dim = query.shape
        kv_heads, source = key.shape[1:3]
        if (value.shape[:3] != key.shape[:3] or query.shape[0] != key.shape[0]
                or dim != key.shape[-1] or not source or not length):
            raise ValueError("Incompatible Q/K/V shapes in bounded PRM attention")
        if (gqa and heads % kv_heads) or (not gqa and kv_heads not in (1, heads)):
            raise ValueError("Incompatible GQA heads in bounded PRM attention")
        if mask is not None:
            if mask.ndim > 4:
                raise ValueError("Unsupported bounded PRM attention mask rank")
            mask = mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))
            if any(a not in (1, b) for a, b in zip(mask.shape, (batch, heads, length, source))):
                raise ValueError("Incompatible bounded PRM attention mask shape")
        group = heads // kv_heads if gqa or kv_heads == 1 else 1
        head_block, query_block = min(heads, 8), min(length, 1024)
        if group > 1:
            head_block = min(head_block, group)
        accum_bytes = max(4, query.element_size())

        def scratch(h, q):
            # Allow four score-sized buffers, FP32 K/V copies and casts,
            # query/output buffers, and a causal mask. All are tile-sized.
            return (batch * h * accum_bytes *
                    (4 * q * source + 2 * source * (dim + value.shape[-1])
                     + 2 * q * (dim + value.shape[-1])) + 2 * q * source)

        while scratch(head_block, query_block) > budget:
            if head_block > 1:
                head_block = max(1, head_block // 2)
            elif query_block > 1:
                query_block = max(1, query_block // 2)
            else:
                raise RuntimeError("Insufficient workspace for one bounded PRM attention row")
        output = torch.empty((*query.shape[:-1], value.shape[-1]),
                             dtype=query.dtype, device=query.device)
        with self._kernel("MATH"):
            for start in range(0, length, query_block):
                end = min(length, start + query_block)
                visible = min(source, end) if causal else source
                causal_mask = None
                if causal:
                    causal_mask = (torch.arange(visible, device=query.device)[None, :] <=
                                   torch.arange(start, end, device=query.device)[:, None])
                head = 0
                while head < heads:
                    stop = min(heads, head + head_block)
                    if group > 1:
                        stop = min(stop, (head // group + 1) * group)
                        k = key[:, head // group:head // group + 1, :visible]
                        v = value[:, head // group:head // group + 1, :visible]
                        k = k.expand(batch, stop - head, visible, dim)
                        v = v.expand(batch, stop - head, visible, value.shape[-1])
                    else:
                        k, v = key[:, head:stop, :visible], value[:, head:stop, :visible]
                    block_mask = None
                    if mask is not None:
                        block_mask = mask[:, slice(head, stop) if mask.shape[1] != 1 else slice(None),
                                          slice(start, end) if mask.shape[2] != 1 else slice(None),
                                          :visible]
                    if causal_mask is not None:
                        if block_mask is None:
                            block_mask = causal_mask
                        elif block_mask.dtype == torch.bool:
                            block_mask = block_mask & causal_mask
                        else:
                            block_mask = block_mask.masked_fill(~causal_mask, float("-inf"))
                    kwargs = dict(attn_mask=block_mask, dropout_p=0.0, is_causal=False)
                    if scale is not None:
                        kwargs["scale"] = scale
                    result = self.original(query[:, head:stop, start:end], k, v, **kwargs)
                    output[:, head:stop, start:end].copy_(result)
                    del result, block_mask, k, v
                    head = stop
                del causal_mask
        return output


@contextmanager
def recovery_attention(runtime):
    # This process has one scoring thread and exits after the failed window.
    # Restore the function even when a model or unsupported operation fails.
    functional = runtime.nn.functional
    original = functional.scaled_dot_product_attention
    functional.scaled_dot_product_attention = _RecoverySDPA(runtime, original)
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original
