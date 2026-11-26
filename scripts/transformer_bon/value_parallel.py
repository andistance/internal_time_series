"""Single-process, persistent DP + head/FFN tensor parallel value inference.

Only the small trajectory model is handled here. No process groups, vLLM
engines, checkpoint copies on disk, or changes to benchmark records are needed.
PyTorch imports stay lazy so CPU-only CLI parsing remains lightweight.
"""
from __future__ import annotations

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import re
import threading


def parse_value_devices(value):
    """Commas join TP devices; semicolons separate DP replicas.

    Ordinary torch devices (including auto/cpu/cuda:0) keep their old meaning.
    Integers and cuda:N are accepted in an explicit multi-device layout.
    Indices always refer to the caller's CUDA_VISIBLE_DEVICES namespace.
    """
    if "," not in value and ";" not in value:
        return ()
    groups, used = [], set()
    for raw_group in value.split(";"):
        group = []
        for item in raw_group.split(","):
            match = re.fullmatch(r"(?:cuda:)?([0-9]+)", item.strip())
            if match is None:
                raise ValueError("value_device must use GPU indices or cuda:N, e.g. '0,1;2,3'")
            device = "cuda:" + str(int(match.group(1)))
            if device in used:
                raise ValueError(f"value_device assigns {device} more than once")
            used.add(device)
            group.append(device)
        groups.append(tuple(group))
    if len({len(group) for group in groups}) != 1:
        raise ValueError("value_device TP groups must have the same number of GPUs")
    return tuple(groups)


class TensorParallelValueModel:
    """Shard QKV by complete heads and shard the two FFN projections.

    Residuals/norms and the small input/output heads live on the group leader.
    Independent shards launch asynchronously; partial row projections are
    reduced with CUDA peer copies, without a global distributed environment.
    CPU devices are also supported for numerical parity tests.
    """
    def __init__(self, model, devices):
        import torch
        self.devices = tuple(torch.device(device) for device in devices)
        self.primary = self.devices[0]
        self.causal = model.transformer_type == "causal"
        self.input_proj = copy.deepcopy(model.input_proj).requires_grad_(False)
        self.pos_encoder = copy.deepcopy(model.pos_encoder)
        self.output_head = copy.deepcopy(model.output_head).requires_grad_(False)
        self.final_norm = copy.deepcopy(model.encoder.norm)
        self.layers = []
        width = len(devices)
        for layer in model.encoder.layers:
            attn = layer.self_attn
            if not layer.norm_first or not attn.batch_first or not attn._qkv_same_embed_dim:
                raise ValueError("Value TP requires the trained pre-norm, batch-first encoder")
            if width > min(attn.num_heads, layer.linear1.out_features):
                raise ValueError(
                    f"Value TP size {width} exceeds nhead={attn.num_heads} or FFN width; "
                    "use smaller TP groups and more DP replicas"
                )
            head_dim = attn.embed_dim // attn.num_heads
            shards = []
            for rank in range(width):
                # Uneven partitions allow TP sizes that do not divide nhead.
                lo = rank * attn.num_heads // width
                hi = (rank + 1) * attn.num_heads // width
                start, stop = lo * head_dim, hi * head_dim
                fstart = rank * layer.linear1.out_features // width
                fstop = (rank + 1) * layer.linear1.out_features // width
                qkv = attn.in_proj_weight.detach().reshape(3, attn.embed_dim, attn.embed_dim)
                bias = attn.in_proj_bias.detach().reshape(3, attn.embed_dim)
                shards.append({
                    "qkv": qkv[:, start:stop].reshape(-1, attn.embed_dim).clone(),
                    "qkv_bias": bias[:, start:stop].reshape(-1).clone(),
                    "out": attn.out_proj.weight.detach()[:, start:stop].clone(),
                    "ff1": layer.linear1.weight.detach()[fstart:fstop].clone(),
                    "ff1_bias": layer.linear1.bias.detach()[fstart:fstop].clone(),
                    "ff2": layer.linear2.weight.detach()[:, fstart:fstop].clone(),
                })
            self.layers.append({
                "norm1": copy.deepcopy(layer.norm1).requires_grad_(False),
                "norm2": copy.deepcopy(layer.norm2).requires_grad_(False),
                "out_bias": attn.out_proj.bias.detach().clone(),
                "ff2_bias": layer.linear2.bias.detach().clone(),
                "head_dim": head_dim,
                "activation": layer.activation, "shards": shards,
            })
        self.to(self.primary)
        self.output_head.eval()

    def to(self, device):
        import torch
        cpu = torch.device(device).type == "cpu"
        self.primary = torch.device("cpu") if cpu else self.devices[0]
        for module in (self.input_proj, self.pos_encoder, self.output_head, self.final_norm):
            if module is not None:
                module.to(self.primary)
        for layer in self.layers:
            for name in ("norm1", "norm2"):
                layer[name].to(self.primary)
            for name in ("out_bias", "ff2_bias"):
                layer[name] = layer[name].to(self.primary)
            for rank, shard in enumerate(layer["shards"]):
                target = "cpu" if cpu else self.devices[rank]
                for name, tensor in shard.items():
                    shard[name] = tensor.to(target)
        return self

    def __call__(self, x, attention_mask):
        import torch
        from torch.nn import functional as F
        with torch.inference_mode():
            hidden = self.pos_encoder(self.input_proj(x.to(self.primary, non_blocking=True)))
            batch, length, _ = hidden.shape
            # Inputs are right-padded. Valid causal queries can never see padding,
            # so is_causal replaces the legacy L x L causal/padding mask entirely.
            masks = {}
            for shard in self.layers[0]["shards"]:
                device = shard["qkv"].device
                if device not in masks:
                    masks[device] = (None if self.causal else
                                     attention_mask.to(device, non_blocking=True)[:, None, None, :])
            for layer in self.layers:
                normalized = layer["norm1"](hidden)
                # Broadcast before launching the leader's matmuls. Otherwise a
                # peer copy on its default stream would sit behind those kernels
                # and serialize the other GPUs' work with the leader's work.
                locals_ = [normalized.to(shard["qkv"].device, non_blocking=True)
                           for shard in layer["shards"]]
                partials = []
                for shard, local in zip(layer["shards"], locals_):
                    heads = shard["qkv"].shape[0] // (3 * layer["head_dim"])
                    qkv = F.linear(local, shard["qkv"], shard["qkv_bias"])
                    q, k, v = qkv.reshape(batch, length, 3, heads, layer["head_dim"]).permute(2, 0, 3, 1, 4).unbind(0)
                    attended = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=masks[local.device], dropout_p=0.0,
                        is_causal=self.causal,
                    ).transpose(1, 2).reshape(batch, length, -1)
                    partials.append(F.linear(attended, shard["out"]))
                hidden = hidden + (self._reduce(partials) + layer["out_bias"])
                normalized = layer["norm2"](hidden)
                locals_ = [normalized.to(shard["ff1"].device, non_blocking=True)
                           for shard in layer["shards"]]
                partials = []
                for shard, local in zip(layer["shards"], locals_):
                    activated = layer["activation"](F.linear(local, shard["ff1"], shard["ff1_bias"]))
                    partials.append(F.linear(activated, shard["ff2"]))
                hidden = hidden + (self._reduce(partials) + layer["ff2_bias"])
            if self.final_norm is not None:
                hidden = self.final_norm(hidden)
            return torch.sigmoid(self.output_head(hidden).squeeze(-1))

    def _reduce(self, partials):
        total = partials[0].to(self.primary, non_blocking=True)
        for partial in partials[1:]:
            total = total + partial.to(self.primary, non_blocking=True)
        return total


class ParallelValueModel:
    """One persistent executor per DP replica, with bounded ordered lookahead."""
    input_device = "cpu"

    def __init__(self, model, groups):
        self.replicas = [TensorParallelValueModel(model, group) for group in groups]
        self.executors = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"value-dp-{i}")
                          for i in range(len(groups))]
        self.local = threading.local()
        self.batch_limits = [None] * len(groups)

    def to(self, device):
        for replica in self.replicas:
            replica.to(device)
        return self

    def __call__(self, x, attention_mask):
        import torch
        rank = getattr(self.local, "rank", None)
        if rank is not None:
            return self._predict(rank, x, attention_mask)
        # Interactive/single-request path: share this batch among the replicas.
        count = min(len(self.replicas), len(x))
        futures = [self.executors[i].submit(
            self._predict, i, x[i * len(x) // count:(i + 1) * len(x) // count],
            attention_mask[i * len(x) // count:(i + 1) * len(x) // count],
        ) for i in range(count)]
        try:
            return torch.cat([future.result() for future in futures])
        finally:
            for future in futures:
                future.cancel()
            for future in futures:
                if not future.cancelled():
                    try:
                        future.result()
                    except BaseException:
                        pass

    def _predict(self, rank, x, mask):
        import torch
        from torch.nn import functional as F
        # inference_mode is thread local. CPU transfer synchronizes this replica
        # once per batch, rather than once per trajectory or prediction.
        with torch.inference_mode():
            limit = self.batch_limits[rank]
            if limit is None or len(x) <= limit:
                try:
                    # Splitting a single request over DP can leave short groups
                    # with padding inherited from a different GPU's longest row.
                    length = int(mask.sum(1).max())
                    scores = self.replicas[rank](x[:, :length], mask[:, :length]).cpu()
                    return F.pad(scores, (0, x.shape[1] - length))
                except torch.OutOfMemoryError:
                    if len(x) == 1:
                        raise
                # Leave the exception handler before retrying so its traceback
                # no longer retains the failed forward's activation tensors.
                self.batch_limits[rank] = max(1, len(x) // 2)
                for device in self.replicas[rank].devices:
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
            limit = self.batch_limits[rank]
            return torch.cat([self._predict(rank, x[start:start + limit], mask[start:start + limit])
                              for start in range(0, len(x), limit)])

    def map_requests(self, score, entries):
        def run(rank, entry):
            self.local.rank = rank
            try:
                return score(entry)
            finally:
                del self.local.rank

        source, pending = iter(entries), deque()
        try:
            for rank, executor in enumerate(self.executors):
                entry = next(source, None)
                if entry is None:
                    break
                pending.append((rank, executor.submit(run, rank, entry)))
            while pending:
                rank, future = pending.popleft()
                result = future.result()
                source_error = None
                try:
                    entry = next(source, None)
                except BaseException as error:
                    # Preserve the last successful result if input decoding of
                    # the next request fails during the refill.
                    entry, source_error = None, error
                if entry is not None:
                    pending.append((rank, self.executors[rank].submit(run, rank, entry)))
                yield result
                if source_error is not None:
                    raise source_error
        finally:
            for _, future in pending:
                future.cancel()
            # No worker may outlive the iterator and race model parking/close.
            for _, future in pending:
                if not future.cancelled():
                    try:
                        future.result()
                    except BaseException:
                        pass

    def close(self):
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)


def load_parallel_value_components(legacy, args, groups):
    torch = legacy.torch
    if not torch.cuda.is_available():
        raise ValueError("Multi-GPU value_device requires CUDA")
    count = torch.cuda.device_count()
    for group in groups:
        for device in group:
            if int(device.split(":")[1]) >= count:
                raise ValueError(f"value_device {device} is outside the {count} visible CUDA devices")
    # Load exactly once on CPU, then copy only each GPU's shards. Never put a
    # complete checkpoint on the group leader as an intermediate allocation.
    cpu_args = copy.copy(args)
    cpu_args.value_device = "cpu"
    model, normalizer, _, metadata = legacy.load_value_components(cpu_args)
    parallel = ParallelValueModel(model, groups)
    print(f"[INFO] Value parallelism: DP={len(groups)}, TP={len(groups[0])}, devices={groups}")
    return parallel, normalizer, torch.device(groups[0][0]), metadata
