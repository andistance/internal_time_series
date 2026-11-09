"""Minimal subprocess entry point: no benchmark, vLLM or legacy script import."""
from __future__ import annotations

import os
import sys
import traceback
from multiprocessing.connection import Client
from types import SimpleNamespace


def main():
    address = ("127.0.0.1", int(sys.argv[1]))
    authkey = bytes.fromhex(os.environ.pop("_BON_PHASE_AUTH"))
    connection = Client(address, authkey=authkey)
    connection.send(int(sys.argv[2]))
    try:
        config, directory = connection.recv()
        args = SimpleNamespace(**config)
        import torch
        from .gpu_phases import select_cached_batch
        from .phase_structure import StructureCache, compile_structure

        if not torch.cuda.is_available():
            raise RuntimeError(f"Phase DP worker cannot access configured GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
        torch.cuda.set_device(0)
        torch.empty(0, device="cuda:0")
        # The parent supplies cache hits. Avoid scanning every shard once per GPU.
        cache, tokenizer = None, None
        connection.send(("ready", None))
        with torch.inference_mode():
            while True:
                batch = connection.recv()
                if batch is None:
                    return
                ready, results = [], []
                for index, descriptor, key, generation, entropy in batch:
                    try:
                        if descriptor is None:
                            if cache is None:
                                cache = StructureCache(directory)
                            descriptor = cache.entries.get(key)
                            if descriptor is None:
                                if tokenizer is None and "_bon_source_structure" not in generation:
                                    from transformers import AutoTokenizer
                                    tokenizer = AutoTokenizer.from_pretrained(
                                        args.model_path, trust_remote_code=getattr(args, "trust_remote_code", False),
                                    )
                                arrays = compile_structure(generation, entropy, tokenizer)
                                descriptor = cache.save(key, *arrays)
                        ready.append((index, descriptor))
                    except Exception:
                        raise
                if ready:
                    try:
                        selected = select_cached_batch([item[1] for item in ready], args, "cuda:0")
                        results.extend((index, descriptor, positions, count, None)
                                       for (index, descriptor), (positions, count) in zip(ready, selected))
                    except Exception:
                        raise
                connection.send(("results", results))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
