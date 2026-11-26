"""Numerical/lifecycle regression tests; run with unittest, without vLLM.

The model and phase aggregation definitions are extracted from the actual
legacy script so these checks exercise the training architecture, not a mock.
CUDA parity cases run automatically on hosts with at least two visible GPUs.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from scripts.transformer_bon.engine import BestOfNConfig, BestOfNService
from scripts.transformer_bon.models import BestOfNRequest
from scripts.transformer_bon.value_parallel import (
    ParallelValueModel, TensorParallelValueModel, parse_value_devices,
    load_parallel_value_components,
)


def legacy_definitions(script="3_best_of_n_via_phases_scores_qwen.py"):
    path = Path(__file__).resolve().parents[1] / script
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {"SinusoidalPositionalEncoding", "SmallTransformerEncoderValueModel",
             "aggregate_phase_scores", "attach_phase_scores_to_steps"}
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = dict(torch=torch, nn=nn, math=math, np=np, Any=Any, Dict=Dict,
                     List=List, Sequence=Sequence)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def model_and_inputs(kind="bidirectional", script="3_best_of_n_via_phases_scores_qwen.py"):
    torch.manual_seed(123)
    legacy = legacy_definitions(script)
    model = legacy.SmallTransformerEncoderValueModel(
        d_model=24, nhead=6, dim_feedforward=49, num_layers=2,
        max_len=32, transformer_type=kind,
    ).eval()
    x = torch.randn(5, 11, 4)
    lengths = torch.tensor([11, 3, 8, 1, 6])
    mask = torch.arange(11)[None, :] < lengths[:, None]
    return model, x, mask


class ValueParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_device_layout_validation_and_existing_devices(self):
        for value in ("auto", "cpu", "cuda", "cuda:0"):
            self.assertEqual(parse_value_devices(value), ())
        self.assertEqual(parse_value_devices("0, cuda:1; 2,3"),
                         (("cuda:0", "cuda:1"), ("cuda:2", "cuda:3")))
        for value in ("0,0", "0;cuda:0", "0,1;2", "0,", ";0", "cpu;1", "-1;0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_value_devices(value)
        with self.assertRaisesRegex(ValueError, "residency"):
            BestOfNConfig(model_path="m", checkpoint_path="c", n=1,
                          enable_thinking=True, value_device="0;1", value_residency="cpu")

    def test_tp_matches_both_legacy_models_for_both_attention_modes(self):
        for script in ("3_best_of_n_via_phases_scores_qwen.py",
                       "3_best_of_n_via_phases_scores_qwen_vllm_multi_signal.py"):
            for kind in ("causal", "bidirectional"):
                model, x, mask = model_and_inputs(kind, script)
                with torch.inference_mode():
                    expected = model(x, mask)
                for size in (1, 2, 4):  # includes uneven head/FFN sharding
                    with self.subTest(script=script, kind=kind, tp=size):
                        replica = TensorParallelValueModel(model, ["cpu"] * size)
                        actual = replica(x, mask)
                        torch.testing.assert_close(actual[mask], expected[mask], atol=2e-6, rtol=2e-6)
                        # The shards together hold each large projection once.
                        for original, layer in zip(model.encoder.layers, replica.layers):
                            self.assertEqual(sum(s["qkv"].numel() for s in layer["shards"]),
                                             original.self_attn.in_proj_weight.numel())
                            self.assertEqual(sum(s["ff2"].numel() for s in layer["shards"]),
                                             original.linear2.weight.numel())
                        replica.to("cpu")
                        torch.testing.assert_close(replica(x, mask)[mask], actual[mask])

    def test_causal_sdpa_does_not_allocate_a_quadratic_mask(self):
        model, x, mask = model_and_inputs("causal")
        replica = TensorParallelValueModel(model, ["cpu", "cpu"])
        original = torch.nn.functional.scaled_dot_product_attention
        with mock.patch.object(torch.nn.functional, "scaled_dot_product_attention", wraps=original) as sdpa:
            replica(x, mask)
        for call in sdpa.call_args_list:
            self.assertIsNone(call.kwargs["attn_mask"])
            self.assertTrue(call.kwargs["is_causal"])

    def test_tp_size_error_and_cuda_validation_before_checkpoint_load(self):
        model, _, _ = model_and_inputs()
        with self.assertRaisesRegex(ValueError, "nhead"):
            TensorParallelValueModel(model, ["cpu"] * 7)
        legacy = SimpleNamespace(torch=torch, load_value_components=mock.Mock())
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                load_parallel_value_components(legacy, SimpleNamespace(), parse_value_devices("0;1"))
        legacy.load_value_components.assert_not_called()

    def test_checkpoint_is_loaded_once_on_cpu_without_changing_args(self):
        args = SimpleNamespace(value_device="0,1;2,3")
        legacy = SimpleNamespace(torch=torch, load_value_components=mock.Mock(
            return_value=("model", "normalizer", "cpu", {"transformer_type": "causal"})))
        groups = parse_value_devices(args.value_device)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "device_count", return_value=4), \
                mock.patch("scripts.transformer_bon.value_parallel.ParallelValueModel") as constructor:
            loaded = load_parallel_value_components(legacy, args, groups)
        self.assertEqual(legacy.load_value_components.call_count, 1)
        self.assertEqual(legacy.load_value_components.call_args.args[0].value_device, "cpu")
        self.assertEqual(args.value_device, "0,1;2,3")
        constructor.assert_called_once_with("model", groups)
        self.assertEqual(str(loaded[2]), "cuda:0")

    def test_dp_batch_order_oom_backoff_and_non_oom_propagation(self):
        model, x, mask = model_and_inputs("causal")
        parallel = ParallelValueModel(model, [("cpu", "cpu")] * 2)
        try:
            with torch.inference_mode():
                expected = model(x, mask)
            torch.testing.assert_close(parallel(x, mask)[mask], expected[mask], atol=2e-6, rtol=2e-6)
            original = parallel.replicas[0]

            def limited(batch, attention_mask):
                if len(batch) > 1:
                    raise torch.OutOfMemoryError("test OOM")
                return original(batch, attention_mask)

            parallel.replicas[0] = mock.Mock(side_effect=limited, devices=original.devices)
            actual = parallel._predict(0, x, mask)
            torch.testing.assert_close(actual[mask], expected[mask], atol=2e-6, rtol=2e-6)
            self.assertEqual(parallel.batch_limits[0], 1)
            parallel.replicas[0] = mock.Mock(side_effect=RuntimeError("invalid model"))
            with self.assertRaisesRegex(RuntimeError, "invalid model"):
                parallel._predict(0, x[:1], mask[:1])
        finally:
            parallel.close()

    def test_bounded_ordered_dp_requests_and_early_close(self):
        model, _, _ = model_and_inputs()
        parallel = ParallelValueModel(model, [("cpu",), ("cpu",)])
        seen = []
        barrier = threading.Barrier(2, timeout=5)

        def entries():
            for i in range(20):
                seen.append(i)
                yield i

        def score(entry):
            if entry < 2:
                barrier.wait()
            return entry, parallel.local.rank

        try:
            results = parallel.map_requests(score, entries())
            self.assertEqual(next(results), (0, 0))
            # Refill the released replica before handing results to the writer.
            self.assertEqual(seen, [0, 1, 2])
            self.assertEqual(next(results), (1, 1))
            results.close()
            self.assertEqual(list(parallel.map_requests(lambda i: i, range(5))), list(range(5)))
        finally:
            parallel.close()
        self.assertTrue(all(not thread.is_alive() for executor in parallel.executors
                            for thread in executor._threads))

    def test_service_preserves_record_schema_order_and_phase_scores(self):
        model, x, mask = model_and_inputs("causal")
        legacy = legacy_definitions()
        legacy.build_value_features = lambda row, **_: x[int(row["sequence_idx"]), :row["num_steps"]].numpy()
        normalizer = SimpleNamespace(transform_one=lambda array: array)
        config = BestOfNConfig(model_path="m", checkpoint_path="c", n=5,
                              enable_thinking=True, scoring_batch_size=2)
        service = BestOfNService(config)
        service._value_components = (model, normalizer, torch.device("cpu"))
        service._value_transformer_type = "causal"
        request = BestOfNRequest(request_id="q", sample_id="s", task_text="question", raw_prompt="Q")
        records = [{"status": "ok", "sequence_idx": i, "num_steps": int(mask[i].sum()),
                    "steps": [{} for _ in range(int(mask[i].sum()))],
                    "step_avg_confidences": [0.5] * int(mask[i].sum()),
                    "phase_ranges": [[1, int(mask[i].sum())]],
                    "num_generated_tokens": int(mask[i].sum()), "full_output": f"answer {i}"}
                   for i in (3, 0, 4, 1, 2)]
        expected = service._score(legacy, request, records, config.legacy_args())
        parallel = ParallelValueModel(model, [("cpu", "cpu")] * 2)
        service._value_components = (parallel, normalizer, torch.device("cpu"))
        try:
            actual = service._score(legacy, request, records, config.legacy_args())
            self.assertEqual([c.sequence_idx for c in actual], [3, 0, 4, 1, 2])
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a.score, b.score, places=6)
                self.assertEqual(a.summary_record.keys(), b.summary_record.keys())
                self.assertEqual(a.steps_record.keys(), b.steps_record.keys())
                self.assertEqual(a.steps_record["steps"][0].keys(), b.steps_record["steps"][0].keys())
            # JSON volume stays comparable; no tensor/shard payloads are added.
            size = lambda values: len(json.dumps([c.__dict__ for c in values]))
            self.assertLess(size(actual), size(expected) * 1.05)
            # Exercise the benchmark streaming route: request records and
            # progress callbacks remain in input order on the main thread.
            config.value_device = "0,1;2,3"
            requests = [BestOfNRequest(request_id=str(i), sample_id=str(i),
                                      task_text="q", raw_prompt="Q") for i in range(3)]
            entries = [{"request_id": r.request_id, "records": records} for r in requests]
            callbacks = []
            for req, candidates in service._score_prepared_entries(
                    legacy, entries, {r.request_id: r for r in requests}, config.legacy_args(),
                    lambda stage: callbacks.append((stage, threading.get_ident()))):
                self.assertEqual(req.request_id, str(len(callbacks) // 5 - 1))
                self.assertEqual([c.sequence_idx for c in candidates], [3, 0, 4, 1, 2])
            self.assertEqual(callbacks, [("scoring", threading.get_ident())] * 15)
            with mock.patch.object(service, "_get_value_components", side_effect=AssertionError("loaded empty stage")):
                self.assertEqual(list(service._score_prepared_entries(
                    legacy, [], {}, config.legacy_args())), [])
        finally:
            parallel.close()

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_real_cuda_dp_tp_parity_and_residency(self):
        for kind in ("causal", "bidirectional"):
            model, x, mask = model_and_inputs(kind)
            with torch.inference_mode():
                reference = model.cuda(0)(x.cuda(0), mask.cuda(0)).cpu()
            model.cpu()
            layouts = [(('cuda:0', 'cuda:1'),), (('cuda:0',), ('cuda:1',))]
            if torch.cuda.device_count() >= 4:
                layouts.append((('cuda:0', 'cuda:1'), ('cuda:2', 'cuda:3')))
            for groups in layouts:
                parallel = ParallelValueModel(model, groups)
                try:
                    torch.testing.assert_close(parallel(x, mask)[mask], reference[mask], atol=3e-6, rtol=3e-6)
                    parallel.to("cpu").to("cuda:0")
                    torch.testing.assert_close(parallel(x, mask)[mask], reference[mask], atol=3e-6, rtol=3e-6)
                finally:
                    parallel.close()


if __name__ == "__main__":
    unittest.main()
