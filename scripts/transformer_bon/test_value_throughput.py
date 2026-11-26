"""Exact feature/result parity, bounded scheduling, and NaturalPlan recovery."""
from __future__ import annotations

import ast
import copy
import importlib
import json
import math
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from .engine import BestOfNConfig, BestOfNService, _iter_stage_request_entries
from .models import BestOfNCandidate, BestOfNRequest, BestOfNResult
from .naturalplan_records import NaturalPlanResultWriter, completed_request_ids
from .test_value_parallel import legacy_definitions, model_and_inputs
from .value_features import build_value_features, copy_steps, prepare_value_items
from .value_parallel import ParallelValueModel


ROOT = Path(__file__).resolve().parents[2]
RUNTIMES = sorted((ROOT / "scripts").glob("3_best_of_n_via_phases_*.py"))


def feature_runtime(path):
    names = {"build_value_features", "compute_local_std_and_slope", "linear_slope",
             "safe_float", "safe_int", "compute_step_start_end_positions", "estimate_text_token_len",
             "FeatureNormalizer"}
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = ast.parse("from __future__ import annotations").body
    nodes += [node for node in tree.body if getattr(node, "name", None) in names]
    namespace = dict(np=np, math=math, re=re, copy=copy)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def record(sequence=0, length=8):
    return dict(status="ok", sequence_idx=sequence, num_steps=length, num_generated_tokens=length * 8,
                steps=[dict(step="reasoning step", token_avg_confidence=.1 + .03 * ((i + sequence) % 19),
                            first_token_index=i * 8, phase_frontier=1) for i in range(length)],
                phase_ranges=[[1, length]], step_avg_confidences=[.5] * length,
                full_output="answer", token_entropies=[1.2] * length)


def request(index=0):
    return BestOfNRequest(request_id=f"r{index}", sample_id=str(index), task_text="q", raw_prompt="q")


class ValueThroughputTests(unittest.TestCase):
    def test_legacy_stage_index_keeps_offsets_and_preserves_mixed_last_row_wins(self):
        original = [dict(sequence_idx=2, value="original"), dict(sequence_idx=0, value="zero")]
        replacement = dict(sequence_idx=2, value="latest")
        other = dict(sequence_idx=1, value="other")
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            path = Path(folder) / "stage.jsonl"
            rows = [dict(request_id="b", records=original), dict(request_id="a", record=other),
                    dict(request_id="b", record=replacement)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            stream = _iter_stage_request_entries(path)
            try:
                self.assertEqual(next(stream), dict(request_id="b", records=[original[1], replacement]))
                grouped = stream.gi_frame.f_locals["grouped"]
                self.assertTrue(all(isinstance(location[0], int)
                                    for slots in grouped.values() for location in slots.values()))
                self.assertEqual(next(stream), dict(request_id="a", records=[other]))
                with self.assertRaises(StopIteration):
                    next(stream)
            finally:
                stream.close()

    def test_features_match_original_bytes_including_fold_normalization(self):
        rng = np.random.default_rng(23)
        for path in RUNTIMES:
            legacy = feature_runtime(path)
            for kind in ("causal", "bidirectional"):
                for length in (1, 2, 3, 4, 31, 256):
                    for scale in (0, 1e-12, 1e-4, 1, 1e6):
                        row = record(length=length)
                        values = .5 + rng.normal(size=length) * scale
                        for step, value in zip(row["steps"], values):
                            step["token_avg_confidence"] = float(value)
                        before = legacy.build_value_features(row, kind)
                        after = build_value_features(legacy, row, kind)
                        self.assertEqual(before.tobytes(), after.tobytes())
                        for mean, std in (([0] * 4, [1] * 4), ([.1, .3, -.2, .5], [.5, 1, .3, .8])):
                            normalizer = legacy.FeatureNormalizer.from_dict(dict(mean=mean, std=std))
                            self.assertEqual(normalizer.transform_one(before).tobytes(),
                                             normalizer.transform_one(after).tobytes())

    def test_invalid_features_and_multi_signal_suffix_keep_legacy_behavior(self):
        for path in RUNTIMES:
            legacy = feature_runtime(path)
            for confidences in ([None, None], [.5, None, .7], [.5, .7, None], [.2, float("inf")]):
                row = record(length=len(confidences))
                for step, value in zip(row["steps"], confidences):
                    step["token_avg_confidence"] = value
                try:
                    expected = legacy.build_value_features(row, "causal")
                except ValueError as error:
                    with self.assertRaisesRegex(ValueError, re.escape(str(error))):
                        build_value_features(legacy, row, "causal")
                else:
                    self.assertEqual(build_value_features(legacy, row, "causal").tobytes(), expected.tobytes())
            for steps in ([], [dict(first_token_index=None, token_avg_confidence=.5)]):
                with self.assertRaises(ValueError):
                    build_value_features(legacy, dict(steps=steps), "causal")

    def test_copy_steps_preserves_nested_and_alias_semantics(self):
        for steps in ([dict(step="text", x=1)], [dict(nested=dict(x=[1]))]):
            actual = copy_steps(steps)
            self.assertEqual(actual, copy.deepcopy(steps))
            actual[0]["new"] = 2
            self.assertNotIn("new", steps[0])
        row = dict(nested=[1])
        copied = copy_steps([row, row])
        self.assertIs(copied[0], copied[1])
        copied[0]["nested"].append(2)
        self.assertEqual(row["nested"], [1])

    def _service(self, kind="causal", device="cpu", profile="throughput"):
        model, _, _ = model_and_inputs(kind)
        model.to(device)
        legacy = legacy_definitions()
        features = feature_runtime(RUNTIMES[1])
        legacy.__dict__.update(features.__dict__)
        normalizer = features.FeatureNormalizer.from_dict(dict(mean=[.1] * 4, std=[.7] * 4))
        service = BestOfNService(BestOfNConfig(model_path="m", checkpoint_path="c", n=3,
                                              enable_thinking=True, scoring_batch_size=2,
                                              performance_profile=profile))
        service._value_components = model, normalizer, torch.device(device)
        service._value_transformer_type = kind
        return service, legacy, model

    def _exact_results(self, device):
        for kind in ("causal", "bidirectional"):
            service, legacy, model = self._service(kind, device)
            req = request()
            rows = [record(0, 7), record(1, 2), record(2, 5)]
            original = copy.deepcopy(rows)
            batches = []
            hook = model.register_forward_pre_hook(lambda _, args, kwargs:
                batches.append((args[0].detach().cpu().numpy().tobytes(),
                                kwargs["attention_mask"].detach().cpu().numpy().tobytes())), with_kwargs=True)
            def reference_items(legacy, rows, normalizer, kind):
                return sorted([(row, normalizer.transform_one(legacy.build_value_features(row, kind)))
                               for row in rows], key=lambda item: item[1].shape[0])
            try:
                with mock.patch("scripts.transformer_bon.engine.prepare_value_items", reference_items), \
                        mock.patch("scripts.transformer_bon.engine.copy_steps", copy.deepcopy):
                    expected = service._score(legacy, req, rows, service.config.legacy_args())
                before = list(batches)
                batches.clear()
                entries = [dict(request_id=req.request_id, records=rows)]
                actual = list(service._score_prepared_entries(legacy, entries, {req.request_id: req},
                                                              service.config.legacy_args()))[0][1]
                self.assertEqual(before, batches)
                self.assertEqual(json.dumps([c.__dict__ for c in expected]), json.dumps([c.__dict__ for c in actual]))
                self.assertEqual(rows, original)
            finally:
                hook.remove()

    def test_forward_batches_and_all_output_bytes_unchanged(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self._exact_results("cpu")
        finally:
            torch.set_num_threads(old)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_outputs_are_bitwise_identical(self):
        self._exact_results("cuda:0")

    def test_prefetch_runs_while_consumer_waits_and_remains_bounded(self):
        service, legacy, model = self._service()
        started, second = [], threading.Event()
        def features(row, **_):
            started.append(row["sequence_idx"])
            if len(started) >= 2:
                second.set()
            return np.ones((2, 4), dtype=np.float32)
        legacy.build_value_features = features
        reqs = [request(i) for i in range(12)]
        source = [dict(request_id=r.request_id, records=[record(i, 2)]) for i, r in enumerate(reqs)]
        callbacks = []
        scored = service._score_prepared_entries(legacy, source, {r.request_id: r for r in reqs},
                    service.config.legacy_args(), lambda _: callbacks.append(threading.get_ident()))
        try:
            self.assertEqual(next(scored)[0], reqs[0])
            self.assertTrue(second.wait(5))
            # One current result, one score in flight, one feature lookahead.
            self.assertLessEqual(len(started), 3)
            self.assertEqual(callbacks, [threading.get_ident()])
        finally:
            scored.close()
        self.assertFalse(any(t.name == "prm-ordered-pipeline" for t in threading.enumerate()))

    def test_memory_profile_does_not_prefetch_and_errors_propagate(self):
        service, legacy, _ = self._service(profile="memory")
        reqs = [request(i) for i in range(2)]
        entries = [dict(request_id=r.request_id, records=[record(i, 2)]) for i, r in enumerate(reqs)]
        legacy.build_value_features = mock.Mock(return_value=np.ones((2, 4), dtype=np.float32))
        scored = service._score_prepared_entries(legacy, entries, {r.request_id: r for r in reqs}, service.config.legacy_args())
        next(scored)
        self.assertEqual(legacy.build_value_features.call_count, 1)
        scored.close()
        service.config.performance_profile = "throughput"
        legacy.build_value_features.side_effect = ValueError("bad feature")
        with self.assertRaisesRegex(ValueError, "bad feature"):
            list(service._score_prepared_entries(legacy, entries, {r.request_id: r for r in reqs}, service.config.legacy_args()))

    def test_next_features_overlap_current_forward(self):
        service, legacy, model = self._service()
        forward_started, feature_ready = threading.Event(), threading.Event()
        def features(row, **_):
            if row["sequence_idx"] == 1:
                self.assertTrue(forward_started.wait(5))
                feature_ready.set()
            return np.ones((2, 4), dtype=np.float32)
        legacy.build_value_features = features
        original_forward = model.forward
        calls = []
        def forward(*args, **kwargs):
            if not calls:
                calls.append(True)
                forward_started.set()
                self.assertTrue(feature_ready.wait(5))
            return original_forward(*args, **kwargs)
        reqs = [request(i) for i in range(2)]
        entries = [dict(request_id=r.request_id, records=[record(i, 2)]) for i, r in enumerate(reqs)]
        with mock.patch.object(model, "forward", side_effect=forward):
            values = list(service._score_prepared_entries(legacy, entries, {r.request_id: r for r in reqs}, service.config.legacy_args()))
        self.assertEqual([r.request_id for r, _ in values], [r.request_id for r in reqs])

    def test_dp_refill_preserves_success_before_input_error_and_cleans_up(self):
        model, _, _ = model_and_inputs()
        parallel = ParallelValueModel(model, [("cpu",), ("cpu",)])
        def entries():
            yield 0
            yield 1
            raise ValueError("input read failed")
        values = parallel.map_requests(lambda item: item, entries())
        try:
            self.assertEqual(next(values), 0)
            with self.assertRaisesRegex(ValueError, "input read failed"):
                next(values)
        finally:
            values.close()
            parallel.close()


def result(index=0, text="answer"):
    req = request(index)
    candidates = [BestOfNCandidate(i, f"{req.request_id}:c{i}", text, .5,
                                  dict(full_output=text, num_generated_tokens=4), dict(steps=[dict(step=text)]))
                  for i in range(2)]
    return BestOfNResult(req, "product", 0, candidates[0].candidate_key, text, .5, candidates)


class NaturalPlanWriterTests(unittest.TestCase):
    def test_resume_keys_match_legacy_including_budget_and_partial_pairs(self):
        runner = importlib.import_module("NaturalPlan.bon_runner_trip")
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            directory = Path(folder)
            rows = [dict(sample_id=str(i)) for i in range(4)]
            with NaturalPlanResultWriter(directory) as writer:
                for index in range(3):
                    writer.write(result(index), {str(index): {}}, lambda *args: True)
            with (directory / "summary.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(dict(sample_id="3", sequence_idx=0, num_generated_tokens=4)) + "\n")
                file.write('{"truncated":')
            for budget in (3, 4, 10):
                summary = runner._read_jsonl(directory / "summary.jsonl")
                steps = runner._read_jsonl(directory / "steps.jsonl")
                expected = {f"naturalplan:trip:{row['sample_id']}" for row in rows
                            if all((row["sample_id"], i) in summary and (row["sample_id"], i) in steps
                                   for i in range(2))
                            and not any(summary[(row["sample_id"], i)].get("num_generated_tokens", 0) > budget
                                        for i in range(2))}
                self.assertEqual(completed_request_ids(directory, "", rows, 2, budget, "trip"), expected)

    def test_all_three_runner_entrypoints_use_writer_and_resume_without_rescoring(self):
        for subset in ("calendar", "meeting", "trip"):
            runner = importlib.import_module(f"NaturalPlan.bon_runner_{subset}")
            generated, closed = [], []
            class Service:
                def __init__(self, config):
                    self.config = config
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    pass
                def generate_and_select_many(self, requests, **kwargs):
                    try:
                        for req in requests:
                            if req.request_id not in kwargs["skip_request_ids"]:
                                generated.append(req.request_id)
                                value = result(int(req.sample_id))
                                value.request = req
                                yield value
                    finally:
                        closed.append(True)
            with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
                directory = Path(folder)
                data = directory / "input.json"
                data.write_text(json.dumps({str(i): dict(prompt_0shot="q", golden_plan="answer")
                                            for i in (2, 0, 1)}), encoding="utf-8")
                output = directory / "results"
                argv = ["--input_json", str(data), "--model_path", "m", "--checkpoint_path", "c",
                        "--output_dir", str(output), "--BoN_N", "2", "--ns", "1", "2"]
                with mock.patch.object(runner, "BestOfNService", Service), \
                        mock.patch.object(runner, "evaluate_nested_bon"), \
                        mock.patch.object(runner, "correctness_fn", return_value=lambda *args: True):
                    runner.run_bon(subset, "unused", argv)
                    first = (output / "summary.jsonl").read_bytes(), (output / "steps.jsonl").read_bytes()
                    runner.run_bon(subset, "unused", argv + ["--resume"])
                    self.assertEqual(first, ((output / "summary.jsonl").read_bytes(), (output / "steps.jsonl").read_bytes()))
                self.assertEqual(len(generated), 3)
                self.assertEqual(len(closed), 2)

    def test_exact_legacy_files_for_three_subsets_and_replacement_resume(self):
        for subset in ("calendar", "meeting", "trip"):
            runner = importlib.import_module(f"NaturalPlan.bon_runner_{subset}")
            with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
                before, after = Path(folder) / "before", Path(folder) / "after"
                before.mkdir(); after.mkdir()
                rows = {str(i): {} for i in range(4)}
                for indices in ([3, 1], [2, 0, 1]):
                    with NaturalPlanResultWriter(after, "_suffix") as writer:
                        for index in indices:
                            value = result(index, "中文" + str(len(indices)))
                            runner._write_final_record_pair(before, value, rows, lambda *args: True, "_suffix")
                            writer.write(value, rows, lambda *args: True)
                    for filename in ("summary_suffix.jsonl", "steps_suffix.jsonl"):
                        self.assertEqual((before / filename).read_bytes(), (after / filename).read_bytes())
                self.assertEqual({p.name for p in after.iterdir()}, {"summary_suffix.jsonl", "steps_suffix.jsonl"})

    def test_partial_tail_unpaired_rows_missing_newline_and_duplicates(self):
        runner = importlib.import_module("NaturalPlan.bon_runner_trip")
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            directory = Path(folder)
            row = dict(sample_id="0", sequence_idx=0, full_output="old")
            (directory / "summary.jsonl").write_text(json.dumps(row) + '\n{"partial":', encoding="utf-8")
            (directory / "steps.jsonl").write_text(json.dumps(dict(row, steps=[])), encoding="utf-8")
            with NaturalPlanResultWriter(directory) as writer:
                writer.write(result(0, "new"), {"0": {}}, lambda *args: False)
                writer.write(result(0, "latest"), {"0": {}}, lambda *args: True)
            for filename in ("summary.jsonl", "steps.jsonl"):
                rows = runner._read_jsonl(directory / filename)
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["is_correct"] for row in rows.values()))
                self.assertEqual(len((directory / filename).read_text().splitlines()), 2)

    def test_append_does_not_decode_or_rewrite_historical_payloads(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            directory = Path(folder)
            with NaturalPlanResultWriter(directory) as writer:
                writer.write(result(0), {"0": {}}, lambda *args: True)
                with mock.patch("scripts.transformer_bon.naturalplan_records.json.loads", side_effect=AssertionError("decoded history")), \
                        mock.patch("scripts.transformer_bon.naturalplan_records.os.replace", side_effect=AssertionError("rewrote history")):
                    writer.write(result(1), {"1": {}}, lambda *args: True)
                self.assertTrue(all(isinstance(value, int) for value in writer.summary.locations.values()))

    def test_grading_failure_closes_files_without_leaving_temp_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            directory = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "grading failed"):
                with NaturalPlanResultWriter(directory) as writer:
                    writer.write(result(0), {"0": {}}, lambda *args: True)
                    writer.write(result(1), {"1": {}}, mock.Mock(side_effect=RuntimeError("grading failed")))
            self.assertTrue(writer.summary.file.closed and writer.steps.file.closed)
            self.assertEqual(len((directory / "summary.jsonl").read_text().splitlines()), 2)
            self.assertFalse(list(directory.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
