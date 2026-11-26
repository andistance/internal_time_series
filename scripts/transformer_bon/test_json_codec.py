"""Lossless JSON IPC, exact phase aggregation, bounds and worker lifecycle."""
import copy
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from . import json_codec as codec
from .naturalplan_records import NaturalPlanResultWriter
from .test_value_parallel import legacy_definitions
from .test_value_throughput import result
from .value_features import attach_phase_scores
from .scoring_profile import ScoringProfile
from . import test_value_throughput as helpers
from .record_index import write_records


ROOT = Path(__file__).resolve().parents[2]


class CustomFloat(float):
    pass


class JsonCodecTests(unittest.TestCase):
    def tearDown(self):
        codec.close_codecs()

    def row(self, index=0):
        rng = np.random.default_rng(19 + index)
        return dict(token_entropies=rng.normal(size=12000).tolist(), steps=[],
                    full_output="中文🙂\n\\\"", index=index, zero=-0.0, integer=2**90)

    def test_exact_encoding_and_float_bits_in_both_directions(self):
        rows = [self.row(i) for i in range(5)]
        expected = [json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8") for row in rows]
        encoded = list(codec.iter_encoded(rows))
        self.assertEqual([payload for _, payload in encoded], expected)
        for row, payload in encoded:
            decoded = codec.decode_json(payload)
            self.assertEqual(decoded, row)
            self.assertEqual(np.asarray(decoded["token_entropies"]).tobytes(), np.asarray(row["token_entropies"]).tobytes())
            self.assertEqual(struct.pack("d", decoded["zero"]), struct.pack("d", row["zero"]))
        self.assertEqual(len(codec.codec_stats()), 3)

    def test_small_oversized_and_disabled_rows_do_not_start_workers(self):
        for row in (dict(x=1), dict(full_output="x" * (codec.MAX_BYTES + 100))):
            self.assertEqual(codec.encode_json(row), json.dumps(row, ensure_ascii=False, allow_nan=False).encode())
        with mock.patch.dict(os.environ, TRANSFORMER_BON_JSON_WORKERS="0"):
            row = self.row()
            self.assertEqual(codec.encode_json(row), json.dumps(row, ensure_ascii=False, allow_nan=False).encode())
            self.assertEqual(codec.decode_json(json.dumps(row)), row)
        self.assertEqual(codec.codec_stats(), {})

    def test_custom_float_stays_compatible_without_importing_model_modules(self):
        row = self.row()
        row["token_entropies"][0] = CustomFloat(.25)
        self.assertEqual(codec.encode_json(row), json.dumps(row, ensure_ascii=False, allow_nan=False).encode())

    def test_json_errors_match_and_worker_recovers(self):
        row = self.row()
        row["token_entropies"][2] = float("nan")
        with self.assertRaises(ValueError):
            codec.encode_json(row)
        invalid = b'{"padding":"' + b"x" * codec.MIN_BYTES
        for decode in (json.loads, codec.decode_json):
            with self.assertRaises(json.JSONDecodeError) as error:
                decode(invalid)
            self.assertEqual(error.exception.pos, 11)
        valid = self.row()
        self.assertEqual(codec.decode_json(codec.encode_json(valid)), valid)

    def test_early_iterator_close_and_worker_exit_are_not_silent(self):
        encoded = codec.iter_encoded([self.row(i) for i in range(6)])
        next(encoded)
        processes = [worker.process for worker in codec._workers.values()]
        encoded.close()
        codec.close_codecs()
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertFalse(any(t.name.startswith("value-json-pipe") for t in threading.enumerate()))
        codec.encode_json(self.row())
        worker = codec._workers[(os.getpid(), "encode")]
        worker.process.kill()
        worker.process.wait()
        with self.assertRaises((RuntimeError, OSError)):
            codec.encode_json(self.row())

    def test_compaction_copies_new_bytes_without_json_decoding(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            directory = Path(folder)
            writer = NaturalPlanResultWriter(directory)
            writer.write(result(3), {"3": {}}, lambda *args: True)
            writer.write(result(1), {"1": {}}, lambda *args: True)
            with mock.patch("scripts.transformer_bon.naturalplan_records.decode_json", side_effect=AssertionError("redecoded")), \
                    mock.patch("scripts.transformer_bon.naturalplan_records.encode_json", side_effect=AssertionError("reencoded")):
                writer.__exit__(None, None, None)
            rows = [json.loads(line) for line in (directory / "summary.jsonl").read_bytes().splitlines()]
            self.assertEqual([row["sample_id"] for row in rows], ["1", "1", "3", "3"])

    def test_writer_snapshots_reused_record_metadata_and_bytes(self):
        row = self.row()
        def records():
            for index in range(4):
                row["candidate_key"] = str(index)
                row["token_entropies"][0] = float(index)
                yield row
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as folder:
            path = Path(folder) / "out.jsonl"
            refs = write_records(path, records())
            values = [json.loads(line) for line in path.read_bytes().splitlines()]
            self.assertEqual([v["candidate_key"] for v in values], list(map(str, range(4))))
            self.assertEqual([v["token_entropies"][0] for v in values], list(map(float, range(4))))
            self.assertEqual([ref[0] for ref in refs], list(map(str, range(4))))

    def test_host_batch_bytes_match_original_torch_collation(self):
        import torch
        for kind in ("causal", "bidirectional"):
            service, legacy, model = helpers.ValueThroughputTests()._service(kind)
            rows = [helpers.record(0, 7), helpers.record(1, 2), helpers.record(2, 5)]
            items = sorted([(row, service._value_components[1].transform_one(legacy.build_value_features(row, kind)))
                            for row in rows], key=lambda item: item[1].shape[0])
            expected = []
            for start in range(0, len(items), 2):
                arrays = [array for _, array in items[start:start + 2]]
                width = max(len(array) for array in arrays)
                x = torch.zeros(len(arrays), width, 4, dtype=torch.float32)
                mask = torch.zeros(len(arrays), width, dtype=torch.bool)
                for index, array in enumerate(arrays):
                    x[index, :len(array)] = torch.from_numpy(array)
                    mask[index, :len(array)] = True
                expected.append((x.numpy().tobytes(), mask.numpy().tobytes()))
            observed = []
            hook = model.register_forward_pre_hook(lambda _, args, kwargs:
                observed.append((args[0].numpy().tobytes(), kwargs["attention_mask"].numpy().tobytes())), with_kwargs=True)
            try:
                service._score(legacy, helpers.request(), rows, service.config.legacy_args())
                self.assertEqual(observed, expected)
            finally:
                hook.remove()

    def test_exact_phase_reductions_for_all_runtimes(self):
        rng = np.random.default_rng(73)
        names = [p.name for p in (ROOT / "scripts").glob("3_best_of_n_via_phases_*.py")]
        for name in names:
            legacy = legacy_definitions(name)
            for count in (1, 2, 3, 8, 33, 128):
                for _ in range(8):
                    predictions = rng.uniform(-.1, 1.1, count).astype("float32").tolist()
                    ranges = [[i + 1, i + 1] for i in range(count)]
                    before, after = [{} for _ in range(count)], [{} for _ in range(count)]
                    expected = legacy.attach_phase_scores_to_steps(before, ranges, predictions)
                    actual = attach_phase_scores(legacy, after, ranges, predictions)
                    self.assertEqual(json.dumps((before, expected)), json.dumps((after, actual)))

    def test_profile_is_opt_in_and_reports_codec_work(self):
        with mock.patch.dict(os.environ, TRANSFORMER_BON_PROFILE="0"):
            profile = ScoringProfile()
            self.assertIsNone(profile.start())
        with mock.patch.dict(os.environ, TRANSFORMER_BON_PROFILE="1"):
            profile = ScoringProfile()
            profile.add("forward_with_transfers", profile.start())
            output = io.StringIO()
            with mock.patch("sys.stdout", output):
                profile.report()
            self.assertIn("forward_with_transfers=", output.getvalue())
            self.assertIn("Concurrent stages overlap", output.getvalue())


if __name__ == "__main__":
    unittest.main()
