"""Regression checks for the opt-in service and six step-only entry points."""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'LiveCodeBench'))
import test_prm_throughput as fixtures
from test_adaptive_phase_partition import CharacterTokenizer, load_cpu_runtime, make_generation
from scripts.transformer_bon import StepPRMBestOfNConfig, StepPRMBestOfNService
from scripts.transformer_bon.engine import BestOfNService, _append_stage_records
from scripts.transformer_bon.entropy_cache import append_entropy_record, build_token_entropy_record
from scripts.transformer_bon.prm import PRMBestOfNService, aggregate_phase_scores
from scripts.transformer_bon.step_prm import build_step_prepared_record
from scripts.transformer_bon.step_entry import load_step_entry, step_parser, livecodebench_step_service


def config(**kwargs):
    return StepPRMBestOfNConfig(model_path='model', checkpoint_path='prm', prm_model_path='prm',
                                n=3, chat_template='raw', scoring_batch_size=2, **kwargs)


class StepModeTests(unittest.TestCase):
    def test_flags_and_standard_preparation_identity(self):
        service = StepPRMBestOfNService(config())
        self.assertTrue(service.config.legacy_args().prm_step_scoring)
        self.assertIs(service._prepare_with_entropy_model, BestOfNService._prepare_with_entropy_model)
        self.assertIs(service._score.__func__, PRMBestOfNService._score)

    def test_all_backends_score_all_steps_and_match_singleton_phase_results(self):
        from scripts.transformer_bon.prms import available_prm_names
        for name in available_prm_names():
            with self.subTest(backend=name):
                request, records = fixtures.make_request(0), fixtures.make_records(0)
                for row in records:
                    row['phase_ranges'] = [[1, 2]]
                step = StepPRMBestOfNService(config(prm_name=name))
                step._prm_scorer = fixtures.FakeScorer()
                actual = step._score(None, request, records, step.config.legacy_args())
                phase = PRMBestOfNService(config(prm_name=name))
                phase._prm_scorer = fixtures.FakeScorer()
                baseline = fixtures.make_records(0)
                for row in baseline:
                    for i, item in enumerate(row['steps']):
                        item['phase_frontier'] = 0 if i == 0 else 2
                expected = phase._score(None, request, baseline, phase.config.legacy_args())
                self.assertEqual([asdict(c) for c in actual], [asdict(c) for c in expected])
                self.assertTrue(all(row['phase_ends'] == [1, 2] for batch in step._prm_scorer.batches for row in batch))
                self.assertEqual(records[0]['phase_ranges'], [[1, 2]])
                step.close(); phase.close()

    def test_preparation_preserves_metadata_and_includes_steps_after_think(self):
        legacy = load_cpu_runtime()
        generation = make_generation()
        args = config().legacy_args()
        entropies = [.5] * len(generation['response_token_ids'])
        with mock.patch.object(legacy, 'select_entropy_boundary_token_positions', side_effect=AssertionError('phase split')):
            prepared = build_step_prepared_record(legacy, generation, CharacterTokenizer(), entropies, args)
        self.assertEqual(prepared['phase_ranges'], [[i, i] for i in range(1, 6)])
        self.assertEqual(prepared['num_steps'], 5)
        self.assertEqual(prepared['steps'][-1]['phase_frontier'], 2)
        self.assertEqual(prepared['token_entropies'], entropies)
        expected = legacy.build_prepared_record(generation, CharacterTokenizer(), entropies, args)
        self.assertEqual(set(prepared), set(expected))
        for field in ('full_output', 'step_avg_confidences', 'num_generated_tokens', 'num_steps', 'token_entropies'):
            self.assertEqual(prepared[field], expected[field])
        args.no_save_token_entropies = True
        self.assertNotIn('token_entropies', build_step_prepared_record(legacy, generation, CharacterTokenizer(), entropies, args))

    def test_cached_stage_uses_no_phase_workers_and_resumes_without_extra_rows(self):
        legacy = load_cpu_runtime()
        legacy.load_tokenizer_without_vllm = mock.Mock(return_value=CharacterTokenizer())
        legacy.load_entropy_model = mock.Mock(side_effect=AssertionError('cached entropy loaded a model'))
        service = StepPRMBestOfNService(config())
        service._prm_scorer = fixtures.FakeScorer()
        service._prm_scorer.score_batch = lambda texts: [[.75] * len(row['phase_ends']) for row in texts]
        request = replace(fixtures.make_request(0), request_id='r', sample_id='q', task_text='question')
        rows = [make_generation(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _append_stage_records(root / '.bon_generation.jsonl', 'r', rows)
            for row in rows:
                append_entropy_record(str(root / '.bon_token_entropies.jsonl'),
                                      build_token_entropy_record(row, [.5] * len(row['response_token_ids']), service.config.legacy_args()))
            with mock.patch('scripts.transformer_bon.engine.load_legacy_module', return_value=legacy), \
                    mock.patch('scripts.transformer_bon.engine.run_parallel_phases', side_effect=AssertionError('phase worker')), \
                    mock.patch('scripts.transformer_bon.engine.run_serial_phases', side_effect=AssertionError('phase split')), \
                    mock.patch.object(service, '_preflight_prm_inputs'):
                first = list(service.generate_and_select_many([request], staging_dir=root, resume=True))
                files = {p.name: p.read_bytes() for p in root.glob('*.jsonl')}
                second = list(service.generate_and_select_many([request], staging_dir=root, resume=True))
                self.assertEqual(files, {p.name: p.read_bytes() for p in root.glob('*.jsonl')})
                self.assertEqual([asdict(c) for c in first[0].candidates], [asdict(c) for c in second[0].candidates])
                self.assertEqual(first[0].candidates[0].summary_record['num_steps'], 5)
                self.assertEqual(len((root / '.bon_prm.jsonl').read_text(encoding='utf-8').splitlines()), 3)
            service.close()

    def test_prefix_aggregates_and_invalid_steps(self):
        service = StepPRMBestOfNService(config())
        scores = np.random.default_rng(17).uniform(.5, 1, 200).tolist()
        for i, result in enumerate(service._cumulative_prm_scores(scores)):
            expected = aggregate_phase_scores(scores[:i+1])
            np.testing.assert_allclose(list(result.values()), list(expected.values()), rtol=2e-13)
        rows = fixtures.make_records(0); rows[0]['steps'].pop()
        with self.assertRaisesRegex(ValueError, 'step count'):
            service._score(None, fixtures.make_request(0), rows, service.config.legacy_args())
        service.close()

    def test_prepared_metadata_matches_original_for_unicode_and_empty_steps(self):
        legacy = load_cpu_runtime()
        for text in ('', ' a ', ' a\n\n \n\n b </think>\n\n答案🙂', 'x\n\ny\n\nz'):
            generation = make_generation(text=text)
            args = config().legacy_args()
            entropies = [.1] * len(generation['response_token_ids'])
            expected = legacy.build_prepared_record(generation, CharacterTokenizer(), entropies, args)
            with mock.patch('scripts.transformer_bon.phase_structure.np.argsort', side_effect=AssertionError('phase entropy sort')):
                actual = build_step_prepared_record(legacy, generation, CharacterTokenizer(), entropies, args)
            for field in set(expected) - {'phase_ranges', 'boundary_token_positions', 'num_entropy_threshold_candidates', 'steps'}:
                self.assertEqual(actual[field], expected[field], (text, field))
            for a, b in zip(actual['steps'], expected['steps']):
                self.assertEqual({k:v for k,v in a.items() if k != 'phase_frontier'},
                                 {k:v for k,v in b.items() if k != 'phase_frontier'})

    def test_many_steps_use_one_job_and_one_selection_aggregate_per_candidate(self):
        from scripts.transformer_bon import prm
        service = StepPRMBestOfNService(config())
        service._prm_scorer = fixtures.FakeScorer()
        service._prm_scorer.score_batch = lambda rows: [[.8] * len(r['phase_ends']) for r in rows]
        rows = fixtures.make_records(0)
        for row in rows:
            row.update(full_output='\n\n'.join(['x'] * 256), num_steps=256,
                       step_avg_confidences=[.9] * 256, steps=[{'step': 'x'} for _ in range(256)],
                       phase_ranges=[[1, 256]])
        with mock.patch.object(prm, 'aggregate_phase_scores', wraps=prm.aggregate_phase_scores) as aggregate, \
                mock.patch.object(service, '_run_prm_jobs', wraps=service._run_prm_jobs) as scoring:
            values = service._score(None, fixtures.make_request(0), rows, service.config.legacy_args())
        self.assertEqual(len(scoring.call_args.args[0]), 3)
        self.assertEqual(aggregate.call_count, 3)
        self.assertTrue(all(len(c.summary_record['phase_scores']) == 256 for c in values))
        service.close()

    def test_single_request_prepare_hook_skips_phase_selection(self):
        legacy = load_cpu_runtime()
        generation = make_generation()
        legacy.compute_response_token_entropies_exact = mock.Mock(
            return_value=[.2] * len(generation['response_token_ids']))
        legacy.build_prepared_record = mock.Mock(side_effect=AssertionError('phase builder'))
        rows = BestOfNService._prepare_with_entropy_model(
            legacy, [generation], config().legacy_args(), CharacterTokenizer(), object())
        self.assertEqual(rows[0]['request_id'], 'r')
        self.assertEqual(rows[0]['phase_ranges'], [[i, i] for i in range(1, 6)])
        legacy.compute_response_token_entropies_exact.assert_called_once()


class EntryTests(unittest.TestCase):
    def test_gpqa_and_bbh_parsers_keep_every_other_action(self):
        import GPQA.baselines.run_prm_bon as gpqa
        import GPQA.baselines.run_step_prm_bon as gpqa_step
        import BBH.run_prm_bon as bbh
        import BBH.run_step_prm_bon as bbh_step
        removed = {'num_segments', 'entropy_threshold', 'phase_partition_algorithm', 'high_entropy_ratio', 'target_phase_tokens'}
        for original, new in ((gpqa, gpqa_step), (bbh, bbh_step)):
            parser, reference = new.build_parser(), original.build_parser()
            expected = {a.dest: a for a in reference._actions if a.dest.removeprefix('bon_') not in removed}
            actual = {a.dest: a for a in parser._actions}
            self.assertEqual(set(expected), set(actual))
            for name, action in actual.items():
                for attr in ('option_strings', 'default', 'required', 'choices', 'type', 'nargs'):
                    self.assertEqual(getattr(action, attr), getattr(expected[name], attr))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(['--bon_K', '4'])
            self.assertIs(new._entry.PRMBestOfNService if original is gpqa else new._entry._RuntimePRMBestOfNService,
                          StepPRMBestOfNService)
        self.assertIs(gpqa.PRMBestOfNService, PRMBestOfNService)

    def test_naturalplan_parsers_are_prm_only_and_drop_all_partition_options(self):
        for subset in ('calendar', 'meeting', 'trip'):
            entry = load_step_entry(ROOT / f'NaturalPlan/bon_runner_{subset}.py', f'NaturalPlan._test_step_{subset}')
            class Captured(Exception):
                pass
            captured = []
            class Parser(argparse.ArgumentParser):
                def parse_args(self, *args, **kwargs):
                    captured.append(step_parser(self))
                    raise Captured
            entry.argparse = SimpleNamespace(**{**vars(argparse), 'ArgumentParser': Parser})
            with self.assertRaises(Captured):
                entry.run_bon(subset, 'unused')
            parser = captured[0]
            dests = {a.dest for a in parser._actions}
            self.assertFalse({'num_segments', 'entropy_threshold', 'phase_partition_algorithm', 'high_entropy_ratio', 'target_phase_tokens'} & dests)
            score = next(a for a in parser._actions if a.dest == 'score_model')
            self.assertEqual(score.choices, ['prm']); self.assertEqual(score.default, 'prm')
            self.assertIs(entry.PRMBestOfNService, StepPRMBestOfNService)

    def test_livecodebench_config_and_evaluator_are_unchanged_except_step_service(self):
        from lcb_runner.runner.transformer_bon_runner import TransformerBoNRunner
        from scripts.transformer_bon.step_entry import load_step_entry
        parser = load_step_entry(ROOT / 'LiveCodeBench/lcb_runner/runner/parser.py', 'lcb_runner.runner._test_step_parser')
        with mock.patch.object(sys, 'argv', ['lcb', '--bon_model_path', 'generator', '--bon_checkpoint_path', 'reward',
                                           '--bon_prm_model_path', 'reward', '--bon_output_dir', 'unused', '--pass_k', '1']):
            args = parser.get_args()
        self.assertEqual(args.bon_scoring_backend, 'prm')
        self.assertEqual(args.generation_backend, 'transformer_bon')
        with mock.patch('scripts.transformer_bon.PRMBestOfNService') as ctor:
            TransformerBoNRunner(args, object())
            expected = ctor.call_args.args[0]
        service = livecodebench_step_service(args)
        self.assertEqual(asdict(service.config), asdict(expected))
        service.close()
        def functions(path):
            return {n.name: n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)}
        path = ROOT / 'LiveCodeBench/lcb_runner/runner/main.py'
        old, new = functions(path), functions(path.with_name('main_step_prm.py'))
        for node in ast.walk(new['_run_transformer_bon']):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'TransformerBoNRunner':
                node.keywords = []
        self.assertEqual({k: ast.dump(v) for k, v in old.items()}, {k: ast.dump(v) for k, v in new.items()})


if __name__ == '__main__':
    unittest.main()
