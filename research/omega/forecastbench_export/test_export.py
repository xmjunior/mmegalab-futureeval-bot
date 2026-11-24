import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from export import build, read_json

HERE = Path(__file__).parent


class ExportTests(unittest.TestCase):
    def setUp(self):
        sample = read_json(HERE / 'sample.json')
        self.questions = sample['questions']
        self.predictions = sample['predictions']

    def test_historical_full_round_excludes_unresolved_and_imputed(self):
        _, counts = build(read_json(HERE / 'round-index.json.gz'), read_json(HERE / 'resolved-archive.json.gz'))
        self.assertEqual((counts['market']['complete_questions'], counts['market']['total_questions']), (117, 250))
        self.assertEqual((counts['dataset']['present_slots'], counts['dataset']['requested_slots']), (1220, 1996))
        self.assertEqual(counts['dataset']['complete_questions'], 0)
        self.assertFalse(any(x['pass'] for x in counts.values()))

    def test_small_index_cannot_be_presented_as_a_complete_official_round(self):
        self.questions['question_set'] = self.predictions['question_set'] = '2025-10-26-llm.json'
        with self.assertRaisesRegex(ValueError, 'incomplete official question index'):
            build(self.questions, self.predictions)

    def test_archived_fixture_and_report_match_the_offline_sources(self):
        full_question_set = read_json(HERE / 'round-index.json.gz')
        full_predictions = read_json(HERE / 'resolved-archive.json.gz')
        full_rows = {(r['source'], r['id'], r['resolution_date']): r['forecast'] for r in full_predictions['forecasts']}
        requested_dates = {(q['source'], q['id']): q['resolution_dates'] for q in full_question_set['questions']}
        for question in self.questions['questions']:
            source_dates = requested_dates[question['source'], question['id']]
            fixture_dates = question['resolution_dates']
            if source_dates == 'N/A':
                self.assertEqual(fixture_dates, 'N/A')
            else:
                self.assertTrue(set(fixture_dates) <= set(source_dates))
        for row in self.predictions['forecasts']:
            self.assertEqual(row['forecast'], full_rows[row['source'], row['id'], row['resolution_date']])
        report = read_json(HERE / 'results.json')
        for key, questions, predictions in (
            ('full_round_closed_rows', full_question_set, full_predictions),
            ('small_resolved_fixture', self.questions, self.predictions),
        ):
            with self.subTest(case=key):
                result, coverage = build(questions, predictions)
                self.assertEqual(coverage, report[key]['coverage'])
                self.assertEqual(len(result['forecasts']), report[key]['present_forecasts'])

    def test_projection_and_exact_threshold(self):
        out, counts = build(self.questions, self.predictions)
        self.assertEqual(out['organization'], 'ForecastBench')
        self.assertEqual(out['model_organization'], 'OpenAI')
        self.assertEqual(set(out), {'question_set', 'forecasts', 'organization', 'model', 'model_organization'})
        self.assertEqual(len(out['forecasts']), 40)
        self.assertEqual(set(out['forecasts'][0]), {'id', 'source', 'forecast', 'reasoning', 'resolution_date'})
        self.assertTrue(all(c['complete_questions'] == 20 and c['pass'] for c in counts.values()))
        self.predictions['forecasts'].pop()
        self.predictions['forecasts'].pop(0)
        _, counts = build(self.questions, self.predictions)
        self.assertTrue(all(c['complete_questions'] == 19 and c['pass'] for c in counts.values()))
        self.predictions['forecasts'].pop()
        _, counts = build(self.questions, self.predictions)
        self.assertEqual((counts['dataset']['pass'], counts['market']['pass']), (False, True))

    def test_duplicates_missing_fields_and_forged_horizons(self):
        rows = self.predictions['forecasts']
        rows.append(rows[0])
        with self.assertRaisesRegex(ValueError, 'duplicate forecast'):
            build(self.questions, self.predictions)
        rows.pop()
        for field in ('resolution_date', 'id', 'source', 'forecast'):
            with self.subTest(field=field):
                data = copy.deepcopy(self.predictions)
                del data['forecasts'][0][field]
                with self.assertRaises(KeyError):
                    build(self.questions, data)
        rows[0]['resolution_date'] = '2026-01-01'
        with self.assertRaisesRegex(ValueError, 'unknown horizon'):
            build(self.questions, self.predictions)

    def test_no_cross_round_and_probability_validation(self):
        for bad in [True, '0.5', float('inf'), float('nan'), -0.01, 1.01, 10**400]:
            with self.subTest(bad=bad):
                self.predictions['forecasts'][0]['forecast'] = bad
                with self.assertRaisesRegex(ValueError, 'finite'):
                    build(self.questions, self.predictions)
        self.predictions['question_set'] = 'unrelated-set'
        with self.assertRaisesRegex(ValueError, 'different question sets'):
            build(self.questions, self.predictions)

    def test_a_missing_dataset_horizon_removes_the_question_from_coverage(self):
        self.questions['questions'][-1]['resolution_dates'].append('2030-01-01')
        _, result = build(self.questions, self.predictions)
        self.assertEqual(result['dataset']['complete_questions'], 19)
        self.assertEqual(result['dataset']['requested_slots'], 21)

    def test_cli_exports_only_passing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            dst = Path(temp) / 'historical-fixture-export.json'
            cmd = [sys.executable, str(HERE / 'export.py')]
            result = subprocess.run(cmd + [str(HERE / 'round-index.json.gz'), str(HERE / 'resolved-archive.json.gz'), '--output', str(dst)], capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(dst.exists())
            paths = [Path(temp) / name for name in ('q.json', 'p.json')]
            for path, data in zip(paths, (self.questions, self.predictions)):
                path.write_text(json.dumps(data))
            result = subprocess.run(cmd + [*paths, '--output', str(dst)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_json(dst)['question_set'], self.questions['question_set'])


if __name__ == '__main__':
    unittest.main()
