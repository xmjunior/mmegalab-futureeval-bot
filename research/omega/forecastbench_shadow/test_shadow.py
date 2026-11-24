from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import model
import shadow
from export import read_json

UTC = timezone.utc
START = shadow.START
MID = datetime.combine(START, datetime.min.time(), tzinfo=UTC) + timedelta(hours=10)


def historical_fixture():
    doc = read_json(shadow.HERE.parent / 'forecastbench_export' / 'round-index.json.gz')
    old = date.fromisoformat(doc['forecast_due_date'])
    for q in doc['questions']:
        if isinstance(q['resolution_dates'], list):
            q['resolution_dates'] = [(START + (date.fromisoformat(t)-old)).isoformat() for t in q['resolution_dates']]
    doc.update(forecast_due_date=START.isoformat(), question_set=START.isoformat()+'-llm.json')
    return doc


def provenance(time=MID):
    return {'committed_at_utc':time.isoformat(), 'revision':'1'*40, 'sha256':'2'*64, 'url':'offline synthetic fixture'}


class ShadowTests(unittest.TestCase):
    def test_frozen_training_and_replay(self):
        panel = model.rows()
        fitted = model.fit(panel)
        expected = json.loads((shadow.HERE/'results.json').read_bytes())
        self.assertEqual(model.sha(model.canonical(fitted)), shadow.MODEL_SHA)
        self.assertEqual(model.replay(), expected)
        for row in panel:
            if row[0] != 'train':
                row[-2] = 1-row[-2]
            row[6:9] = [0.0]*3
        self.assertEqual(model.fit(panel), fitted)
        self.assertEqual(model.prediction(fitted, 'new-source', 1), (0.5, 'neutral'))
        self.assertEqual(model.prediction(fitted, 'fred', 3650)[1], 'source')

    def test_schedule_utc_and_expired_issues(self):
        self.assertTrue(shadow.active(START))
        self.assertTrue(shadow.active(START+timedelta(days=14)))
        self.assertFalse(shadow.active(START+timedelta(days=1)))
        self.assertFalse(shadow.active(date(2027,4,11)))
        with tempfile.TemporaryDirectory() as root, patch.object(shadow,'now',return_value=MID+timedelta(days=1)), patch.object(shadow,'snapshot') as network:
            state = Path(root) / 'state'
            shadow.run(state, state/'receipts')
            network.assert_not_called()
            self.assertFalse(state.exists())

    def test_malformed_or_cross_midnight_issues_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            q = historical_fixture()
            for actual, upstream in ((MID+timedelta(days=1),MID),(MID,MID+timedelta(minutes=1))):
                with self.assertRaises(ValueError):
                    shadow.seal(state,START,(provenance(upstream),q),actual)
            q['forecast_due_date'] = '2025-10-26'
            with self.assertRaises(ValueError):
                shadow.seal(state,START,(provenance(),q),MID)
            self.assertFalse(list(state.iterdir()))

    def test_seal_coverage_immutability_and_first_release_scoring(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            receipts = state / 'receipts'
            path = state / (START.isoformat()+'.json')
            q = historical_fixture()
            with patch.object(shadow, 'now', return_value=MID), patch.object(shadow,'snapshot', return_value=(provenance(),q)) as net:
                shadow.run(state, receipts)
                raw = path.read_bytes()
                shadow.run(state, receipts)
                self.assertEqual(net.call_count, 1)
                self.assertEqual(path.read_bytes(),raw)
            with self.assertRaises(FileExistsError):
                shadow.seal(state, START, (provenance(),q), MID)
            issue = shadow.read_seal(path)
            self.assertEqual(issue['coverage']['market']['complete_questions'], 250)
            self.assertEqual(issue['coverage']['dataset']['complete_questions'], 250)
            self.assertEqual(issue['coverage']['dataset']['present_slots'], 1996)
            self.assertEqual(issue['routes']['market/neutral'],250)
            self.assertFalse(shadow.receipt_ok(path, receipts, issue))
            receipts.mkdir()
            for stamp, digest, allowed in ((MID, 'bad', False),(MID-timedelta(seconds=1), model.sha(raw), False),
                (MID+timedelta(days=1), model.sha(raw), False),(MID+timedelta(minutes=1), model.sha(raw), True)):
                record = {'issue_sha256':digest,'published_at_utc':stamp.isoformat()}
                (receipts/path.name).write_text(json.dumps(record))
                self.assertEqual(shadow.receipt_ok(path,receipts,issue),allowed)
            first = next(row for row in issue['forecast_set']['forecasts'] if row['resolution_date'] is None)
            row = {'source':first['source'],'id':first['id'],'direction':None,'resolved':True,'resolved_to':1,
                   'resolution_date':(START+timedelta(days=1)).isoformat()}
            outcome = {'forecast_due_date':START.isoformat(),'question_set':q['question_set'],'resolutions':[row]}
            then = MID+timedelta(days=2)
            with patch.object(shadow,'now',return_value=then), patch.object(shadow,'snapshot',return_value=(provenance(then),outcome)):
                shadow.run(state, receipts)
                scorefile = state/'scores'/path.name
                prev = scorefile.read_bytes()
                self.assertEqual(json.loads(prev)['scores']['eligible_scored'],1)
                self.assertEqual(json.loads(prev)['scores']['forecast']['brier'],0.25)
                row['resolved_to'] = 0
                shadow.run(state, receipts)
                self.assertEqual(scorefile.read_bytes(), prev)
                changed_revision = provenance(then)
                changed_revision['revision'] = '3'*40
                with patch.object(shadow, 'snapshot',return_value=(changed_revision,outcome)):
                    shadow.run(state, receipts)
                self.assertEqual(scorefile.read_bytes(), prev)
            outcome['resolutions'][0]['resolution_date'] = START.isoformat()
            self.assertIsNone(shadow.record_scores(issue,None,model.sha(raw),(provenance(then),outcome),then))
            self.assertEqual(path.read_bytes(), raw)

    def test_upstream_is_revision_pinned_and_rejects_future(self):
        commit = {'sha':'a'*40,'commit':{'committer':{'date':MID.isoformat()}}}
        with patch.object(shadow,'get',side_effect=[json.dumps([commit]).encode(), b'{"data": 1}']) as get:
            prov, doc = shadow.snapshot('datasets/test.json', MID)
            self.assertEqual(doc['data'], 1)
            self.assertIn('/'+'a'*40+'/datasets/test.json',get.call_args_list[1].args[0])
            self.assertEqual(prov['sha256'],model.sha(b'{"data": 1}'))
        with patch.object(shadow,'get',return_value=json.dumps([commit]).encode()):
            with self.assertRaises(ValueError):
                shadow.snapshot('datasets/test.json', MID-timedelta(seconds=1))
        with patch.object(shadow,'get',return_value=b'[]'):
            self.assertIsNone(shadow.snapshot('datasets/test.json',MID))


if __name__ == '__main__':
    unittest.main()
