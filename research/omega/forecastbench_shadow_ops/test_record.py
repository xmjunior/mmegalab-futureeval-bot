from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import record

ISSUE = datetime(2026, 12, 6, 7, tzinfo=timezone.utc)
SHA = 'a' * 40
CLOCK = '2026-12-06T07:00:04Z'
RUN = '823'
NAME = '2026-12-06.json'


class Publication(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        self.repo = base / 'repo'
        self.remote = base / 'origin.git'
        for path in (self.repo, self.remote):
            path.mkdir()
        record.git(self.remote, 'init', '--quiet', '--bare')
        record.git(self.repo, 'init', '--quiet', '-b', record.BRANCH)
        record.git(self.repo, 'config', 'user.name', 'tester')
        record.git(self.repo, 'config', 'user.email', 'test@example.org')
        record.git(self.repo, 'commit', '--allow-empty', '-m', 'isolated ledger')
        record.git(self.repo, 'remote', 'add', 'origin', str(self.remote))
        record.git(self.repo, 'push', '--quiet', 'origin', 'HEAD')
        self.statuses = {}
        self.clock = CLOCK

    def api(self, commit, payload=None):
        entries = self.statuses.setdefault(commit, [])
        if payload is None:
            return entries
        row = dict(payload, id=4242, created_at=self.clock,
                   creator={'login': 'github-actions[bot]'},
                   url=f'https://api.github.com/repos/{record.REPO}/statuses/{commit}')
        entries.append(row)
        return row

    def issue(self, folder, receipts):
        folder.mkdir(exist_ok=True)
        target = folder / NAME
        if not target.exists():
            record.write_once(target, {'fair-coin': .5})

    def run_cycle(self, when, driver=None, api=None):
        record.operate(self.repo, SHA, RUN, when, driver or self.issue, api or self.api)

    def committed(self):
        return record.git_text(self.remote, 'ls-tree', '--name-only', '-r', 'refs/heads/' + record.BRANCH).splitlines()

    def test_today_must_be_on_the_two_week_calendar(self):
        def deny(*args):
            self.fail('Attempted to call a service outside the calendar')
        self.run_cycle(ISSUE-timedelta(days=12), deny, deny)
        self.run_cycle(ISSUE+timedelta(days=7), deny, deny)
        self.assertEqual(self.committed(), [])
        self.assertEqual(sum(record.eligible((ISSUE+timedelta(days=i)).date()) for i in range(119)), 9)

    def test_server_attestation_bound_to_immutable_bytes_and_round(self):
        self.run_cycle(ISSUE)
        receipt = json.loads((self.repo/'forecastbench/receipts'/NAME).read_bytes())
        self.assertEqual(receipt['published_at_utc'], CLOCK)
        self.assertIn(receipt['commit_sha'], self.statuses)
        remote_sha = record.git_text(self.repo, 'rev-parse', 'HEAD')
        self.run_cycle(ISSUE)
        self.assertEqual(record.git_text(self.repo, 'rev-parse', 'HEAD'), remote_sha)
        self.assertEqual(sum(map(len, self.statuses.values())), 1)

    def test_late_status_does_not_certify_forecast_or_score(self):
        self.clock = '2026-12-07T00:00:00Z'
        self.run_cycle(ISSUE)
        def bad_score(folder, _):
            record.write_once(folder/'scores'/NAME, {'metric': 0})
        with self.assertRaisesRegex(ValueError, 'without a timely'):
            self.run_cycle(ISSUE+timedelta(days=2), bad_score)
        self.assertNotIn('forecastbench/scores/'+NAME, self.committed())
        self.assertNotIn('forecastbench/receipts/'+NAME, self.committed())

    def test_existing_status_can_recover_a_lost_api_reply_after_deadline(self):
        def interrupted(commit, payload=None):
            result = self.api(commit, payload)
            if payload is not None:
                raise TimeoutError('response lost')
            return result
        with self.assertRaisesRegex(ValueError, 'TimeoutError'):
            self.run_cycle(ISSUE, api=interrupted)
        self.run_cycle(ISSUE+timedelta(days=2))
        self.assertIn('forecastbench/receipts/'+NAME, self.committed())
        self.assertEqual(sum(map(len, self.statuses.values())), 1)

    def test_score_is_verified_remotely_and_issued_bytes_cannot_change(self):
        self.run_cycle(ISSUE)
        def score(folder, _):
            record.write_once(folder/'scores'/NAME, {'brier': .25})
        self.run_cycle(ISSUE+timedelta(days=2), score)
        self.assertIn('forecastbench/scores/'+NAME, self.committed())
        with self.assertRaisesRegex(ValueError, 'previously published'):
            self.run_cycle(ISSUE+timedelta(days=2), lambda f,_: (f/NAME).write_text('changed'))
        self.assertEqual(json.loads(record.git(self.remote, 'show', 'refs/heads/'+record.BRANCH+':forecastbench/'+NAME)), {'fair-coin':.5})

    def test_other_bot_or_wrong_status_url_cannot_pass(self):
        self.run_cycle(ISSUE)
        commit, rows = next(iter(self.statuses.items()))
        for modified in (dict(rows[0], creator={'login':'visitor'}), dict(rows[0], url='https://example.org'),
                         dict(rows[0], target_url='https://example.org'), dict(rows[0], context='manual'),
                         dict(rows[0], created_at='2026-12-05T23:59:59Z')):
            self.assertFalse(record.timely_status(modified, commit, ISSUE.date()))

    def test_stale_branch_cannot_overwrite_another_writer(self):
        second = self.repo.parent/'second'
        subprocess.run(['git','clone','--quiet','-b',record.BRANCH,str(self.remote),str(second)],check=True)
        record.git(second, 'config','user.name','tester')
        record.git(second, 'config','user.email','test@example.org')
        record.git(second, 'commit','--allow-empty','-m','concurrent ledger advance')
        record.git(second, 'push','--quiet','origin','HEAD')
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_cycle(ISSUE)
        self.assertEqual(self.committed(), [])
        self.assertEqual(self.statuses, {})


if __name__ == '__main__':
    unittest.main()
