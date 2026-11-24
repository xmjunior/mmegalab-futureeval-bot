import os
import base64
import json
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import recovery_clock as c

NOW = datetime(2026, 11, 24, 20, 29, tzinfo=UTC)
ENV = {
    "name": c.NAME,
    "id": 321,
    "protection_rules": [{"type": "wait_timer", "wait_timer": 50}],
}
ANCHOR = int(NOW.timestamp()) - 3000


def run(file, age, ident, **extra):
    return {
        "id": ident,
        "event": "schedule",
        "head_branch": "main",
        "path": ".github/workflows/" + file,
        "created_at": (NOW - timedelta(minutes=age)).isoformat(),
        "status": "completed",
        **extra,
    }


class Recovery(unittest.TestCase):
    def test_elapsed_and_identity_require_real_50_minutes(self):
        for e, i, t in [
            ({}, 321, ANCHOR),
            (dict(ENV, protection_rules=[]), 321, ANCHOR),
            (
                dict(ENV, protection_rules=[{"type": "wait_timer", "wait_timer": 49}]),
                321,
                ANCHOR,
            ),
            (dict(ENV, id=123), 321, ANCHOR),
            (ENV, 321, ANCHOR + 1),
            (ENV, 321, int(NOW.timestamp()) + 5000),
        ]:
            engine = Mock()
            engine.environment.return_value = e
            with self.subTest(e=e, t=t), self.assertRaises(ValueError):
                c.wake(engine, NOW, lambda: NOW, 0, 0, i, t)
            engine.send.assert_not_called()
            engine.original_coverage.assert_not_called()
            engine.runs.assert_not_called()
        c.check_elapsed(ENV, 321, ANCHOR, NOW)

    def test_real_events_across_three_workflows_and_distinct_runs(self):
        engine = Mock()
        fs = list(c.watchdog.TARGETS.values())
        rows = {f: [run(f, 20, 2), run(f, 50, 1)] for f in fs}
        rows[c.MARKET] = [run(c.MARKET, 35, 3)]
        engine.runs.side_effect = lambda f, **k: rows[f]
        engine.evaluation_receipts.return_value = []
        self.assertFalse(c.native_healthy(engine, NOW))
        engine.evaluation_receipts.return_value = [
            ("abc", run(c.CONTROL, 50, 9, event="workflow_dispatch"))
        ]
        self.assertFalse(c.native_healthy(engine, NOW))
        engine.evaluation_receipts.return_value = [("abc", run(c.CONTROL, 50, 9))]
        self.assertTrue(c.native_healthy(engine, NOW))
        engine.environment.return_value = ENV
        self.assertEqual(c.wake(engine, NOW, lambda: NOW, 0, 0, 321, ANCHOR), "ended")
        engine.send.assert_not_called()
        for corrupt in [
            dict(id=2),
            dict(head_branch="other"),
            dict(event="workflow_dispatch"),
            dict(path=".github/workflows/mock.yaml"),
            dict(created_at=(NOW - timedelta(days=1)).isoformat()),
        ]:
            rows[fs[0]][1] = dict(run(fs[0], 50, 1), **corrupt)
            self.assertFalse(c.native_healthy(engine, NOW))
        rows[fs[0]][1] = run(fs[0], 50, 1)
        rows[c.MARKET] = []
        self.assertFalse(c.native_healthy(engine, NOW))

    def test_market_active_race_and_scheduled_window(self):
        e = Mock()
        e.runs.side_effect = [[], [run(c.MARKET, 7, 2)]]
        e.market_has_open_candidate.return_value = True
        self.assertIn("another run", c.recover_market(e, NOW))
        e.send.assert_not_called()
        e = Mock()
        e.runs.return_value = []
        e.market_has_open_candidate.return_value = False
        self.assertIn("no approved", c.recover_market(e, NOW))
        e.send.assert_not_called()
        e = Mock()
        self.assertEqual(c.recover_market(e, NOW + timedelta(hours=4)), "outside slot")
        e.runs.assert_not_called()
        e.runs.return_value = []
        e.market_has_open_candidate.return_value = True
        self.assertIn("dispatched", c.recover_market(e, NOW))
        e.send.assert_called_once_with(c.MARKET)

    def test_backoff_failure_cap_and_end(self):
        e = Mock()
        e.environment.return_value = ENV
        e.runs.return_value = []
        e.original_coverage.side_effect = TimeoutError()
        e.evaluation_receipts.return_value = [("abc", run(c.CONTROL, 50, 9))]
        e.market_has_open_candidate.return_value = False
        with self.assertRaises(RuntimeError):
            c.wake(e, NOW, lambda: NOW, 17, 0, 321, ANCHOR)
        e.send.assert_called_once_with(
            c.CONTROL,
            {"mode": "recovery", "recovery_cycle": "18", "recovery_errors": "1"},
        )
        e.send.reset_mock()
        with self.assertRaises(RuntimeError):
            c.wake(e, NOW, lambda: NOW, 18, 2, 321, ANCHOR)
        e.send.assert_not_called()
        for cycle, errors in ((0, 3), (1420, 0), (-1, 0)):
            self.assertEqual(
                c.wake(e, NOW, lambda: NOW, cycle, errors, 321, ANCHOR), "ended"
            )
        self.assertEqual(c.wake(e, c.STOP, lambda: c.STOP, 0, 0, 321, ANCHOR), "ended")

    def test_rule_removed_before_successor_prevents_recursive_dispatch(self):
        e = Mock()
        e.environment.side_effect = [ENV, dict(ENV, protection_rules=[])]
        e.evaluation_receipts.return_value = [("abc", run(c.CONTROL, 50, 9))]
        e.runs.return_value = []
        e.market_has_open_candidate.return_value = False
        with self.assertRaisesRegex(ValueError, "50-minute"):
            c.wake(e, NOW, lambda: NOW, 0, 0, 321, ANCHOR)
        e.send.assert_not_called()

    def test_missing_environment_cannot_create_or_start_any_job(self):
        e = Mock()
        e.environment.side_effect = HTTPError(
            "https://api.github.com", 404, "missing", {}, None
        )
        with tempfile.TemporaryDirectory() as folder:
            outputs = Path(folder) / "out"
            outputs.touch()
            env = {
                "GH_TOKEN": "placeholder",
                "GITHUB_REPOSITORY": "xmjunior/mmegalab-futureeval-bot",
                "GITHUB_REF": "refs/heads/main",
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_RUN_ATTEMPT": "1",
                "RECOVERY_ERRORS": "0",
                "RECOVERY_CYCLE": "0",
                "GITHUB_OUTPUT": str(outputs),
                "RECOVERY_MODE": "recovery_check",
            }
            with (
                patch.dict(os.environ, env),
                patch.object(sys, "argv", ["clock", "preflight"]),
                patch.object(c, "Dispatcher", return_value=e),
            ):
                c.main()
                self.assertEqual(outputs.read_text(), "")
                os.environ["RECOVERY_MODE"] = "recovery"
                with self.assertRaises(SystemExit):
                    c.main()
            e.send.assert_not_called()
            e.original_coverage.assert_not_called()
            self.assertEqual(outputs.read_text(), "")

    def test_control_recovery_checks_score_receipt_then_mode_specific_history(self):
        engine = Mock()
        engine.evaluation_receipts.return_value = []
        engine.evaluation_attempted.return_value = False
        self.assertEqual(c.control_slot(NOW).hour, 19)
        self.assertEqual(c.control_slot(NOW.replace(hour=19, minute=30)).hour, 7)
        self.assertIn("dispatched", c.recover_control(engine, NOW))
        engine.send.assert_called_once_with(c.CONTROL, {"mode": "control"})
        engine.send.reset_mock()
        engine.evaluation_attempted.return_value = True
        c.recover_control(engine, NOW)
        engine.send.assert_not_called()
        engine.evaluation_receipts.return_value = [
            ("recorded-hash", run(c.CONTROL, 50, 12))
        ]
        self.assertIn("recorded-hash", c.recover_control(engine, NOW))
        engine.send.assert_not_called()

    def test_receipt_must_trace_main_and_complete_successful_results(self):
        d = object.__new__(c.Dispatcher)
        sha = "f" * 40
        record = {
            "GITHUB_RUN_ID": "100",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_EVENT_NAME": "schedule",
            "GITHUB_SHA": "abc",
            **dict.fromkeys(["PROMO", "TREASURY", "MINI", "AUDIT"], "success"),
        }
        row = run(c.CONTROL, 45, 100, head_sha="abc")

        def api(endpoint, **kwargs):
            if endpoint == "/commits":
                self.assertEqual(kwargs["path"], "records/control.md")
                return [
                    {
                        "sha": sha,
                        "commit": {
                            "message": "Record autonomous forecast checks for run 100/2"
                        },
                    }
                ]
            if endpoint == "/contents/records/runs/100-2/run.json":
                self.assertEqual(kwargs["ref"], sha)
                return {
                    "encoding": "base64",
                    "content": base64.b64encode(json.dumps(record).encode()),
                }
            self.assertEqual(endpoint, "/actions/runs/100/attempts/2")
            return row

        d.github = api
        self.assertEqual(d.evaluation_receipts(c.control_slot(NOW)), [(sha, row)])
        record["PROMO"] = "failure"
        self.assertEqual(d.evaluation_receipts(c.control_slot(NOW)), [])
        record["PROMO"] = "success"
        row["head_branch"] = "verify/champion-v2-control"
        self.assertEqual(d.evaluation_receipts(c.control_slot(NOW)), [])

    def test_action_events_cannot_impersonate_semidaily_evaluations(self):
        d = object.__new__(c.Dispatcher)
        d.github = Mock()
        for ev, title, state, valid in [
            ("workflow_dispatch", "Champion recovery", "in_progress", False),
            ("schedule", "Champion schedule 27 * * * *", "in_progress", False),
            (
                "workflow_run",
                "Champion research and competition control",
                "in_progress",
                False,
            ),
            ("workflow_dispatch", "Champion control", "completed", True),
            ("schedule", "Champion schedule 17 7,19 * * *", "in_progress", True),
            ("schedule", "Champion schedule 17 7,19 * * *", "completed", False),
        ]:
            d.github.return_value = {
                "workflow_runs": [
                    run(c.CONTROL, 20, 62, event=ev, display_title=title, status=state)
                ]
            }
            self.assertEqual(d.evaluation_attempted(c.control_slot(NOW)), valid)

    def test_duration_of_second_wait_is_independent(self):
        e = Mock()
        e.environment.return_value = ENV
        first_anchor = int(NOW.timestamp())
        t2 = NOW + timedelta(minutes=50)
        c.check_elapsed(ENV, 321, first_anchor, t2)
        with self.assertRaises(ValueError):
            c.check_elapsed(
                ENV, 321, int(t2.timestamp()), t2 + timedelta(minutes=49, seconds=59)
            )
        c.check_elapsed(ENV, 321, int(t2.timestamp()), t2 + timedelta(minutes=50))


if __name__ == "__main__":
    unittest.main()
