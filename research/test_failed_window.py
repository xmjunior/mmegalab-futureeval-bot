"""Exercise the actual dispatch guard with synthetic history or known closed windows."""
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

import watchdog

UTC = timezone.utc
CHECK = datetime(2026, 11, 24, 19, 57, 46, tzinfo=UTC)
BEGIN = "2026-11-24T18:00:00Z"
END = "2026-11-24T21:00:00Z"
SEASON = "/actions/workflows/run_bot_on_tournament.yaml"


def completed(start, end, result="failure"):
    return {"status": "completed", "head_branch": "main", "created_at": start,
            "updated_at": end, "conclusion": result, "event": "workflow_dispatch"}


FAILED = completed("2026-11-24T19:33:12Z", "2026-11-24T19:34:48Z")


def replay(now=CHECK, latest=FAILED, *, guard=watchdog, closing=END, opened=BEGIN, answered=False,
           race=None, history_complete=True, second_now=None):
    dispatches = []
    reads = []
    times = iter([now, second_now or now])
    class Clock(datetime):
        @classmethod
        def now(cls, tz):
            return next(times, second_now or now)

    def api(base, path, token, method="GET", payload=None, **filters):
        if path.endswith("/runs"):
            if not path.startswith(SEASON):
                return {"workflow_runs": [{**latest, "status": "in_progress"}]}
            reads.append(path)
            return {"workflow_runs": [race if race and len(reads) > 1 else latest]}
        if path == "/users/me/":
            return {"id": 313890, "username": "megalab-mario-bot", "is_primary_bot": True}
        if path == "/projects/tournaments/33121/":
            return {"id": 33121}
        if path == "/posts/":
            return {"results": [{"question": {"id": 46894, "status": "open",
                    "open_time": opened, "scheduled_close_time": closing}}]}
        if path == "/questions/bulk-forecast-read/":
            assert filters["question_ids"] == [46894]
            assert method == "GET"
            return {"results": [{"question_id": 46894,
                    "forecasts": [{"author_id": 313890}] if answered else []}] if history_complete else []}
        if path == SEASON + "/dispatches":
            assert method == "POST" and payload == {"ref":"main"}
            dispatches.append(path)
            return
        raise AssertionError(path)

    with patch.object(guard, "call", api), patch.object(guard, "datetime", Clock), \
         patch.dict(os.environ, GH_TOKEN="test", METACULUS_TOKEN="test"), \
         patch.object(sys, "argv", ["watchdog", "--dispatch"]), redirect_stdout(io.StringIO()):
        guard.main()
    return len(dispatches)


class FailedWindowRecovery(unittest.TestCase):
    def test_closed_incident_reaches_dispatch_after_cooldown(self):
        self.assertEqual(replay(), 1)

    def test_reach_cadence_boundary_but_never_retry_earlier(self):
        cutoff = datetime.fromisoformat(FAILED["updated_at"]) + timedelta(minutes=20)
        self.assertEqual(replay(cutoff - timedelta(seconds=1)), 0)
        self.assertEqual(replay(cutoff), 1)
        long_failure = {**FAILED, "created_at": (cutoff-timedelta(hours=1)).isoformat()}
        self.assertEqual(replay(cutoff-timedelta(seconds=1), long_failure), 0)

    def test_success_cancellation_and_unknown_conclusion_remain_covered(self):
        for outcome in ("success", "cancelled", None):
            with self.subTest(outcome=outcome):
                self.assertEqual(replay(latest={**FAILED, "conclusion":outcome}), 0)
        self.assertEqual(replay(latest={**FAILED, "updated_at":None}), 0)

    def test_accepted_or_incomplete_receipt_never_dispatches(self):
        self.assertEqual(replay(answered=True), 0)
        with self.assertRaisesRegex(RuntimeError, "history incomplete"):
            replay(history_complete=False)

    def test_retry_requires_known_live_clock_in_both_reads(self):
        self.assertEqual(replay(closing=None), 0)
        self.assertEqual(replay(opened=None), 0)
        self.assertEqual(replay(closing=CHECK.isoformat()), 0)
        self.assertEqual(replay(second_now=datetime.fromisoformat(END)), 0)
        self.assertEqual(replay(opened=(CHECK+timedelta(seconds=1)).isoformat()), 0)

    def test_racing_original_or_later_success_suppresses_retry(self):
        self.assertEqual(replay(race={**FAILED,"status":"queued"}), 0)
        self.assertEqual(replay(race={**FAILED,"conclusion":"success"}), 0)

    def test_replay_four_observed_historical_starts(self):
        archive = json.loads((Path(__file__).parent / "evidence/seasonal-coverage-20261125.json").read_bytes())
        self.assertEqual(historical_decisions(watchdog, archive), archive["comparison"]["candidate_dispatched"])


def historical_decisions(guard, archive):
    previous = archive["previous_run"]
    decisions = []
    for run in archive["attempts_in_second_window"]:
        as_of = datetime.fromisoformat(run["started"]) - timedelta(seconds=1)
        prior = completed(previous["started"], previous["ended"], previous["conclusion"])
        decisions.append(replay(as_of, prior, guard=guard))
        previous = run
    return decisions


if __name__ == "__main__":
    # Load the exact predecessor exported by Git; both sides use the same mock transport.
    archive = json.loads((Path(__file__).parent / "evidence/seasonal-coverage-20261125.json").read_bytes())
    spec = importlib.util.spec_from_file_location("frozen_watchdog", sys.argv[1])
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    for name, guard in [("base_dispatched", baseline), ("candidate_dispatched", watchdog)]:
        decisions = historical_decisions(guard, archive)
        print(name, decisions)
        if decisions != archive["comparison"][name]:
            raise SystemExit(f"Closed-incident decision mismatch for {name}")
