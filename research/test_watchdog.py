"""Coverage recovery uses only the original main workflows and verified account metadata."""
import contextlib
import io
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import watchdog


class CoverageRecovery(unittest.TestCase):
    def exercise(self, age, answered, enabled, expected, opened=None, ended=None, racing=False):
        checks = {}
        now = datetime.now(timezone.utc)
        sent = []
        meta_calls = []

        def api(base, path, token, method="GET", payload=None, **params):
            if path.endswith("/runs"):
                checks[path] = checks.get(path, 0) + 1
                status = "in_progress" if racing and checks[path] > 1 else "completed"
                return {"workflow_runs": [{
                    "head_branch": "main", "status": status, "event": "workflow_dispatch",
                    "created_at": (now - timedelta(minutes=age)).isoformat(),
                    "updated_at": (now - timedelta(minutes=ended)).isoformat() if ended is not None else None,
                }]}
            if base == watchdog.MC:
                meta_calls.append(path)
            if path == "/users/me/":
                return {"id": 313890, "username": "megalab-mario-bot", "is_primary_bot": True}
            if path.startswith("/projects/tournaments/"):
                return {"id": 33169 if "minibench" in path else 33121}
            if path == "/posts/":
                rows = [{"question": {"id": 123, "status": "open", "open_time": (now - timedelta(minutes=opened)).isoformat() if opened is not None else None}}] if params["tournaments"] == 33121 else []
                return {"results": rows, "next": "speculative"}
            if path == "/questions/bulk-forecast-read/":
                self.assertEqual(params["question_ids"], [123])
                return {"results": [{"question_id": 123, "forecasts": [{"author_id": 313890}] if answered else []}]}
            if path.endswith("/dispatches"):
                sent.append((path, method, payload))
                return None
            self.fail("Unexpected request")

        with patch.dict(os.environ, {"GH_TOKEN": "test", "METACULUS_TOKEN": "test"}), patch.object(
            sys, "argv", ["watchdog"] + (["--dispatch"] if enabled else [])
        ), patch.object(watchdog, "call", api), contextlib.redirect_stdout(io.StringIO()):
            watchdog.main()
        self.assertEqual(sent, [("/actions/workflows/run_bot_on_tournament.yaml/dispatches", "POST", {"ref":"main"})] if expected else [])
        if age < 3:
            self.assertFalse(meta_calls)

    def test_recent(self):
        self.exercise(2, False, True, False)

    def test_answered(self):
        self.exercise(90, True, True, False)

    def test_dry(self):
        self.exercise(90, False, False, False)

    def test_unanswered(self):
        self.exercise(90, False, True, True)

    def test_new_window_after_recent_completed_run(self):
        self.exercise(25, False, True, True, opened=12, ended=23)

    def test_recent_overlapping_run(self):
        self.exercise(25, False, True, False, opened=27, ended=20)

    def test_unknown_finish_or_opening(self):
        self.exercise(25, False, True, False, opened=12)
        self.exercise(25, False, True, False, ended=23)

    def test_concurrent_dispatch_suppressed(self):
        self.exercise(90, False, True, False, racing=True)

    def test_new_window_already_forecast(self):
        self.exercise(25, True, True, False, opened=12, ended=23)
