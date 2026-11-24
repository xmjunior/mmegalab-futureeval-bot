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
    def exercise(self, age, answered, enabled, expected):
        sent = []
        meta_calls = []

        def api(base, path, token, method="GET", payload=None, **params):
            if path.endswith("/runs"):
                return {"workflow_runs": [{
                    "head_branch": "main", "status": "completed", "event": "workflow_dispatch",
                    "created_at": (datetime.now(timezone.utc) - timedelta(minutes=age)).isoformat(),
                }]}
            if base == watchdog.MC:
                meta_calls.append(path)
            if path == "/users/me/":
                return {"id": 313890, "username": "megalab-mario-bot", "is_primary_bot": True}
            if path.startswith("/projects/tournaments/"):
                return {"id": 33169 if "minibench" in path else 33121}
            if path == "/posts/":
                rows = [{"question": {"id": 123, "status": "open"}}] if params["tournaments"] == 33121 else []
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
        if age < 42:
            self.assertFalse(meta_calls)

    def test_recent(self):
        self.exercise(3, False, True, False)

    def test_answered(self):
        self.exercise(90, True, True, False)

    def test_dry(self):
        self.exercise(90, False, False, False)

    def test_unanswered(self):
        self.exercise(90, False, True, True)
