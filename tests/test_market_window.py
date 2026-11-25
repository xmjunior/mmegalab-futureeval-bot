from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.market_window import rehearsal_windows


class MarketWindowTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 25, tzinfo=timezone.utc)
        self.rows = {46010: {}}
        self.market = SimpleNamespace(
            PROJECT=33131,
            cross_quarter_menu=lambda: {str(group): [1.25] for group in self.rows},
            timestamp=datetime.fromisoformat,
        )
        self.account = SimpleNamespace(get=self.get)

    def get(self, path):
        group = int(path.strip("/").split("/")[-1])
        return {"id": group, "projects": {"default_project": {"id": 33131}, "tournament": [{"id": 33131}]},
                "group_of_questions": {"questions": list(self.rows[group].values())}}

    def add(self, status, spot_minutes, close_minutes, group=46010):
        row = {"status": status, "type": "numeric",
               "spot_scoring_time": (self.now + timedelta(minutes=spot_minutes)).isoformat(),
               "scheduled_close_time": (self.now + timedelta(minutes=close_minutes)).isoformat()}
        qs = self.rows.setdefault(group, {})
        qs[len(qs)] = row

    def windows(self):
        return rehearsal_windows(self.market, self.account, self.now)

    def test_closed_expired_and_five_minute_boundary_wait(self):
        for status, spot, close in [("resolved", 30, 30), ("closed", 30, 30),
                                    ("open", 5, 60), ("upcoming", 60, -1)]:
            self.add(status, spot, close)
        self.assertEqual(self.windows(), 0)

    def test_upcoming_beyond_seven_days_is_valid_for_rehearsal(self):
        self.add("open", 6, 8)
        self.add("upcoming", 14000, 15000, group=46011)
        self.assertEqual(self.windows(), 2)

    def test_empty_scorecard_and_missing_group_are_errors(self):
        with self.assertRaises(ValueError):
            self.windows()
        self.rows.clear()
        with self.assertRaises(ValueError):
            self.windows()

    def test_unrecognized_status_and_missing_deadline_are_errors(self):
        self.add("new_state", 7, 8)
        with self.assertRaises(ValueError):
            self.windows()
        row = self.rows[46010][0]
        row["status"] = "open"
        del row["spot_scoring_time"]
        with self.assertRaises(KeyError):
            self.windows()

    def test_mixed_schema_rejects_unsupported_type_even_when_other_questions_are_closed(self):
        self.add("closed", 20, 20)
        self.add("upcoming", 200, 180)
        self.rows[46010][1]["type"] = "discrete"
        with self.assertRaisesRegex(ValueError, "does not support.*discrete"):
            self.windows()
        self.market.MARKET_QUESTION_TYPES = {"numeric", "discrete"}
        with self.assertRaisesRegex(ValueError, "Discrete geometry is unqualified"):
            self.windows()
        self.market.MARKET_DISCRETE_GROUPS = {46010}
        self.assertEqual(self.windows(), 1)

    def test_active_window_requires_the_pinned_parser_to_accept_the_group(self):
        self.add("resolved", 2, 2)
        self.add("upcoming", 15, 30)
        self.market.parse_market_question = lambda post, row: None if row["status"] == "resolved" else row
        with self.assertRaisesRegex(ValueError, "Adapter rejected"):
            self.windows()

    def test_foreign_project_rejected(self):
        self.market.PROJECT = 3
        self.add("open", 20, 20)
        with self.assertRaisesRegex(ValueError, "Unexpected Market Pulse project or post"):
            self.windows()

    def test_scheduler_receives_wait_only_after_successful_metadata_read(self):
        self.add("closed", 10, 10)
        self.market.Account = lambda: self.account
        script = Path(__file__).resolve().parents[1] / "research" / "market_window.py"
        with TemporaryDirectory() as temp:
            output = Path(temp) / "outputs"
            with patch.dict(sys.modules, market_pulse=self.market), patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
                runpy.run_path(str(script), run_name="__main__")
                self.assertEqual(output.read_text(), "available=false\n")
                output.unlink()
                self.rows.clear()
                with self.assertRaises(ValueError):
                    runpy.run_path(str(script), run_name="__main__")
                self.assertFalse(output.exists())
