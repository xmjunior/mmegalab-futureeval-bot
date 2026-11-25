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
            cross_quarter_menu=lambda: {str(group): [1.25] for group in self.rows},
            timestamp=datetime.fromisoformat,
        )
        self.account = SimpleNamespace(group=self.rows.__getitem__)

    def add(self, status, spot_minutes, close_minutes, group=46010):
        row = {"status": status,
               "spot_scoring_time": (self.now + timedelta(minutes=spot_minutes)).isoformat(),
               "scheduled_close_time": (self.now + timedelta(minutes=close_minutes)).isoformat()}
        qs = self.rows.setdefault(group, {})
        qs[len(qs)] = SimpleNamespace(api_json={"question": row})

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
        row = self.rows[46010][0].api_json["question"]
        row["status"] = "open"
        del row["spot_scoring_time"]
        with self.assertRaises(KeyError):
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
