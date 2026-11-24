import copy
import json
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import arena


class ArenaTests(unittest.TestCase):
    def setUp(self):
        self.record = arena.promo.read_json(arena.SEALED)
        self.spec = arena.panel()
        self.book = arena.promo.new_state()
        probabilities = self.record["payload"]["probabilities"]
        for name in self.record["payload"]["source_forecast_sha256"]:
            self.book["arms"][name] = {"probabilities": probabilities[name].copy()}

    def test_future_and_unreleased_labels_do_not_change_fit(self):
        rows = arena.load_history()
        cutoff = date(2025, 1, 1)
        contaminated = copy.deepcopy(rows)
        for row in contaminated:
            if date.fromisoformat(row["origin"])+timedelta(days=40) >= cutoff:
                row.update(reference=0.99, scaled=0.01, outcome=1-row["outcome"])
        first = arena.fit(rows, cutoff)
        self.assertEqual(first, arena.fit(contaminated, cutoff))
        self.assertEqual((first["last_training_origin"], first["training_cases"]), ("2024-11-01", 92))

    def test_metrics_reproduce_frozen_holdout(self):
        actual = arena.backtest()
        frozen = arena.promo.read_json(arena.HERE / "historical.json")
        self.assertEqual((actual["test_months"], actual["test_cases"]), (22,88))
        for name in frozen["scores"]:
            for key in ("brier", "log_loss_nats", "ece_5_bins"):
                self.assertAlmostEqual(actual["scores"][name][key], frozen["scores"][name][key], places=10)
        self.assertEqual(actual["scores"]["coin"]["brier"], 0.25)
        self.assertEqual(actual["paired_calibration_gain_95pct_four_month_blocks"],
                         frozen["paired_calibration_gain_95pct_four_month_blocks"])

    def test_irreversible_issue_and_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/"seal.json"
            arena.write_once(dest, {"x":1})
            with self.assertRaises(FileExistsError):
                arena.write_once(dest, {"x":2})
            self.assertEqual(json.loads(dest.read_bytes()), {"x":1})
            with self.assertRaisesRegex(ValueError, "deadline"):
                arena.issue(Path(tmp)/"absent", datetime(2026,11,25,tzinfo=UTC))

    def test_model_members_and_fixed_combinations(self):
        arms = arena.checked_seal(self.record, self.spec)
        fit = self.record["payload"]["fit"]
        for case in self.spec["cases"]:
            pool = sum(arms[m][case] for m in arena.MODELS)/len(arena.MODELS)
            self.assertAlmostEqual(arms["model_mean"][case], pool)
            self.assertAlmostEqual(arms["model_reference_mean"][case], (pool+arms["C"][case])/2)
            self.assertAlmostEqual(arms["calibrated_reference"][case], arena.calibrated(arms["C"][case], fit))

    def test_modified_seal_and_changed_journal_forecast_rejected(self):
        bad = copy.deepcopy(self.record)
        key = next(iter(self.spec["cases"]))
        bad["payload"]["probabilities"]["openai"][key] = 0.90
        bad["sha256"] = arena.promo.fingerprint(bad["payload"])
        with self.assertRaisesRegex(ValueError, "seal mismatch"):
            arena.score(bad, self.book)
        self.book["arms"]["openai"]["probabilities"][key] = 0.90
        with self.assertRaisesRegex(ValueError, "sealed forecast"):
            arena.score(self.record, self.book)

    def test_pending_uses_no_network_and_no_dummy_scores(self):
        original = copy.deepcopy(self.book)
        s = arena.score(self.record, self.book)
        self.assertEqual((s["pending"], s["resolved"], len(s["scores"])), (4, 0, 8))
        self.assertTrue(all(r["mean_brier"] is None for r in s["scores"].values()))
        self.assertEqual(arena.score(self.record, self.book), s)
        self.assertEqual(self.book, original)

    def test_paired_scoring_waits_for_publication_and_uses_strict_threshold(self):
        key = "ust-promo-v1-20261202-2y"
        case = self.spec["cases"][key]
        outcome = {"status":"resolved", "observed_date":case["target"],
                   "threshold":case["threshold"], "value":case["threshold"], "outcome":0,
                   "recorded_utc":"2026-12-04T12:00:00+00:00"}
        self.book["resolutions"][key] = outcome
        self.book["resolutions"]["ust-promo-v1-20261209-2y"] = {
            "status":"void", "reason":"no_evaluable_official_observation_in_window",
            "recorded_utc":"2026-12-17T12:00:00+00:00"}
        with self.assertRaisesRegex(ValueError, "unpublished"):
            arena.score(self.record, self.book, datetime(2026,11,25,tzinfo=UTC))
        settled_at = datetime(2026,12,18,tzinfo=UTC)
        scored = arena.score(self.record, self.book, settled_at)
        self.assertEqual((scored["pending"], scored["resolved"], scored["void"]), (2,1,1))
        for name, values in self.record["payload"]["probabilities"].items():
            self.assertAlmostEqual(scored["scores"][name]["mean_brier"], values[key]**2)
        outcome["recorded_utc"] = "2026-12-03T23:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "published resolution"):
            arena.score(self.record, self.book, settled_at)
        outcome.update(recorded_utc="2026-12-04T00:00:00+00:00",outcome=1)
        with self.assertRaisesRegex(ValueError, "published resolution"):
            arena.score(self.record, self.book, settled_at)


if __name__ == "__main__":
    unittest.main()
