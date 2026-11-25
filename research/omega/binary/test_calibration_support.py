import unittest
import subprocess
import sys
from datetime import date, timedelta

import numpy as np
import evaluate as ev


def rows(split, date_string, horizons):
    origin = date.fromisoformat(date_string)
    return [[split, date_string, "source", str(i), str(origin + timedelta(days=h)), "[]",
             0.2 + i / 100, 0.3 + i / 100, 0.4 - i / 100, i % 2]
            for i, h in enumerate(horizons)]


class CalibrationSupportTests(unittest.TestCase):
    def setUp(self):
        self.train = rows("train", "2025-12-01", [7] * 8)
        self.valid = rows("valid", "2026-01-01", [7, 30] * 4)
        self.test = rows("test", "2026-03-01", [7, 14, 30, 90])

    def test_guard_applies_only_at_observed_horizons(self):
        guarded, fit = ev.predict(self.train, self.valid, self.test)
        historical, _ = ev.predict(self.train, self.valid, self.test, archived_20261124=True)
        self.assertEqual(fit["venn_support"], {"calibration_cases_by_horizon_days": {7: 4, 30: 4},
                                               "applied": 2, "base_forecast_used": 2})
        for base, arm in (("mean", "venn_mean"), ("correlation_pool", "venn_corr")):
            np.testing.assert_array_equal(guarded[arm][[0, 2]], historical[arm][[0, 2]])
            np.testing.assert_array_equal(guarded[arm][[1, 3]], guarded[base][[1, 3]])
            self.assertFalse(np.array_equal(guarded[arm][[1, 3]], historical[arm][[1, 3]]))

    def test_cli_refuses_archive_before_loading_cohort(self):
        command = [sys.executable, ev.__file__, "--out", str(ev.HERE / "results.json")]
        outcome = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(outcome.returncode, 2)
        self.assertIn("distinct from the archived results", outcome.stderr)

    def test_censored_all_and_empty_validation(self):
        guarded, fit = ev.predict(self.train, [], self.test)
        self.assertEqual(fit["venn_support"]["applied"], 0)
        for base, arm in (("mean", "venn_mean"), ("correlation_pool", "venn_corr")):
            np.testing.assert_array_equal(guarded[arm], guarded[base])

    def test_archived_formula_and_correlation_estimand(self):
        historical, old_fit = ev.predict(self.train, self.valid, self.test, archived_20261124=True)
        guarded, new_fit = ev.predict(self.train, self.valid, self.test)
        errors = np.asarray([r[6:9] for r in self.train]) - np.asarray([r[9] for r in self.train])[:, None]
        np.testing.assert_allclose(new_fit["signed_forecast_error_correlation"], np.corrcoef(errors.T))
        self.assertIn("Bernoulli", new_fit["correlation_estimand"])
        self.assertNotIn("residual_correlation", new_fit)
        self.assertNotIn("venn_support", old_fit)
        np.testing.assert_array_equal(old_fit["residual_correlation"], new_fit["signed_forecast_error_correlation"])
        mean_cal = np.mean([r[6:9] for r in self.valid], axis=1)
        calibrator = ev.venn(mean_cal, [r[9] for r in self.valid])
        np.testing.assert_array_equal(historical["venn_mean"], [calibrator(p) for p in guarded["mean"]])
        changed = [r[:-1] + [1 - r[-1]] for r in self.test]
        altered, _ = ev.predict(self.train, self.valid, changed)
        for arm in guarded:
            np.testing.assert_array_equal(guarded[arm], altered[arm])
