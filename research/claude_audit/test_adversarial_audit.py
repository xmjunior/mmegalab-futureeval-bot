"""Automated Adversarial Mathematical Test Suite for MEGALAB FutureEval / Market Pulse / OMEGA.

Distinguishes strictly between:
  [VERIFIED CODE DEFECT] — Deterministic, reproducible mathematical/numerical bug or
      specification mismatch in existing repository/SDK code (`main.py`,
      `forecasting-tools==0.3.5`, `market_prices.py`, `paired_economic.py`,
      `omega/binary/evaluate.py`, `omega/continuous/evaluate.py`).
  [MATHEMATICAL INVARIANT] — Strict properness, range/scale invariance, boundary
      continuity, causal prefix-invariance (zero future leakage), and malformed-input
      rejection required of all candidate remedies.

Zero network calls. Zero API inference. Zero modification to production files.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import unittest

import numpy as np
from scipy import stats

try:
    from .binary_aggregation import (
        HEBLoPAggregator,
        decompose_residual_covariance,
        online_heb_lop_predict,
    )
    from .builders import metalog_cdf, normal_fit_cdf, tq_cdf
    from .macro_fhs_aci import (
        TruncatedStudentT,
        fhs_standardized_quantiles,
        run_causal_aci_sequence,
    )
    from .metaculus_math import (
        Geometry,
        baseline_score,
        expected_baseline_score,
        pmf_from_cdf,
        sdk_cdf,
        sdk_standardize,
        true_bucket_probs,
        validate_cdf,
    )
except ImportError:
    from binary_aggregation import (
        HEBLoPAggregator,
        decompose_residual_covariance,
        online_heb_lop_predict,
    )
    from builders import metalog_cdf, normal_fit_cdf, tq_cdf
    from macro_fhs_aci import (
        TruncatedStudentT,
        fhs_standardized_quantiles,
        run_causal_aci_sequence,
    )
    from metaculus_math import (
        Geometry,
        baseline_score,
        expected_baseline_score,
        pmf_from_cdf,
        sdk_cdf,
        sdk_standardize,
        true_bucket_probs,
        validate_cdf,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
LEVELS_6 = (0.10, 0.20, 0.40, 0.60, 0.80, 0.90)


class TestDefect1ContinuousCdfPath(unittest.TestCase):
    """[VERIFIED CODE DEFECTS] in NumericDistribution.get_cdf() & internal continuous scorers."""

    def test_d1a_sdk_open_bound_5pct_tail_tax_and_range_non_invariance(self):
        """SDK forces >5% mass outside each open bound and loses >25 expected points when range widens."""
        belief = stats.norm(500.0, 20.0)
        pct = [(p, float(belief.ppf(p))) for p in LEVELS_6]

        narrow = Geometry(440.0, 560.0, open_lower=True, open_upper=True)
        wide = Geometry(0.0, 1000.0, open_lower=True, open_upper=True)

        c_sdk_wide = sdk_cdf(pct, wide)
        c_tq_wide = tq_cdf(pct, wide)
        c_oracle_wide = sdk_standardize(belief.cdf(wide.edges()), wide)

        self.assertEqual(validate_cdf(c_sdk_wide, wide), [])
        self.assertEqual(validate_cdf(c_tq_wide, wide), [])

        # Defect reproduction: SDK hard-codes ~5.04% below 0 and ~5.04% above 1000 (>50x API min 0.001),
        # even though P(X < 0) = P(X > 1000) < 1e-130 under N(500, 20)!
        self.assertGreater(c_sdk_wide[0], 0.05)
        self.assertGreater(1.0 - c_sdk_wide[-1], 0.05)
        self.assertLess(c_tq_wide[0], 0.002)
        self.assertLess(1.0 - c_tq_wide[-1], 0.002)

        tp_wide = true_bucket_probs(belief.cdf, wide)
        score_sdk = expected_baseline_score(c_sdk_wide, wide, tp_wide)
        score_tq = expected_baseline_score(c_tq_wide, wide, tp_wide)
        score_oracle = expected_baseline_score(c_oracle_wide, wide, tp_wide)

        # SDK loses >27 expected tournament points on N(500,20) and >140 points on a 2.25-sigma outcome (545)
        self.assertGreater(score_oracle - score_sdk, 27.0)
        self.assertGreater(score_tq - score_sdk, 26.0)
        self.assertLess(abs(score_oracle - score_tq), 1.5)
        self.assertGreater(baseline_score(c_tq_wide, wide, 545.0) - baseline_score(c_sdk_wide, wide, 545.0), 130.0)

        # Range-invariance check: widening nominal bounds changes normalized inbound shoulder mass drastically in SDK
        # P(525.6 < X <= 560) under belief is ~9.86%; SDK shrinks it to ~0.34% on wide vs ~5.94% under TQ (>15x larger!)
        idx_560 = wide.bucket(560.0)
        idx_526 = wide.bucket(525.63)
        sdk_shoulder_wide = float(c_sdk_wide[idx_560] - c_sdk_wide[idx_526])
        tq_shoulder_wide = float(c_tq_wide[idx_560] - c_tq_wide[idx_526])
        self.assertLess(sdk_shoulder_wide, 0.006)
        self.assertGreater(tq_shoulder_wide, 0.050)

    def test_d1a_sdk_open_bound_discontinuity_at_upper_bound(self):
        """Shifting P90 from 99.99 to 100.0 across upper_bound jumps SDK upper-tail mass from 5.04% to 9.98%."""
        geo = Geometry(0.0, 100.0, open_lower=True, open_upper=True)
        pct_below = [(0.1, 40.0), (0.2, 50.0), (0.4, 65.0), (0.6, 78.0), (0.8, 90.0), (0.9, 99.99)]
        pct_at = [(0.1, 40.0), (0.2, 50.0), (0.4, 65.0), (0.6, 78.0), (0.8, 90.0), (0.9, 100.00)]

        sdk_below = sdk_cdf(pct_below, geo)
        sdk_at = sdk_cdf(pct_at, geo)
        # Discontinuity > 0.049 in upper tail mass and artificial boundary spike from a 0.01% shift in P90!
        self.assertGreater(abs((1.0 - sdk_at[-1]) - (1.0 - sdk_below[-1])), 0.048)
        self.assertGreater(pmf_from_cdf(sdk_below)[-2], 0.05)

        # Remedy `tq_cdf` is Lipschitz-continuous across the boundary
        tq_below = tq_cdf(pct_below, geo)
        tq_at = tq_cdf(pct_at, geo)
        self.assertLess(np.max(np.abs(tq_at - tq_below)), 1e-3)

    def test_d1b_sdk_log_scale_closed_bound_buffer_destroys_lower_half_of_axis(self):
        """[CRITICAL] On log-scaled [0.001, 1000] closed-lower, SDK buffer=1 overwrites ALL percentiles <= 1.001!"""
        geo = Geometry(0.001, 1000.0, open_lower=False, open_upper=False, zero_point=0.0)
        # True log-normal belief centered at 0.05 (well inside the lower half [0.001, 1.0] of the log axis)
        log_belief = stats.lognorm(s=0.8, scale=0.05)
        pct = [(p, float(log_belief.ppf(p))) for p in LEVELS_6]
        self.assertTrue(all(0.005 < v < 0.5 for _, v in pct))

        c_sdk = sdk_cdf(pct, geo)
        c_tq = tq_cdf(pct, geo)
        self.assertEqual(validate_cdf(c_sdk, geo), [])
        self.assertEqual(validate_cdf(c_tq, geo), [])

        # Because range_size = 999.999 > 100, SDK sets buffer = 1.0 and clamps ALL 6 percentiles to 1.001 (bucket 100)!
        pmf_sdk = pmf_from_cdf(c_sdk)
        modal_bucket_sdk = int(np.argmax(pmf_sdk))
        # Bucket 101 corresponds to nominal value ~1.001 (20x larger than the true median 0.05!)
        self.assertGreaterEqual(modal_bucket_sdk, 99)
        self.assertGreater(pmf_sdk[modal_bucket_sdk], 0.18)

        tp = true_bucket_probs(log_belief.cdf, geo)
        gain = expected_baseline_score(c_tq, geo, tp) - expected_baseline_score(c_sdk, geo, tp)
        self.assertGreater(gain, 75.0)

    def test_d1b_sdk_closed_lower_bound_zero_point_mass_displacement(self):
        """On closed-lower count range [0, 50], SDK pushes P10=P20=P40=0 to 0.50 (bucket 3), starving bucket 1."""
        geo = Geometry(0.0, 50.0, open_lower=False, open_upper=True)
        pct = [(0.1, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 2.0), (0.8, 5.0), (0.9, 9.0)]

        # Metaculus formulas.py maps exact outcome 0.0 to bucket 1
        self.assertEqual(geo.bucket(0.0), 1)

        c_sdk = sdk_cdf(pct, geo)
        c_tq = tq_cdf(pct, geo)
        self.assertEqual(validate_cdf(c_sdk, geo), [])
        self.assertEqual(validate_cdf(c_tq, geo), [])

        # SDK gives bucket 1 only ~6.07% despite 40% declared probability at <= 0; TQ gives the full 19% cap
        self.assertLess(pmf_from_cdf(c_sdk)[1], 0.07)
        self.assertGreater(pmf_from_cdf(c_tq)[1], 0.189)
        self.assertGreater(baseline_score(c_tq, geo, 0.0) - baseline_score(c_sdk, geo, 0.0), 55.0)

    def test_d1c_sdk_repeating_values_breaks_monotonicity_and_crashes_get_cdf(self):
        """Repeated in-bounds quantiles on narrow ranges (<1e-5) invert order AND crash get_cdf() with NaN!"""
        from forecasting_tools import NumericDistribution, Percentile

        geo = Geometry(0.0, 2e-6, open_lower=True, open_upper=True)
        # Mode A: Silent order inversion! SDK mutates P20 (1.0e-6) -> 0.2e-6 < P10 (0.8e-6), putting >35% mass below P10!
        pct_invert = [
            (0.10, 0.8e-6),
            (0.20, 1.0e-6),
            (0.40, 1.0e-6),
            (0.60, 1.4e-6),
            (0.80, 1.6e-6),
            (0.90, 1.8e-6),
        ]
        dist_inv = NumericDistribution(
            declared_percentiles=[Percentile(percentile=p, value=v) for p, v in pct_invert],
            open_upper_bound=True, open_lower_bound=True, upper_bound=2e-6, lower_bound=0.0, zero_point=None, cdf_size=201,
        )
        # Monotonicity invariant is silently violated inside the validated Pydantic model:
        self.assertLess(dist_inv.declared_percentiles[1].value, dist_inv.declared_percentiles[0].value)
        # Mode B: Round-tripping the SDK's own validated object (`model_dump` -> `model_validate`, used in report saving) crashes!
        with self.assertRaises(ValueError):
            NumericDistribution.model_validate(dist_inv.model_dump())
        # Mode C: And `get_cdf()` creates a discontinuous cliff right above P10=0.80e-6 (jumping to >0.28 at 0.82e-6)!
        c_corrupt = sdk_cdf(pct_invert, geo)
        self.assertGreater(c_corrupt[geo.bucket(0.82e-6)], 0.28)

        # TQ operates in dimensionless u-space with +1e-9 tie lift: preserves P10 (~11% at 0.82e-6) and passes validation!
        c_tq = tq_cdf(pct_invert, geo)
        self.assertEqual(validate_cdf(c_tq, geo), [])
        self.assertLess(c_tq[geo.bucket(0.82e-6)], 0.14)

    def test_d1d_internal_scorers_mismatch_official_metaculus_buckets_and_floors(self):
        """Reproduce off-by-one boundary bucket in market_prices._mass and <0.001 tail violation in paired_economic."""
        # 1. market_prices._mass uses floor((y - low)*bins/(high - low)) -> index k -> pmf[k+1]
        # whereas Metaculus formulas.py uses max(int(u*bins + 1 - 1e-10), 1) -> bucket k -> pmf[k]
        geo = Geometry(-25.0, 25.0, open_lower=True, open_upper=True, cdf_size=201)
        exact_edge_outcome = -24.75  # u = 0.25 / 50 = 1 / 200 -> exact boundary of bucket 1
        official_bucket = geo.bucket(exact_edge_outcome)
        market_prices_bucket = min(200 - 1, int(math.floor((exact_edge_outcome - (-25.0)) * 200 / 50.0))) + 1
        self.assertEqual(official_bucket, 1)
        self.assertEqual(market_prices_bucket, 2)
        self.assertNotEqual(official_bucket, market_prices_bucket)

        # 2. paired_economic.py (champion-v2) sets VIX (n=120) open-bound floor to 0.01 / (120 + 2) = 8.197e-5 < 0.001
        geo_vix = Geometry(0.0, 120.0, open_lower=True, open_upper=True, cdf_size=121)
        q = np.array([15.0, 18.0, 21.0, 25.0, 30.0])
        probs = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
        edges = np.linspace(0.0, 120.0, 121)
        res = np.interp(edges, q, probs)
        left_s = (q[1] - q[0]) * probs[0] / (probs[1] - probs[0])
        right_s = (q[-1] - q[-2]) * (1 - probs[-1]) / (probs[-1] - probs[-2])
        res[edges < q[0]] = probs[0] * np.exp((edges[edges < q[0]] - q[0]) / left_s)
        res[edges > q[-1]] = 1.0 - (1.0 - probs[-1]) * np.exp(-(edges[edges > q[-1]] - q[-1]) / right_s)
        weights = 0.99 * np.diff(np.concatenate(([0.0], res, [1.0]))) + 0.01 / 122.0
        paired_econ_cdf = np.cumsum(weights)[:-1]
        violations = validate_cdf(paired_econ_cdf, geo_vix)
        self.assertTrue(any("open upper bound mass" in v or "open lower bound mass" in v for v in violations))

    def test_d1e_main_py_date_bound_message_truncates_intraday_utc_timestamps(self):
        """main.py formats DateQuestion bounds with .date().isoformat(), placing YYYY-MM-DD below intraday lower_bound."""
        src = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("question.lower_bound.date().isoformat()", src)
        lower_dt = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
        displayed_lower = lower_dt.date().isoformat()  # "2026-11-25"
        parsed_ts = datetime.fromisoformat(displayed_lower).replace(tzinfo=timezone.utc).timestamp()
        self.assertLess(parsed_ts, lower_dt.timestamp())


class TestDefect2BinaryAggregationAndCalibration(unittest.TestCase):
    """[VERIFIED CODE DEFECTS] in OMEGA binary residual correlation, Venn-Abers under censoring, and main.py clipping."""

    def test_d2a_aleatoric_bernoulli_variance_inflates_omega_residual_correlation(self):
        """For 3 independent forecasters, Corr(p - Y) > 0.93 while exact variance-cancelling identity recovers rho ~ 0."""
        rng = np.random.default_rng(20261124)
        n = 6000
        # 1. Direct unbiased probability-space independent errors: p_k = clip(pi + eps_k), Corr(eps_j, eps_k) == 0
        pi_u = rng.uniform(0.18, 0.82, size=n)
        y_u = rng.binomial(1, pi_u)
        probs_u = np.clip(pi_u[:, None] + rng.normal(0.0, 0.07, size=(n, 3)), 0.01, 0.99)
        dec_u = decompose_residual_covariance(probs_u, y_u)
        self.assertGreater(dec_u["naive_outcome_residual_corr_mean"], 0.95)
        self.assertLess(dec_u["debiased_epistemic_corr_mean"], 0.08)

        # 2. Bayesian log-odds signal observers: naive alpha is crushed (~1.03) while debiased alpha recovers >1.30
        z = rng.normal(0.0, 1.2, size=n)
        pi = 1.0 / (1.0 + np.exp(-z))
        y = rng.binomial(1, pi)
        sig = z[:, None] + rng.normal(0.0, 0.8, size=(n, 3))
        s1 = 1.44 / (1.44 + 0.64)
        vp1 = 1.44 * 0.64 / (1.44 + 0.64)
        probs = 1.0 / (1.0 + np.exp(-sig * s1 / math.sqrt(1.0 + math.pi / 8.0 * vp1)))

        decomp = decompose_residual_covariance(probs, y)
        self.assertGreater(decomp["naive_outcome_residual_corr_mean"], 0.93)
        self.assertGreater(decomp["satopaa_alpha_debiased"] - decomp["satopaa_alpha_naive"], 0.25)

    def test_d2b_right_censored_validation_degrades_isotonic_venn_abers(self):
        """Short-horizon right-censoring shifts validation base rate and biases non-parametric isotonic Venn-Abers."""
        rng = np.random.default_rng(77)
        n_va, n_te = 600, 2000
        z_va = rng.normal(-1.1, 1.2, size=n_va)  # right-censored short-horizon validation only!
        y_va = rng.binomial(1, 1.0 / (1.0 + np.exp(-z_va)))
        sig_va = z_va[:, None] + rng.normal(0.0, 0.6, size=(n_va, 3))
        p_va = 1.0 / (1.0 + np.exp(-0.75 * sig_va))

        z_te = np.concatenate([rng.normal(-1.1, 1.2, n_te // 2), rng.normal(0.4, 1.2, n_te // 2)])
        pi_te = 1.0 / (1.0 + np.exp(-z_te))
        y_te = rng.binomial(1, pi_te)
        p_te = 1.0 / (1.0 + np.exp(-0.75 * (z_te[:, None] + rng.normal(0.0, 0.6, size=(n_te, 3)))))

        order = np.argsort(p_va.mean(1))
        xs = p_va.mean(1)[order]
        blocks = []
        for xv, yv in zip(xs, y_va[order]):
            blocks.append([xv, xv, float(yv), 1.0])
            while len(blocks) > 1 and blocks[-2][2] * blocks[-1][3] > blocks[-1][2] * blocks[-2][3]:
                a, b = blocks[-2:]
                blocks[-2:] = [[a[0], b[1], a[2] + b[2], a[3] + b[3]]]
        knot_x = np.array([0.5 * (b[0] + b[1]) for b in blocks])
        knot_y = np.array([b[2] / b[3] for b in blocks])
        iso_pred = np.interp(p_te.mean(1), knot_x, knot_y)

        heb = HEBLoPAggregator().fit(p_va, y_va)
        heb_pred = heb.predict(p_te)
        self.assertLess(np.mean((heb_pred - y_te) ** 2), np.mean((iso_pred - y_te) ** 2))

    def test_d2c_main_py_binary_clipping_is_10x_tighter_than_metaculus_api(self):
        """main.py clamps binary forecasts to [0.01, 0.99] whereas Metaculus allows [0.001, 0.999]."""
        src = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertTrue(re.search(r"max\(\s*0\.01\s*,\s*min\(\s*0\.99", src))
        score_999 = 100.0 * math.log(2.0 * 0.999) / math.log(2.0)
        score_990 = 100.0 * math.log(2.0 * 0.990) / math.log(2.0)
        self.assertGreater(score_999 - score_990, 1.30)


class TestDefect3MacroVolatilityAndCausalInvariants(unittest.TestCase):
    """[VERIFIED CODE DEFECTS] in Market Pulse / OMEGA volatility scaling + strict prefix-invariance checks."""

    def test_d3a_market_prices_505_vs_252_volatility_denominator_bias(self):
        """505-day raw differences scaled by 252-day long_vol systematically under-disperses both IQR (>35%) and W80 (>15%)."""
        rng = np.random.default_rng(1234)
        # Year -2 calm (sigma=0.4), Year -1 + current 30d high-vol regime (sigma=2.0)
        r_calm = rng.normal(0.0, 0.4, size=253)
        r_high = rng.normal(0.0, 2.0, size=251)
        values = 100.0 + np.cumsum(np.concatenate([[0.0], r_calm, r_high]))  # 505 prices
        h = 10
        levels = (0.10, 0.25, 0.50, 0.75, 0.90)

        hist_changes = values[h:] - values[:-h]
        daily = np.diff(values)
        mult = float(np.clip(math.sqrt(np.std(daily[-30:], ddof=1) / np.std(daily[-252:], ddof=1)), 0.7, 2.0))
        champ_q = values[-1] + mult * np.quantile(hist_changes - np.median(hist_changes), levels)

        realized_cur_sd = float(np.std(daily[-30:], ddof=1))
        true_w80 = 2.0 * 1.2815515655 * realized_cur_sd * math.sqrt(h)
        true_w50 = 2.0 * 0.6744897502 * realized_cur_sd * math.sqrt(h)
        champ_w80 = float(champ_q[-1] - champ_q[0])
        champ_w50 = float(champ_q[3] - champ_q[1])
        fhs_q = fhs_standardized_quantiles(values, h, levels=levels)
        fhs_w80 = float(fhs_q[-1] - fhs_q[0])
        fhs_w50 = float(fhs_q[3] - fhs_q[1])

        self.assertLess(champ_w50 / true_w50, 0.60)  # >40% IQR under-dispersion!
        self.assertLess(champ_w80 / true_w80, 0.80)  # >20% W80 under-dispersion!
        self.assertAlmostEqual(fhs_w80 / true_w80, 1.0, delta=0.15)
        self.assertAlmostEqual(fhs_w50 / true_w50, 1.0, delta=0.15)

    def test_d3b_strict_prefix_invariance_zero_future_leakage_binary_and_continuous(self):
        """[CAUSAL INVARIANT] Corrupting future outcomes/inputs changes 0 bits of past predictions."""
        # 1. Binary online_heb_lop_predict: corrupt outcomes that resolve AFTER round 1
        train_valid = [
            ["train", "2025-10-01", "fred", f"q{i}", "2025-10-15", "[]", 0.2 + 0.05 * (i % 10), 0.25 + 0.05 * (i % 10), 0.3 + 0.05 * (i % 10), i % 2, "f"]
            for i in range(40)
        ]
        test_clean = [
            ["test", "2026-03-01", "fred", "t1", "2026-04-15", "[]", 0.30, 0.40, 0.35, 0, "f"],
            ["test", "2026-03-15", "fred", "t2", "2026-03-20", "[]", 0.70, 0.75, 0.65, 1, "f"],
        ]
        # Corrupt both test labels AND add an overlapping unresolved row before 2026-03-15
        test_corrupted = [
            ["test", "2026-03-01", "fred", "t1", "2026-04-15", "[]", 0.30, 0.40, 0.35, 1, "f"],  # resolves Apr 15 > Mar 15!
            ["test", "2026-03-15", "fred", "t2", "2026-03-20", "[]", 0.70, 0.75, 0.65, 0, "f"],
        ]
        p_clean, _ = online_heb_lop_predict(train_valid, test_clean)
        p_corrupt, _ = online_heb_lop_predict(train_valid, test_corrupted)
        self.assertTrue(np.array_equal(p_clean, p_corrupt))

        # 2. Continuous run_causal_aci_sequence: corrupting an outcome published AFTER next origin changes 0 bits
        from datetime import date
        rng = np.random.default_rng(9)
        h1 = 5.0 + np.cumsum(rng.normal(0.0, 0.15, 120))
        h2 = 5.2 + np.cumsum(rng.normal(0.0, 0.15, 120))
        cases_a = [
            (date(2020, 3, 31), date(2020, 2, 1), 6, date(2020, 8, 1), date(2020, 9, 4), 5.5, h1),
            (date(2020, 6, 30), date(2020, 5, 1), 3, date(2020, 8, 1), date(2020, 9, 4), 5.4, h2),
        ]
        cases_b = [
            (date(2020, 3, 31), date(2020, 2, 1), 6, date(2020, 8, 1), date(2020, 9, 4), 99.9, h1),  # published Sep 4 > Jun 30!
            (date(2020, 6, 30), date(2020, 5, 1), 3, date(2020, 8, 1), date(2020, 9, 4), 0.1, h2),
        ]
        d_a = run_causal_aci_sequence(cases_a)
        d_b = run_causal_aci_sequence(cases_b)
        grid = np.linspace(0.05, 0.95, 19)
        self.assertTrue(np.array_equal(d_a[0].quantile(grid), d_b[0].quantile(grid)))
        self.assertTrue(np.array_equal(d_a[1].quantile(grid), d_b[1].quantile(grid)))


class TestMathematicalInvariantsAndMalformedInputs(unittest.TestCase):
    """[MATHEMATICAL INVARIANTS] Properness, extreme scales, heavy tails, and malformed input rejection."""

    def test_strict_properness_of_expected_metaculus_baseline_score(self):
        """By Gibbs' inequality, expected continuous Baseline score is uniquely maximized at the true bucket PMF."""
        geo = Geometry(0.0, 100.0, open_lower=True, open_upper=True)
        truth = stats.t(df=4.0, loc=42.0, scale=9.0)
        p_true = true_bucket_probs(truth.cdf, geo)
        cdf_true = np.cumsum(p_true)[:-1]
        s_opt = expected_baseline_score(cdf_true, geo, p_true)
        for shift in (-8.0, -2.0, 2.0, 8.0):
            p_wrong = true_bucket_probs(stats.t(df=4.0, loc=42.0 + shift, scale=9.0).cdf, geo)
            s_wrong = expected_baseline_score(np.cumsum(p_wrong)[:-1], geo, p_true)
            self.assertLess(s_wrong, s_opt - 0.5)

    def test_tq_cdf_extreme_scales_heavy_tails_and_discrete_cdf_sizes(self):
        """TQ produces Metaculus-valid CDFs across 1e-9..1e15, Pareto/Cauchy tails, log scales, and discrete sizes."""
        configs = [
            (Geometry(1e-9, 1e-6, True, True), [(0.1, 2e-7), (0.5, 5e-7), (0.9, 8e-7)]),
            (Geometry(1e12, 1e15, False, True, zero_point=0.0), [(0.1, 1e12), (0.5, 3e13), (0.9, 5e14)]),
            (Geometry(-50.0, 50.0, True, True, cdf_size=52), [(0.1, -49.9), (0.5, 0.0), (0.9, 49.9)]),
            (Geometry(0.0, 100.0, False, False, cdf_size=26), [(0.1, 0.0), (0.25, 0.0), (0.5, 1.0), (0.75, 5.0), (0.9, 100.0)]),
        ]
        for geo, pct in configs:
            c = tq_cdf(pct, geo)
            self.assertEqual(validate_cdf(c, geo), [], f"failed on {geo}")
            # Verify normal_fit_cdf also respects Metaculus rules
            c_norm = normal_fit_cdf(pct, geo)
            self.assertEqual(validate_cdf(c_norm, geo), [])

    def test_malformed_inputs_rejected_cleanly(self):
        """NaN, Inf, non-increasing levels, decreasing values, and below-zero-point inputs raise ValueError."""
        geo_log = Geometry(1.0, 1000.0, False, True, zero_point=0.0)
        bad_inputs = [
            [(0.1, float("nan")), (0.9, 10.0)],
            [(0.1, 2.0), (0.9, float("inf"))],
            [(0.5, 10.0), (0.5, 20.0)],            # duplicate probability level
            [(0.0, 10.0), (0.9, 20.0)],            # level 0.0 outside (0, 1)
            [(0.1, 50.0), (0.9, 20.0)],            # decreasing values
            [(0.1, 0.0), (0.9, 20.0)],             # value == zero_point on log question
        ]
        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                tq_cdf(bad, geo_log)
        with self.assertRaises(ValueError):
            TruncatedStudentT(float("nan"), 1.0, 3.5)
        with self.assertRaises(ValueError):
            TruncatedStudentT(5.0, 1.0, 0.5)       # df <= 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
