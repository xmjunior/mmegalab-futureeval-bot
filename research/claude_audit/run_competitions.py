"""Reproducible Scientific Competition Runner (Zero Paid APIs, Zero Production Edits).

Executes three out-of-sample competitions corresponding to the Top 3 Mathematical Defects:
  1. Continuous CDF Construction (`tq_cdf` vs `sdk_cdf`, `normal_fit_cdf`, `metalog_cdf`)
     on an 8-regime x 50-case exact mathematical benchmark AND frozen Market Pulse 26Q4 ALFRED cases.
  2. Multi-Model Binary Aggregation & Calibration (`heb_lop_static`, `heb_lop_online`,
     `satopaa_debiased` vs OMEGA `correlation_pool`, `venn_mean`, `hedge`, `mini`, `mean`, `median`)
     on the SHA-verified 8,027-row ForecastBench archive (`test` N = 5,723).
  3. Point-in-Time Continuous Macro Forecasting (`robust_t_aci`, `holt_damped_t` vs
     `champion_rule`, `cqr_tails`, `naive_rw`, `empirical`, `three_scales`, `ridge_ar`)
     on the SHA-verified 1,471-row ALFRED `UNRATE` real-time vintage archive (`test` N = 108).

Writes complete JSON evidence to `research/claude_audit/results_competition.json`.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from binary_aggregation import (
    HEBLoPAggregator,
    decompose_residual_covariance,
    inverse_brier_linear_pool,
    online_heb_lop_predict,
    satopaa_debiased_pool,
)
from builders import metalog_cdf, normal_fit_cdf, tq_cdf
from macro_fhs_aci import holt_damped_t_distribution, run_causal_aci_sequence
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

OMEGA_COMMIT = "80bdeaaf9c099005cd7ab9aa838332dd497d390e"
MARKET_COMMIT = "199e336d333f74b47b71a1df607cc430107dab39"


def _git_blob(commit: str, relpath: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relpath}"], cwd=REPO)


def holm_step_down(p_values: dict[str, float]) -> dict[str, float]:
    """Holm (1979) strong family-wise error rate step-down adjustment."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    running = 0.0
    out = {}
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * float(p)))
        out[name] = running
    return out


def discrete_bucket_crps(cdf: np.ndarray, true_pmf: np.ndarray) -> float:
    """Exact discrete RPS across the 201 Metaculus CDF boundaries separating the N+2 buckets."""
    F_true = np.cumsum(true_pmf)[:-1]
    return float(np.mean((np.asarray(cdf, dtype=float) - F_true) ** 2))


# ======================================================================================
# COMPETITION 1: CONTINUOUS CDF GEOMETRY & METACULUS SCORING
# ======================================================================================

def competition_1_continuous_cdf() -> dict:
    rng = np.random.default_rng(20261124)
    levels_6 = [0.10, 0.20, 0.40, 0.60, 0.80, 0.90]
    regimes = [
        "tight_gaussian_open",
        "moderate_gaussian_open",
        "heavy_tail_t3_open",
        "skew_lognormal_open",
        "boundary_near_upper_open",
        "log_scale_closed_lower",
        "censored_zero_atom_count",
        "narrow_scale_tied_grid",
    ]
    n_per_regime = 50
    methods = ("sdk_cdf", "normal_fit_cdf", "metalog_cdf", "tq_cdf", "oracle_true_cdf")
    records = []
    timings_s = defaultdict(float)
    failures = defaultdict(int)

    for reg in regimes:
        for idx in range(n_per_regime):
            if reg == "tight_gaussian_open":
                geo = Geometry(0.0, 1000.0, True, True)
                mu = float(rng.uniform(200.0, 800.0))
                sd = float(rng.uniform(15.0, 45.0))
                dist_true = stats.norm(mu, sd)
                pct = [(p, float(dist_true.ppf(p))) for p in levels_6]
                tp = true_bucket_probs(dist_true.cdf, geo)
            elif reg == "moderate_gaussian_open":
                geo = Geometry(-100.0, 100.0, True, True)
                mu = float(rng.uniform(-40.0, 40.0))
                sd = float(rng.uniform(16.0, 42.0))
                dist_true = stats.norm(mu, sd)
                pct = [(p, float(dist_true.ppf(p))) for p in levels_6]
                tp = true_bucket_probs(dist_true.cdf, geo)
            elif reg == "heavy_tail_t3_open":
                geo = Geometry(0.0, 500.0, True, True)
                mu = float(rng.uniform(120.0, 380.0))
                sc = float(rng.uniform(10.0, 40.0))
                dist_true = stats.t(df=3.0, loc=mu, scale=sc)
                pct = [(p, float(dist_true.ppf(p))) for p in levels_6]
                tp = true_bucket_probs(dist_true.cdf, geo)
            elif reg == "skew_lognormal_open":
                geo = Geometry(0.0, 400.0, True, True)
                s = float(rng.uniform(0.35, 0.85))
                scale = float(rng.uniform(25.0, 110.0))
                dist_true = stats.lognorm(s=s, scale=scale)
                pct = [(p, float(dist_true.ppf(p))) for p in levels_6]
                tp = true_bucket_probs(dist_true.cdf, geo)
            elif reg == "boundary_near_upper_open":
                geo = Geometry(0.0, 100.0, True, True)
                mu = float(rng.uniform(76.0, 94.0))
                sd = float(rng.uniform(8.0, 18.0))
                dist_true = stats.norm(mu, sd)
                pct = [(p, float(dist_true.ppf(p))) for p in levels_6]
                tp = true_bucket_probs(dist_true.cdf, geo)
            elif reg == "log_scale_closed_lower":
                geo = Geometry(0.001, 1000.0, False, False, zero_point=0.0)
                med = float(10.0 ** rng.uniform(-2.1, 1.2))
                s = float(rng.uniform(0.45, 1.05))
                raw_ln = stats.lognorm(s=s, scale=med)
                # Censor latent distribution cleanly onto closed [0.001, 1000]
                edges = geo.edges()
                F_e = np.clip(raw_ln.cdf(edges), 0.0, 1.0)
                tp = np.zeros(geo.n + 2)
                tp[1] = F_e[1]  # all mass <= edge_1 censored at closed lower bound lands in bucket 1
                tp[2:-2] = np.diff(F_e[1:-1])
                tp[-2] = 1.0 - F_e[-2]  # all mass >= edge_{n-1} lands in bucket n
                tp = np.clip(tp, 0.0, None); tp /= tp.sum()
                pct = [(p, float(np.clip(raw_ln.ppf(p), 0.001, 1000.0))) for p in levels_6]
            elif reg == "censored_zero_atom_count":
                geo = Geometry(0.0, 50.0, False, True)
                mu = float(rng.uniform(-4.0, 3.0))
                sd = float(rng.uniform(3.0, 8.0))
                latent = stats.norm(mu, sd)
                edges = geo.edges()
                F_e = latent.cdf(edges)
                tp = np.zeros(geo.n + 2)
                tp[1] = F_e[1]  # Tobit censoring max(0, X): atom at 0 lands in bucket 1!
                tp[2:-1] = np.diff(F_e[1:])
                tp[-1] = 1.0 - F_e[-1]
                tp = np.clip(tp, 0.0, None); tp /= tp.sum()
                pct = [(p, float(max(0.0, latent.ppf(p)))) for p in levels_6]
            else:  # narrow_scale_tied_grid
                geo = Geometry(0.0, 4e-6, False, False)
                mu = float(rng.uniform(0.8e-6, 3.2e-6))
                sd = float(rng.uniform(0.12e-6, 0.45e-6))
                latent = stats.norm(mu, sd)
                edges = geo.edges()
                F_e = latent.cdf(edges)
                tp = np.zeros(geo.n + 2)
                tp[1] = F_e[1]
                tp[2:-2] = np.diff(F_e[1:-1])
                tp[-2] = 1.0 - F_e[-2]
                tp = np.clip(tp, 0.0, None); tp /= tp.sum()
                # Snap to 101-point grid just like main.py:587-608 choose_integers!
                grid_101 = np.linspace(geo.lower, geo.upper, 101)
                pct = [(p, float(grid_101[int(np.argmin(np.abs(grid_101 - latent.ppf(p))))])) for p in levels_6]

            oracle_raw = np.cumsum(tp)[:-1]
            if not geo.open_lower:
                oracle_raw[0] = 0.0
            if not geo.open_upper:
                oracle_raw[-1] = 1.0
            c_oracle = sdk_standardize(oracle_raw, geo)

            row_scores = {"regime": reg, "case": idx}
            row_crps = {"regime": reg, "case": idx}

            for m_name, fn in [
                ("sdk_cdf", lambda: sdk_cdf(pct, geo)),
                ("normal_fit_cdf", lambda: normal_fit_cdf(pct, geo)),
                ("metalog_cdf", lambda: metalog_cdf(pct, geo)),
                ("tq_cdf", lambda: tq_cdf(pct, geo)),
                ("oracle_true_cdf", lambda: c_oracle),
            ]:
                t0 = time.perf_counter()
                try:
                    c = fn()
                    timings_s[m_name] += time.perf_counter() - t0
                    errs = validate_cdf(c, geo)
                    if errs:
                        raise ValueError("; ".join(errs))
                except Exception:
                    timings_s[m_name] += time.perf_counter() - t0
                    failures[m_name] += 1
                    # Uniform legal fallback when a method crashes/fails so paired expectations remain well-defined
                    c = sdk_standardize(np.linspace(0.0, 1.0, geo.cdf_size), geo)
                row_scores[m_name] = expected_baseline_score(c, geo, tp)
                row_crps[m_name] = discrete_bucket_crps(c, tp)
            records.append((row_scores, row_crps))

    # Summarize synthetic benchmark + block bootstrap over regimes + Holm control
    n_total = len(records)
    summary_syn = {}
    for m in methods:
        vals = np.array([r[0][m] for r in records])
        crps_v = np.array([r[1][m] for r in records])
        summary_syn[m] = {
            "mean_expected_baseline_score": float(vals.mean()),
            "mean_discrete_rps": float(crps_v.mean()),
            "failures_out_of_400": int(failures[m]),
            "valid_rate": float((n_total - failures[m]) / n_total),
            "mean_latency_ms": float(1000.0 * timings_s[m] / n_total),
            "by_regime": {
                reg: float(np.mean([r[0][m] for r in records if r[0]["regime"] == reg]))
                for reg in regimes
            },
        }

    # Contrasts of candidates vs sdk_cdf across the 8 independent regime blocks & 400 cases
    contrasts_syn = {}
    raw_p = {}
    for cand in ("tq_cdf", "normal_fit_cdf", "metalog_cdf"):
        diff_all = np.array([r[0][cand] - r[0]["sdk_cdf"] for r in records])
        reg_means = np.array([
            np.mean([r[0][cand] - r[0]["sdk_cdf"] for r in records if r[0]["regime"] == reg])
            for reg in regimes
        ])
        # Exact 2^8 = 256 regime-block sign-flip permutation p-value (one-sided cand > sdk_cdf)
        obs = float(reg_means.mean())
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(regimes))))
        p_reg_perm = float(np.mean((signs @ reg_means) / len(regimes) >= obs - 1e-12))
        # Paired case-level Wilcoxon signed-rank p-value (one-sided)
        p_wilcoxon = float(stats.wilcoxon(diff_all, alternative="greater").pvalue)
        b_rng = np.random.default_rng(77411)
        boot = diff_all[b_rng.integers(0, len(diff_all), size=(4000, len(diff_all)))].mean(axis=1)
        contrasts_syn[cand + "_vs_sdk_cdf"] = {
            "mean_baseline_score_gain": float(diff_all.mean()),
            "median_case_gain": float(np.median(diff_all)),
            "positive_regimes_out_of_8": int(np.sum(reg_means > 0)),
            "case_win_rate": float(np.mean(diff_all > 1e-9)),
            "ci95_bootstrap_gain": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "p_regime_sign_permutation_exact": p_reg_perm,
            "p_wilcoxon_one_sided": p_wilcoxon,
        }
        raw_p[cand + "_vs_sdk_cdf"] = p_reg_perm
    for k, ph in holm_step_down(raw_p).items():
        contrasts_syn[k]["p_holm_regime_exact"] = ph

    # Part 1B: Frozen point-in-time Market Pulse 26Q4 FRED subquestions (`199e336:market_prices.py`)
    mp_src = _git_blob(MARKET_COMMIT, "market_prices.py").decode("utf-8")
    spec = importlib.util.spec_from_loader("mp_frozen", loader=None)
    mp_mod = importlib.util.module_from_spec(spec)
    sys.modules["mp_frozen"] = mp_mod
    exec(compile(mp_src, "market_prices_199e336.py", "exec"), mp_mod.__dict__)

    levels_mp = list(mp_mod.LEVELS)  # [0.1, 0.25, 0.5, 0.75, 0.9]
    fred_rows = []
    closed_lookup = {
        (r["post_id"], r["spot"].astimezone(mp_mod.timezone.utc).isoformat()): r
        for r in mp_mod.HISTORY
    }
    for (post_id, asof_s, spot_s, res_s), replay in sorted(mp_mod._FRED_REPLAY.items()):
        row = closed_lookup[(post_id, spot_s)]
        geo = Geometry(float(row["low"]), float(row["high"]), True, True, cdf_size=int(row["bins"]) + 1)
        adj_q = list(zip(levels_mp, replay[5]))
        y = float(row["outcome"])
        c_sdk = sdk_cdf(adj_q, geo)
        c_tq = tq_cdf(adj_q, geo)
        c_norm = normal_fit_cdf(adj_q, geo)
        fred_rows.append({
            "post_id": post_id,
            "question_id": row["question_id"],
            "spot": spot_s,
            "outcome": y,
            "sdk_baseline_score": baseline_score(c_sdk, geo, y),
            "tq_baseline_score": baseline_score(c_tq, geo, y),
            "normal_baseline_score": baseline_score(c_norm, geo, y),
            "tq_minus_sdk_points": baseline_score(c_tq, geo, y) - baseline_score(c_sdk, geo, y),
        })

    gains_mp = [r["tq_minus_sdk_points"] for r in fred_rows]
    return {
        "synthetic_8_regimes_400_cases": {
            "n_cases": n_total,
            "n_regimes": len(regimes),
            "methods": summary_syn,
            "contrasts": contrasts_syn,
        },
        "market_pulse_26q4_frozen_alfred_6_cases": {
            "source_commit": MARKET_COMMIT,
            "n_cases": len(fred_rows),
            "mean_sdk_baseline_score": float(np.mean([r["sdk_baseline_score"] for r in fred_rows])),
            "mean_tq_baseline_score": float(np.mean([r["tq_baseline_score"] for r in fred_rows])),
            "mean_normal_baseline_score": float(np.mean([r["normal_baseline_score"] for r in fred_rows])),
            "mean_tq_minus_sdk_points": float(np.mean(gains_mp)),
            "wins_ties_losses_tq_vs_sdk": [
                sum(g > 1e-9 for g in gains_mp),
                sum(abs(g) <= 1e-9 for g in gains_mp),
                sum(g < -1e-9 for g in gains_mp),
            ],
            "cases": fred_rows,
        },
    }


# ======================================================================================
# COMPETITION 2: MULTI-MODEL BINARY AGGREGATION & CALIBRATION (FORECASTBENCH N = 5,723)
# ======================================================================================

def competition_2_binary_aggregation() -> dict:
    manifest = json.loads(_git_blob(OMEGA_COMMIT, "research/omega/binary/inputs.json"))
    raw_gz = _git_blob(OMEGA_COMMIT, "research/omega/binary/rows.jsonl.gz")
    raw_bytes = gzip.decompress(raw_gz)
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != manifest["rows_jsonl_sha256"]:
        raise RuntimeError("ForecastBench rows.jsonl.gz SHA-256 mismatch")

    # Load exact OMEGA binary evaluator to keep identical metrics, block bootstraps & sign tests
    tmp_dir = Path(tempfile.mkdtemp())
    (tmp_dir / "evaluate.py").write_bytes(_git_blob(OMEGA_COMMIT, "research/omega/binary/evaluate.py"))
    spec = importlib.util.spec_from_file_location("omega_bin", tmp_dir / "evaluate.py")
    ob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ob)

    rows = [json.loads(line) for line in raw_bytes.splitlines()]
    train = [r for r in rows if r[0] == "train"]
    valid = [r for r in rows if r[0] == "valid"]
    test = [r for r in rows if r[0] == "test"]

    # 1. Existing OMEGA baseline & ensemble predictions on `test`
    t0 = time.perf_counter()
    preds_all, omega_fit = ob.predict(train, valid, test)
    omega_ms = 1000.0 * (time.perf_counter() - t0)

    train_valid = train + valid
    p_tv = np.array([r[6:9] for r in train_valid], dtype=float)
    y_tv = np.array([r[9] for r in train_valid], dtype=float)
    d_tv = [r[2] for r in train_valid]

    p_te = np.array([r[6:9] for r in test], dtype=float)
    y_te = np.array([r[9] for r in test], dtype=float)
    d_te = [r[2] for r in test]
    dates_te = [r[1] for r in test]
    fams_te = [r[10] for r in test]

    # 2. Simple & published comparators
    t0 = time.perf_counter()
    preds_all["inverse_brier_pool"] = inverse_brier_linear_pool(p_tv, y_tv, p_te)
    inv_ms = 1000.0 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    preds_all["satopaa_debiased"] = satopaa_debiased_pool(p_tv, y_tv, p_te)
    sat_ms = 1000.0 * (time.perf_counter() - t0)

    # 3. Proposed Candidate A: Static Hierarchical Empirical-Bayes Log-Odds Pool (fitted strictly on train+valid)
    t0 = time.perf_counter()
    heb_static = HEBLoPAggregator().fit(p_tv, y_tv, d_tv)
    preds_all["heb_lop_static"] = heb_static.predict(p_te, d_te)
    heb_static_ms = 1000.0 * (time.perf_counter() - t0)

    # 4. Proposed Candidate B: Strictly-Causal Prequential Online HEB-LoP (updates as rounds resolve <= due_date)
    t0 = time.perf_counter()
    preds_online, online_diag = online_heb_lop_predict(train_valid, test)
    preds_all["heb_lop_online"] = preds_online
    heb_online_ms = 1000.0 * (time.perf_counter() - t0)

    scores = {arm: ob.summarize(prob, y_te) for arm, prob in preds_all.items()}

    # Head-to-head contrasts against BOTH `mean` AND `mini` (the undefeated single-model leader in OMEGA!)
    candidate_arms = ("heb_lop_static", "heb_lop_online", "satopaa_debiased", "inverse_brier_pool",
                      "correlation_pool", "venn_mean", "hedge")
    contrasts = {}
    raw_p_vs_mean = {}
    raw_p_vs_mini = {}
    for arm in candidate_arms:
        gain_vs_mean = ob.losses(preds_all["mean"], y_te)[0] - ob.losses(preds_all[arm], y_te)[0]
        gain_vs_mini = ob.losses(preds_all["mini"], y_te)[0] - ob.losses(preds_all[arm], y_te)[0]
        ll_gain_vs_mini = ob.losses(preds_all["mini"], y_te)[1] - ob.losses(preds_all[arm], y_te)[1]

        ci_2rd_mean = ob.boot_interval(gain_vs_mean, dates_te, True)
        ci_fam_mean = ob.boot_interval(gain_vs_mean, fams_te, False)
        ci_2rd_mini = ob.boot_interval(gain_vs_mini, dates_te, True)
        ci_fam_mini = ob.boot_interval(gain_vs_mini, fams_te, False)

        p_rs_mean = ob.round_sign_p(gain_vs_mean, dates_te)
        p_rs_mini = ob.round_sign_p(gain_vs_mini, dates_te)
        raw_p_vs_mean[arm] = p_rs_mean
        raw_p_vs_mini[arm] = p_rs_mini

        contrasts[arm] = {
            "gain_brier_vs_mean": float(gain_vs_mean.mean()),
            "brier_skill_vs_mean": float(gain_vs_mean.mean() / scores["mean"]["brier"]),
            "ci95_vs_mean_envelope": [min(ci_2rd_mean[0], ci_fam_mean[0]), max(ci_2rd_mean[1], ci_fam_mean[1])],
            "p_round_sign_vs_mean": p_rs_mean,
            "gain_brier_vs_mini": float(gain_vs_mini.mean()),
            "brier_skill_vs_mini": float(gain_vs_mini.mean() / scores["mini"]["brier"]),
            "gain_logloss_vs_mini": float(ll_gain_vs_mini.mean()),
            "ci95_vs_mini_two_round_blocks": ci_2rd_mini,
            "ci95_vs_mini_family_blocks": ci_fam_mini,
            "p_round_sign_vs_mini": p_rs_mini,
            "p_adjacent_block_sign_vs_mini": ob.block_sign_p(gain_vs_mini, dates_te),
            "tail_p95_beats_or_matches_mean": bool(scores[arm]["log_loss_p95"] <= scores["mean"]["log_loss_p95"] + 1e-6),
        }

    holm_mean = holm_step_down(raw_p_vs_mean)
    holm_mini = holm_step_down(raw_p_vs_mini)
    for arm in candidate_arms:
        c = contrasts[arm]
        c["p_holm_vs_mean"] = holm_mean[arm]
        c["p_holm_vs_mini"] = holm_mini[arm]
        c["strict_omega_screen_pass"] = bool(
            c["brier_skill_vs_mean"] >= 0.02
            and c["brier_skill_vs_mini"] >= 0.02
            and scores[arm]["log_loss"] < min(scores["mean"]["log_loss"], scores["mini"]["log_loss"])
            and scores[arm]["log_loss_p95"] <= scores["mean"]["log_loss_p95"]
            and c["ci95_vs_mean_envelope"][0] > 0
            and holm_mean[arm] < 0.05
        )
        c["dominates_best_single_model_mini"] = bool(
            c["gain_brier_vs_mini"] > 0
            and c["gain_logloss_vs_mini"] > 0
            and c["ci95_vs_mini_two_round_blocks"][0] > 0
            and c["ci95_vs_mini_family_blocks"][0] > 0
            and holm_mini[arm] < 0.05
        )

    # Horizon breakdown (`short` <= 30d vs `long` > 30d where valid had n=0!)
    short_mask = np.array([ob.bucket(r)[1] for r in test])
    horizon_scores = {
        "short_le_30d": {
            "n": int(short_mask.sum()),
            "scores": {a: ob.summarize(preds_all[a][short_mask], y_te[short_mask])
                       for a in ("mini", "mean", "correlation_pool", "venn_mean", "heb_lop_static", "heb_lop_online")},
        },
        "long_gt_30d": {
            "n": int((~short_mask).sum()),
            "scores": {a: ob.summarize(preds_all[a][~short_mask], y_te[~short_mask])
                       for a in ("mini", "mean", "correlation_pool", "venn_mean", "heb_lop_static", "heb_lop_online")},
        },
    }

    return {
        "source_commit": OMEGA_COMMIT,
        "rows_jsonl_sha256": actual_sha,
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "covariance_audit_train_valid": decompose_residual_covariance(p_tv, y_tv),
        "heb_lop_static_fit": heb_static.diagnostics_,
        "heb_lop_online_summary": {
            "rounds": len(online_diag["resolved_history_by_round"]),
            "first_round_history": min(online_diag["resolved_history_by_round"].values()),
            "last_round_history": max(online_diag["resolved_history_by_round"].values()),
            "last_fit": online_diag["last_fit"],
        },
        "runtime_ms_total_for_5723_forecasts": {
            "omega_all_arms": round(omega_ms, 2),
            "inverse_brier_pool": round(inv_ms, 2),
            "satopaa_debiased": round(sat_ms, 2),
            "heb_lop_static": round(heb_static_ms, 2),
            "heb_lop_online": round(heb_online_ms, 2),
        },
        "test_scores": scores,
        "contrasts": contrasts,
        "horizon_breakdown": horizon_scores,
    }


# ======================================================================================
# COMPETITION 3: POINT-IN-TIME MACRO CONTINUOUS DISTRIBUTIONS (ALFRED UNRATE N = 108)
# ======================================================================================

def competition_3_macro_continuous() -> dict:
    csv_bytes = _git_blob(OMEGA_COMMIT, "research/omega/continuous/unrate_realtime.csv")
    csv_sha = hashlib.sha256(csv_bytes).hexdigest()

    tmp_dir = Path(tempfile.mkdtemp())
    (tmp_dir / "evaluate.py").write_bytes(_git_blob(OMEGA_COMMIT, "research/omega/continuous/evaluate.py"))
    (tmp_dir / "unrate_realtime.csv").write_bytes(csv_bytes)
    spec = importlib.util.spec_from_file_location("omega_cont", tmp_dir / "evaluate.py")
    oc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oc)

    all_cases = oc.cohort(oc.archive())
    non_purged = [c for c in all_cases if c[-1] != "purged"]
    expansion = oc.calibration(all_cases)

    # Pre-run strictly causal online ACI sequence across chronological non-purged cases
    t0 = time.perf_counter()
    aci_dists = run_causal_aci_sequence(non_purged, mom_weight=0.50, df=3.5, aci_lr=0.06, target_miscover=0.20)
    aci_ms = 1000.0 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    holt_dists = [holt_damped_t_distribution(c[6], c[2]) for c in non_purged]
    holt_ms = 1000.0 * (time.perf_counter() - t0)

    arms_names = ("naive_rw", "empirical", "champion_rule", "three_scales", "ridge_ar", "cqr_tails",
                  "holt_damped_t", "robust_t_aci")
    test_rows_by_arm: dict[str, list[dict]] = {k: [] for k in arms_names}
    valid_rows_by_arm: dict[str, list[dict]] = {k: [] for k in arms_names}

    for idx, c in enumerate(non_purged):
        origin, last, h, target, known, y, hist, revised, group = c
        if group not in ("valid", "test"):
            continue
        thresh = round(float(hist[-1]) + 0.2, 2)
        bundle = oc.arms(hist, h, expansion[h])
        bundle["holt_damped_t"] = holt_dists[idx]
        bundle["robust_t_aci"] = aci_dists[idx]

        dest = test_rows_by_arm if group == "test" else valid_rows_by_arm
        covid = 2020 <= origin.year <= 2021
        for name in arms_names:
            m = oc.metrics(bundle[name], y, thresh)
            m.update(origin=str(origin), horizon=h, covid=covid)
            dest[name].append(m)

    valid_summary = {k: oc.summary(v) for k, v in valid_rows_by_arm.items()}
    test_summary = {k: oc.summary(v) for k, v in test_rows_by_arm.items()}

    # Era breakdown (pre/post COVID non-crisis vs 2020-2021 COVID shock!)
    test_non_covid = {k: oc.summary([r for r in v if not r["covid"]]) for k, v in test_rows_by_arm.items()}
    test_covid = {k: oc.summary([r for r in v if r["covid"]]) for k, v in test_rows_by_arm.items()}

    # Exact OMEGA protocol moving-block bootstrap (4 quarters, 2000 replicates, seed 20261124) + quarterly sign test
    origins_sorted = sorted({r["origin"] for r in test_rows_by_arm["champion_rule"]})
    n_orig = len(origins_sorted)  # 36 quarters

    def paired_origin_contrast(cand: str, ref: str) -> dict:
        by_o = defaultdict(list)
        for rc, rr in zip(test_rows_by_arm[cand], test_rows_by_arm[ref]):
            assert rc["origin"] == rr["origin"] and rc["horizon"] == rr["horizon"]
            # Positive gain = candidate has LOWER CRPS than reference
            by_o[rc["origin"]].append(rr["crps_q99"] - rc["crps_q99"])
        q_gains = np.array([np.mean(by_o[o]) for o in origins_sorted], dtype=float)

        # 4-quarter circular moving-block bootstrap (2000 reps, seed 20261124 per protocol.json)
        rng = np.random.default_rng(20261124)
        block_len = 4
        n_blocks = math.ceil(n_orig / block_len)
        starts = rng.integers(0, n_orig, size=(2000, n_blocks))
        idx_mat = ((starts[:, :, None] + np.arange(block_len)[None, None, :]) % n_orig).reshape(2000, -1)[:, :n_orig]
        boot_means = q_gains[idx_mat].mean(axis=1)

        # Monte Carlo / exact sign-flip randomization across non-overlapping 4-quarter blocks (9 blocks -> 2^9 = 512 exact!)
        macro_blocks = np.array([q_gains[i:i + 4].mean() for i in range(0, n_orig, 4)])
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(macro_blocks))))
        obs_b = float(macro_blocks.mean())
        p_block_sign = float(np.mean((signs @ macro_blocks) / len(macro_blocks) >= obs_b - 1e-12))

        return {
            "mean_crps_gain": float(q_gains.mean()),
            "median_quarter_crps_gain": float(np.median(q_gains)),
            "quarter_wins_losses": [int(np.sum(q_gains > 1e-12)), int(np.sum(q_gains < -1e-12))],
            "ci95_4q_moving_block_bootstrap": [float(x) for x in np.quantile(boot_means, [0.025, 0.975])],
            "p_4q_block_sign_exact_one_sided": p_block_sign,
        }

    contrasts = {
        "robust_t_aci_vs_champion_rule": paired_origin_contrast("robust_t_aci", "champion_rule"),
        "robust_t_aci_vs_cqr_tails": paired_origin_contrast("robust_t_aci", "cqr_tails"),
        "robust_t_aci_vs_naive_rw": paired_origin_contrast("robust_t_aci", "naive_rw"),
        "holt_damped_t_vs_champion_rule": paired_origin_contrast("holt_damped_t", "champion_rule"),
        "cqr_tails_vs_champion_rule": paired_origin_contrast("cqr_tails", "champion_rule"),
    }
    holm_p = holm_step_down({k: v["p_4q_block_sign_exact_one_sided"] for k, v in contrasts.items()})
    for k, ph in holm_p.items():
        contrasts[k]["p_holm_4q_block_exact"] = ph

    # Protocol E5 admissibility check (cover80 in [0.74, 0.86], CRPS & NLL < quantile and Gaussian baselines)
    protocol_e5_check = {}
    for name, s in test_summary.items():
        cov_ok = 0.74 <= s["cover80"] <= 0.86
        beats_bases = (
            s["crps_q99"] < min(test_summary["naive_rw"]["crps_q99"], test_summary["empirical"]["crps_q99"])
            and s["nll"] < min(test_summary["naive_rw"]["nll"], test_summary["empirical"]["nll"])
        )
        protocol_e5_check[name] = {
            "coverage_in_74_86_pct": cov_ok,
            "beats_gaussian_and_empirical_on_crps_and_nll": beats_bases,
            "statistical_e5_criteria_met": bool(cov_ok and beats_bases),
        }

    return {
        "source_commit": OMEGA_COMMIT,
        "unrate_realtime_sha256": csv_sha,
        "split_sizes": {
            "train": sum(c[-1] == "train" for c in all_cases),
            "valid": len(valid_rows_by_arm["champion_rule"]),
            "purged": sum(c[-1] == "purged" for c in all_cases),
            "test": len(test_rows_by_arm["champion_rule"]),
        },
        "runtime_ms_all_404_cases": {
            "robust_t_aci": round(aci_ms, 2),
            "holt_damped_t": round(holt_ms, 2),
        },
        "valid_summary_2007_2015": valid_summary,
        "test_summary_2016_2024": test_summary,
        "test_non_covid_summary_n84": test_non_covid,
        "test_covid_2020_2021_summary_n24": test_covid,
        "paired_contrasts_test": contrasts,
        "protocol_e5_statistical_check": protocol_e5_check,
    }


def main() -> None:
    t_start = time.perf_counter()
    c1 = competition_1_continuous_cdf()
    c2 = competition_2_binary_aggregation()
    c3 = competition_3_macro_continuous()
    elapsed = time.perf_counter() - t_start

    payload = {
        "protocol": "MEGALAB OMEGA INFINITY · CLAUDE OPUS SURGICAL MATHEMATICAL AUDIT",
        "branch": "research/claude-opus-audit",
        "api_cost_usd": 0.0,
        "network_calls": 0,
        "total_wall_time_seconds": round(elapsed, 3),
        "competition_1_continuous_cdf": c1,
        "competition_2_binary_aggregation": c2,
        "competition_3_macro_continuous": c3,
    }
    out_path = HERE / "results_competition.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} in {elapsed:.2f}s (cost=$0.00)")


if __name__ == "__main__":
    main()
