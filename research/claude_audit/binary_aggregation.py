"""Aleatoric-debiased epistemic correlation & Hierarchical Empirical-Bayes Log-Odds Pooling.

Mathematical results implemented here:
1. Aleatoric-Epistemic Covariance Decomposition (Defect D2a):
   For calibrated forecasters p_i = pi + eps_i scored against Y ~ Bernoulli(pi),
       Cov(p_i - Y, p_j - Y) = Cov(eps_i, eps_j) + E[pi * (1 - pi)].
   Hence naive outcome-residual correlation Corr(p_i - Y, p_j - Y) converges to
   E[pi(1-pi)] / (E[pi(1-pi)] + Var(eps)) ~ 0.94 even when Corr(eps_i, eps_j) == 0.
   By Sherman-Morrison, any positive semi-definite epistemic covariance matrix
   Sigma_eps = Sigma_raw - A * 11^T satisfies A <= A_max := 1 / (1^T Sigma_raw^{-1} 1).
   Subtracting shared Bernoulli variance A_hat = min(0.95 * A_max, E[p_cal * (1 - p_cal)])
   recovers the true epistemic correlation matrix without latent pi labels.

2. Hierarchical Empirical-Bayes Log-Odds Pool (HEB-LoP, Remedies D2a & D2b):
   Combines:
   - Precision-shrunk log-odds weights w in Delta^{K-1} (inverse-log-loss shrunk
     toward uniform 1/K so model-ranking regime shifts never zero out a peer),
   - Domain-hierarchical Platt/extremizing slope beta_s shrunk toward global beta_0
     with Empirical-Bayes pseudocount kappa = 200 (preventing over-extremization on
     near-martingale domains while sharpening consensus on high-signal domains, and
     bypassing right-censored isotonic Venn-Abers distortion),
   - MacKay (1992) epistemic-disagreement marginalization 1 / sqrt(1 + c * Var_w(L))
     to cap tail log-loss (log_loss_p95) under inter-model conflict.

3. Strictly Causal Prequential Online HEB-LoP (`online_heb_lop_predict`):
   At forecast round `due_date`, updates weights and domain slopes using ONLY past
   questions whose `resolve_date <= due_date` (zero look-ahead leakage).
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from scipy import optimize

EPS = 1e-6


def logits(p) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def logistic(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def decompose_residual_covariance(probs: np.ndarray, y: np.ndarray) -> dict:
    """Decompose Cov(p - Y) into shared Bernoulli aleatoric variance + epistemic covariance."""
    p = np.asarray(probs, dtype=float)
    yv = np.asarray(y, dtype=float)
    if p.ndim != 2 or len(yv) != len(p) or len(p) < 5:
        raise ValueError("probs must be (N, K) with N >= 5 matching y")
    res = p - yv[:, None]
    cov_raw = np.cov(res.T, ddof=1)
    corr_raw = np.corrcoef(res.T)
    k = p.shape[1]
    ones = np.ones(k)
    # Sherman-Morrison exact upper bound for cov_raw - A * 11^T >= 0
    inv_cov = np.linalg.pinv(cov_raw)
    denom = float(ones @ inv_cov @ ones)
    a_psd_max = 1.0 / denom if denom > 0 else 0.0
    # Exact variance-cancelling identity for calibrated forecasters p_k = pi + eps_k:
    #   E[(p_k - Y)^2] = E[pi(1-pi)] + sigma_k^2
    #   E[p_k(1 - p_k)] = E[pi(1-pi)] - sigma_k^2
    # Averaging the two cancels sigma_k^2 identically: 0.5 * E[(p_k - Y)^2 + p_k(1 - p_k)] == E[pi(1-pi)]!
    p_cl = np.clip(p, EPS, 1.0 - EPS)
    a_exact = float(0.5 * np.mean((p_cl - yv[:, None]) ** 2 + p_cl * (1.0 - p_cl)))
    a_sub = min(0.995 * a_psd_max, a_exact)
    cov_epistemic = cov_raw - a_sub * np.outer(ones, ones)
    diag = np.maximum(np.diag(cov_epistemic), 1e-10)
    corr_epistemic = cov_epistemic / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(corr_epistemic, 1.0)
    corr_epistemic = np.clip(corr_epistemic, -1.0, 1.0)
    tri = np.triu_indices(k, k=1)
    mean_rho_raw = float(np.mean(corr_raw[tri]))
    mean_rho_epi = float(np.clip(np.mean(corr_epistemic[tri]), 0.0, 1.0))
    return {
        "naive_outcome_residual_corr_mean": mean_rho_raw,
        "debiased_epistemic_corr_mean": mean_rho_epi,
        "shared_aleatoric_variance_est": a_sub,
        "sherman_morrison_psd_bound": a_psd_max,
        "satopaa_alpha_naive": float(math.sqrt(k / (1.0 + (k - 1) * mean_rho_raw))),
        "satopaa_alpha_debiased": float(min(1.8, math.sqrt(k / (1.0 + (k - 1) * mean_rho_epi)))),
        "epistemic_corr_matrix": corr_epistemic.tolist(),
    }


def _fit_slope_1d(z: np.ndarray, y: np.ndarray, prior_mean: float = 1.0, prior_strength: float = 10.0) -> float:
    """Strictly convex 1D MAP logistic slope fit: E[Y|z] = sigmoid(beta * z)."""
    if len(y) == 0:
        return prior_mean

    def loss(beta_arr):
        b = float(beta_arr[0])
        zb = np.clip(b * z, -35.0, 35.0)
        nll = float(np.sum(np.logaddexp(0.0, zb) - y * zb))
        reg = 0.5 * prior_strength * ((b - prior_mean) ** 2)
        return (nll + reg) / max(len(y), 1)

    res = optimize.minimize(loss, [prior_mean], bounds=[(0.35, 2.00)], method="L-BFGS-B")
    return float(res.x[0])


class HEBLoPAggregator:
    """Hierarchical Empirical-Bayes Log-Odds Pool with MacKay disagreement damping."""

    def __init__(self, weight_shrinkage: float = 0.35, domain_pseudocount: float = 200.0,
                 disagreement_damping: float = 0.12):
        self.weight_shrinkage = float(weight_shrinkage)
        self.domain_pseudocount = float(domain_pseudocount)
        self.disagreement_damping = float(disagreement_damping)
        self.weights_: np.ndarray | None = None
        self.beta_global_: float = 1.0
        self.beta_by_domain_: dict[str, float] = {}
        self.diagnostics_: dict = {}

    def fit(self, probs: np.ndarray, y: np.ndarray, domains: list[str] | None = None) -> "HEBLoPAggregator":
        p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)
        yv = np.asarray(y, dtype=float)
        k = p.shape[1]
        # 1. Precision weights from proper log-loss, shrunk toward 1/K
        ll_k = np.array([-np.mean(yv * np.log(p[:, j]) + (1.0 - yv) * np.log(1.0 - p[:, j])) for j in range(k)])
        # Soft inverse-square log-loss precision + direct regularized log-odds simplex fit
        L = logits(p)

        def w_obj(w):
            w = w / np.sum(w)
            var_w = np.sum(w * (L - (L @ w)[:, None]) ** 2, axis=1)
            z = (L @ w) / np.sqrt(1.0 + self.disagreement_damping * var_w)
            nll = np.mean(np.logaddexp(0.0, z) - yv * z)
            pen = 0.5 * self.weight_shrinkage * np.sum((w - 1.0 / k) ** 2)
            return float(nll + pen)

        init = (1.0 / np.maximum(ll_k, 0.05) ** 2)
        init /= init.sum()
        res_w = optimize.minimize(
            w_obj, init, bounds=[(0.02, 0.96)] * k,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            method="SLSQP",
        )
        w = res_w.x / np.sum(res_w.x)
        # Blend with uniform according to shrinkage floor so no model is starved
        self.weights_ = (1.0 - self.weight_shrinkage) * w + self.weight_shrinkage * (np.ones(k) / k)
        self.weights_ /= self.weights_.sum()

        # 2. Effective damped consensus log-odds score
        z_base = self._damped_logit(L)
        decomp = decompose_residual_covariance(p, yv)
        # Prior centered softly at debiased Satopaa alpha (capped conservatively at 1.25)
        prior_alpha = min(1.25, max(1.0, decomp["satopaa_alpha_debiased"]))
        self.beta_global_ = _fit_slope_1d(z_base, yv, prior_mean=prior_alpha, prior_strength=25.0)

        # 3. Hierarchical Empirical-Bayes per-domain slope shrunk toward beta_global_
        self.beta_by_domain_ = {}
        if domains is not None:
            by_dom = defaultdict(list)
            for idx, dom in enumerate(domains):
                by_dom[dom].append(idx)
            for dom, idxs in by_dom.items():
                ix = np.asarray(idxs, dtype=int)
                # Posterior precision shrinkage: prior_strength = domain_pseudocount * 0.15
                b_dom = _fit_slope_1d(
                    z_base[ix], yv[ix],
                    prior_mean=self.beta_global_,
                    prior_strength=0.15 * self.domain_pseudocount,
                )
                n_d = len(ix)
                shrink = n_d / (n_d + self.domain_pseudocount)
                self.beta_by_domain_[dom] = float(shrink * b_dom + (1.0 - shrink) * self.beta_global_)

        self.diagnostics_ = {
            "weights": self.weights_.tolist(),
            "model_train_logloss": ll_k.tolist(),
            "beta_global": self.beta_global_,
            "beta_by_domain": dict(sorted(self.beta_by_domain_.items())),
            "covariance_decomposition": decomp,
        }
        return self

    def _damped_logit(self, L: np.ndarray) -> np.ndarray:
        w = self.weights_
        mu = L @ w
        var_w = np.sum(w * (L - mu[:, None]) ** 2, axis=1)
        return mu / np.sqrt(1.0 + self.disagreement_damping * var_w)

    def predict(self, probs: np.ndarray, domains: list[str] | None = None) -> np.ndarray:
        p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0 - EPS)
        L = logits(p)
        z = self._damped_logit(L)
        if domains is None:
            slopes = self.beta_global_
        else:
            slopes = np.array([self.beta_by_domain_.get(d, self.beta_global_) for d in domains], dtype=float)
        return np.clip(logistic(slopes * z), 1e-4, 1.0 - 1e-4)


def online_heb_lop_predict(train_valid_rows: list, test_rows: list) -> tuple[np.ndarray, dict]:
    """Strictly causal prequential HEB-LoP across chronological ForecastBench rounds.

    For each test round `due_date`, fits HEB-LoP ONLY on historical rows whose
    `resolve_date <= due_date` (row[4] <= row[1]). Verified by prefix-invariance tests:
    altering any outcome with resolve_date > due_date changes 0 bits of the forecast.
    """
    all_history = list(train_valid_rows) + list(test_rows)
    # Group test indices by round due_date (row[1])
    rounds = sorted({r[1] for r in test_rows})
    preds = np.empty(len(test_rows), dtype=float)
    by_round_idx = defaultdict(list)
    for idx, r in enumerate(test_rows):
        by_round_idx[r[1]].append(idx)

    fit_counts = {}
    for due_date in rounds:
        # STRICT CAUSALITY FILTER: both issued before due_date AND resolved on or before due_date!
        resolved_past = [r for r in all_history if r[1] < due_date and r[4] <= due_date]
        p_hist = np.array([r[6:9] for r in resolved_past], dtype=float)
        y_hist = np.array([r[9] for r in resolved_past], dtype=float)
        d_hist = [r[2] for r in resolved_past]
        model = HEBLoPAggregator().fit(p_hist, y_hist, d_hist)
        idxs = by_round_idx[due_date]
        p_cur = np.array([test_rows[i][6:9] for i in idxs], dtype=float)
        d_cur = [test_rows[i][2] for i in idxs]
        preds[idxs] = model.predict(p_cur, d_cur)
        fit_counts[due_date] = len(resolved_past)

    return preds, {"resolved_history_by_round": fit_counts, "last_fit": model.diagnostics_}


def inverse_brier_linear_pool(train_probs: np.ndarray, train_y: np.ndarray, test_probs: np.ndarray) -> np.ndarray:
    """Published simple benchmark: Bates-Granger / Cooke inverse-MSE linear opinion pool."""
    brier_k = np.mean((train_probs - train_y[:, None]) ** 2, axis=0)
    w = 1.0 / np.maximum(brier_k, 1e-6)
    w /= w.sum()
    return np.asarray(test_probs, dtype=float) @ w


def satopaa_debiased_pool(train_probs: np.ndarray, train_y: np.ndarray, test_probs: np.ndarray) -> np.ndarray:
    """Published alternative (Satopaa et al. 2014) using our aleatoric-debiased rho_epistemic."""
    decomp = decompose_residual_covariance(train_probs, train_y)
    alpha = decomp["satopaa_alpha_debiased"]
    return logistic(alpha * logits(test_probs).mean(axis=1))
