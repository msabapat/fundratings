# fund_replication/analysis.py
# Replication methods: constrained OLS (RBSA), LASSO selection, rolling window.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LassoCV


# ── Core constrained regression (Sharpe RBSA) ────────────────────────────────

def constrained_ols(
    fund_ret:      np.ndarray,
    etf_ret:       np.ndarray,
    min_vol_ratio: float = 0.0,
) -> np.ndarray:
    """
    Non-negative weights summing to 1, minimising sum-of-squared residuals.
    Implements Sharpe (1992) Return-Based Style Analysis.

    min_vol_ratio > 0 adds an inequality constraint:
        w^T Σ w  >=  min_vol_ratio^2 * Var(fund)
    forcing the replica to reach at least that fraction of the fund's volatility.
    Use 1.0 to match fund vol exactly; 1.1 to add a 10% concentration buffer.
    """
    n = etf_ret.shape[1]
    w0 = np.ones(n) / n

    def objective(w):
        return np.sum((fund_ret - etf_ret @ w) ** 2)

    def gradient(w):
        return -2.0 * etf_ret.T @ (fund_ret - etf_ret @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    if min_vol_ratio > 0.0:
        # Capture by value — avoids closure-over-loop-variable bugs
        _target_var = min_vol_ratio ** 2 * float(np.var(fund_ret, ddof=1))
        _cov        = np.cov(etf_ret.T) if n > 1 else np.array([[float(np.var(etf_ret))]])
        constraints.append({
            "type": "ineq",
            # >= 0  →  portfolio variance must be >= target
            "fun":  lambda w, cov=_cov, tv=_target_var: float(w @ cov @ w) - tv,
            "jac":  lambda w, cov=_cov:                 2.0 * cov @ w,
        })

    result = minimize(
        objective, w0, jac=gradient, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    w = np.clip(result.x, 0.0, 1.0)
    w /= w.sum()
    return w


def replication_metrics(
    fund_ret:   np.ndarray,
    replica_ret: np.ndarray,
) -> dict:
    """Annualised return diff, tracking error, information ratio, R²."""
    diff   = fund_ret - replica_ret
    te     = diff.std() * np.sqrt(12)
    ir     = diff.mean() / diff.std() * np.sqrt(12) if diff.std() > 0 else np.nan
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((fund_ret - fund_ret.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ann_fund    = (1 + fund_ret.mean())  ** 12 - 1
    ann_replica = (1 + replica_ret.mean()) ** 12 - 1
    return {
        "ann_fund_ret":     round(ann_fund,    4),
        "ann_replica_ret":  round(ann_replica, 4),
        "ret_diff":         round(ann_fund - ann_replica, 4),
        "tracking_error":   round(te, 4),
        "info_ratio":       round(ir, 4) if not np.isnan(ir) else np.nan,
        "r_squared":        round(r2, 4) if not np.isnan(r2) else np.nan,
    }


# ── Full in-sample regression ─────────────────────────────────────────────────

def full_is_regression(
    fund_ret:      pd.Series,
    etf_ret:       pd.DataFrame,
    min_vol_ratio: float = 0.0,
) -> dict:
    """Fit constrained OLS on all available data, return weights + metrics."""
    w = constrained_ols(fund_ret.values, etf_ret.values, min_vol_ratio=min_vol_ratio)
    replica = etf_ret.values @ w
    metrics = replication_metrics(fund_ret.values, replica)
    weights = pd.Series(w, index=etf_ret.columns).sort_values(ascending=False)
    # Keep only non-trivial weights
    weights_trimmed = weights[weights > 0.01].round(4)
    return {"weights": weights_trimmed, **metrics}


# ── IS / OOS split ────────────────────────────────────────────────────────────

def is_oos_split(
    fund_ret:      pd.Series,
    etf_ret:       pd.DataFrame,
    oos_start:     str,
    min_vol_ratio: float = 0.0,
) -> dict:
    """Train on IS, evaluate on OOS, report both periods."""
    mask_is  = fund_ret.index < oos_start
    mask_oos = fund_ret.index >= oos_start

    if mask_is.sum() < 24 or mask_oos.sum() < 12:
        return {}

    w = constrained_ols(fund_ret[mask_is].values, etf_ret[mask_is].values,
                        min_vol_ratio=min_vol_ratio)

    is_rep  = etf_ret[mask_is].values  @ w
    oos_rep = etf_ret[mask_oos].values @ w

    return {
        "weights":      pd.Series(w, index=etf_ret.columns).sort_values(ascending=False),
        "is_metrics":   replication_metrics(fund_ret[mask_is].values,  is_rep),
        "oos_metrics":  replication_metrics(fund_ret[mask_oos].values, oos_rep),
        "n_is_months":  int(mask_is.sum()),
        "n_oos_months": int(mask_oos.sum()),
    }


# ── Rolling window with quarterly rebalance ───────────────────────────────────

def rolling_replication(
    fund_ret:      pd.Series,
    etf_ret:       pd.DataFrame,
    train_months:  int   = 36,
    rebal_months:  int   = 3,
    min_vol_ratio: float = 0.0,
) -> dict:
    """
    Fit constrained OLS on trailing `train_months` months, hold for
    `rebal_months` then refit. Returns OOS monthly returns alongside fund.

    min_vol_ratio is passed to each constrained_ols call to enforce a
    variance floor: replica_var >= min_vol_ratio^2 * fund_var, estimated
    within each training window (fully out-of-sample).
    """
    n = len(fund_ret)
    if n <= train_months:
        raise ValueError(f"Not enough data: {n} months <= train window {train_months}")

    dates        = fund_ret.index.tolist()
    replica_rets = {}
    weights_log  = {}

    for i in range(train_months, n, rebal_months):
        # Train on [i-train:i]
        etf_tr  = etf_ret.iloc[i - train_months : i].values
        fund_tr = fund_ret.iloc[i - train_months : i].values
        w = constrained_ols(fund_tr, etf_tr, min_vol_ratio=min_vol_ratio)

        # Apply weights for next rebal_months (or until end)
        end = min(i + rebal_months, n)
        for j in range(i, end):
            replica_rets[dates[j]] = float(etf_ret.iloc[j].values @ w)
            weights_log[dates[j]]  = pd.Series(w, index=etf_ret.columns)

    rep_series = pd.Series(replica_rets, name="replica")
    fund_oos   = fund_ret.loc[rep_series.index]
    wgt_df     = pd.DataFrame(weights_log).T.sort_index()   # dates × ETFs

    return {
        "returns": pd.DataFrame({"fund": fund_oos, "replica": rep_series}),
        "weights": wgt_df,
    }


# ── LASSO sparse ETF selection ────────────────────────────────────────────────

def lasso_selection(
    fund_ret:    pd.Series,
    etf_ret:     pd.DataFrame,
    n_alphas:    int = 50,
    cv_folds:    int = 5,
) -> dict:
    """
    Use LASSO (positive=True, no intercept) to identify the sparse subset of
    ETFs that explain fund returns. Then run constrained OLS on that subset.
    Reports selected tickers, constrained weights, and replication metrics.
    """
    alphas = np.logspace(-6, 0, n_alphas)
    lasso  = LassoCV(
        alphas=alphas,
        cv=cv_folds,
        positive=True,
        fit_intercept=False,
        max_iter=10_000,
    )
    # Standardise ETF returns for LASSO to equalise coefficient scale
    etf_vals   = etf_ret.values
    etf_std    = etf_vals.std(axis=0)
    etf_std[etf_std == 0] = 1.0
    etf_scaled = etf_vals / etf_std

    lasso.fit(etf_scaled, fund_ret.values)

    # Recover selected tickers (non-zero LASSO coefficients)
    raw_coef  = lasso.coef_ / etf_std
    selected  = etf_ret.columns[raw_coef > 0].tolist()

    if len(selected) < 2:
        # Fallback: take top-5 by |correlation|
        corr     = etf_ret.corrwith(fund_ret).abs().sort_values(ascending=False)
        selected = corr.head(5).index.tolist()

    # Constrained OLS on selected subset
    etf_sub = etf_ret[selected]
    w_sub   = constrained_ols(fund_ret.values, etf_sub.values)
    replica = etf_sub.values @ w_sub
    metrics = replication_metrics(fund_ret.values, replica)
    weights = pd.Series(w_sub, index=selected).sort_values(ascending=False).round(4)

    return {
        "selected_etfs": selected,
        "n_selected":    len(selected),
        "best_alpha":    round(lasso.alpha_, 6),
        "weights":       weights,
        **metrics,
    }
