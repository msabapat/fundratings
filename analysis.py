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

    Uses quadprog (pure-C QP solver, no Python callbacks → GIL released for the
    entire solve, true thread parallelism). Falls back to SLSQP when quadprog is
    unavailable or when the vol constraint can't be satisfied by the QP solution.

    min_vol_ratio > 0 adds: w^T Σ w >= min_vol_ratio^2 * Var(fund)
    """
    n = etf_ret.shape[1]

    # ── quadprog fast path ────────────────────────────────────────────────────
    # quadprog solves: min 0.5 w^T G w - a^T w  s.t. C^T w >= b
    # Our problem:     min w^T (X^T X) w - 2(X^T f)^T w
    #   → G = 2 X^T X,  a = 2 X^T f
    #   Constraints: sum(w)=1 (equality, meq=1), w_i >= 0 (inequality)
    try:
        import quadprog
        G   = 2.0 * (etf_ret.T @ etf_ret) + 1e-10 * np.eye(n)
        a   = 2.0 * (etf_ret.T @ fund_ret)
        C   = np.column_stack([np.ones(n), np.eye(n)])   # (n, n+1)
        b   = np.concatenate([[1.0], np.zeros(n)])
        w   = quadprog.solve_qp(G, a, C, b, 1)[0]        # meq=1
        w   = np.clip(w, 0.0, 1.0)
        s   = w.sum()
        if s <= 0:
            raise ValueError("zero-weight solution")
        w /= s

        # Check vol constraint — quadprog can't enforce the quadratic inequality,
        # so verify it. For equity funds this is almost always satisfied.
        if min_vol_ratio > 0.0:
            _target_var = min_vol_ratio ** 2 * float(np.var(fund_ret, ddof=1))
            _cov        = np.cov(etf_ret.T) if n > 1 else np.array([[float(np.var(etf_ret))]])
            if float(w @ _cov @ w) >= _target_var * 0.999:
                return w
            # Constraint not met → fall through to SLSQP
        else:
            return w
    except Exception:
        pass   # quadprog unavailable or failed → use SLSQP

    # ── SLSQP fallback (handles quadratic vol constraint) ────────────────────
    w0 = np.ones(n) / n

    def objective(w):
        return np.sum((fund_ret - etf_ret @ w) ** 2)

    def gradient(w):
        return -2.0 * etf_ret.T @ (fund_ret - etf_ret @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    if min_vol_ratio > 0.0:
        _target_var = min_vol_ratio ** 2 * float(np.var(fund_ret, ddof=1))
        _cov        = np.cov(etf_ret.T) if n > 1 else np.array([[float(np.var(etf_ret))]])
        constraints.append({
            "type": "ineq",
            "fun":  lambda w, cov=_cov, tv=_target_var: float(w @ cov @ w) - tv,
            "jac":  lambda w, cov=_cov:                 2.0 * cov @ w,
        })

    result = minimize(
        objective, w0, jac=gradient, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-10},
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
    max_etfs:      int   = 30,
    n_jobs:        int   = 0,   # 0 = auto (use all CPU cores)
) -> dict:
    """
    Fit constrained OLS on trailing `train_months` months, hold for
    `rebal_months` then refit. Returns OOS monthly returns alongside fund.

    Pre-screens to `max_etfs` by absolute correlation before the rolling loop
    (SLSQP is much faster on 30 ETFs than 80 with near-identical results).
    Fits are parallelised across rebalance periods via ThreadPoolExecutor
    (scipy releases the GIL during SLSQP).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    if n_jobs <= 0:
        n_jobs = os.cpu_count() or 4

    n = len(fund_ret)
    if n <= train_months:
        raise ValueError(f"Not enough data: {n} months <= train window {train_months}")

    # Pre-screen: keep only the most correlated ETFs to reduce SLSQP size
    if len(etf_ret.columns) > max_etfs:
        top_cols   = etf_ret.corrwith(fund_ret).abs().nlargest(max_etfs).index.tolist()
        etf_screen = etf_ret[top_cols]
    else:
        etf_screen = etf_ret

    dates        = fund_ret.index.tolist()
    period_idxs  = list(range(train_months, n, rebal_months))

    def _fit(i):
        etf_tr  = etf_screen.iloc[i - train_months : i].values
        fund_tr = fund_ret.iloc[i - train_months : i].values
        return i, constrained_ols(fund_tr, etf_tr, min_vol_ratio=min_vol_ratio)

    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        fits = dict(ex.map(_fit, period_idxs))

    replica_rets = {}
    weights_log  = {}
    for i in period_idxs:
        w   = fits[i]
        end = min(i + rebal_months, n)
        for j in range(i, end):
            replica_rets[dates[j]] = float(etf_screen.iloc[j].values @ w)
            weights_log[dates[j]]  = pd.Series(w, index=etf_screen.columns)

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
