"""
grade_v2.py
Second-generation fund grading, replacing the single-benchmark / fixed-period
compute_grade() in batch_rbsa.py. Design decisions (2026-07 session):

  - Benchmark selection: composite score (40% annual-return RMSE match, 35%
    volatility match, 25% monthly correlation) instead of pure R^2 maximisation.
    Pure R^2 conflates shared-factor correlation (e.g. two bond funds both
    moving with interest rates) with genuine style match, and can pick a
    Treasury-only ETF for a credit fund purely because rate sensitivity
    dominates the correlation even though the fund's real risk (and its
    divergence from Treasuries in credit-stress periods) is spread-driven.
    Among near-tied candidates (within BM_TIE_TOLERANCE of the top composite
    score), the lowest-expense-ratio ETF wins -- free cost improvement, no
    fit quality given up.

  - Non-overlapping year buckets (0-3y, 3-5y, 0-5y, 5-10y, 10y+) instead of
    nested trailing windows (1y/3y/5y/10y/full). Nested windows double-count
    shared history (a 15yr fund's 10y and full-history windows overlap by
    10/15 of their data) and silently collapse to a single window's worth of
    signal for young funds. Non-overlapping buckets avoid both.

  - Each bucket's score blends three comparators: 60% OOS 3-ETF replica (fit
    on the PRIOR window only, frozen weights tested on the bucket -- the most
    rigorous, "would this have kept working" test), 25% single-ETF benchmark,
    15% IS 3-ETF replica (fit and tested on the same window -- the most
    generous/flattering of the three, kept as a minority signal only).

  - Recent grade = 70% (0-3y) + 30% (3-5y). Overall grade = 60% (0-5y) +
    25% (5-10y) + 15% (10y+). Diffs are blended BEFORE scoring, not scores
    blended after -- averaging independently-scored buckets mechanically
    compresses the tails regardless of how well each bucket's own band is
    calibrated, which is why the original Overall grade showed almost no
    funds at the 1/5 extremes even though each bucket's own distribution
    warranted some.

  - Grade bands are calibrated per-population from the data itself: a
    symmetric-around-median band (half-width = avg of |P5-median| and
    |P95-median|) rather than a raw [P5, P95] band, since raw percentiles
    skew the resulting grade scale when one tail is much fatter than the
    other (which happens for low-volatility funds -- see next point).

  - Low-volatility funds (full-history annualised vol below LOW_VOL_THRESHOLD,
    e.g. short-duration/cash-like bond funds) get their OWN band, calibrated
    on their own population. Sharpe ratio = excess return / vol, so a tiny
    vol denominator mechanically amplifies small, economically trivial return
    differences into huge Sharpe swings -- grading them against the general
    population's band pushed a hugely disproportionate share of these funds
    to the 1/5 extremes for reasons unrelated to genuine skill or its absence.

Run standalone: `python grade_v2.py` (recomputes from scratch, no --reset flag
needed since it always does a full pass -- the population-band calibration
step requires every fund's diff before any fund can be scored).
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import config as cfg
from analysis import constrained_ols
from batch_rbsa import load_etf_returns, load_fund_returns, load_tbill

DB_PATH = Path(__file__).parent / "fund_universe.duckdb"

MIN_MONTHS_RBSA = 24

# Non-overlapping year buckets: (months-back start, months-back end); end=None -> to inception
BUCKETS: dict[str, tuple[int, int | None]] = {
    "0-3y":  (0, 36),
    "3-5y":  (36, 60),
    "0-5y":  (0, 60),
    "5-10y": (60, 120),
    "10y+":  (120, None),
}
RECENT_WEIGHTS  = {"0-3y": 0.70, "3-5y": 0.30}
OVERALL_WEIGHTS = {"0-5y": 0.60, "5-10y": 0.25, "10y+": 0.15}
COMPARATOR_BLEND = {"bm": 0.25, "is3": 0.15, "oos3": 0.60}

LOW_VOL_ANN_VOL_THRESHOLD = 0.04  # full-history annualised vol below this -> own grading band

BM_WEIGHT_RET, BM_WEIGHT_VOL, BM_WEIGHT_CORR = 0.40, 0.35, 0.25
BM_TIE_TOLERANCE = 0.01  # near-tie band (absolute composite-score gap) for the ER tie-break

# ETFs whose config.py description has no embedded "x.xx%" expense ratio --
# well-known published figures, filled in manually.
_ER_OVERRIDES = {
    "IVW": 0.18, "IVE": 0.18, "IWF": 0.19, "IWD": 0.19, "IWO": 0.24, "IWN": 0.24, "IJJ": 0.18,
    "GOVT": 0.05, "TIP": 0.19, "CWB": 0.40,
    "XLK": 0.09, "XLF": 0.09, "XLE": 0.09, "XLV": 0.09, "XLY": 0.09, "XLP": 0.09,
    "XLI": 0.09, "XLB": 0.09, "XLU": 0.09, "XLRE": 0.09,
    "SMH": 0.35, "FDN": 0.51, "KRE": 0.35, "XME": 0.35, "XOP": 0.35, "XRT": 0.35,
    "ITB": 0.39, "XHB": 0.35, "JETS": 0.60, "AMLP": 0.85,
}


def _build_er_map() -> dict[str, float]:
    er_map: dict[str, float] = {}
    for t, desc in {**cfg.PASSIVE_ETFS, **cfg.ACTIVE_ETFS}.items():
        m = re.search(r'([\d.]+)\s*%\)?', desc)
        if m:
            er_map[t] = float(m.group(1))
    er_map.update(_ER_OVERRIDES)
    return er_map


ER_MAP = _build_er_map()


# ── Fund-type mapping (for browse-by-type UI) ────────────────────────────────
# Keyed off single_etf_benchmark, since Morningstar's raw category field is
# only populated for ~11% of the universe -- this gives 100% coverage.
FUND_TYPE_MAP: dict[str, str] = {
    'SPY': 'Large Blend', 'IWB': 'Large Blend', 'VTI': 'Large Blend',
    'IVW': 'Large Growth', 'VUG': 'Large Growth', 'IWF': 'Large Growth', 'QQQ': 'Large Growth',
    'FDN': 'Large Growth', 'MTUM': 'Large Growth',
    'IVE': 'Large Value', 'VTV': 'Large Value', 'IWD': 'Large Value', 'VLUE': 'Large Value',
    'QUAL': 'Large Blend', 'USMV': 'Large Blend',
    'VYM': 'Dividend/Value', 'SDY': 'Dividend/Value', 'DLN': 'Dividend/Value',
    'MDY': 'Mid Cap', 'IJH': 'Mid Cap', 'IJJ': 'Mid Cap', 'DON': 'Mid Cap',
    'IWM': 'Small Cap', 'IWO': 'Small Cap', 'IWN': 'Small Cap', 'IJR': 'Small Cap', 'DES': 'Small Cap',
    'EFA': 'International', 'VEA': 'International', 'ACWI': 'International',
    'EEM': 'Emerging Markets', 'VWO': 'Emerging Markets', 'IEMG': 'Emerging Markets',
    'BND': 'FI Multi-Sector/Core', 'GOVT': 'FI Government', 'BNDX': 'FI International',
    'SHY': 'FI Short-Term', 'VCSH': 'FI Short-Term',
    'HYG': 'FI High Yield', 'LQD': 'FI Corporate/IG', 'MUB': 'FI Municipal', 'EMB': 'FI EM Debt',
    'TIP': 'FI TIPS', 'TLT': 'FI Long Treasury', 'CWB': 'FI Convertibles',
    'XLV': 'Sector', 'XLU': 'Sector', 'XLF': 'Sector', 'XLK': 'Sector', 'XLY': 'Sector', 'XLI': 'Sector',
    'XLE': 'Sector', 'XLB': 'Sector', 'XLP': 'Sector', 'XHB': 'Sector', 'KRE': 'Sector', 'SMH': 'Sector',
    'VNQ': 'Real Estate', 'GLD': 'Commodities/Alt', 'DBC': 'Commodities/Alt', 'AMLP': 'Commodities/Alt',
    'ARKF': 'Thematic/Active', 'ARKQ': 'Thematic/Active',
}


# ── Series helpers ────────────────────────────────────────────────────────────

def _ann_returns_by_year(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)


def _sharpe_excess(s: pd.Series, tbill: pd.Series) -> float | None:
    rf = tbill.reindex(s.index).ffill().fillna(0)
    excess = s - rf
    std = float(excess.std() * np.sqrt(12))
    return float(excess.mean() * 12 / std) if std > 0 else None


def _r2_of(f: np.ndarray, pred: np.ndarray) -> float | None:
    ss_res = float(np.sum((f - pred) ** 2))
    ss_tot = float(np.sum((f - f.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else None


def _greedy_topk(f_vals: np.ndarray, X_all: np.ndarray, max_k: int):
    """Forward stepwise selection of up to max_k ETFs (see analysis.constrained_ols
    for the per-step QP fit). Approximates best-subset at a fraction of the cost --
    empirically, K=3 greedy captures ~all the R^2 of the full 85-ETF fit."""
    n = X_all.shape[1]
    chosen: list[int] = []
    remaining = list(range(n))
    best = None
    for _ in range(max_k):
        best_r2, best_j, best_w = -np.inf, None, None
        for j in remaining:
            idx = chosen + [j]
            Xs = X_all[:, idx]
            w = constrained_ols(f_vals, Xs)
            r2 = _r2_of(f_vals, Xs @ w)
            if r2 is not None and r2 > best_r2:
                best_r2, best_j, best_w = r2, j, w
        if best_j is None:
            break
        chosen.append(best_j)
        remaining.remove(best_j)
        best = (chosen.copy(), best_w.copy())
    return best


def _clean_window(f: pd.Series, etf: pd.DataFrame):
    if len(f) == 0:
        return None
    first_f = f.first_valid_index()
    valid_cols = [c for c in etf.columns if etf[c].first_valid_index() is not None
                  and etf[c].first_valid_index() <= first_f]
    if not valid_cols:
        return None
    etf_s = etf[valid_cols].ffill(limit=1).dropna()
    f_s = f.reindex(etf_s.index).dropna()
    etf_s = etf_s.reindex(f_s.index).dropna()
    if len(f_s) < MIN_MONTHS_RBSA:
        return None
    return f_s, etf_s


def _slice_bucket(fund_sub: pd.Series, etf_sub: pd.DataFrame, start_mo: int, end_mo: int | None):
    n = len(fund_sub)
    end_idx = n - start_mo
    start_idx = 0 if end_mo is None else max(0, n - end_mo)
    if start_idx >= end_idx:
        return None
    return fund_sub.iloc[start_idx:end_idx], etf_sub.iloc[start_idx:end_idx]


def _prior_slice(fund_sub: pd.Series, etf_sub: pd.DataFrame, start_mo: int):
    """Everything before this bucket's most-recent edge -- the fit window for OOS."""
    n = len(fund_sub)
    end_idx = n - start_mo
    if end_idx <= 0:
        return None
    return fund_sub.iloc[:end_idx], etf_sub.iloc[:end_idx]


def _full_period_matrices(fund_s: pd.Series, etf_ret: pd.DataFrame):
    if len(fund_s) < MIN_MONTHS_RBSA:
        return None
    common_idx = etf_ret.index.intersection(fund_s.index)
    etf_sub = etf_ret.reindex(common_idx)
    fund_sub = fund_s.reindex(common_idx).dropna()
    etf_sub = etf_sub.reindex(fund_sub.index)
    etf_full = etf_sub.ffill(limit=1).dropna(how="all", axis=1)
    valid = [c for c in etf_full.columns if etf_full[c].notna().sum() >= MIN_MONTHS_RBSA]
    if not valid:
        return None
    etf_full = etf_full[valid].dropna()
    f_full = fund_sub.reindex(etf_full.index).dropna()
    etf_full = etf_full.reindex(f_full.index).dropna()
    if len(f_full) < MIN_MONTHS_RBSA:
        return None
    return f_full, etf_full


# ── Benchmark selection: composite score + ER tie-break ─────────────────────

def select_best_single_etf(f_full: pd.Series, etf_full: pd.DataFrame) -> tuple[str | None, float | None]:
    """
    Best single-ETF benchmark, scored on a composite of annual-return fit (40%),
    volatility match (35%), and monthly correlation (25%) rather than raw R^2 --
    see module docstring for why pure R^2 misleads for e.g. credit-risk bond
    funds. Ties within BM_TIE_TOLERANCE go to the lowest-expense-ratio candidate.
    """
    fund_vol = float(f_full.std())
    ann_fund = _ann_returns_by_year(f_full)
    scores: dict[str, float] = {}
    for t in etf_full.columns:
        etf_s = etf_full[t]
        ann_etf = _ann_returns_by_year(etf_s)
        common_years = ann_fund.index.intersection(ann_etf.index)
        if len(common_years) < 2:
            continue
        rmse = float(np.sqrt(np.mean(
            (ann_fund.reindex(common_years).values - ann_etf.reindex(common_years).values) ** 2)))
        ret_score = float(np.exp(-10.0 * rmse))

        etf_vol = float(etf_s.std())
        vol_ratio = etf_vol / fund_vol if fund_vol > 0 else 1.0
        vol_score = float(np.exp(-5.0 * (np.log(vol_ratio)) ** 2)) if vol_ratio > 0 else 0.0

        corr = float(np.corrcoef(f_full.values, etf_s.values)[0, 1])

        scores[t] = BM_WEIGHT_RET * ret_score + BM_WEIGHT_VOL * vol_score + BM_WEIGHT_CORR * corr

    if not scores:
        return None, None

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_t, best_s = ranked[0]
    near_tied = [t for t, s in ranked if best_s - s <= BM_TIE_TOLERANCE and t in ER_MAP]
    if near_tied:
        best_t = min(near_tied, key=lambda t: ER_MAP[t])

    r2 = _r2_of(f_full.values, etf_full[best_t].values)
    return best_t, r2


# ── Per-bucket BM/IS/OOS diff ─────────────────────────────────────────────────

def bucket_diff(fund_sub: pd.Series, etf_sub: pd.DataFrame, bm_ticker: str | None,
                 start_mo: int, end_mo: int | None, tbill: pd.Series) -> float | None:
    """Blended (60% OOS / 25% BM / 15% IS) Sharpe diff for one non-overlapping bucket."""
    sl = _slice_bucket(fund_sub, etf_sub, start_mo, end_mo)
    if sl is None:
        return None
    bucket_f_raw, bucket_etf_raw = sl
    comps: dict[str, float] = {}

    if bm_ticker and bm_ticker in bucket_etf_raw.columns:
        bm_series = bucket_etf_raw[bm_ticker].dropna()
        f_aligned = bucket_f_raw.reindex(bm_series.index).dropna()
        bm_aligned = bm_series.reindex(f_aligned.index).dropna()
        if len(f_aligned) >= 12:
            fs = _sharpe_excess(f_aligned, tbill)
            bs = _sharpe_excess(bm_aligned, tbill)
            if fs is not None and bs is not None:
                comps["bm"] = fs - bs

    cw = _clean_window(bucket_f_raw, bucket_etf_raw)
    if cw is not None:
        f_s, etf_s = cw
        res3 = _greedy_topk(f_s.values, etf_s.values, 3)
        if res3 is not None:
            idx, w = res3
            is3_series = pd.Series(etf_s.values[:, idx] @ w, index=f_s.index)
            fs = _sharpe_excess(f_s, tbill)
            is3_sh = _sharpe_excess(is3_series, tbill)
            if fs is not None and is3_sh is not None:
                comps["is3"] = fs - is3_sh

    ps = _prior_slice(fund_sub, etf_sub, start_mo)
    if ps is not None:
        prior_f_raw, prior_etf_raw = ps
        cw_prior = _clean_window(prior_f_raw, prior_etf_raw)
        if cw_prior is not None:
            f_p, etf_p = cw_prior
            res_prior = _greedy_topk(f_p.values, etf_p.values, 3)
            if res_prior is not None:
                idx, w = res_prior
                chosen = [etf_p.columns[j] for j in idx]
                if all(t in bucket_etf_raw.columns for t in chosen):
                    tc = bucket_etf_raw[chosen].ffill(limit=1).dropna()
                    fa = bucket_f_raw.reindex(tc.index).dropna()
                    tc = tc.reindex(fa.index).dropna()
                    if len(fa) >= 12:
                        oos_series = pd.Series(tc.values @ w, index=tc.index)
                        fs_oos = _sharpe_excess(fa, tbill)
                        oos_sh = _sharpe_excess(oos_series, tbill)
                        if fs_oos is not None and oos_sh is not None:
                            comps["oos3"] = fs_oos - oos_sh

    if not comps:
        return None
    eff = {k: w for k, w in COMPARATOR_BLEND.items() if k in comps}
    if not eff:
        return None
    tot = sum(eff.values())
    return sum(comps[k] * (w / tot) for k, w in eff.items())


def all_bucket_diffs(fund_sub: pd.Series, etf_sub: pd.DataFrame,
                      bm_ticker: str | None, tbill: pd.Series) -> dict[str, float]:
    out = {}
    for bname, (start_mo, end_mo) in BUCKETS.items():
        d = bucket_diff(fund_sub, etf_sub, bm_ticker, start_mo, end_mo, tbill)
        if d is not None:
            out[bname] = d
    return out


def blend_buckets(bucket_vals: dict[str, float], weights: dict[str, float]) -> float | None:
    eff = {k: w for k, w in weights.items() if bucket_vals.get(k) is not None}
    if not eff:
        return None
    tot = sum(eff.values())
    return sum(bucket_vals[k] * (w / tot) for k, w in eff.items())


# ── Live per-bucket raw stats (for the per-fund detail page's Overall/Recent
# weighted-avg rows -- same bucket definitions and comparators as the grade,
# but exposing ann_ret/vol/sharpe directly instead of just the blended diff) ──

def _ann_vol_sharpe(s: pd.Series, tbill: pd.Series) -> tuple[float, float, float | None]:
    n = len(s)
    ann = float((1 + s).prod() ** (12 / n) - 1)
    vol = float(s.std() * np.sqrt(12))
    return ann, vol, _sharpe_excess(s, tbill)


def bucket_raw_stats(fund_sub: pd.Series, etf_sub: pd.DataFrame, bm_ticker: str | None,
                      start_mo: int, end_mo: int | None, tbill: pd.Series,
                      etf_ret_raw: pd.DataFrame) -> dict | None:
    """
    Fund / IS-3ETF-replica / single-benchmark (vol-matched "bm_adj") / QQQ
    ann_ret, vol, sharpe for one non-overlapping bucket -- the same comparators
    used for grading, in the shape the detail page's Returns/Risk tables expect
    (benchmark aliased to "spy_*" keys, matching how the rest of the page
    already aliases whatever benchmark is selected into that key name).
    """
    sl = _slice_bucket(fund_sub, etf_sub, start_mo, end_mo)
    if sl is None:
        return None
    bucket_f, bucket_etf = sl
    if len(bucket_f) < 12:
        return None

    out: dict = {}
    out["fund_ret"], out["fund_std"], out["fund_sharpe"] = _ann_vol_sharpe(bucket_f, tbill)

    cw = _clean_window(bucket_f, bucket_etf)
    if cw is not None:
        f_s, etf_s = cw
        res3 = _greedy_topk(f_s.values, etf_s.values, 3)
        if res3 is not None:
            idx, w = res3
            rep_series = pd.Series(etf_s.values[:, idx] @ w, index=f_s.index)
            out["replica_ret"], out["replica_std"], out["replica_sharpe"] = _ann_vol_sharpe(rep_series, tbill)
            diff = f_s.values - rep_series.reindex(f_s.index).values
            out["tracking_error"] = float(np.nanstd(diff) * np.sqrt(12))

    # bm_ticker may not survive etf_sub's coverage filtering (e.g. it wasn't
    # picked as one of this fund's valid RBSA candidates) even though it has
    # perfectly good standalone history -- fall back to the raw ETF universe,
    # same as the older trailing-period code path already does.
    _bm_src = bucket_etf if (bm_ticker and bm_ticker in bucket_etf.columns) else etf_ret_raw
    if bm_ticker and bm_ticker in _bm_src.columns:
        bm_s = _bm_src[bm_ticker].reindex(bucket_f.index).ffill().dropna()
        f_aligned = bucket_f.reindex(bm_s.index).dropna()
        bm_aligned = bm_s.reindex(f_aligned.index).dropna()
        if len(f_aligned) >= 12:
            out["spy_ret"], out["spy_std"], out["spy_sharpe"] = _ann_vol_sharpe(bm_aligned, tbill)
            fund_vol = float(f_aligned.std() * np.sqrt(12))
            if out["spy_std"] > 0:
                bm_w = fund_vol / out["spy_std"]
                rf = tbill.reindex(bm_aligned.index).ffill().fillna(0)
                bm_adj_series = bm_w * bm_aligned + (1.0 - bm_w) * rf
                out["bm_adj_ret"], out["bm_adj_std"], out["bm_adj_sharpe"] = _ann_vol_sharpe(bm_adj_series, tbill)

    if "QQQ" in etf_ret_raw.columns:
        qqq_s = etf_ret_raw["QQQ"].reindex(bucket_f.index).ffill().dropna()
        f_aligned_q = bucket_f.reindex(qqq_s.index).dropna()
        if len(f_aligned_q) >= 12:
            qqq_aligned = qqq_s.reindex(f_aligned_q.index)
            out["qqq_ret"], out["qqq_std"], out["qqq_sharpe"] = _ann_vol_sharpe(qqq_aligned, tbill)

    return out


def blend_bucket_stats(bucket_stats: dict[str, dict], weights: dict[str, float]) -> dict:
    """Weighted-average every numeric field across buckets (spill redistributed
    to whichever buckets have that specific field, matching blend_buckets)."""
    all_keys = {k for stats in bucket_stats.values() if stats for k in stats}
    result: dict = {}
    for key in all_keys:
        vals = {b: stats[key] for b, stats in bucket_stats.items()
                if stats and stats.get(key) is not None}
        eff = {b: w for b, w in weights.items() if b in vals}
        if not eff:
            continue
        tot = sum(eff.values())
        result[key] = sum(vals[b] * (w / tot) for b, w in eff.items())
    return result


def live_weighted_periods(fund_sub: pd.Series, etf_sub: pd.DataFrame, bm_ticker: str | None,
                           tbill: pd.Series, etf_ret_raw: pd.DataFrame) -> dict[str, dict]:
    """Returns {"overall": {...}, "recent": {...}} in the detail page's period-row shape."""
    stats = {b: bucket_raw_stats(fund_sub, etf_sub, bm_ticker, start_mo, end_mo, tbill, etf_ret_raw)
              for b, (start_mo, end_mo) in BUCKETS.items()}
    return {
        "overall": blend_bucket_stats(stats, OVERALL_WEIGHTS),
        "recent":  blend_bucket_stats(stats, RECENT_WEIGHTS),
    }


# ── Population-calibrated scoring bands ──────────────────────────────────────

def fit_band(diffs: pd.Series) -> tuple[float, float]:
    """
    Symmetric-around-median band: half-width = average of |P5-median| and
    |P95-median|. A plain [P5, P95] band skews the resulting grade scale when
    one tail is much fatter than the other (as with low-vol funds' Sharpe
    diffs) -- the far-away extreme anchor makes the median performer look
    artificially better (or worse) than 3.0 just because of tail asymmetry.
    """
    median = float(diffs.median())
    half_width = ((median - diffs.quantile(0.05)) + (diffs.quantile(0.95) - median)) / 2
    return median - half_width, median + half_width


def score_from_diff(diff: float, low: float, high: float) -> float:
    s = 1.0 + 4.0 * (diff - low) / (high - low)
    return max(1.0, min(5.0, s))


# ── Main orchestration ────────────────────────────────────────────────────────

def run(con: duckdb.DuckDBPyConnection) -> None:
    etf_ret = load_etf_returns(con)
    fund_all = load_fund_returns(con)
    tbill = load_tbill(con)
    tickers = con.execute("""
        SELECT DISTINCT ticker FROM batch_summary WHERE grade IS NOT NULL
    """).df()["ticker"].tolist()
    print(f"grade_v2: {len(tickers)} funds, {etf_ret.shape[1]} ETFs")

    t0 = time.time()
    bm_ticker_map: dict[str, str] = {}
    bm_r2_map: dict[str, float] = {}
    fund_vol_map: dict[str, float] = {}
    recent_diff: dict[str, float] = {}
    overall_diff: dict[str, float] = {}

    for i, ticker in enumerate(tickers, 1):
        if ticker not in fund_all.columns:
            continue
        fund_s = fund_all[ticker].dropna()
        m = _full_period_matrices(fund_s, etf_ret)
        if m is None:
            continue
        f_full, etf_full = m
        fund_vol_map[ticker] = float(f_full.std() * np.sqrt(12))

        bm_t, bm_r2 = select_best_single_etf(f_full, etf_full)
        if bm_t is not None:
            bm_ticker_map[ticker] = bm_t
            bm_r2_map[ticker] = bm_r2

        common_idx = etf_ret.index.intersection(fund_s.index)
        fund_sub = fund_s.reindex(common_idx).dropna()
        etf_sub = etf_ret.reindex(common_idx).reindex(fund_sub.index)

        buckets = all_bucket_diffs(fund_sub, etf_sub, bm_t, tbill)
        rd = blend_buckets(buckets, RECENT_WEIGHTS)
        od = blend_buckets(buckets, OVERALL_WEIGHTS)
        if rd is not None:
            recent_diff[ticker] = rd
        if od is not None:
            overall_diff[ticker] = od

        if i % 100 == 0:
            elapsed = time.time() - t0
            remaining = (len(tickers) - i) / (i / elapsed) / 60
            print(f"  [{i:4d}/{len(tickers)}]  ~{remaining:.0f} min left")

    print(f"\nDiff computation done in {(time.time()-t0)/60:.1f} min")

    # ── Population-split band calibration (low-vol funds get their own band) ──
    low_vol_tickers = {t for t, v in fund_vol_map.items() if v < LOW_VOL_ANN_VOL_THRESHOLD}
    print(f"{len(low_vol_tickers)} low-vol funds (< {LOW_VOL_ANN_VOL_THRESHOLD:.0%} ann. vol) get their own band")

    def _calibrate_and_score(diff_map: dict[str, float]) -> dict[str, float]:
        s = pd.Series(diff_map)
        lv_mask = s.index.isin(low_vol_tickers)
        result: dict[str, float] = {}
        if (~lv_mask).sum() >= 30:
            lo, hi = fit_band(s[~lv_mask])
            for t, v in s[~lv_mask].items():
                result[t] = score_from_diff(v, lo, hi)
        if lv_mask.sum() >= 10:
            lo, hi = fit_band(s[lv_mask])
            for t, v in s[lv_mask].items():
                result[t] = score_from_diff(v, lo, hi)
        return result

    recent_grade = _calibrate_and_score(recent_diff)
    overall_grade = _calibrate_and_score(overall_diff)

    # ── Persist ────────────────────────────────────────────────────────────────
    con.execute("ALTER TABLE batch_summary ADD COLUMN IF NOT EXISTS single_etf_benchmark VARCHAR")
    con.execute("ALTER TABLE batch_summary ADD COLUMN IF NOT EXISTS single_etf_benchmark_r2 DOUBLE")
    con.execute("ALTER TABLE batch_summary ADD COLUMN IF NOT EXISTS single_etf_benchmark_low_r2 BOOLEAN")
    con.execute("ALTER TABLE batch_summary ADD COLUMN IF NOT EXISTS fund_type VARCHAR")
    con.execute("ALTER TABLE batch_summary ADD COLUMN IF NOT EXISTS recent_grade DOUBLE")
    con.execute("ALTER TABLE batch_summary ADD COLUMN IF NOT EXISTS overall_grade DOUBLE")
    con.execute("ALTER TABLE batch_summary ADD COLUMN IF NOT EXISTS is_low_vol_fund BOOLEAN")

    rows = []
    all_tickers = set(bm_ticker_map) | set(recent_grade) | set(overall_grade)
    for t in all_tickers:
        bm_t = bm_ticker_map.get(t)
        rows.append((
            t,
            bm_t,
            bm_r2_map.get(t),
            (bm_r2_map.get(t) is not None and bm_r2_map[t] < 0.70),
            FUND_TYPE_MAP.get(bm_t, "Other") if bm_t else None,
            recent_grade.get(t),
            overall_grade.get(t),
            t in low_vol_tickers,
        ))
    df = pd.DataFrame(rows, columns=[
        "ticker", "single_etf_benchmark", "single_etf_benchmark_r2", "single_etf_benchmark_low_r2",
        "fund_type", "recent_grade", "overall_grade", "is_low_vol_fund",
    ])
    con.register("_upd", df)
    con.execute("""
        UPDATE batch_summary SET
            single_etf_benchmark = _upd.single_etf_benchmark,
            single_etf_benchmark_r2 = _upd.single_etf_benchmark_r2,
            single_etf_benchmark_low_r2 = _upd.single_etf_benchmark_low_r2,
            fund_type = _upd.fund_type,
            recent_grade = _upd.recent_grade,
            overall_grade = _upd.overall_grade,
            is_low_vol_fund = _upd.is_low_vol_fund
        FROM _upd
        WHERE batch_summary.ticker = _upd.ticker
    """)
    con.execute("CHECKPOINT")
    print(f"\nPersisted grade_v2 results for {len(df)} funds.")

    print("\nRecent grade distribution:")
    print(pd.Series(recent_grade).round().value_counts().sort_index())
    print("\nOverall grade distribution:")
    print(pd.Series(overall_grade).round().value_counts().sort_index())


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    try:
        run(con)
    finally:
        con.execute("CHECKPOINT")
        con.close()
