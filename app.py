# fund_replication/app.py
# Flask backend for the Fund Replication web dashboard.
#
# Run:  py app.py
# Then open http://localhost:5050

import json, os, sys, warnings, threading
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, abort

import config as cfg
from data     import load_etf_returns, load_fund_returns, load_tbill_monthly, align
from analysis import constrained_ols, rolling_replication

app = Flask(__name__, static_folder="static")
ANALYSIS_CACHE_PATH = Path(__file__).parent / "analysis_cache.json"

# ── Module-level cache ────────────────────────────────────────────────────────
_etf_returns: pd.DataFrame | None = None
_etf_lock    = threading.Lock()
_fund_cache: dict  = {}
_info_cache: dict  = {}

_tbill_cache: pd.Series | None = None
_tbill_lock  = threading.Lock()


def _get_tbill_monthly() -> pd.Series:
    """Monthly T-bill rate as a decimal. Reads from Parquet; falls back to yfinance."""
    global _tbill_cache
    if _tbill_cache is not None:
        return _tbill_cache
    with _tbill_lock:
        if _tbill_cache is not None:
            return _tbill_cache
        try:
            _tbill_cache = load_tbill_monthly()
        except Exception:
            _tbill_cache = pd.Series(dtype=float, name="tbill")
    return _tbill_cache


def _get_etf_returns() -> pd.DataFrame:
    global _etf_returns
    if _etf_returns is None:
        with _etf_lock:
            if _etf_returns is None:
                _etf_returns = load_etf_returns(
                    list(cfg.PASSIVE_ETFS.keys()),
                    start=cfg.DEFAULT_START,
                    end=cfg.DEFAULT_END,
                )
    return _etf_returns


def _bm_stats(s: pd.Series, months: int) -> dict:
    """Annualised return, std dev, Sharpe for a benchmark slice of `months`."""
    ann = float(s.add(1).prod() ** (12 / months) - 1)
    std = float(s.std() * np.sqrt(12))
    return dict(ret=round(ann, 4), std=round(std, 4),
                sharpe=round(ann / std, 2) if std > 0 else None)


def _trailing(months: int, fund: pd.Series, replica: pd.Series,
              benchmarks: dict[str, pd.Series] | None = None) -> dict | None:
    if len(fund) < months:
        return None
    f    = fund.iloc[-months:]
    r    = replica.iloc[-months:]
    diff = f - r
    te   = float(diff.std() * np.sqrt(12))
    ann_f = float(f.add(1).prod() ** (12 / months) - 1)
    ann_r = float(r.add(1).prod() ** (12 / months) - 1)
    std_f = float(f.std() * np.sqrt(12))
    std_r = float(r.std() * np.sqrt(12))
    sr_f  = ann_f / std_f if std_f > 0 else np.nan
    sr_r  = ann_r / std_r if std_r > 0 else np.nan
    result = dict(
        fund_ret=round(ann_f, 4),        replica_ret=round(ann_r, 4),
        fund_std=round(std_f, 4),        replica_std=round(std_r, 4),
        tracking_error=round(te, 4),
        fund_sharpe   =round(sr_f, 2) if not np.isnan(sr_f) else None,
        replica_sharpe=round(sr_r, 2) if not np.isnan(sr_r) else None,
    )
    if benchmarks:
        for name, bm in benchmarks.items():
            if len(bm) >= months:
                st = _bm_stats(bm.iloc[-months:], months)
                key = name.lower()
                result[f"{key}_ret"]    = st["ret"]
                result[f"{key}_std"]    = st["std"]
                result[f"{key}_sharpe"] = st["sharpe"]
    return result


def _infer_fi_category(w: np.ndarray, columns) -> str:
    """
    Returns a non-empty string when FI ETFs dominate (>50% weight), so equity
    funds pass through unchanged and get the vol-based SPY/QQQ benchmark.
    """
    wd    = dict(zip(columns, w))
    fi_wt = sum(v for k, v in wd.items() if k in cfg.FI_ETFS)
    return "Fixed Income" if fi_wt >= 0.50 else ""


# Representative FI passives used for data-driven benchmark selection.
# Ordered roughly from high-credit-risk to low-risk / long-duration.
_FI_BENCHMARK_CANDIDATES = [
    "HYG",   # high yield
    "CWB",   # convertibles
    "EMB",   # emerging markets bonds
    "LQD",   # investment grade corporate
    "VCSH",  # short-term corporate
    "MUB",   # munis
    "BND",   # total bond market
    "BNDX",  # international bonds
    "IEF",   # 7-10yr treasuries
    "TIP",   # TIPS
    "TLT",   # 20+yr treasuries
    "SHY",   # 1-3yr treasuries
]


def _pick_fi_benchmark(roll_fund: "pd.Series", etf_ret_raw: "pd.DataFrame") -> str:
    """
    Pick the FI passive whose annual returns most closely match the fund.

    Monthly correlations and vols are dominated by the shared interest-rate
    factor, so they discriminate poorly between credit-risk tiers. Annual
    return RMSE cuts through that: an IG-corporate fund will track LQD closely
    year-by-year even when both move with rates.

    Scoring:
      70%  annual return RMSE  — exp(-10 * rmse_annual); primary discriminator
      30%  monthly correlation — co-movement direction / secondary check
      ×    coverage ratio      — (years_overlap / fund_oos_years); penalises
                                 ETFs with short history that appear to fit well
                                 over fewer years

    Falls back to BND if no candidate reaches 5 annual observations.
    """
    roll_idx   = pd.DatetimeIndex(roll_fund.index)
    fund_years = roll_idx.year.nunique()

    def _ann(s: "pd.Series") -> "pd.Series":
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        return s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)

    ann_fund = _ann(roll_fund)

    best_ticker = "BND"
    best_score  = -np.inf

    for t in _FI_BENCHMARK_CANDIDATES:
        if t not in etf_ret_raw.columns:
            continue
        etf_s    = etf_ret_raw[t].reindex(roll_fund.index).ffill()
        overlap  = etf_s.notna() & roll_fund.notna()
        if overlap.sum() < 24:
            continue

        # Annual RMSE across shared calendar years
        ann_etf  = _ann(etf_s.dropna())
        common   = ann_fund.index.intersection(ann_etf.index)
        if len(common) < 5:
            continue
        f_ann = ann_fund[common].values
        e_ann = ann_etf[common].values
        ann_rmse  = float(np.sqrt(np.mean((f_ann - e_ann) ** 2)))
        ret_score = float(np.exp(-10.0 * ann_rmse))

        # Monthly correlation (secondary)
        f_mo = roll_fund[overlap].values
        e_mo = etf_s[overlap].values
        corr = float(np.corrcoef(f_mo, e_mo)[0, 1])

        # Penalise ETFs that only cover part of the fund's OOS history
        coverage = len(common) / max(fund_years, 1)

        score = (0.70 * ret_score + 0.30 * corr) * coverage
        if score > best_score:
            best_score  = score
            best_ticker = t

    return best_ticker


def _grade_description(f_sr, r_sr, b_sr, bm_label: str = "benchmark") -> str:
    """Narrative sentence describing fund Sharpe vs replica and vs risk-adj benchmark."""
    if f_sr is None:
        return "Insufficient data to assess fund performance."

    def _verdict(d, pos, neg, match):
        if   d >=  0.10: return pos + " by a wide margin"
        elif d >=  0.03: return pos + " modestly"
        elif d >= -0.03: return match
        elif d >= -0.10: return neg + " modestly"
        else:            return neg + " by a wide margin"

    parts = []
    if r_sr is not None:
        d   = f_sr - r_sr
        v   = _verdict(d, "outperforms replica", "trails replica", "matches replica")
        parts.append(f"{v} ({d:+.2f} Sharpe)")
    if b_sr is not None:
        d   = f_sr - b_sr
        bm  = f"risk-adj {bm_label}"
        v   = _verdict(d, f"outperforms {bm}", f"trails {bm}", f"matches {bm}")
        parts.append(f"{v} ({d:+.2f} Sharpe)")

    if not parts:
        return "Insufficient benchmark data for detailed assessment."
    return "Fund " + "; ".join(parts) + "."


def _fund_grade(periods: dict, bm_label: str = "benchmark") -> dict:
    """
    Multi-period weighted grade on a 1–5 scale.
    Uses GRADE_TIME_WEIGHTS, GRADE_BLEND_REP_WT, GRADE_HIGH_DIFF, GRADE_LOW_DIFF from config.
    Score 5.0 = fund Sharpe exceeds 50/50 blend of (replica, bm_adj) by +GRADE_HIGH_DIFF.
    Score 3.0 = even; 1.0 = trails by GRADE_LOW_DIFF. Linear interpolation in between.
    Also returns wtd_avg metrics dict for the weighted-average table row.
    """
    TW = cfg.GRADE_TIME_WEIGHTS

    _metric_keys = [
        'fund_ret', 'replica_ret', 'fund_std', 'replica_std', 'tracking_error',
        'fund_sharpe', 'replica_sharpe',
        'spy_ret', 'spy_std', 'spy_sharpe',
        'bm_adj_ret', 'bm_adj_std', 'bm_adj_sharpe',
        'qqq_ret', 'qqq_std', 'qqq_sharpe',
    ]
    wtd_sums = {k: 0.0 for k in _metric_keys}
    wtd_wts  = {k: 0.0 for k in _metric_keys}

    for key, tw in TW.items():
        p = periods.get(key)
        if not p:
            continue
        for mk in _metric_keys:
            v = p.get(mk)
            if v is not None:
                wtd_sums[mk] += tw * v
                wtd_wts[mk]  += tw

    wtd_avg = {
        mk: (round(wtd_sums[mk] / wtd_wts[mk], 4) if wtd_wts[mk] > 0 else None)
        for mk in _metric_keys
    }

    # Scoring
    f_sr     = wtd_avg.get('fund_sharpe')
    r_sr     = wtd_avg.get('replica_sharpe')
    b_sr     = wtd_avg.get('bm_adj_sharpe')
    blend_wt = cfg.GRADE_BLEND_REP_WT

    if r_sr is not None and b_sr is not None:
        blend_sr = blend_wt * r_sr + (1.0 - blend_wt) * b_sr
    elif r_sr is not None:
        blend_sr = r_sr
    elif b_sr is not None:
        blend_sr = b_sr
    else:
        blend_sr = None

    diff  = (f_sr - blend_sr) if (f_sr is not None and blend_sr is not None) else 0.0
    HIGH  = cfg.GRADE_HIGH_DIFF
    LOW   = cfg.GRADE_LOW_DIFF

    if diff >= HIGH:
        score = 5.0
    elif diff <= LOW:
        score = 1.0
    elif diff >= 0:
        score = 3.0 + 2.0 * diff / HIGH
    else:
        score = 3.0 + 2.0 * diff / abs(LOW)
    score = round(max(1.0, min(5.0, score)), 1)

    thresholds = [
        (4.7, "A"), (4.4, "A-"), (4.1, "B+"), (3.8, "B"),
        (3.4, "B-"), (3.1, "C+"), (2.8, "C"), (2.4, "C-"),
        (2.0, "D+"), (1.6, "D"),
    ]
    grade = "F"
    for t, g in thresholds:
        if score >= t:
            grade = g
            break

    return {
        "score":   score,
        "grade":   grade,
        "desc":    _grade_description(f_sr, r_sr, b_sr, bm_label),
        "wtd_avg": wtd_avg,
    }


def _replica_er(weights: dict) -> float:
    """Weighted-average ER of the replication ETF portfolio."""
    # ER map built from PASSIVE_ETFS descriptions — parse "x.xx%)" pattern
    er_map: dict[str, float] = {}
    for t, desc in cfg.PASSIVE_ETFS.items():
        import re
        m = re.search(r'([\d.]+)%\)', desc)
        if m:
            er_map[t] = float(m.group(1)) / 100
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    rep_er = sum(weights.get(t, 0) * er_map.get(t, 0.002) for t in weights) / total_w
    return round(rep_er, 4)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/funds")
def get_funds():
    funds = []
    for t, meta in cfg.ACTIVE_MUTUAL_FUNDS.items():
        funds.append(dict(
            ticker=t,
            name=meta.get("name", t),
            er=meta.get("er"),
            stars=meta.get("stars"),
            category=meta.get("category", "Unknown"),
            source="mutual_fund",
        ))
    for t, desc in cfg.ACTIVE_ETFS.items():
        funds.append(dict(
            ticker=t,
            name=desc.split("(")[0].strip(),
            er=0.0075,
            stars=None,
            category="Active ETF",
            source="active_etf",
        ))
    funds.sort(key=lambda x: x["name"])
    return jsonify(funds)


@app.route("/api/fundinfo/<ticker>")
def fund_info(ticker: str):
    """Fetch live fund metadata from yfinance .info (cached per session)."""
    ticker = ticker.upper()
    if ticker in _info_cache:
        return jsonify(_info_cache[ticker])

    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        # Convert inception timestamp to date string if present
        incep = info.get("fundInceptionDate")
        if incep:
            from datetime import date
            incep = str(date.fromtimestamp(incep))

        assets = info.get("totalAssets") or info.get("netAssets")
        assets_str = None
        if assets:
            if assets >= 1e9:
                assets_str = f"${assets/1e9:.1f}B"
            else:
                assets_str = f"${assets/1e6:.0f}M"

        result = dict(
            long_name    = info.get("longName", ticker),
            description  = info.get("longBusinessSummary", ""),
            category     = info.get("category", ""),
            fund_family  = info.get("fundFamily", ""),
            inception    = incep,
            total_assets = assets_str,
            er           = info.get("annualReportExpenseRatio") or info.get("totalExpenseRatio"),
            ytd_return   = info.get("ytdReturn"),
            three_yr     = info.get("threeYearAverageReturn"),
            five_yr      = info.get("fiveYearAverageReturn"),
            ms_risk      = info.get("morningStarRiskRating"),
            ms_overall   = info.get("morningStarOverallRating"),
        )
        _info_cache[ticker] = result
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def _run_analysis(ticker: str, bm_override: str = "") -> dict:
    """
    Core analysis logic, extracted so the pre-computation script can call it directly
    without going through Flask. Results are stored in _fund_cache.
    """
    cache_key = f"{ticker}|{bm_override}"
    if cache_key in _fund_cache:
        return _fund_cache[cache_key]

    etf_ret = _get_etf_returns()

    # Derive metadata early — needed for category-based benchmark selection below
    if ticker in cfg.ACTIVE_MUTUAL_FUNDS:
        meta = cfg.ACTIVE_MUTUAL_FUNDS[ticker]
    elif ticker in cfg.ACTIVE_ETFS:
        desc = cfg.ACTIVE_ETFS[ticker]
        meta = {"name": desc.split("(")[0].strip(), "er": 0.0075, "stars": None, "category": "Active ETF"}
    else:
        meta = {"name": ticker, "er": None, "stars": None, "category": ""}

    # Load fund returns
    if ticker in cfg.ACTIVE_ETFS:
        raw      = load_etf_returns([ticker], start=cfg.DEFAULT_START, end=cfg.DEFAULT_END)
        if ticker not in raw.columns:
            raise ValueError(f"{ticker} not found in DB")
        fund_raw = raw[ticker].dropna()
    else:
        raw      = load_fund_returns([ticker], start=cfg.DEFAULT_START, end=cfg.DEFAULT_END)
        if ticker not in raw.columns:
            raise ValueError(f"{ticker} not found via yfinance")
        fund_raw = raw[ticker].dropna()

    fund_ret, etf_aligned = align(fund_raw, etf_ret)

    # Full-IS constrained regression (weights for the pie/bar chart)
    w          = constrained_ols(fund_ret.values, etf_aligned.values,
                                 min_vol_ratio=cfg.MIN_VOL_RATIO)
    replica_is = pd.Series(etf_aligned.values @ w, index=fund_ret.index, name="replica")

    # Rolling OOS replication — ALL performance numbers derived from here
    # so table and chart are consistent (both use actual geometric compounding)
    train_months = min(cfg.TRAIN_MONTHS, len(fund_ret) // 2)
    try:
        roll_result = rolling_replication(fund_ret, etf_aligned,
                                          train_months=train_months,
                                          rebal_months=cfg.REBAL_MONTHS,
                                          min_vol_ratio=cfg.MIN_VOL_RATIO)
        roll_df     = roll_result["returns"]
        roll_wgt_df = roll_result["weights"]   # dates × ETFs
        roll_fund   = roll_df["fund"]
        roll_rep    = roll_df["replica"]
    except ValueError:
        roll_fund   = fund_ret
        roll_rep    = replica_is
        roll_wgt_df = None

    # IS replica over the same OOS window (not truly OOS — same weights for all history)
    replica_is_oos = replica_is.reindex(roll_fund.index)
    cum_is_oos     = (1 + replica_is_oos).cumprod()

    # Fund vol needed for benchmark selection and vol-adjustment below
    std_f_full = float(roll_fund.std() * np.sqrt(12))

    # For uncategorised funds, detect FI from RBSA weights; then pick the closest
    # FI passive by return correlation + vol match rather than a fixed category map.
    _etf_ret_raw = _get_etf_returns()
    _inferred_fi_bm: str = ""
    if not bm_override and not meta.get("category"):
        if _infer_fi_category(w, etf_aligned.columns):
            _inferred_fi_bm = _pick_fi_benchmark(roll_fund, _etf_ret_raw)

    # Determine the category-appropriate benchmark (used when no bm_override given).
    # Configured funds with an explicit category use CATEGORY_BM_MAP;
    # inferred FI funds use the data-driven picker above.
    if _inferred_fi_bm:
        _cat_bm = _inferred_fi_bm
    elif not bm_override:
        _cat_bm = cfg.CATEGORY_BM_MAP.get(meta.get("category", ""), "")
    else:
        _cat_bm = ""

    # Load benchmark series. SPY/QQQ come from etf_aligned (full history guaranteed).
    # Category/FI benchmarks may have later inception than the fund, so fall back to
    # raw etf_ret restricted to the OOS window.
    benchmarks: dict[str, pd.Series] = {}
    for _bmt in {"SPY", "QQQ", _cat_bm} - {""}:
        _src = etf_aligned if _bmt in etf_aligned.columns else _etf_ret_raw
        if _bmt in _src.columns:
            _bm_s = _src[_bmt].reindex(roll_fund.index)
            _bm_s = _bm_s.ffill()
            if _bm_s.notna().sum() >= 12:
                benchmarks[_bmt] = _bm_s

    # Custom benchmark override — replaces "SPY" in benchmarks dict so all
    # downstream period stats (spy_ret/spy_std/spy_sharpe) and grade logic work unchanged.
    bm_label = "SPY"
    if bm_override:
        bm_label = bm_override
        if bm_override in benchmarks:
            benchmarks["SPY"] = benchmarks[bm_override]
        elif bm_override != "SPY":
            try:
                _bm_raw = load_fund_returns([bm_override],
                                            start=cfg.DEFAULT_START,
                                            end=cfg.DEFAULT_END)
                if bm_override in _bm_raw.columns:
                    _bm_s = (_bm_raw[bm_override]
                             .dropna()
                             .reindex(roll_fund.index)
                             .ffill()
                             .fillna(0.0))
                    benchmarks["SPY"] = _bm_s
                else:
                    bm_label = "SPY"
            except Exception:
                bm_label = "SPY"

    # Select primary benchmark: explicit override > category map > vol-based auto (SPY/QQQ).
    # Category map ensures mid-cap, small-cap, and international funds are compared
    # against a style-matched benchmark rather than penalised vs SPY.
    bm_oos         = None
    bm_oos_chart   = None
    bm_label_chart = bm_label
    bm_w           = None    # scaling factor, e.g. 0.62 → "62 % SPY + 38 % T-bill"

    if "SPY" in benchmarks:
        if bm_override:
            bm_oos = benchmarks["SPY"]
        elif _cat_bm and _cat_bm in benchmarks:
            bm_oos   = benchmarks[_cat_bm]
            bm_label = _cat_bm
            benchmarks["SPY"] = benchmarks[_cat_bm]
        else:
            qqq_std  = (float(benchmarks["QQQ"].std() * np.sqrt(12))
                        if "QQQ" in benchmarks else 999.0)
            if std_f_full >= qqq_std:
                bm_oos   = benchmarks["QQQ"]
                bm_label = "QQQ"
                # Make spy_ret slot hold QQQ data so the primary-bm column is correct
                benchmarks["SPY"] = benchmarks["QQQ"]
            else:
                bm_oos = benchmarks["SPY"]

    if bm_oos is not None and std_f_full > 0:
        std_bm = float(bm_oos.std() * np.sqrt(12))
        if std_bm > 0:
            bm_w         = round(std_f_full / std_bm, 4)
            tbill_oos    = (_get_tbill_monthly()
                            .reindex(roll_fund.index)
                            .ffill()
                            .fillna(0.0))
            bm_oos_adj   = (bm_w * bm_oos + (1.0 - bm_w) * tbill_oos).rename("bm_adj")
            benchmarks["bm_adj"] = bm_oos_adj   # included in every period row
            bm_oos_chart         = bm_oos_adj
            bm_label_chart       = f"{bm_label} risk-adj"
        else:
            bm_oos_chart = bm_oos

    # Trailing period performance — bm_adj automatically appears as bm_adj_* keys
    periods = {}
    for label, months in [("1y", 12), ("3y", 36), ("5y", 60), ("10y", 120)]:
        periods[label] = _trailing(months, roll_fund, roll_rep, benchmarks=benchmarks)

    # Cumulative series — rolling OOS; compute first so table and chart share same values
    cum_fund = (1 + roll_fund).cumprod()
    cum_rep  = (1 + roll_rep).cumprod()
    n_oos    = len(roll_fund)

    # Full OOS period — annualised returns derived from cumulative product to guarantee
    # they match the chart's final values exactly (avoids NaN/length edge cases)
    ann_f_full = float(cum_fund.iloc[-1] ** (12 / n_oos) - 1)
    ann_r_full = float(cum_rep.iloc[-1]  ** (12 / n_oos) - 1)
    std_r_full = float(roll_rep.std()  * np.sqrt(12))
    diff_oos   = roll_fund - roll_rep

    periods["full"] = dict(
        fund_ret       = round(ann_f_full, 4),
        replica_ret    = round(ann_r_full, 4),
        fund_std       = round(std_f_full, 4),
        replica_std    = round(std_r_full, 4),
        tracking_error = round(float(diff_oos.std() * np.sqrt(12)), 4),
        fund_sharpe    = round(ann_f_full / std_f_full, 2) if std_f_full > 0 else None,
        replica_sharpe = round(ann_r_full / std_r_full, 2) if std_r_full > 0 else None,
        label          = f"Full OOS ({str(roll_fund.index[0].date())} to {str(roll_fund.index[-1].date())})",
    )
    # Add all benchmark stats (spy, qqq, bm_adj) to the full period
    for _bm_key, _bm_series in benchmarks.items():
        st  = _bm_stats(_bm_series, n_oos)
        key = _bm_key.lower()
        periods["full"][f"{key}_ret"]    = st["ret"]
        periods["full"][f"{key}_std"]    = st["std"]
        periods["full"][f"{key}_sharpe"] = st["sharpe"]

    # Weights dict (full IS, sorted descending)
    weights = {
        etf_aligned.columns[i]: round(float(w[i]), 4)
        for i in np.argsort(w)[::-1]
        if w[i] > 0.005
    }

    # Replica cost
    rep_er     = _replica_er(weights)
    fund_er    = cfg.ACTIVE_MUTUAL_FUNDS.get(ticker, {}).get("er") or 0.0075
    fee_saving = round(fund_er - rep_er, 4)

    # Fund grade — multi-period weighted (all periods passed, function handles NaNs)
    grade = _fund_grade(periods, bm_label)
    # Move weighted-average row into periods so the frontend table can render it
    wtd_avg = grade.pop("wtd_avg", None)
    if wtd_avg:
        periods["wtd_avg"] = wtd_avg

    # Monthly OOS returns — used by frontend to compute rolling trailing returns
    monthly_oos = dict(
        dates      = [str(d.date()) for d in roll_fund.index],
        fund       = [round(float(v), 4) for v in roll_fund.values],
        replica    = [round(float(v), 4) for v in roll_rep.values],
        replica_is = [round(float(v), 4) for v in replica_is_oos.values],
        benchmark  = ([round(float(v), 4) for v in bm_oos_chart.values]
                      if bm_oos_chart is not None else None),
        benchmark_ticker = bm_label_chart,
    )

    # Rolling weights over time — filter to ETFs with meaningful average allocation
    if roll_wgt_df is not None and not roll_wgt_df.empty:
        avg_w    = roll_wgt_df.mean()
        sig_etfs = avg_w[avg_w > 0.02].sort_values(ascending=False).index.tolist()[:12]
        wgt_series = {e: [round(float(v), 4) for v in roll_wgt_df[e].values]
                      for e in sig_etfs}
        # Residual "Other" so the stacked chart always sums to 100 %
        sig_sum = roll_wgt_df[sig_etfs].values.sum(axis=1)
        wgt_series["Other"] = [round(float(max(0.0, 1.0 - v)), 4) for v in sig_sum]
        rolling_weights = dict(
            dates  = [str(d.date()) for d in roll_wgt_df.index],
            etfs   = sig_etfs + ["Other"],
            series = wgt_series,
        )
    else:
        rolling_weights = None

    result = dict(
        ticker   = ticker,
        name     = meta.get("name", ticker),
        er       = meta.get("er"),
        stars    = meta.get("stars"),
        category = meta.get("category", "Unknown"),
        rep_er   = rep_er,
        fee_saving = fee_saving,
        passive_etfs = sorted(cfg.PASSIVE_ETFS.keys()),
        period   = dict(
            start  = str(fund_ret.index[0].date()),
            end    = str(fund_ret.index[-1].date()),
            months = int(len(fund_ret)),
            oos_start = str(roll_fund.index[0].date()),
        ),
        periods  = periods,
        weights  = weights,
        monthly_oos     = monthly_oos,
        rolling_weights = rolling_weights,
        cumulative = dict(
            dates      = [str(d.date()) for d in cum_fund.index],
            fund       = [round(float(v), 4) for v in cum_fund.values],
            replica    = [round(float(v), 4) for v in cum_rep.values],
            replica_is = [round(float(v), 4) for v in cum_is_oos.values],
            benchmark  = ([round(float(v), 4)
                           for v in (1 + bm_oos_chart).cumprod().values]
                          if bm_oos_chart is not None else None),
            benchmark_ticker = bm_label_chart,
        ),
        grade    = grade,
        bm_label = bm_label,
        bm_w     = bm_w,      # vol-scaling factor, e.g. 0.62 → "62% SPY + 38% T-bill"
    )

    _fund_cache[cache_key] = result
    return result


@app.route("/api/analyze/<ticker>")
def analyze(ticker: str):
    ticker      = ticker.upper()
    bm_override = request.args.get("benchmark", "").upper().strip()
    try:
        return jsonify(_run_analysis(ticker, bm_override))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def _load_analysis_cache() -> None:
    """Load pre-computed analysis results into _fund_cache at startup."""
    if not ANALYSIS_CACHE_PATH.exists():
        return
    try:
        with open(ANALYSIS_CACHE_PATH) as f:
            loaded = json.load(f)
        _fund_cache.update(loaded)
        print(f"Pre-computation cache: loaded {len(loaded)} fund analyses")
    except Exception as e:
        warnings.warn(f"Failed to load analysis cache: {e}")


# Pre-load ETF data and T-bill rate at import time (gunicorn workers).
# T-bill is fetched in a background thread so it doesn't block worker startup.
if __name__ != "__main__":
    _get_etf_returns()
    threading.Thread(target=_get_tbill_monthly, daemon=True).start()
    _load_analysis_cache()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print("Pre-loading ETF returns...")
    _get_etf_returns()
    print(f"  Ready — {_etf_returns.shape[1]} ETFs loaded")
    _load_analysis_cache()
    print(f"Starting server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
