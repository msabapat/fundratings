# fund_replication/app.py
# Flask backend for the Fund Replication web dashboard.
#
# Run:  py app.py
# Then open http://localhost:5050

import os, sys, warnings, threading
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, abort

import config as cfg
from data     import load_etf_returns, load_fund_returns, align
from analysis import constrained_ols, rolling_replication

app = Flask(__name__, static_folder="static")

# ── Module-level cache ────────────────────────────────────────────────────────
_etf_returns: pd.DataFrame | None = None
_etf_lock    = threading.Lock()
_fund_cache: dict  = {}
_info_cache: dict  = {}

_tbill_cache: pd.Series | None = None
_tbill_lock  = threading.Lock()


def _get_tbill_monthly() -> pd.Series:
    """
    Monthly 3-month T-bill rate as a decimal (not annualised). Cached per session.
    Source: ^IRX (13-week T-bill yield) from yfinance.
    """
    global _tbill_cache
    if _tbill_cache is not None:
        return _tbill_cache
    with _tbill_lock:
        if _tbill_cache is not None:
            return _tbill_cache
        try:
            import yfinance as yf
            hist = yf.Ticker("^IRX").history(start="2000-01-01", auto_adjust=False)
            hist.index = hist.index.tz_localize(None)
            # ^IRX is an annualised percentage; monthly decimal = value / 100 / 12
            s = (hist["Close"] / 100 / 12).resample("ME").last()
            _tbill_cache = s.rename("tbill")
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


def _fund_grade(periods: dict) -> dict:
    """
    Multi-period weighted grade on a 1–5 scale.
    Time weights: Full 30%, 10y 25%, 5y 15%, 3y 15%, 1y 15%.
    Factor weights: 60% vs replica Sharpe, 40% vs SPY Sharpe.
    Baseline 3.0 = C → fund matches both its replica AND SPY on risk-adj basis.
    """
    TIME_W = {'full': 0.30, '10y': 0.25, '5y': 0.15, '3y': 0.15, '1y': 0.15}

    sum_vs_rep = 0.0;  tot_rep_w = 0.0
    sum_vs_spy = 0.0;  tot_spy_w = 0.0

    for key, tw in TIME_W.items():
        p = periods.get(key)
        if not p:
            continue
        f_sr = p.get('fund_sharpe')
        r_sr = p.get('replica_sharpe')
        s_sr = p.get('spy_sharpe')

        if f_sr is not None and r_sr is not None:
            sum_vs_rep += tw * (f_sr - r_sr)
            tot_rep_w  += tw
        if f_sr is not None and s_sr is not None:
            sum_vs_spy += tw * (f_sr - s_sr)
            tot_spy_w  += tw

    avg_vs_rep = sum_vs_rep / tot_rep_w if tot_rep_w > 0 else 0.0
    avg_vs_spy = sum_vs_spy / tot_spy_w if tot_spy_w > 0 else 0.0

    composite = 0.6 * avg_vs_rep + 0.4 * avg_vs_spy
    score = round(max(1.0, min(5.0, 3.0 + composite * 4.0)), 1)

    thresholds = [
        (4.7, "A"),
        (4.4, "A-"),
        (4.1, "B+"),
        (3.8, "B"),
        (3.4, "B-"),
        (3.1, "C+"),
        (2.8, "C"),
        (2.4, "C-"),
        (2.0, "D+"),
        (1.6, "D"),
    ]
    grade = "F"
    for t, g in thresholds:
        if score >= t:
            grade = g
            break

    return {"score": score, "grade": grade}


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

        assets = info.get("totalAssets")
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
            ms_risk      = info.get("morningstarRiskRating"),
            ms_overall   = info.get("morningstarOverallRating"),
        )
        _info_cache[ticker] = result
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/analyze/<ticker>")
def analyze(ticker: str):
    ticker      = ticker.upper()
    bm_override = request.args.get("benchmark", "").upper().strip()
    cache_key   = f"{ticker}|{bm_override}"
    if cache_key in _fund_cache:
        return jsonify(_fund_cache[cache_key])

    etf_ret = _get_etf_returns()

    # Load fund returns
    try:
        if ticker in cfg.ACTIVE_ETFS:
            raw      = load_etf_returns([ticker], start=cfg.DEFAULT_START, end=cfg.DEFAULT_END)
            if ticker not in raw.columns:
                abort(404, f"{ticker} not found in DB")
            fund_raw = raw[ticker].dropna()
        else:
            raw      = load_fund_returns([ticker], start=cfg.DEFAULT_START, end=cfg.DEFAULT_END)
            if ticker not in raw.columns:
                abort(404, f"{ticker} not found via yfinance")
            fund_raw = raw[ticker].dropna()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

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

    # Benchmark series (SPY, QQQ) over the same OOS window as roll_fund
    benchmarks: dict[str, pd.Series] = {}
    for _bmt in ["SPY", "QQQ"]:
        if _bmt in etf_aligned.columns:
            _bm_s = etf_aligned[_bmt].reindex(roll_fund.index)
            if _bm_s.notna().all():
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

    # Select primary benchmark (SPY vs QQQ auto, or custom) and build the
    # vol-adjusted version BEFORE trailing-period computation so bm_adj stats
    # are included in every period row automatically (via _trailing's benchmarks loop).
    bm_oos         = None
    bm_oos_chart   = None
    bm_label_chart = bm_label
    bm_w           = None    # scaling factor, e.g. 0.62 → "62 % SPY + 38 % T-bill"

    if "SPY" in benchmarks:
        if bm_override:
            bm_oos = benchmarks["SPY"]
        else:
            qqq_std  = (float(benchmarks["QQQ"].std() * np.sqrt(12))
                        if "QQQ" in benchmarks else 999.0)
            if std_f_full >= qqq_std:
                bm_oos   = benchmarks["QQQ"]
                bm_label = "QQQ"
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

    # Config metadata
    if ticker in cfg.ACTIVE_MUTUAL_FUNDS:
        meta = cfg.ACTIVE_MUTUAL_FUNDS[ticker]
    else:
        desc = cfg.ACTIVE_ETFS.get(ticker, ticker)
        meta = {"name": desc.split("(")[0].strip(), "er": 0.0075, "stars": None, "category": "Active ETF"}

    # Fund grade — multi-period weighted (all periods passed, function handles NaNs)
    grade = _fund_grade(periods)

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
    return jsonify(result)


# Pre-load ETF data and T-bill rate at import time (gunicorn workers).
# T-bill is fetched in a background thread so it doesn't block worker startup.
if __name__ != "__main__":
    _get_etf_returns()
    threading.Thread(target=_get_tbill_monthly, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print("Pre-loading ETF returns...")
    _get_etf_returns()
    print(f"  Ready — {_etf_returns.shape[1]} ETFs loaded")
    print(f"Starting server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
