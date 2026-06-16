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
import universe_data
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

_universe_meta: dict | None = None
_universe_lock = threading.Lock()
_browse_cache: list | None = None
_browse_lock   = threading.Lock()


def _get_universe_meta() -> dict:
    """ticker -> fund_universe row, for fast metadata/NAV lookups (no live API calls)."""
    global _universe_meta
    if _universe_meta is None:
        with _universe_lock:
            if _universe_meta is None:
                try:
                    _universe_meta = universe_data.load_universe_meta()
                except Exception:
                    _universe_meta = {}
    return _universe_meta


def _get_browse_funds() -> list:
    """Cached list of analysed universe funds for the Browse Universe table."""
    global _browse_cache
    if _browse_cache is None:
        with _browse_lock:
            if _browse_cache is None:
                try:
                    _browse_cache = universe_data.list_browse_funds()
                except Exception:
                    _browse_cache = []
    return _browse_cache


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

# Representative equity passives — broad style/cap/region splits, deliberately
# excluding near-duplicates of SPY (e.g. VTI, IWB) so the fit can't be ambiguous
# between two near-identical large-blend trackers.
_EQUITY_BENCHMARK_CANDIDATES = [
    "SPY",   # large blend
    "QQQ",   # large growth / tech-heavy
    "IWF",   # large growth (broader than QQQ)
    "IWD",   # large value
    "IJH",   # mid blend
    "IWM",   # small blend
    "EFA",   # developed international
    "ACWI",  # global all-country
]


def _pick_benchmark_by_fit(roll_fund: "pd.Series", etf_ret_raw: "pd.DataFrame",
                            candidates: list[str], fallback: str) -> str:
    """
    Pick whichever candidate ETF's annual returns most closely match the fund's.

    Monthly correlations and vols are dominated by shared macro factors (rates
    for bonds, market beta for equities), so they discriminate poorly between
    style/credit tiers. Annual return RMSE cuts through that: an IG-corporate
    fund will track LQD closely year-by-year even when both move with rates;
    a large-blend equity fund will track SPY closely even when both move with
    the market.

    Scoring:
      70%  annual return RMSE  — exp(-10 * rmse_annual); primary discriminator
      30%  monthly correlation — co-movement direction / secondary check
      ×    coverage ratio      — (years_overlap / fund_oos_years); penalises
                                 candidates with short history that appear to
                                 fit well over fewer years
      ×    vol cap             — min(1, MAX_BM_VOL_RATIO / vol_ratio); no penalty
                                 below the cap, proportional penalty above it so
                                 high-vol candidates can't win purely on return
                                 coincidence

    Falls back to `fallback` if no candidate reaches 5 annual observations.
    """
    roll_idx   = pd.DatetimeIndex(roll_fund.index)
    fund_years = roll_idx.year.nunique()
    fund_vol   = float(roll_fund.std())

    def _ann(s: "pd.Series") -> "pd.Series":
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        return s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)

    ann_fund = _ann(roll_fund)

    best_ticker = fallback
    best_score  = -np.inf

    for t in candidates:
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

        # Coverage: penalise ETFs that only cover part of the fund's OOS history
        coverage = len(common) / max(fund_years, 1)

        # Vol cap: soft penalty when benchmark vol exceeds fund vol by more than
        # MAX_BM_VOL_RATIO. No penalty below the cap; proportional above it.
        etf_vol   = float(etf_s[overlap].std())
        vol_ratio = etf_vol / fund_vol if fund_vol > 0 else 1.0
        vol_cap   = min(1.0, cfg.MAX_BM_VOL_RATIO / vol_ratio) if vol_ratio > 0 else 1.0

        score = (0.70 * ret_score + 0.30 * corr) * coverage * vol_cap
        if score > best_score:
            best_score  = score
            best_ticker = t

    return best_ticker


def _pick_fi_benchmark(roll_fund: "pd.Series", etf_ret_raw: "pd.DataFrame") -> str:
    return _pick_benchmark_by_fit(roll_fund, etf_ret_raw, _FI_BENCHMARK_CANDIDATES, fallback="BND")


def _pick_equity_benchmark(roll_fund: "pd.Series", etf_ret_raw: "pd.DataFrame") -> str:
    return _pick_benchmark_by_fit(roll_fund, etf_ret_raw, _EQUITY_BENCHMARK_CANDIDATES, fallback="SPY")


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

    # Redistribute weight from missing periods to 'full' so a fund with only
    # 3 years of history doesn't silently drop 60% of the scoring weight.
    eff_weights: dict[str, float] = {}
    spill = 0.0
    for key, tw in TW.items():
        if periods.get(key):
            eff_weights[key] = tw
        else:
            spill += tw
    if spill > 0:
        if "full_hist" in eff_weights:
            eff_weights["full_hist"] += spill
        elif "full" in eff_weights:
            eff_weights["full"] += spill
        else:
            # neither full-history bucket is present — spread spill across whatever is present
            present = list(eff_weights)
            if present:
                per = spill / len(present)
                for k in present:
                    eff_weights[k] += per

    wtd_sums = {k: 0.0 for k in _metric_keys}
    wtd_wts  = {k: 0.0 for k in _metric_keys}

    for key, tw in eff_weights.items():
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
    score = max(1.0, min(5.0, score))

    # "Closet index" penalty — a high full-history R² against a single static passive
    # blend means the fund hasn't changed style/allocation much over its life, so a
    # buy-and-hold mix would have replicated it just as well. Applied independently
    # of the Sharpe-diff score above, since a decent Sharpe diff can still just be
    # noise within an otherwise static style.
    r2_full = (periods.get("full_hist") or {}).get("r_squared")
    r2_penalty = 0.0
    if r2_full is not None and r2_full > cfg.GRADE_R2_PENALTY_THRESHOLD:
        r2_penalty = (cfg.GRADE_R2_PENALTY_MAX
                      * (r2_full - cfg.GRADE_R2_PENALTY_THRESHOLD)
                      / (1.0 - cfg.GRADE_R2_PENALTY_THRESHOLD))
    score = round(max(1.0, min(5.0, score - r2_penalty)), 1)

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

    desc = _grade_description(f_sr, r_sr, b_sr, bm_label)
    if r2_penalty > 0.05:
        desc += (f" Full-history returns are {r2_full*100:.0f}% explained by a static "
                 f"passive blend (R²) — a {r2_penalty:.1f}-pt deduction reflects "
                 f"limited style differentiation from a buy-and-hold alternative.")

    return {
        "score":          score,
        "grade":          grade,
        "desc":           desc,
        "wtd_avg":        wtd_avg,
        "r_squared_full": r2_full,
        "r2_penalty":     round(r2_penalty, 2),
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


@app.route("/api/universe")
def api_universe():
    """All analysed funds in fund_universe, joined with their batch_summary grade/stats."""
    return jsonify(_get_browse_funds())


@app.route("/api/fundinfo/<ticker>")
def fund_info(ticker: str):
    """Fund metadata. Prefers cached fund_universe data; falls back to live yfinance .info."""
    ticker = ticker.upper()
    if ticker in _info_cache:
        return jsonify(_info_cache[ticker])

    umeta = _get_universe_meta().get(ticker)
    if umeta:
        assets = umeta.get("aum_millions")
        assets_str = None
        if assets:
            assets_str = f"${assets/1000:.1f}B" if assets >= 1000 else f"${assets:.0f}M"
        result = dict(
            long_name    = umeta.get("long_name") or ticker,
            description  = "",
            category     = umeta.get("asset_class") or "",
            fund_family  = umeta.get("fund_family") or "",
            inception    = (str(umeta["inception_date"])[:10] if umeta.get("inception_date") else None),
            total_assets = assets_str,
            er           = umeta.get("expense_ratio"),
            ytd_return   = None,
            three_yr     = None,
            five_yr      = None,
            ms_risk      = umeta.get("ms_risk"),
            ms_overall   = umeta.get("ms_overall"),
        )
        _info_cache[ticker] = result
        return jsonify(result)

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

    _umeta = _get_universe_meta().get(ticker)

    # Derive metadata early — needed for category-based benchmark selection below
    if ticker in cfg.ACTIVE_MUTUAL_FUNDS:
        meta = cfg.ACTIVE_MUTUAL_FUNDS[ticker]
    elif ticker in cfg.ACTIVE_ETFS:
        desc = cfg.ACTIVE_ETFS[ticker]
        meta = {"name": desc.split("(")[0].strip(), "er": 0.0075, "stars": None, "category": "Active ETF"}
    elif _umeta:
        meta = {
            "name":     _umeta.get("long_name") or ticker,
            "er":       _umeta.get("expense_ratio"),
            "stars":    _umeta.get("ms_overall"),
            "category": _umeta.get("category") or "",
        }
    else:
        meta = {"name": ticker, "er": None, "stars": None, "category": ""}

    # Load fund returns
    if ticker in cfg.ACTIVE_ETFS:
        raw      = load_etf_returns([ticker], start=cfg.DEFAULT_START, end=cfg.DEFAULT_END)
        if ticker not in raw.columns:
            raise ValueError(f"{ticker} not found in DB")
        fund_raw = raw[ticker].dropna()
    elif ticker not in cfg.ACTIVE_MUTUAL_FUNDS and _umeta:
        fund_raw = universe_data.load_universe_fund_nav(ticker)
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

    # Nothing matched yet (no override, no FI signal, category unset/unmapped) —
    # pick the best-fitting equity style benchmark from real return data instead
    # of a coarse fund-vol-vs-QQQ-vol threshold. Covers the common case of
    # broad/diversified funds whose volatility happens to sit near QQQ's despite
    # holding no real tech concentration (e.g. FCTDX, a total-market blend fund).
    if not bm_override and not _cat_bm:
        _cat_bm = _pick_equity_benchmark(roll_fund, _etf_ret_raw)

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
        elif bm_override in _etf_ret_raw.columns:
            _bm_s = (_etf_ret_raw[bm_override]
                     .dropna()
                     .reindex(roll_fund.index)
                     .ffill()
                     .fillna(0.0))
            benchmarks["SPY"] = _bm_s
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

    # Select primary benchmark: explicit override > category map / FI pick /
    # data-driven equity pick (all funnelled into _cat_bm above).
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

    # Full-history benchmark set — same chosen tickers as `benchmarks`, but reindexed
    # over the ENTIRE fund history (not just the post-training OOS window), so the
    # "full_hist" period below and the full-history chart aren't blind to the training years.
    tbill_full = _get_tbill_monthly().reindex(fund_ret.index).ffill().fillna(0.0)
    benchmarks_full: dict[str, pd.Series] = {}
    for _bmt in {"SPY", "QQQ"}:
        _src = etf_aligned if _bmt in etf_aligned.columns else _etf_ret_raw
        if _bmt in benchmarks and _bmt in _src.columns:
            _s = _src[_bmt].reindex(fund_ret.index).ffill()
            if _s.notna().sum() >= 12:
                benchmarks_full[_bmt] = _s
    if "SPY" in benchmarks_full and bm_w is not None:
        benchmarks_full["bm_adj"] = (bm_w * benchmarks_full["SPY"]
                                      + (1.0 - bm_w) * tbill_full).rename("bm_adj")

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

    # Full-history period — ENTIRE fund history (including the training years the
    # rolling-OOS view above discards), replica = single static full-IS weight vector.
    # By construction this replica was fit to minimise error against this exact data,
    # so a high R² here means a static, never-rebalanced passive blend tracks the fund
    # closely across its whole life — itself a signal of low style/allocation drift.
    cum_fund_full = (1 + fund_ret).cumprod()
    cum_is_full   = (1 + replica_is).cumprod()
    n_full        = len(fund_ret)
    ann_f_full2   = float(cum_fund_full.iloc[-1] ** (12 / n_full) - 1)
    ann_is_full   = float(cum_is_full.iloc[-1]   ** (12 / n_full) - 1)
    std_f_full2   = float(fund_ret.std() * np.sqrt(12))
    std_is_full   = float(replica_is.std() * np.sqrt(12))
    diff_is_full  = fund_ret - replica_is
    ss_res = float((diff_is_full ** 2).sum())
    ss_tot = float(((fund_ret - fund_ret.mean()) ** 2).sum())
    r2_full = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None

    periods["full_hist"] = dict(
        fund_ret       = round(ann_f_full2, 4),
        replica_ret    = round(ann_is_full, 4),
        fund_std       = round(std_f_full2, 4),
        replica_std    = round(std_is_full, 4),
        tracking_error = round(float(diff_is_full.std() * np.sqrt(12)), 4),
        fund_sharpe    = round(ann_f_full2 / std_f_full2, 2) if std_f_full2 > 0 else None,
        replica_sharpe = round(ann_is_full / std_is_full, 2) if std_is_full > 0 else None,
        r_squared      = r2_full,
        label          = (f"Full History, R² to static blend: {round(r2_full*100)}%"
                          if r2_full is not None else "Full History"),
    )
    for _bm_key, _bm_series in benchmarks_full.items():
        st  = _bm_stats(_bm_series, n_full)
        key = _bm_key.lower()
        periods["full_hist"][f"{key}_ret"]    = st["ret"]
        periods["full_hist"][f"{key}_std"]    = st["std"]
        periods["full_hist"][f"{key}_sharpe"] = st["sharpe"]

    # Weights dict (full IS, sorted descending)
    weights = {
        etf_aligned.columns[i]: round(float(w[i]), 4)
        for i in np.argsort(w)[::-1]
        if w[i] > 0.005
    }

    # Replica cost
    rep_er     = _replica_er(weights)
    fund_er    = meta.get("er") or 0.0075
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

    # Full-history cumulative series — Fund/IS-Replica/Risk-Adj-Benchmark from true
    # inception; the rolling-OOS replica is overlaid starting at the OOS start date,
    # rescaled so it picks up at the fund's actual cumulative level at that point
    # instead of resetting to 1.0 (which would look like a discontinuous jump).
    oos_start_date   = roll_fund.index[0]
    base             = float(cum_fund_full.loc[oos_start_date])
    cum_rep_rescaled = base * cum_rep / float(cum_rep.iloc[0])
    rep_oos_full = pd.Series(np.nan, index=fund_ret.index)
    rep_oos_full.loc[cum_rep_rescaled.index] = cum_rep_rescaled.values

    cum_bmadj_full = ((1 + benchmarks_full["bm_adj"]).cumprod()
                       if "bm_adj" in benchmarks_full else None)

    cumulative_full = dict(
        dates       = [str(d.date()) for d in fund_ret.index],
        fund        = [round(float(v), 4) for v in cum_fund_full.values],
        replica_is  = [round(float(v), 4) for v in cum_is_full.values],
        replica_oos = [None if pd.isna(v) else round(float(v), 4) for v in rep_oos_full.values],
        benchmark   = ([round(float(v), 4) for v in cum_bmadj_full.values]
                       if cum_bmadj_full is not None else None),
        benchmark_ticker = bm_label_chart,
        oos_start   = str(oos_start_date.date()),
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
        cumulative_full = cumulative_full,
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
