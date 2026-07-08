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
from grade_v2 import select_best_single_etf, _greedy_topk, live_weighted_periods, _full_period_matrices

app = Flask(__name__, static_folder="static")
ANALYSIS_CACHE_PATH = Path(__file__).parent / "analysis_cache.json"


def _safe_round(v, ndigits=4):
    """
    round(float(v), n) but returns None for NaN/inf instead of a float that
    still prints as the bare token `NaN`/`Infinity` -- valid to Python's
    json.dumps (allow_nan=True by default) but NOT valid JSON per spec, so
    the browser's strict response.json() throws a SyntaxError and the
    frontend hangs on "Running analysis..." forever with no visible error
    banner (only visible via the console).
    """
    v = float(v)
    return None if not np.isfinite(v) else round(v, ndigits)

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
                try:
                    _etf_returns = universe_data.load_etf_returns_db()
                except Exception:
                    # DB unavailable (e.g. local dev without fund_universe.duckdb) --
                    # fall back to the parquet snapshot. Benchmark selection may then
                    # disagree slightly with the batch grade for the reason documented
                    # on load_etf_returns_db().
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


def _trailing(months: int, fund_full: pd.Series, replica_oos: pd.Series | None,
              benchmarks_full: dict[str, pd.Series] | None = None) -> dict | None:
    """
    Fund and benchmark stats always come from the fund's FULL history, so a
    fund with only e.g. 3-5 years of life still gets 3y/5y rows -- OOS
    replication needs an initial training window before it can start, so a
    short-history fund can have real 3y/5y fund/benchmark data with no OOS
    replica available for those windows at all. Replica columns are simply
    omitted (not the whole row) when there isn't enough OOS history yet.
    """
    if len(fund_full) < months:
        return None
    f     = fund_full.iloc[-months:]
    ann_f = float(f.add(1).prod() ** (12 / months) - 1)
    std_f = float(f.std() * np.sqrt(12))
    sr_f  = ann_f / std_f if std_f > 0 else np.nan
    result = dict(
        fund_ret=round(ann_f, 4), fund_std=round(std_f, 4),
        fund_sharpe=round(sr_f, 2) if not np.isnan(sr_f) else None,
    )
    if benchmarks_full:
        for name, bm in benchmarks_full.items():
            if len(bm) >= months:
                st = _bm_stats(bm.iloc[-months:], months)
                key = name.lower()
                result[f"{key}_ret"]    = st["ret"]
                result[f"{key}_std"]    = st["std"]
                result[f"{key}_sharpe"] = st["sharpe"]

    if replica_oos is not None and len(replica_oos) >= months:
        r     = replica_oos.iloc[-months:]
        diff  = fund_full.reindex(r.index) - r
        ann_r = float(r.add(1).prod() ** (12 / months) - 1)
        std_r = float(r.std() * np.sqrt(12))
        sr_r  = ann_r / std_r if std_r > 0 else np.nan
        result.update(
            replica_ret=round(ann_r, 4), replica_std=round(std_r, 4),
            tracking_error=round(float(diff.std() * np.sqrt(12)), 4),
            replica_sharpe=round(sr_r, 2) if not np.isnan(sr_r) else None,
        )
    return result


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


_GRADE_LETTER_THRESHOLDS = [
    (4.7, "A"), (4.4, "A-"), (4.1, "B+"), (3.8, "B"),
    (3.4, "B-"), (3.1, "C+"), (2.8, "C"), (2.4, "C-"),
    (2.0, "D+"), (1.6, "D"),
]


def _letter_for_score(score: float) -> str:
    for t, g in _GRADE_LETTER_THRESHOLDS:
        if score >= t:
            return g
    return "F"


def _grade_obj(score: float | None, period: dict | None, bm_label: str) -> dict | None:
    """{score, grade (letter), desc} for one of the Overall/Recent scores, built
    from the corresponding live period row (same shape _grade_description expects)."""
    if score is None:
        return None
    score = round(score, 1)
    p = period or {}
    desc = _grade_description(p.get("fund_sharpe"), p.get("replica_sharpe"),
                              p.get("bm_adj_sharpe"), bm_label)
    return {"score": score, "grade": _letter_for_score(score), "desc": desc}


def _live_score_from_period(period: dict | None) -> float | None:
    """
    Fallback scorer for tickers not yet in the batch_summary Recent/Overall
    grade (e.g. funds outside the RBSA universe) -- same GRADE_HIGH_DIFF/
    GRADE_LOW_DIFF absolute band as the batch grading, applied to a 50/50
    fund-Sharpe-vs-(replica, bm_adj) blend. Less rigorous than grade_v2.py's
    population-calibrated bands (no per-bucket OOS weighting, no low-vol
    carve-out), but keeps the score card populated instead of blank.
    """
    if not period:
        return None
    f_sr = period.get("fund_sharpe")
    r_sr = period.get("replica_sharpe")
    b_sr = period.get("bm_adj_sharpe")
    if f_sr is None:
        return None
    if r_sr is not None and b_sr is not None:
        blend_sr = cfg.GRADE_BLEND_REP_WT * r_sr + (1.0 - cfg.GRADE_BLEND_REP_WT) * b_sr
    elif r_sr is not None:
        blend_sr = r_sr
    elif b_sr is not None:
        blend_sr = b_sr
    else:
        return None
    diff  = f_sr - blend_sr
    HIGH, LOW = cfg.GRADE_HIGH_DIFF, cfg.GRADE_LOW_DIFF
    score = 1.0 + 4.0 * (diff - LOW) / (HIGH - LOW)
    return max(1.0, min(5.0, score))


_BOND_CATEGORY_KEYWORDS = ("bond", "income", "fixed", "muni", "treasury", "credit")


def _infer_asset_class(category: str | None) -> str:
    """
    Fallback asset_class for tickers outside fund_universe.duckdb (e.g. the
    curated cfg.ACTIVE_MUTUAL_FUNDS list), which has a granular Morningstar-
    style category string (e.g. "High Yield Bond") rather than the coarse
    asset_class column select_best_single_etf's asset-class-aware weighting
    expects. Defaults to "Equity" (the safer, correlation-dominant weighting)
    when the category is missing or doesn't look bond-like.
    """
    if category and any(kw in category.lower() for kw in _BOND_CATEGORY_KEYWORDS):
        return "Fixed Income"
    return "Equity"


def _etf_name(ticker: str) -> str:
    """Plain-English name for an ETF, e.g. 'BND' -> 'Total US Bond' (strips
    the '(Provider, ER%)' suffix from the config.py description)."""
    desc = cfg.PASSIVE_ETFS.get(ticker) or cfg.ACTIVE_ETFS.get(ticker)
    if not desc:
        return ticker
    return desc.split("(")[0].strip()


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


@app.route("/admin/upload-db", methods=["POST"])
def admin_upload_db():
    """
    One-off (and repeatable) way to get fund_universe.duckdb onto Railway's
    persistent volume -- there's no direct file-upload path to a Railway
    volume otherwise. Protected by a shared-secret token so it can be left
    in place for future refreshes rather than deployed-then-removed each time.

    Usage: curl -X POST -H "X-Admin-Token: $ADMIN_UPLOAD_TOKEN" \
                -F "file=@fund_universe.duckdb" https://<host>/admin/upload-db
    """
    expected = os.environ.get("ADMIN_UPLOAD_TOKEN")
    if not expected or request.headers.get("X-Admin-Token") != expected:
        abort(403)
    f = request.files.get("file")
    if f is None:
        return jsonify(error="no file provided (expected multipart field 'file')"), 400

    dest = universe_data.DB_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".uploading")
    f.save(tmp)
    tmp.replace(dest)

    # Force the next /api/universe call to re-read from the freshly uploaded file.
    global _browse_cache
    with _browse_lock:
        _browse_cache = None

    return jsonify(ok=True, path=str(dest), size_bytes=dest.stat().st_size)


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
    from data import _with_timeout
    try:
        info = _with_timeout(lambda: yf.Ticker(ticker).info)
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
    if ticker in etf_ret.columns:
        # A handful of active ETFs (ARKK etc.) that a user can analyze directly
        # are also members of the 85-ETF replication candidate universe --
        # drop self from the candidates so align()/select_best_single_etf don't
        # choke on a column colliding with the fund series itself.
        etf_ret = etf_ret.drop(columns=[ticker])

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

    # fund_universe.duckdb's asset_class column (Equity/Fixed Income/Allocation/
    # Alternative, no nulls) is the reliable input for select_best_single_etf's
    # asset-class-aware weighting; tickers outside that DB fall back to
    # keyword-matching their (granular) category string.
    asset_class = _umeta.get("asset_class") if _umeta else _infer_asset_class(meta.get("category"))

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

    fund_ret, etf_full = align(fund_raw, etf_ret)
    _etf_ret_raw = _get_etf_returns()

    # Which tickers get selected (best single benchmark, top-3 replica) is
    # decided using grade_v2's OWN alignment (_full_period_matrices) rather
    # than data.py's align() -- the two apply different NaN/coverage
    # thresholds, so even reading the same tables they could hand
    # select_best_single_etf a slightly different date range and land on a
    # different "best" ETF than the batch grade actually used. The selected
    # tickers are then applied to the existing align()-based etf_full below,
    # so the actual fit/chart data keeps its original row alignment.
    # Single-benchmark selection uses the fund's full raw series against the
    # full ETF table so each candidate is scored on its own max overlap with
    # the fund (see select_best_single_etf docstring) -- NOT the pre-
    # intersected _full_period_matrices() output, which truncates every
    # fund's window to whichever ETF (e.g. ARKF, 2019) started trading most
    # recently among all 85 candidates.
    if not bm_override:
        _cat_bm, _ = select_best_single_etf(fund_raw.dropna(), etf_ret, asset_class)
        _cat_bm = _cat_bm or ""
    else:
        _cat_bm = ""

    # The top-3 replica IS a joint multi-ETF regression, so it still needs a
    # common window across whichever ETFs end up chosen -- that's what
    # _full_period_matrices is for here.
    _gv2 = _full_period_matrices(fund_raw.dropna(), etf_ret)
    _top3_tickers: list[str] | None = None
    if _gv2 is not None:
        _f_full_gv2, _etf_full_gv2 = _gv2
        _top3_gv2 = _greedy_topk(_f_full_gv2.values, _etf_full_gv2.values, 3)
        if _top3_gv2 is not None:
            _top3_tickers = [_etf_full_gv2.columns[i] for i in _top3_gv2[0]]

    # Restrict the displayed replica to the SAME top-3 ETFs (greedy selection,
    # full history) used for the IS/OOS replica components of the grade --
    # a full-85-ETF fit barely improves on 3 (see grade_v2.py docstring) but
    # is much harder to read as "what does this fund actually look like".
    if _top3_tickers and all(t in etf_full.columns for t in _top3_tickers):
        etf_aligned = etf_full[_top3_tickers]
    else:
        _top3 = _greedy_topk(fund_ret.values, etf_full.values, 3)
        etf_aligned = etf_full.iloc[:, _top3[0]] if _top3 is not None else etf_full

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

    # Fund vol needed for vol-adjustment below
    std_f_full = float(roll_fund.std() * np.sqrt(12))

    # Load benchmark series. etf_aligned is now the 3-ETF replica subset, so
    # SPY/QQQ/the chosen benchmark usually aren't in it -- fall back to the raw
    # (unrestricted) ETF universe, restricted to the OOS window, for all three.
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
    for _bmt in {"SPY", "QQQ", _cat_bm, bm_override.upper()} - {""}:
        _src = etf_aligned if _bmt in etf_aligned.columns else _etf_ret_raw
        if _bmt in _src.columns:
            _s = _src[_bmt].reindex(fund_ret.index).ffill()
            if _s.notna().sum() >= 12:
                benchmarks_full[_bmt] = _s
    # Alias "SPY" to whichever benchmark this fund is actually being judged
    # against, mirroring the aliasing already done for `benchmarks` above.
    # Without this, bm_adj (and the full-history chart/description) silently
    # fall back to comparing against real SPY for any fund whose selected
    # benchmark differs -- e.g. a high-yield bond fund benchmarked to HYG
    # would get its full-history performance graded against equity returns.
    if bm_override and bm_override in benchmarks_full:
        benchmarks_full["SPY"] = benchmarks_full[bm_override]
    elif _cat_bm and _cat_bm in benchmarks_full:
        benchmarks_full["SPY"] = benchmarks_full[_cat_bm]
    if "SPY" in benchmarks_full and bm_w is not None:
        benchmarks_full["bm_adj"] = (bm_w * benchmarks_full["SPY"]
                                      + (1.0 - bm_w) * tbill_full).rename("bm_adj")

    # Trailing period performance — bm_adj automatically appears as bm_adj_* keys
    periods = {}
    for label, months in [("1y", 12), ("3y", 36), ("5y", 60), ("10y", 120)]:
        periods[label] = _trailing(months, fund_ret, roll_rep, benchmarks_full=benchmarks_full)

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

    # Overall/Recent weighted-avg rows -- same non-overlapping year buckets and
    # comparators (single-ETF benchmark, IS-3ETF replica) used for the fund's
    # Overall/Recent grade, so this table is consistent with the score shown above
    # instead of the old GRADE_TIME_WEIGHTS-based "Weighted Avg" row.
    _live_bm = bm_override.upper() if bm_override else _cat_bm
    _live_periods = live_weighted_periods(fund_ret, etf_full, _live_bm or None,
                                          _get_tbill_monthly(), _etf_ret_raw)
    for _k in ("overall", "recent"):
        if _live_periods.get(_k):
            periods[_k] = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                           for kk, vv in _live_periods[_k].items()}

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

    # Fund grade — prefer the batch-precomputed Overall/Recent grade (same
    # numbers shown in the Browse Universe table); fall back to a live score
    # from the periods just computed for tickers outside the RBSA batch.
    _batch_grades = universe_data.get_grade_v2(ticker) or {}
    _overall_score = _batch_grades.get("overall_grade")
    _recent_score  = _batch_grades.get("recent_grade")
    if _overall_score is None:
        _overall_score = _live_score_from_period(periods.get("overall"))
    if _recent_score is None:
        _recent_score = _live_score_from_period(periods.get("recent"))
    overall_grade = _grade_obj(_overall_score, periods.get("overall"), bm_label)
    recent_grade  = _grade_obj(_recent_score,  periods.get("recent"),  bm_label)

    # Monthly OOS returns — used by frontend to compute rolling trailing returns
    monthly_oos = dict(
        dates      = [str(d.date()) for d in roll_fund.index],
        fund       = [_safe_round(v) for v in roll_fund.values],
        replica    = [_safe_round(v) for v in roll_rep.values],
        replica_is = [_safe_round(v) for v in replica_is_oos.values],
        benchmark  = ([_safe_round(v) for v in bm_oos_chart.values]
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
        fund        = [_safe_round(v) for v in cum_fund_full.values],
        replica_is  = [_safe_round(v) for v in cum_is_full.values],
        replica_oos = [_safe_round(v) for v in rep_oos_full.values],
        benchmark   = ([_safe_round(v) for v in cum_bmadj_full.values]
                       if cum_bmadj_full is not None else None),
        benchmark_ticker = bm_label_chart,
        oos_start   = str(oos_start_date.date()),
    )

    # Rolling weights over time — filter to ETFs with meaningful average allocation
    if roll_wgt_df is not None and not roll_wgt_df.empty:
        avg_w    = roll_wgt_df.mean()
        sig_etfs = avg_w[avg_w > 0.02].sort_values(ascending=False).index.tolist()[:12]
        wgt_series = {e: [_safe_round(v) for v in roll_wgt_df[e].values]
                      for e in sig_etfs}
        # Residual "Other" so the stacked chart always sums to 100 %
        sig_sum = roll_wgt_df[sig_etfs].values.sum(axis=1)
        wgt_series["Other"] = [_safe_round(max(0.0, v)) for v in (1.0 - sig_sum)]
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
            fund       = [_safe_round(v) for v in cum_fund.values],
            replica    = [_safe_round(v) for v in cum_rep.values],
            replica_is = [_safe_round(v) for v in cum_is_oos.values],
            benchmark  = ([_safe_round(v)
                           for v in (1 + bm_oos_chart).cumprod().values]
                          if bm_oos_chart is not None else None),
            benchmark_ticker = bm_label_chart,
        ),
        cumulative_full = cumulative_full,
        overall_grade = overall_grade,
        recent_grade  = recent_grade,
        bm_label = bm_label,
        bm_name  = _etf_name(bm_label),
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
