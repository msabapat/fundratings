"""
batch_rbsa.py
Runs RBSA on every fund in fund_universe and stores structured results.

Tables written to fund_universe.duckdb:
  batch_periods   — one row per (ticker, period): RBSA weights + metrics
  batch_summary   — one row per ticker: weighted grade + peer percentile

Usage:
  py batch_rbsa.py              # analyse all funds not yet done (resumable)
  py batch_rbsa.py --reset      # wipe and re-run everything
  py batch_rbsa.py --stats      # summary only (no analysis)
  py batch_rbsa.py --ticker FCNTX  # re-run a single fund
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from analysis import constrained_ols

DB_PATH = Path(__file__).parent / "fund_universe.duckdb"

PERIODS = {
    "1y":   12,
    "3y":   36,
    "5y":   60,
    "10y": 120,
    "full":  0,   # 0 = all available data
}
MIN_MONTHS_RBSA = 24   # minimum months needed to run RBSA
MIN_MONTHS_FULL = 24   # minimum months for 'full' period

# ── Database ──────────────────────────────────────────────────────────────────

def get_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH))


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS batch_periods (
            ticker          VARCHAR NOT NULL,
            period          VARCHAR NOT NULL,
            n_months        INTEGER,
            benchmark       VARCHAR,
            weights_json    VARCHAR,
            -- Fund metrics
            fund_ann_ret    DOUBLE,
            fund_vol        DOUBLE,
            fund_sharpe     DOUBLE,
            -- Benchmark metrics
            bm_ann_ret      DOUBLE,
            bm_vol          DOUBLE,
            bm_sharpe       DOUBLE,
            -- Active metrics
            alpha           DOUBLE,
            tracking_error  DOUBLE,
            info_ratio      DOUBLE,
            r_squared       DOUBLE,
            -- Capture ratios
            up_capture      DOUBLE,
            down_capture    DOUBLE,
            -- Beat rate (% months fund > benchmark)
            beat_rate       DOUBLE,
            PRIMARY KEY (ticker, period)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS batch_summary (
            ticker              VARCHAR PRIMARY KEY,
            benchmark           VARCHAR,
            weights_json        VARCHAR,
            grade               DOUBLE,
            peer_n              INTEGER,
            peer_percentile     DOUBLE,
            wtd_alpha           DOUBLE,
            wtd_info_ratio      DOUBLE,
            wtd_sharpe_diff     DOUBLE,
            wtd_beat_rate       DOUBLE,
            computed_at         TIMESTAMP DEFAULT now()
        )
    """)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_etf_returns(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT date, ticker, return_monthly FROM etf_returns ORDER BY date
    """).df()
    wide = df.pivot(index="date", columns="ticker", values="return_monthly")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def load_fund_returns(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT date, ticker, return_monthly FROM fund_nav ORDER BY date
    """).df()
    wide = df.pivot(index="date", columns="ticker", values="return_monthly")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def load_tbill(con: duckdb.DuckDBPyConnection) -> pd.Series:
    df = con.execute("SELECT date, rate_monthly FROM tbill_returns ORDER BY date").df()
    s = df.set_index("date")["rate_monthly"]
    s.index = pd.to_datetime(s.index)
    return s


# ── Benchmark selection ───────────────────────────────────────────────────────

_FI_CANDIDATES     = ["HYG","CWB","EMB","LQD","VCSH","MUB","BND","BNDX","IEF","TIP","TLT","SHY"]
# Broad style/cap/region splits — deliberately excludes near-duplicates of SPY
# (VTI, IWB) so the fit can't be ambiguous between two near-identical trackers.
_EQUITY_CANDIDATES = ["SPY","QQQ","IWF","IWD","IJH","IWM","EFA","ACWI"]


def _pick_best_fit(fund_s: pd.Series, etf_ret: pd.DataFrame,
                    candidates: list[str], fallback: str) -> str:
    """
    Pick whichever candidate ETF's annual returns most closely match the fund's.
    Same scoring as app.py::_pick_benchmark_by_fit — 70% annual-return RMSE +
    30% monthly correlation, scaled by history-coverage ratio and a vol cap that
    stops high-vol candidates winning purely on return coincidence.
    """
    fund_vol   = float(fund_s.std())
    fund_years = pd.DatetimeIndex(fund_s.index).year.nunique()

    def _ann(s: pd.Series) -> pd.Series:
        s = s.copy(); s.index = pd.to_datetime(s.index)
        return s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)

    ann_fund  = _ann(fund_s)
    best, best_score = fallback, -np.inf

    for t in candidates:
        if t not in etf_ret.columns:
            continue
        etf_s   = etf_ret[t].reindex(fund_s.index).ffill()
        overlap = etf_s.notna() & fund_s.notna()
        if overlap.sum() < 24:
            continue
        ann_etf = _ann(etf_s.dropna())
        common  = ann_fund.index.intersection(ann_etf.index)
        if len(common) < 5:
            continue
        ann_rmse  = float(np.sqrt(np.mean((ann_fund[common].values - ann_etf[common].values) ** 2)))
        ret_score = float(np.exp(-10.0 * ann_rmse))
        f_mo, e_mo = fund_s[overlap].values, etf_s[overlap].values
        corr      = float(np.corrcoef(f_mo, e_mo)[0, 1])
        coverage  = len(common) / max(fund_years, 1)
        etf_vol   = float(etf_s[overlap].std())
        vol_ratio = etf_vol / fund_vol if fund_vol > 0 else 1.0
        vol_cap   = min(1.0, cfg.MAX_BM_VOL_RATIO / vol_ratio) if vol_ratio > 0 else 1.0
        score = (0.70 * ret_score + 0.30 * corr) * coverage * vol_cap
        if score > best_score:
            best_score, best = score, t

    return best


def select_benchmark(
    ticker: str,
    category: str | None,
    weights: np.ndarray,
    etf_cols: list[str],
    fund_s: pd.Series,
    etf_ret: pd.DataFrame,
) -> str:
    # 1. Category → CATEGORY_BM_MAP
    if category and category in cfg.CATEGORY_BM_MAP:
        bm = cfg.CATEGORY_BM_MAP[category]
        if bm in etf_ret.columns:
            return bm

    # 2. FI detection via RBSA weights → data-driven FI benchmark
    wd = dict(zip(etf_cols, weights))
    fi_wt = sum(v for k, v in wd.items() if k in cfg.FI_ETFS)
    if fi_wt >= 0.50:
        return _pick_best_fit(fund_s, etf_ret, _FI_CANDIDATES, fallback="BND")

    # 3. Equity: best-fitting style benchmark by annual-return RMSE + correlation,
    # not just a SPY-vs-QQQ volatility-proximity guess (that misclassified broad/
    # diversified funds as QQQ purely because their vol happened to land near it).
    return _pick_best_fit(fund_s, etf_ret, _EQUITY_CANDIDATES, fallback="SPY")


# ── Per-period metrics ────────────────────────────────────────────────────────

def _annualise(s: pd.Series) -> tuple[float, float, float | None]:
    """Returns (ann_ret, ann_vol, sharpe_vs_zero)."""
    n   = len(s)
    ann = float((1 + s).prod() ** (12 / n) - 1)
    vol = float(s.std() * np.sqrt(12))
    sr  = ann / vol if vol > 0 else None
    return ann, vol, sr


def _sharpe_excess(s: pd.Series, tbill: pd.Series) -> float | None:
    rf = tbill.reindex(s.index).ffill().fillna(0)
    excess = s - rf
    std = float(excess.std() * np.sqrt(12))
    if std <= 0:
        return None
    return float(excess.mean() * 12 / std)


def _capture(fund: pd.Series, bm: pd.Series) -> tuple[float | None, float | None]:
    up   = bm > 0
    down = bm < 0
    uc = float(fund[up].mean()  / bm[up].mean()  * 100) if up.sum()   > 3 else None
    dc = float(fund[down].mean()/ bm[down].mean() * 100) if down.sum() > 3 else None
    return uc, dc


def compute_period_metrics(
    fund_s:  pd.Series,
    etf_s:   pd.DataFrame,
    bm_name: str,
    tbill:   pd.Series,
    n_months: int,          # 0 = all
) -> dict | None:
    # Slice to period
    if n_months > 0:
        if len(fund_s) < n_months:
            return None
        f   = fund_s.iloc[-n_months:]
        etf = etf_s.iloc[-n_months:]
    else:
        f   = fund_s
        etf = etf_s

    # Drop ETFs missing at start of this period
    first_f = f.first_valid_index()
    valid_cols = [c for c in etf.columns
                  if etf[c].first_valid_index() is not None
                  and etf[c].first_valid_index() <= first_f]
    if not valid_cols or len(f) < MIN_MONTHS_RBSA:
        return None

    etf_sub = etf[valid_cols].ffill(limit=1).dropna()
    f_sub   = f.reindex(etf_sub.index).dropna()
    etf_sub = etf_sub.reindex(f_sub.index).dropna()

    if len(f_sub) < MIN_MONTHS_RBSA:
        return None

    # RBSA
    w = constrained_ols(f_sub.values, etf_sub.values)
    replica = pd.Series(etf_sub.values @ w, index=f_sub.index)
    weights_dict = {c: round(float(v), 4) for c, v in zip(etf_sub.columns, w) if v > 0.01}

    # Fund metrics
    fund_ann, fund_vol, _ = _annualise(f_sub)
    fund_sharpe = _sharpe_excess(f_sub, tbill)

    # R² of fund vs its own static replica blend — high values mean a single
    # never-rebalanced passive mix tracks the fund closely (see compute_grade's
    # closet-index penalty), distinct from r_squared below (fund vs single benchmark).
    rep_ss_res = np.nansum((f_sub.values - replica.reindex(f_sub.index).values) ** 2)
    rep_ss_tot = np.nansum((f_sub.values - f_sub.mean()) ** 2)
    r2_replica = 1 - rep_ss_res / rep_ss_tot if rep_ss_tot > 0 else None

    # Benchmark metrics
    result = {
        "n_months":         len(f_sub),
        "weights_json":     json.dumps(weights_dict),
        "fund_ann_ret":      round(fund_ann, 4),
        "fund_vol":          round(fund_vol, 4),
        "fund_sharpe":       round(fund_sharpe, 3) if fund_sharpe else None,
        "r_squared_replica": round(r2_replica, 4) if r2_replica is not None else None,
    }

    if bm_name in etf_s.columns:
        bm_s = etf_s[bm_name].reindex(f_sub.index).ffill()
        if bm_s.notna().sum() >= 12:
            bm_ann, bm_vol, _ = _annualise(bm_s.dropna())
            bm_sharpe = _sharpe_excess(bm_s, tbill)
            alpha = fund_ann - bm_ann
            diff  = f_sub.values - bm_s.reindex(f_sub.index).values
            te    = float(np.nanstd(diff) * np.sqrt(12))
            ir    = alpha / te if te > 0 else None
            ss_res = np.nansum((f_sub.values - bm_s.reindex(f_sub.index).values) ** 2)
            ss_tot = np.nansum((f_sub.values - f_sub.mean()) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else None
            uc, dc = _capture(f_sub, bm_s.reindex(f_sub.index).dropna())
            beat   = float((f_sub.values > bm_s.reindex(f_sub.index).values).mean())

            result.update({
                "bm_ann_ret":    round(bm_ann, 4),
                "bm_vol":        round(bm_vol, 4),
                "bm_sharpe":     round(bm_sharpe, 3) if bm_sharpe else None,
                "alpha":         round(alpha, 4),
                "tracking_error":round(te, 4),
                "info_ratio":    round(ir, 3)  if ir else None,
                "r_squared":     round(r2, 4)  if r2 else None,
                "up_capture":    round(uc, 1)  if uc else None,
                "down_capture":  round(dc, 1)  if dc else None,
                "beat_rate":     round(beat, 3),
            })

    return result


# ── Grading ───────────────────────────────────────────────────────────────────

def compute_grade(period_rows: list[dict]) -> dict:
    """
    Weighted grade 1–5 across periods, based on Sharpe diff vs benchmark.
    Also returns weighted alpha, info_ratio, beat_rate for summary table.
    """
    TW = cfg.GRADE_TIME_WEIGHTS
    period_map = {r["period"]: r for r in period_rows}

    eff: dict[str, float] = {}
    spill = 0.0
    for key, tw in TW.items():
        if period_map.get(key) and period_map[key].get("bm_sharpe") is not None:
            eff[key] = tw
        else:
            spill += tw
    if spill > 0 and "full" in eff:
        eff["full"] += spill
    elif spill > 0 and eff:
        per = spill / len(eff)
        for k in eff:
            eff[k] += per

    if not eff:
        return {"grade": None, "wtd_sharpe_diff": None,
                "wtd_alpha": None, "wtd_info_ratio": None, "wtd_beat_rate": None}

    tot_w = sum(eff.values())
    grade_sum = sdiff_sum = alpha_sum = ir_sum = beat_sum = 0.0
    sdiff_w = alpha_w = ir_w = beat_w = 0.0

    HIGH = cfg.GRADE_HIGH_DIFF
    LOW  = cfg.GRADE_LOW_DIFF

    for key, tw in eff.items():
        p  = period_map[key]
        w  = tw / tot_w
        fs = p.get("fund_sharpe")
        bs = p.get("bm_sharpe")
        if fs is None or bs is None:
            continue
        diff  = fs - bs
        score = 1.0 + 4.0 * (diff - LOW) / (HIGH - LOW)
        score = max(1.0, min(5.0, score))
        grade_sum  += score * w
        sdiff_sum  += diff  * w; sdiff_w += w

        al = p.get("alpha")
        ir = p.get("info_ratio")
        br = p.get("beat_rate")
        if al is not None:
            alpha_sum += al * w; alpha_w += w
        if ir is not None:
            ir_sum += ir * w; ir_w += w
        if br is not None:
            beat_sum += br * w; beat_w += w

    grade = round(grade_sum, 2) if grade_sum else None

    # "Closet index" penalty — same formula as the detail page (app.py::_fund_grade).
    # batch's "full" period is already the entire fund history (no OOS/IS split here),
    # so r_squared_replica from that period is the right full-history signal.
    if grade is not None:
        r2_replica = (period_map.get("full") or {}).get("r_squared_replica")
        if r2_replica is not None and r2_replica > cfg.GRADE_R2_PENALTY_THRESHOLD:
            penalty = (cfg.GRADE_R2_PENALTY_MAX
                       * (r2_replica - cfg.GRADE_R2_PENALTY_THRESHOLD)
                       / (1.0 - cfg.GRADE_R2_PENALTY_THRESHOLD))
            grade = round(max(1.0, min(5.0, grade - penalty)), 2)

    return {
        "grade":           grade,
        "wtd_sharpe_diff": round(sdiff_sum / sdiff_w, 3)   if sdiff_w  else None,
        "wtd_alpha":       round(alpha_sum / alpha_w, 4)   if alpha_w  else None,
        "wtd_info_ratio":  round(ir_sum    / ir_w,    3)   if ir_w     else None,
        "wtd_beat_rate":   round(beat_sum  / beat_w,  3)   if beat_w   else None,
    }


# ── Peer percentiles ──────────────────────────────────────────────────────────

def compute_peer_percentiles(con: duckdb.DuckDBPyConnection) -> None:
    """Rank each fund's grade within its asset_class peer group."""
    df = con.execute("""
        SELECT s.ticker, s.grade, u.asset_class, u.category
        FROM batch_summary s
        JOIN fund_universe u ON s.ticker = u.ticker
        WHERE s.grade IS NOT NULL AND u.share_class_role = 'primary'
    """).df()

    if df.empty:
        return

    df["peer_percentile"] = df.groupby("asset_class")["grade"].rank(pct=True) * 100
    df["peer_n"] = df.groupby("asset_class")["ticker"].transform("count")

    for _, row in df.iterrows():
        con.execute("""
            UPDATE batch_summary
            SET peer_percentile = ?, peer_n = ?
            WHERE ticker = ?
        """, [round(row["peer_percentile"], 1), int(row["peer_n"]), row["ticker"]])

    con.execute("CHECKPOINT")


# ── Main analysis loop ────────────────────────────────────────────────────────

def analyse_fund(
    ticker:   str,
    category: str | None,
    fund_all: pd.DataFrame,
    etf_ret:  pd.DataFrame,
    tbill:    pd.Series,
) -> tuple[list[dict], dict | None]:
    """Returns (period_rows, summary_row) or raises on error."""
    if ticker not in fund_all.columns:
        raise ValueError(f"{ticker} not in fund_nav")

    fund_s = fund_all[ticker].dropna()
    if len(fund_s) < MIN_MONTHS_FULL:
        raise ValueError(f"only {len(fund_s)} months of data")

    # Align ETF universe: drop ETFs with no data in fund's period
    common_idx = etf_ret.index.intersection(fund_s.index)
    etf_sub    = etf_ret.reindex(common_idx)
    fund_sub   = fund_s.reindex(common_idx).dropna()
    etf_sub    = etf_sub.reindex(fund_sub.index)

    # Get full-period weights to determine benchmark
    etf_full = etf_sub.ffill(limit=1).dropna(how="all", axis=1)
    valid    = [c for c in etf_full.columns
                if etf_full[c].notna().sum() >= MIN_MONTHS_RBSA]
    if not valid:
        raise ValueError("no ETFs with sufficient overlap")

    etf_full = etf_full[valid].dropna()
    f_full   = fund_sub.reindex(etf_full.index).dropna()
    etf_full = etf_full.reindex(f_full.index).dropna()

    w_full = constrained_ols(f_full.values, etf_full.values)
    bm     = select_benchmark(ticker, category, w_full,
                              list(etf_full.columns), fund_sub, etf_ret)

    # Compute metrics for each period
    period_rows: list[dict] = []
    for period_name, n_mo in PERIODS.items():
        try:
            metrics = compute_period_metrics(fund_sub, etf_sub, bm, tbill, n_mo)
        except Exception:
            metrics = None
        if metrics:
            row = {"ticker": ticker, "period": period_name, "benchmark": bm}
            row.update(metrics)
            period_rows.append(row)

    if not period_rows:
        raise ValueError("no periods computed")

    # Full-period weights for summary
    full_row    = next((r for r in period_rows if r["period"] == "full"), period_rows[-1])
    full_weights = full_row.get("weights_json", "{}")

    grade_info = compute_grade(period_rows)

    summary = {
        "ticker":          ticker,
        "benchmark":       bm,
        "weights_json":    full_weights,
        "grade":           grade_info["grade"],
        "wtd_alpha":       grade_info["wtd_alpha"],
        "wtd_info_ratio":  grade_info["wtd_info_ratio"],
        "wtd_sharpe_diff": grade_info["wtd_sharpe_diff"],
        "wtd_beat_rate":   grade_info["wtd_beat_rate"],
    }
    return period_rows, summary


def run_batch(con: duckdb.DuckDBPyConnection,
              reset: bool = False,
              single_ticker: str | None = None) -> None:

    create_tables(con)

    if reset:
        print("Resetting batch_periods and batch_summary...")
        con.execute("DROP TABLE IF EXISTS batch_periods")
        con.execute("DROP TABLE IF EXISTS batch_summary")
        create_tables(con)
        con.execute("CHECKPOINT")

    # Funds to process
    if single_ticker:
        funds = con.execute("""
            SELECT fu.ticker, fu.category
            FROM fund_universe fu WHERE fu.ticker = ?
        """, [single_ticker]).fetchall()
    else:
        funds = con.execute("""
            SELECT fu.ticker, fu.category
            FROM fund_universe fu
            WHERE fu.share_class_role IN ('primary','secondary')
              AND fu.ticker NOT IN (SELECT DISTINCT ticker FROM batch_summary)
            ORDER BY fu.aum_millions DESC
        """).fetchall()

    if not funds:
        if single_ticker:
            print(f"{single_ticker} not found in fund_universe table.")
        else:
            print("All funds already analysed. Use --reset to re-run.")
        return

    print(f"Loading data from fund_universe.duckdb...")
    etf_ret  = load_etf_returns(con)
    fund_all = load_fund_returns(con)
    tbill    = load_tbill(con)
    print(f"  ETF returns:  {etf_ret.shape[1]} tickers × {etf_ret.shape[0]} months")
    print(f"  Fund returns: {fund_all.shape[1]} tickers × {fund_all.shape[0]} months")
    print(f"\nAnalysing {len(funds):,} funds...\n")

    ok = errors = 0
    t0 = time.time()

    for i, (ticker, category) in enumerate(funds, 1):
        try:
            period_rows, summary = analyse_fund(ticker, category, fund_all, etf_ret, tbill)

            # Write period rows
            for r in period_rows:
                con.execute("""
                    INSERT OR REPLACE INTO batch_periods VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                """, [
                    r["ticker"], r["period"], r.get("n_months"), r.get("benchmark"),
                    r.get("weights_json"),
                    r.get("fund_ann_ret"), r.get("fund_vol"), r.get("fund_sharpe"),
                    r.get("bm_ann_ret"),   r.get("bm_vol"),   r.get("bm_sharpe"),
                    r.get("alpha"), r.get("tracking_error"), r.get("info_ratio"),
                    r.get("r_squared"),
                    r.get("up_capture"), r.get("down_capture"), r.get("beat_rate"),
                ])

            # Write summary (without asset_class/category — join from fund_universe)
            con.execute("""
                INSERT OR REPLACE INTO batch_summary
                    (ticker, benchmark, weights_json, grade,
                     wtd_alpha, wtd_info_ratio, wtd_sharpe_diff, wtd_beat_rate)
                VALUES (?,?,?,?,?,?,?,?)
            """, [
                summary["ticker"], summary["benchmark"], summary["weights_json"],
                summary["grade"],  summary["wtd_alpha"], summary["wtd_info_ratio"],
                summary["wtd_sharpe_diff"], summary["wtd_beat_rate"],
            ])
            ok += 1

            # Self-healing asset_class: the ingestion-time text classifier
            # (build_fund_universe.py) misses abbreviated bond-fund names (e.g.
            # "Tx-Ex", "Investment Grade") and mislabels them "Equity". RBSA
            # weights are a much more reliable signal of what a fund actually
            # holds, so reconcile using the same >=50% FI-weight threshold used
            # for benchmark selection above.
            fi_wt = sum(v for k, v in json.loads(summary["weights_json"]).items()
                        if k in cfg.FI_ETFS)
            if fi_wt >= 0.50:
                con.execute("""
                    UPDATE fund_universe SET asset_class = 'Fixed Income'
                    WHERE ticker = ? AND asset_class != 'Fixed Income'
                """, [ticker])

        except Exception as exc:
            errors += 1
            if single_ticker:
                print(f"  ERROR: {exc}")

        if i % 100 == 0:
            con.execute("CHECKPOINT")
            elapsed   = time.time() - t0
            remaining = (len(funds) - i) / (i / elapsed) / 60
            print(f"  [{i:4d}/{len(funds)}]  ok={ok}  err={errors}  ~{remaining:.0f} min left")

    con.execute("CHECKPOINT")
    elapsed = time.time() - t0
    print(f"\nRBSA done in {elapsed/60:.1f} min: {ok} ok, {errors} errors")

    print("Computing peer percentiles...")
    compute_peer_percentiles(con)
    print("Done.\n")
    print_stats(con)


def print_stats(con: duckdb.DuckDBPyConnection) -> None:
    print(con.execute("""
        SELECT u.asset_class,
               COUNT(DISTINCT s.ticker)             AS analysed,
               ROUND(AVG(s.grade), 2)               AS avg_grade,
               ROUND(AVG(s.wtd_alpha)*100, 2)       AS avg_alpha_pct,
               ROUND(AVG(s.wtd_info_ratio), 2)      AS avg_ir,
               ROUND(AVG(s.wtd_beat_rate)*100, 1)   AS avg_beat_rate_pct,
               SUM(CASE WHEN s.grade >= 4 THEN 1 ELSE 0 END) AS n_grade_4_5
        FROM batch_summary s
        JOIN fund_universe u ON s.ticker = u.ticker
        WHERE u.share_class_role = 'primary'
        GROUP BY u.asset_class ORDER BY analysed DESC
    """).df().to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",  action="store_true")
    parser.add_argument("--stats",  action="store_true")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()

    con = get_con()
    try:
        create_tables(con)
        if args.stats:
            print_stats(con)
        else:
            run_batch(con, reset=args.reset, single_ticker=args.ticker)
    finally:
        con.execute("CHECKPOINT")
        con.close()


if __name__ == "__main__":
    main()
