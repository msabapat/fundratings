# fund_replication/universe_data.py
# Read-only access to fund_universe.duckdb for the Browse Universe dashboard tab.

from __future__ import annotations
import os
from pathlib import Path

import duckdb
import pandas as pd

# Local dev: file lives alongside the code. Railway: FUND_UNIVERSE_DB_PATH
# points at the persistent volume mount (e.g. /data/fund_universe.duckdb) --
# the DB is gitignored (~40MB, changes independently of code deploys) and
# isn't baked into the image, so it has to live on a volume in production.
DB_PATH = Path(os.environ.get("FUND_UNIVERSE_DB_PATH")
               or Path(__file__).parent / "fund_universe.duckdb")


def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def load_universe_meta() -> dict[str, dict]:
    """ticker -> fund_universe row (dict), for every ticker (primary + secondary)."""
    con = _con()
    try:
        df = con.execute("SELECT * FROM fund_universe").df()
    finally:
        con.close()
    df = df.astype(object).where(pd.notna(df), None)
    return {row["ticker"]: row.to_dict() for _, row in df.iterrows()}


def load_universe_fund_nav(ticker: str) -> pd.Series:
    """Monthly return series for one ticker from fund_universe.duckdb::fund_nav."""
    con = _con()
    try:
        df = con.execute(
            "SELECT date, return_monthly FROM fund_nav WHERE ticker = ? ORDER BY date",
            [ticker],
        ).df()
    finally:
        con.close()
    if df.empty:
        raise ValueError(f"{ticker} not found in fund_universe.duckdb::fund_nav")
    s = df.set_index("date")["return_monthly"]
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s.dropna()


# Columns added by grade_v2.py (2026-07). Queried defensively -- an older/stale
# batch_summary (e.g. a not-yet-refreshed deployment DB) won't have these yet,
# and a missing column would otherwise 500 the whole Browse Universe endpoint
# rather than just omitting the new fields.
_GRADE_V2_COLUMNS = [
    "fund_type", "recent_grade", "overall_grade",
    "single_etf_benchmark", "single_etf_benchmark_r2", "single_etf_benchmark_low_r2",
    "is_low_vol_fund",
]


def _batch_summary_columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    return set(con.execute("PRAGMA table_info('batch_summary')").df()["name"])


def get_grade_v2(ticker: str) -> dict | None:
    """Precomputed Recent/Overall grade for one ticker, or None if not batch-graded."""
    con = _con()
    try:
        available = _batch_summary_columns(con)
        if "recent_grade" not in available or "overall_grade" not in available:
            return None
        row = con.execute(
            "SELECT recent_grade, overall_grade FROM batch_summary WHERE ticker = ?",
            [ticker],
        ).fetchone()
    finally:
        con.close()
    if row is None or (row[0] is None and row[1] is None):
        return None
    return {"recent_grade": row[0], "overall_grade": row[1]}


def list_browse_funds() -> list[dict]:
    """
    Primary share class of every analysed fund, joined with its batch_summary
    grade/peer-percentile/alpha metrics, for the Browse Universe table.
    """
    con = _con()
    try:
        available = _batch_summary_columns(con)
        extra_cols = [c for c in _GRADE_V2_COLUMNS if c in available]
        extra_sql = "".join(f", s.{c}" for c in extra_cols)
        df = con.execute(f"""
            SELECT u.ticker, u.long_name, u.fund_family, u.asset_class,
                   u.aum_millions, u.expense_ratio, u.is_active,
                   u.has_front_load, u.has_deferred_load,
                   s.benchmark, s.grade, s.peer_percentile, s.peer_n,
                   s.wtd_alpha, s.wtd_info_ratio, s.wtd_beat_rate, s.wtd_sharpe_diff
                   {extra_sql}
            FROM fund_universe u
            JOIN batch_summary s ON u.ticker = s.ticker
            WHERE u.share_class_role = 'primary'
            ORDER BY u.aum_millions DESC NULLS LAST
        """).df()
    finally:
        con.close()
    df = df.astype(object).where(pd.notna(df), None)
    return df.to_dict(orient="records")
