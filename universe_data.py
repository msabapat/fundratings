# fund_replication/universe_data.py
# Read-only access to fund_universe.duckdb for the Browse Universe dashboard tab.

from __future__ import annotations
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).parent / "fund_universe.duckdb"


def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def load_universe_meta() -> dict[str, dict]:
    """ticker -> fund_universe row (dict), for every ticker (primary + secondary)."""
    con = _con()
    try:
        df = con.execute("SELECT * FROM fund_universe").df()
    finally:
        con.close()
    df = df.where(pd.notna(df), None)
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


def list_browse_funds() -> list[dict]:
    """
    Primary share class of every analysed fund, joined with its batch_summary
    grade/peer-percentile/alpha metrics, for the Browse Universe table.
    """
    con = _con()
    try:
        df = con.execute("""
            SELECT u.ticker, u.long_name, u.fund_family, u.asset_class,
                   u.aum_millions, u.expense_ratio, u.is_active,
                   u.has_front_load, u.has_deferred_load,
                   s.benchmark, s.grade, s.peer_percentile, s.peer_n,
                   s.wtd_alpha, s.wtd_info_ratio, s.wtd_beat_rate, s.wtd_sharpe_diff
            FROM fund_universe u
            JOIN batch_summary s ON u.ticker = s.ticker
            WHERE u.share_class_role = 'primary'
            ORDER BY u.aum_millions DESC NULLS LAST
        """).df()
    finally:
        con.close()
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")
