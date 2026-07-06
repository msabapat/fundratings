# refresh_etf_returns.py
# Rebuilds fund_universe.duckdb::etf_returns from yfinance (the same reliable,
# always-current source download_fund_nav.py uses for mutual fund NAV data).
#
# The table was originally populated once via migrate_to_db.py from a
# Sharadar-sourced local snapshot (etf_prices.duckdb) that stopped updating
# at 2025-12-31 with no refresh mechanism -- since app.py's align() truncates
# every fund's return series to whatever etf_returns covers, this silently
# made every trailing-period return (1y/3y/5y/10y) site-wide up to 7 months
# stale, not just the fund NAV side.
#
# Usage: py refresh_etf_returns.py

from __future__ import annotations
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import yfinance as yf

import config as cfg

DB_PATH = Path(__file__).parent / "fund_universe.duckdb"
START_DATE = "2000-01-01"

ALL_TICKERS = sorted(set(cfg.PASSIVE_ETFS.keys()) | set(cfg.ACTIVE_ETFS.keys()))
print(f"Refreshing {len(ALL_TICKERS)} ETFs from yfinance...")

raw = yf.download(ALL_TICKERS, start=START_DATE, auto_adjust=True, threads=True)
close = raw["Close"]

close.index = pd.to_datetime(close.index).tz_localize(None)
monthly = close.resample("ME").last()
monthly.index = monthly.index + pd.offsets.MonthEnd(0)
returns = monthly.pct_change().iloc[1:]

long = (
    returns.reset_index()
    .melt(id_vars="Date", var_name="ticker", value_name="return_monthly")
    .rename(columns={"Date": "date"})
    .dropna(subset=["return_monthly"])
)
long = long[long["return_monthly"].abs() < 1.0]

print(f"  {long['ticker'].nunique()} tickers, {long['date'].min()} to {long['date'].max()}, {len(long)} rows")

con = duckdb.connect(str(DB_PATH))
con.execute("DELETE FROM etf_returns")
con.register("_new", long)
con.execute("INSERT INTO etf_returns SELECT date, ticker, return_monthly FROM _new")
con.execute("CHECKPOINT")

n = con.execute("SELECT COUNT(*) FROM etf_returns").fetchone()[0]
rng = con.execute("SELECT MIN(date), MAX(date) FROM etf_returns").fetchone()
print(f"\netf_returns refreshed: {n} rows, {rng[0]} to {rng[1]}")
con.close()
