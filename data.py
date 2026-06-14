# fund_replication/data.py
# Load monthly total returns from Parquet (deployment) or DuckDB (local dev).

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH      = Path(__file__).parent.parent / "backtester.duckdb"
PARQUET_PATH = Path(__file__).parent / "etf_returns.parquet"


def _eom_prices_from_db(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Return end-of-month closeadj prices for given tickers from DuckDB."""
    import duckdb
    placeholders = ", ".join(["?"] * len(tickers))
    sql = f"""
        SELECT ticker,
               DATE_TRUNC('month', date)::DATE AS month,
               LAST(closeadj ORDER BY date)    AS eom_price
        FROM   sharadar_fundprices
        WHERE  ticker IN ({placeholders})
          AND  date BETWEEN ? AND ?
        GROUP  BY ticker, month
        ORDER  BY ticker, month
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(sql, tickers + [start, end]).df()
    con.close()
    return df


def load_etf_returns(
    tickers: list[str],
    start: str = "2003-01-01",
    end:   str = "2025-12-31",
) -> pd.DataFrame:
    """
    Monthly total returns for passive/active ETFs.
    Loads from etf_returns.parquet if present (deployment), otherwise DuckDB (local dev).
    Returns a DataFrame indexed by month-end date, columns = ticker symbols.
    """
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        cols = [t for t in tickers if t in df.columns]
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            warnings.warn(f"Tickers not in Parquet (not exported): {missing}", stacklevel=2)
        df = df.loc[start:end, cols]
    else:
        raw = _eom_prices_from_db(tickers, start, end)
        wide = raw.pivot(index="month", columns="ticker", values="eom_price")
        wide.index = pd.to_datetime(wide.index) + pd.offsets.MonthEnd(0)
        df = wide.pct_change().iloc[1:]

    short = [c for c in df.columns if df[c].notna().sum() < 12]
    if short:
        warnings.warn(f"Dropping ETFs with <12 months of data: {short}", stacklevel=2)
        df = df.drop(columns=short)

    return df.sort_index()


def load_fund_returns(
    tickers: list[str],
    start: str = "2005-01-01",
    end:   str = "2025-12-31",
) -> pd.DataFrame:
    """
    Monthly total returns for mutual funds from yfinance (dividend-adjusted NAV).
    Returns a DataFrame indexed by month-end date, columns = ticker symbols.
    """
    import yfinance as yf

    frames: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(
                start=start, end=end, auto_adjust=True
            )
            if len(hist) < 24:
                warnings.warn(f"{t}: only {len(hist)} daily rows — skipping", stacklevel=2)
                continue
            monthly = hist["Close"].resample("ME").last()
            monthly.index = (monthly.index + pd.offsets.MonthEnd(0)).tz_localize(None)
            frames[t] = monthly.pct_change().rename(t)
        except Exception as exc:
            warnings.warn(f"{t}: yfinance error — {exc}", stacklevel=2)

    if not frames:
        raise ValueError("No fund data loaded from yfinance")

    result = pd.DataFrame(frames).sort_index()
    result = result.iloc[1:]
    return result


def align(
    fund_returns: pd.DataFrame | pd.Series,
    etf_returns: pd.DataFrame,
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame]:
    """
    Inner-join fund and ETF returns on date index, drop columns with >20% NaN,
    then forward-fill (max 1 month) and drop remaining NaN rows.
    """
    if isinstance(fund_returns, pd.Series):
        combined = etf_returns.join(fund_returns, how="inner")
        combined = combined.dropna(subset=[fund_returns.name])
    else:
        combined = etf_returns.join(fund_returns, how="inner")
        combined = combined.dropna(subset=list(fund_returns.columns))

    etf_cols    = list(etf_returns.columns)
    cutoff_date = combined.index[min(6, len(combined) - 1)]
    first_valid = combined[etf_cols].apply(lambda s: s.first_valid_index())
    drop        = first_valid[first_valid > cutoff_date].index.tolist()
    if drop:
        warnings.warn(f"Dropping ETFs with late-start overlap: {drop}", stacklevel=2)
        combined = combined.drop(columns=drop)
        etf_cols = [c for c in etf_cols if c not in drop]

    combined[etf_cols] = combined[etf_cols].ffill(limit=1)
    combined = combined.dropna()

    if isinstance(fund_returns, pd.Series):
        return combined[fund_returns.name], combined[etf_cols]
    else:
        fund_cols = list(fund_returns.columns)
        return combined[fund_cols], combined[etf_cols]
