# export_etf_data.py
# Run ONCE locally to generate etf_returns.parquet for deployment.
# Re-run whenever you add ETFs to config or want fresher price history.
#
#   py export_etf_data.py

from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import config as cfg
from data import _eom_prices_from_db

ALL_TICKERS = list(cfg.PASSIVE_ETFS.keys()) + list(cfg.ACTIVE_ETFS.keys())

print(f"Exporting {len(ALL_TICKERS)} tickers from DuckDB ...")
print(f"  Date range: {cfg.DEFAULT_START} → {cfg.DEFAULT_END}")

raw = _eom_prices_from_db(ALL_TICKERS, start=cfg.DEFAULT_START, end=cfg.DEFAULT_END)

wide = raw.pivot(index="month", columns="ticker", values="eom_price")
wide.index = pd.to_datetime(wide.index) + pd.offsets.MonthEnd(0)
returns = wide.pct_change().iloc[1:].sort_index()

# Drop anything with fewer than 12 months of data
short = [c for c in returns.columns if returns[c].notna().sum() < 12]
if short:
    print(f"  Dropping tickers with <12 months: {short}")
    returns = returns.drop(columns=short)

out = Path(__file__).parent / "etf_returns.parquet"
returns.to_parquet(out)

kb = out.stat().st_size / 1024
print(f"  Saved {returns.shape[0]} months × {returns.shape[1]} ETFs")
print(f"  Output: {out}  ({kb:.0f} KB)")
print("Done — commit etf_returns.parquet and deploy.")
