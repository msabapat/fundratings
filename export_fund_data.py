# export_fund_data.py
# Run locally before deploying to pre-populate fund returns and T-bill data.
# Produces fund_returns.parquet and tbill_returns.parquet in this directory.
# Commit both files so Railway doesn't need yfinance for known funds.
#
# Usage: py export_fund_data.py

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

OUTPUT_DIR = Path(__file__).parent
EXPORT_START = "2000-01-01"


def export_tbill() -> None:
    import yfinance as yf
    print("Downloading T-bill data (^IRX)...")
    hist = yf.Ticker("^IRX").history(start=EXPORT_START, auto_adjust=False)
    hist.index = hist.index.tz_localize(None)
    # Monthly decimal rate: annualised pct ÷ 100 ÷ 12
    s = (hist["Close"] / 100 / 12).resample("ME").last()
    s.index = s.index + pd.offsets.MonthEnd(0)
    df = s.rename("tbill").to_frame()
    path = OUTPUT_DIR / "tbill_returns.parquet"
    df.to_parquet(path)
    print(f"  {len(df)} months → {path.name}")


def export_funds() -> None:
    import yfinance as yf
    tickers = list(cfg.ACTIVE_MUTUAL_FUNDS.keys()) + list(cfg.ACTIVE_ETFS.keys())
    print(f"\nDownloading {len(tickers)} active fund/ETF tickers from yfinance...")
    frames: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(start=EXPORT_START, auto_adjust=True)
            if len(hist) < 24:
                print(f"  {t}: skipping (<24 daily rows)")
                continue
            monthly = hist["Close"].resample("ME").last()
            monthly.index = (monthly.index + pd.offsets.MonthEnd(0)).tz_localize(None)
            ret = monthly.pct_change().rename(t).iloc[1:]
            frames[t] = ret
            print(f"  {t}: {len(ret)} months")
        except Exception as exc:
            print(f"  {t}: ERROR — {exc}")

    if not frames:
        print("No data downloaded — aborting.")
        return

    df = pd.DataFrame(frames).sort_index()
    path = OUTPUT_DIR / "fund_returns.parquet"
    df.to_parquet(path)
    print(f"\n  {df.shape[1]} tickers × {df.shape[0]} months → {path.name}")


if __name__ == "__main__":
    export_tbill()
    export_funds()
    print("\nDone. Commit fund_returns.parquet and tbill_returns.parquet to git before deploying.")
