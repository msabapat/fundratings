"""
download_fund_nav.py
Downloads monthly total-return NAV history for all tickers in fund_universe
and stores them in fund_universe.duckdb :: fund_nav table.

Resumable: already-downloaded tickers are skipped on restart.
Existing fund_nav rows (from parquet migration) are preserved.

Usage:
  py download_fund_nav.py            # download all missing tickers
  py download_fund_nav.py --reset    # wipe fund_nav and re-download everything
  py download_fund_nav.py --stats    # show coverage summary only
"""
from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DB_PATH    = Path(__file__).parent / "fund_universe.duckdb"
BATCH_SIZE = 100          # tickers per yf.download call
START_DATE = "2000-01-01"
DELAY      = 1.5          # seconds between batches


def get_con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH))


def coverage_stats(con: duckdb.DuckDBPyConnection) -> None:
    print(con.execute("""
        SELECT
            fu.asset_class,
            COUNT(DISTINCT fu.ticker)                                        AS universe,
            COUNT(DISTINCT fn.ticker)                                        AS downloaded,
            COUNT(DISTINCT fu.ticker) - COUNT(DISTINCT fn.ticker)           AS missing,
            MIN(fn.date)::VARCHAR                                            AS earliest,
            MAX(fn.date)::VARCHAR                                            AS latest,
            ROUND(AVG(months_per_ticker), 0)                                AS avg_months
        FROM fund_universe fu
        LEFT JOIN fund_nav fn ON fu.ticker = fn.ticker
        LEFT JOIN (
            SELECT ticker, COUNT(*) AS months_per_ticker FROM fund_nav GROUP BY ticker
        ) m ON fu.ticker = m.ticker
        WHERE fu.share_class_role IN ('primary','secondary')
        GROUP BY fu.asset_class ORDER BY universe DESC
    """).df().to_string(index=False))


def download(con: duckdb.DuckDBPyConnection, reset: bool = False) -> None:
    if reset:
        print("Resetting fund_nav table...")
        con.execute("DELETE FROM fund_nav")
        con.execute("CHECKPOINT")

    # Tickers we still need
    pending = [r[0] for r in con.execute("""
        SELECT DISTINCT fu.ticker
        FROM fund_universe fu
        WHERE fu.share_class_role IN ('primary', 'secondary')
          AND fu.ticker NOT IN (SELECT DISTINCT ticker FROM fund_nav)
        ORDER BY fu.ticker
    """).fetchall()]

    if not pending:
        print("All fund_universe tickers already have NAV data.")
        coverage_stats(con)
        return

    total    = len(pending)
    n_done   = 0
    n_errors = 0
    t0       = time.time()
    print(f"Downloading NAV history for {total:,} tickers in batches of {BATCH_SIZE}...")
    print(f"Estimated time: ~{total / BATCH_SIZE * DELAY / 60:.0f}-{total / BATCH_SIZE * 3 / 60:.0f} min\n")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = pending[batch_start : batch_start + BATCH_SIZE]

        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            print(f"  Batch {batch_start//BATCH_SIZE + 1}: download error — {exc}")
            n_errors += len(batch)
            time.sleep(DELAY * 2)
            continue

        if raw.empty:
            n_errors += len(batch)
            time.sleep(DELAY)
            continue

        # yf.download returns MultiIndex columns: (field, ticker)
        # For a single ticker it collapses to single-level; normalise both cases.
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0:0]
        else:
            # Single ticker in batch — wrap in DataFrame with ticker as column name
            close = raw[["Close"]].rename(columns={"Close": batch[0]})

        # Resample to month-end, compute pct_change
        close.index = pd.to_datetime(close.index).tz_localize(None)
        monthly = close.resample("ME").last()
        monthly.index = monthly.index + pd.offsets.MonthEnd(0)
        ret = monthly.pct_change().iloc[1:]   # drop first NaN row

        if ret.empty:
            n_errors += len(batch)
            time.sleep(DELAY)
            continue

        # Melt wide → long and store
        ret.index.name = "date"
        long = (
            ret.reset_index()
               .melt(id_vars="date", var_name="ticker", value_name="return_monthly")
               .dropna(subset=["return_monthly"])
        )
        long = long[long["return_monthly"].abs() < 1.0]   # drop implausible >100% moves

        if not long.empty:
            con.register("_batch", long)
            con.execute("""
                INSERT OR IGNORE INTO fund_nav (date, ticker, return_monthly)
                SELECT date, ticker, return_monthly FROM _batch
            """)
            con.unregister("_batch")

        # Count how many tickers got data in this batch
        got_data = set(long["ticker"].unique()) if not long.empty else set()
        n_done  += len(got_data)
        n_errors += len(batch) - len(got_data)

        # Checkpoint every 5 batches
        batch_num = batch_start // BATCH_SIZE + 1
        if batch_num % 5 == 0:
            con.execute("CHECKPOINT")
            elapsed   = time.time() - t0
            remaining = (total - batch_start - len(batch)) / BATCH_SIZE * DELAY / 60
            pct = (batch_start + len(batch)) / total * 100
            print(f"  Batch {batch_num:3d}  [{pct:4.0f}%]  ok={n_done}  err={n_errors}"
                  f"  ~{remaining:.0f} min left")

        time.sleep(DELAY)

    con.execute("CHECKPOINT")
    elapsed = time.time() - t0
    print(f"\nDownload complete in {elapsed/60:.1f} min: {n_done} ok, {n_errors} no data\n")
    coverage_stats(con)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe fund_nav and re-download")
    parser.add_argument("--stats", action="store_true", help="Show coverage stats only")
    args = parser.parse_args()

    con = get_con()
    try:
        if args.stats:
            coverage_stats(con)
        else:
            download(con, reset=args.reset)
    finally:
        con.execute("CHECKPOINT")
        con.close()


if __name__ == "__main__":
    main()
