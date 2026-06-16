"""
migrate_to_db.py
Migrates existing parquet files and analysis_cache.json into fund_universe.duckdb.

Tables created:
  etf_returns   (date, ticker, return_monthly)
  fund_nav      (date, ticker, return_monthly)   -- replaces fund_returns.parquet
  tbill_returns (date, rate_monthly)
  fund_analysis (ticker, period, computed JSON)  -- from analysis_cache.json

Run once:  py migrate_to_db.py
"""
from __future__ import annotations
import json
from pathlib import Path

import duckdb
import pandas as pd

DIR    = Path(__file__).parent
DB     = DIR / "fund_universe.duckdb"
con    = duckdb.connect(str(DB))


# ── ETF returns ───────────────────────────────────────────────────────────────
def migrate_etf_returns() -> None:
    src = DIR / "etf_returns.parquet"
    if not src.exists():
        print("etf_returns.parquet not found — skipping")
        return

    df = pd.read_parquet(src)          # index=date, columns=tickers
    df.index.name = "date"
    long = df.reset_index().melt(id_vars="date", var_name="ticker", value_name="return_monthly")
    long = long.dropna(subset=["return_monthly"])
    long["date"] = pd.to_datetime(long["date"])

    con.execute("DROP TABLE IF EXISTS etf_returns")
    con.execute("""
        CREATE TABLE etf_returns (
            date            DATE    NOT NULL,
            ticker          VARCHAR NOT NULL,
            return_monthly  DOUBLE,
            PRIMARY KEY (date, ticker)
        )
    """)
    con.register("_etf", long)
    con.execute("INSERT INTO etf_returns SELECT date, ticker, return_monthly FROM _etf")
    con.execute("CHECKPOINT")
    n = con.execute("SELECT COUNT(*) FROM etf_returns").fetchone()[0]
    print(f"etf_returns:   {n:,} rows  ({df.shape[1]} tickers × {df.shape[0]} months)")


# ── Fund NAV / returns ────────────────────────────────────────────────────────
def migrate_fund_nav() -> None:
    src = DIR / "fund_returns.parquet"
    if not src.exists():
        print("fund_returns.parquet not found — skipping")
        return

    df = pd.read_parquet(src)
    df.index.name = "date"
    long = df.reset_index().melt(id_vars="date", var_name="ticker", value_name="return_monthly")
    long = long.dropna(subset=["return_monthly"])
    long["date"] = pd.to_datetime(long["date"])

    con.execute("DROP TABLE IF EXISTS fund_nav")
    con.execute("""
        CREATE TABLE fund_nav (
            date            DATE    NOT NULL,
            ticker          VARCHAR NOT NULL,
            return_monthly  DOUBLE,
            PRIMARY KEY (date, ticker)
        )
    """)
    con.register("_nav", long)
    con.execute("INSERT INTO fund_nav SELECT date, ticker, return_monthly FROM _nav")
    con.execute("CHECKPOINT")
    n = con.execute("SELECT COUNT(*) FROM fund_nav").fetchone()[0]
    print(f"fund_nav:      {n:,} rows  ({df.shape[1]} tickers × {df.shape[0]} months)")


# ── T-bill returns ────────────────────────────────────────────────────────────
def migrate_tbill() -> None:
    src = DIR / "tbill_returns.parquet"
    if not src.exists():
        print("tbill_returns.parquet not found — skipping")
        return

    df = pd.read_parquet(src)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    col = [c for c in df.columns if c != "date"][0]
    df = df.rename(columns={col: "rate_monthly"})

    con.execute("DROP TABLE IF EXISTS tbill_returns")
    con.execute("""
        CREATE TABLE tbill_returns (
            date          DATE PRIMARY KEY,
            rate_monthly  DOUBLE
        )
    """)
    con.register("_tbill", df)
    con.execute("INSERT INTO tbill_returns SELECT date, rate_monthly FROM _tbill")
    con.execute("CHECKPOINT")
    n = con.execute("SELECT COUNT(*) FROM tbill_returns").fetchone()[0]
    print(f"tbill_returns: {n:,} rows")


# ── Analysis cache (JSON → flat table) ───────────────────────────────────────
def migrate_analysis_cache() -> None:
    src = DIR / "analysis_cache.json"
    if not src.exists():
        print("analysis_cache.json not found — skipping")
        return

    with open(src) as f:
        cache = json.load(f)

    con.execute("DROP TABLE IF EXISTS fund_analysis")
    con.execute("""
        CREATE TABLE fund_analysis (
            cache_key   VARCHAR PRIMARY KEY,
            ticker      VARCHAR,
            bm_override VARCHAR,
            payload     JSON,
            cached_at   TIMESTAMP DEFAULT now()
        )
    """)

    rows = []
    for key, val in cache.items():
        ticker, _, bm = key.partition("|")
        rows.append((key, ticker, bm or None, json.dumps(val)))

    con.executemany("INSERT INTO fund_analysis VALUES (?, ?, ?, ?, now())", rows)
    con.execute("CHECKPOINT")
    n = con.execute("SELECT COUNT(*) FROM fund_analysis").fetchone()[0]
    print(f"fund_analysis: {n:,} rows  (from analysis_cache.json)")


# ── Summary ───────────────────────────────────────────────────────────────────
def summary() -> None:
    print()
    print("Tables in fund_universe.duckdb:")
    tables = con.execute("""
        SELECT table_name,
               estimated_size AS approx_rows
        FROM duckdb_tables()
        ORDER BY table_name
    """).fetchall()
    for name, rows in tables:
        print(f"  {name:<20s}  ~{rows:>8,} rows")
    size_mb = DB.stat().st_size / 1e6
    print(f"\nDatabase size: {size_mb:.1f} MB")


if __name__ == "__main__":
    print(f"Migrating into {DB.name}...\n")
    migrate_etf_returns()
    migrate_fund_nav()
    migrate_tbill()
    migrate_analysis_cache()
    summary()
    con.close()
    print("\nDone. Parquet files and analysis_cache.json can be kept as backups.")
