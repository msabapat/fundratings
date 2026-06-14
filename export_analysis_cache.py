# export_analysis_cache.py
# Pre-computes analysis for every configured fund locally and saves to analysis_cache.json.
# Commit that file so Railway loads it at startup instead of recomputing on demand.
#
# Run order (first time):
#   py export_etf_data.py       -- creates etf_returns.parquet
#   py export_fund_data.py      -- creates fund_returns.parquet + tbill_returns.parquet
#   py export_analysis_cache.py -- creates analysis_cache.json
#   git add *.parquet analysis_cache.json && git commit -m "..." && git push
#
# Refresh before re-deploying to pick up new fund data / logic changes.

from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import _run_analysis, _get_etf_returns, _get_tbill_monthly
import config as cfg

print("Pre-loading ETF and T-bill data...")
_get_etf_returns()
_get_tbill_monthly()
print()

tickers = list(cfg.ACTIVE_MUTUAL_FUNDS.keys()) + list(cfg.ACTIVE_ETFS.keys())
print(f"Computing analyses for {len(tickers)} funds...\n")

cache: dict = {}
ok = errors = 0

for i, t in enumerate(tickers, 1):
    t0 = time.time()
    try:
        result = _run_analysis(t)
        cache[f"{t}|"] = result
        oos = len(result.get("cumulative", {}).get("dates", []))
        elapsed = time.time() - t0
        print(f"  [{i:2d}/{len(tickers)}] {t:<10s}  {oos} OOS months  ({elapsed:.1f}s)")
        ok += 1
    except Exception as e:
        print(f"  [{i:2d}/{len(tickers)}] {t:<10s}  ERROR — {e}")
        errors += 1

path = Path(__file__).parent / "analysis_cache.json"
with open(path, "w") as f:
    json.dump(cache, f, separators=(',', ':'))

size_kb = path.stat().st_size / 1024
print(f"\n{ok} OK, {errors} errors  →  {path.name}  ({size_kb:.0f} KB)")
print("\nNext steps:")
print("  git add analysis_cache.json fund_returns.parquet tbill_returns.parquet")
print("  git commit -m 'Add pre-computed analysis cache'")
print("  git push")
