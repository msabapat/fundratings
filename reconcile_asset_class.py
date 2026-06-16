"""One-off / re-runnable fix for fund_universe.asset_class mislabeling.

Two independent signals, applied in order of reliability:

1. Text reclassification — re-run the (now-expanded) _infer_asset_class
   regex from build_fund_universe.py against the long_name/category
   already stored in the DB. No yfinance/RBSA work needed, so this is
   cheap and catches funds outside batch_summary too.
2. RBSA-weight override — for funds with a completed batch_rbsa run
   (batch_summary.weights_json), a >=50% weight in cfg.FI_ETFS is a more
   reliable fixed-income signal than fund-name text (catches funds whose
   name gives no textual clue at all, e.g. plain proprietary fund names).
   This takes precedence over step 1 when they disagree.

Safe to re-run any time (idempotent UPDATE ... WHERE asset_class != ...).
"""
import json

import duckdb

import config as cfg
from build_fund_universe import _infer_asset_class

DB_PATH = "fund_universe.duckdb"


def main():
    con = duckdb.connect(DB_PATH)

    rows = con.execute(
        "SELECT ticker, category, long_name, asset_class FROM fund_universe"
    ).fetchall()

    text_changes = []
    for ticker, category, long_name, asset_class in rows:
        new_class = _infer_asset_class(category, long_name)
        if new_class != asset_class:
            text_changes.append((ticker, asset_class, new_class))

    for ticker, old, new in text_changes:
        con.execute(
            "UPDATE fund_universe SET asset_class = ? WHERE ticker = ?",
            [new, ticker],
        )

    print(f"Text reclassification: {len(text_changes)} funds changed")
    for ticker, old, new in text_changes:
        print(f"  {ticker}: {old!r} -> {new!r}")

    weights_rows = con.execute("""
        SELECT b.ticker, b.weights_json, f.asset_class
        FROM batch_summary b
        JOIN fund_universe f ON f.ticker = b.ticker
        WHERE b.weights_json IS NOT NULL
    """).fetchall()

    weight_changes = []
    for ticker, weights_json, asset_class in weights_rows:
        weights = json.loads(weights_json)
        fi_wt = sum(v for k, v in weights.items() if k in cfg.FI_ETFS)
        if fi_wt >= 0.50 and asset_class != "Fixed Income":
            weight_changes.append((ticker, asset_class, fi_wt))

    for ticker, old, fi_wt in weight_changes:
        con.execute(
            "UPDATE fund_universe SET asset_class = 'Fixed Income' WHERE ticker = ?",
            [ticker],
        )

    print(f"\nRBSA-weight override: {len(weight_changes)} funds changed")
    for ticker, old, fi_wt in weight_changes:
        print(f"  {ticker}: {old!r} -> 'Fixed Income' (FI weight={fi_wt:.0%})")

    con.commit()
    con.close()
    print(f"\nTotal: {len(text_changes)} text-based + {len(weight_changes)} weight-based changes committed")


if __name__ == "__main__":
    main()
