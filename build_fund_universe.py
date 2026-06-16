"""
build_fund_universe.py
Builds fund_universe.duckdb with all US mutual funds above $250M AUM.

Phases (each saves progress so the pipeline is resumable):
  py build_fund_universe.py discover   # ~10 min  : page Yahoo screener, save raw tickers
  py build_fund_universe.py enrich     # ~90 min  : pull yfinance info per ticker (resumable)
  py build_fund_universe.py select     # ~1 min   : deduplicate → write fund_universe table
  py build_fund_universe.py all        # run all three sequentially
  py build_fund_universe.py stats      # summary of fund_universe table
"""

from __future__ import annotations
import argparse
import re
import time
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

DB_PATH      = Path(__file__).parent / "fund_universe.duckdb"
MIN_AUM      = 250e6      # $250M AUM floor
MAX_RANK     = 22_000     # screener rank cap (~$250M AUM based on probe data)
PAGE_SIZE    = 250        # Yahoo screener max per call
CALL_DELAY   = 0.40       # seconds between API calls

US_EXCHANGES = ["NAS", "NYQ", "NCM", "NMS", "NGM", "OEM", "ASE", "WCB", "OQB", "OGM", "PNK"]
SCREEN_QUERY = yf.FundQuery("is-in", ["exchange"] + US_EXCHANGES)

# ── Inference helpers ─────────────────────────────────────────────────────────

# Known fund families: (search_string, canonical_name)  — ordered longest-first
_FUND_FAMILIES: list[tuple[str, str]] = [
    ("american funds",       "American Funds"),
    ("franklin templeton",   "Franklin Templeton"),
    ("t. rowe price",        "T. Rowe Price"),
    ("t rowe price",         "T. Rowe Price"),
    ("dodge & cox",          "Dodge & Cox"),
    ("lord abbett",          "Lord Abbett"),
    ("eaton vance",          "Eaton Vance"),
    ("hartford",             "Hartford"),
    ("john hancock",         "John Hancock"),
    ("columbia",             "Columbia"),
    ("blackrock",            "BlackRock"),
    ("goldman sachs",        "Goldman Sachs"),
    ("jpmorgan",             "JPMorgan"),
    ("jp morgan",            "JPMorgan"),
    ("morgan stanley",       "Morgan Stanley"),
    ("state street",         "State Street"),
    ("principal",            "Principal"),
    ("vanguard",             "Vanguard"),
    ("fidelity",             "Fidelity"),
    ("pimco",                "PIMCO"),
    ("invesco",              "Invesco"),
    ("franklin",             "Franklin Templeton"),
    ("templeton",            "Franklin Templeton"),
    ("schwab",               "Schwab"),
    ("putnam",               "Putnam"),
    ("nuveen",               "Nuveen"),
    ("neuberger berman",     "Neuberger Berman"),
    ("baron",                "Baron"),
    ("artisan",              "Artisan"),
    ("oakmark",              "Oakmark"),
    ("primecap",             "Primecap"),
    ("gabelli",              "Gabelli"),
    ("william blair",        "William Blair"),
    ("harbor",               "Harbor"),
    ("mfs",                  "MFS"),
    ("dws",                  "DWS"),
    ("aqr",                  "AQR"),
    ("dimensional",          "Dimensional"),
    ("dfa",                  "Dimensional"),
    ("manning & napier",     "Manning & Napier"),
    ("calvert",              "Calvert"),
    ("parnassus",            "Parnassus"),
    ("american century",     "American Century"),
    ("janus henderson",      "Janus Henderson"),
    ("janus",                "Janus Henderson"),
    ("alliancebernstein",    "AllianceBernstein"),
    ("ab ",                  "AllianceBernstein"),
    ("wells fargo",          "Allspring"),
    ("allspring",            "Allspring"),
    ("natixis",              "Natixis"),
    ("loomis sayles",        "Loomis Sayles"),
    ("legg mason",           "Franklin Templeton"),
    ("pioneer",              "Amundi"),
    ("manning",              "Manning & Napier"),
    ("oppenheimer",          "Invesco"),
    ("western asset",        "Western Asset"),
    ("baird",                "Baird"),
    ("thornburg",            "Thornburg"),
    ("royce",                "Royce"),
    ("dodge",                "Dodge & Cox"),
]

_PASSIVE_PATTERNS = re.compile(
    r"\b(index|idx|s&p\s*500|russell|msci|total\s+market|total\s+stock|total\s+bond|"
    r"total\s+intl|dow\s+jones|500\s+index|2000\s+index|enhanced\s+index|benchmark)\b",
    re.IGNORECASE,
)

_FI_PATTERNS = re.compile(
    r"\b(bond|income|treasury|credit|muni|municipal|inflation|loan|"
    r"fixed\s+income|gilt|note|yield|debt|duration|maturity|"
    r"tax.?exempt|tx.?ex|investment.?grade|inv.?grade|gnma|ginnie.?mae|"
    r"mortgage|corp(orate)?\s*bd|government|govt|federal)\b",
    re.IGNORECASE,
)

_ALLOC_PATTERNS = re.compile(
    r"\b(allocation|balanced|target.date|multi.asset|lifestyle|lifecycle|"
    r"conservative|moderate|aggressive|retirement\s+\d{4})\b",
    re.IGNORECASE,
)

_ALT_PATTERNS = re.compile(
    r"\b(commodity|real\s+estate|reit|alternative|market\s+neutral|"
    r"managed\s+future|arbitrage|long.short|macro|hedged|merger)\b",
    re.IGNORECASE,
)

# Share class → load type inference
_FRONT_LOAD_PATTERNS  = re.compile(r"\b(class\s*a|series\s*a|cl\s*a|type\s*a)\b", re.IGNORECASE)
_BACK_LOAD_PATTERNS   = re.compile(r"\b(class\s*b|series\s*b|cl\s*b)\b",           re.IGNORECASE)
_LEVEL_LOAD_PATTERNS  = re.compile(r"\b(class\s*c|series\s*c|cl\s*c)\b",           re.IGNORECASE)
_NO_LOAD_PATTERNS     = re.compile(
    r"\b(institutional|instl|inst|class\s*i|cl\s*i|cl\s*z|class\s*z|class\s*r6|"
    r"admiral|investor|retail|no.load|no\s+load|class\s*n|cl\s*n|"
    r"select|premier|service)\b",
    re.IGNORECASE,
)


def _extract_fund_family(long_name: str | None) -> str | None:
    if not long_name:
        return None
    nl = long_name.lower()
    for key, canonical in _FUND_FAMILIES:
        if key in nl:
            return canonical
    return None


def _infer_category(long_name: str | None, summary: str | None = None) -> str | None:
    text = (long_name or "") + " " + (summary or "")
    if _FI_PATTERNS.search(text):
        return "Fixed Income"
    if _ALLOC_PATTERNS.search(text):
        return "Allocation"
    if _ALT_PATTERNS.search(text):
        return "Alternative"
    if re.search(r"\b(equity|stock|growth|value|blend|large.cap|mid.cap|small.cap|"
                 r"international|foreign|emerging|global|world)\b", text, re.IGNORECASE):
        return "Equity"
    return None


def _infer_load(long_name: str | None) -> tuple[float | None, float | None]:
    """Returns (front_load_flag, deferred_load_flag). 1.0 = has this load, 0.0 = no load."""
    if not long_name:
        return None, None
    if _NO_LOAD_PATTERNS.search(long_name):
        return 0.0, 0.0
    if _FRONT_LOAD_PATTERNS.search(long_name):
        return 1.0, 0.0   # Class A: front load, no CDSC after 1yr typically
    if _BACK_LOAD_PATTERNS.search(long_name):
        return 0.0, 1.0   # Class B: back-end CDSC
    if _LEVEL_LOAD_PATTERNS.search(long_name):
        return 0.0, 1.0   # Class C: level load / CDSC
    return None, None


def _infer_is_active(long_name: str | None) -> bool:
    if not long_name:
        return True
    return not bool(_PASSIVE_PATTERNS.search(long_name))


def _infer_asset_class(category: str | None, long_name: str | None) -> str:
    text = (category or "") + " " + (long_name or "")
    if _FI_PATTERNS.search(text):
        return "Fixed Income"
    if _ALLOC_PATTERNS.search(text):
        return "Allocation"
    if _ALT_PATTERNS.search(text):
        return "Alternative"
    return "Equity"


# ── Database setup ─────────────────────────────────────────────────────────────

def get_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS raw_tickers (
            symbol   VARCHAR PRIMARY KEY,
            rank_pos INTEGER,
            exchange VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ticker_info (
            symbol          VARCHAR PRIMARY KEY,
            long_name       VARCHAR,
            fund_family     VARCHAR,
            category        VARCHAR,
            total_assets    DOUBLE,
            inception_date  DATE,
            expense_ratio   DOUBLE,
            front_load      DOUBLE,
            deferred_load   DOUBLE,
            ms_overall      INTEGER,
            ms_risk         INTEGER,
            ms_return       INTEGER,
            legal_type      VARCHAR,
            currency        VARCHAR,
            fetched_at      TIMESTAMP DEFAULT now(),
            fetch_error     VARCHAR
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fund_universe (
            ticker              VARCHAR PRIMARY KEY,
            long_name           VARCHAR,
            fund_family         VARCHAR,
            category            VARCHAR,
            asset_class         VARCHAR,
            aum_millions        DOUBLE,
            inception_date      DATE,
            expense_ratio       DOUBLE,
            has_front_load      BOOLEAN,
            has_deferred_load   BOOLEAN,
            ms_overall          INTEGER,
            ms_risk             INTEGER,
            is_active           BOOLEAN,
            share_class_role    VARCHAR,
            updated_at          TIMESTAMP DEFAULT now()
        )
    """)
    return con


# ── Phase 1: Discover ─────────────────────────────────────────────────────────

def discover(con: duckdb.DuckDBPyConnection) -> None:
    max_rank = con.execute("SELECT COALESCE(MAX(rank_pos), 0) FROM raw_tickers").fetchone()[0]

    if max_rank >= MAX_RANK:
        n = con.execute("SELECT COUNT(*) FROM raw_tickers").fetchone()[0]
        print(f"raw_tickers already complete: {n:,} rows, max rank {max_rank:,}.")
        return

    start_off = (max_rank // PAGE_SIZE) * PAGE_SIZE
    if max_rank > 0:
        print(f"Resuming discover from offset {start_off:,} (max rank already saved: {max_rank:,})...")
    else:
        print(f"Paging Yahoo screener sorted by AUM desc, up to rank {MAX_RANK:,}...")

    batch_count = 0
    for offset in range(start_off, MAX_RANK, PAGE_SIZE):
        res    = yf.screen(SCREEN_QUERY, offset=offset, size=PAGE_SIZE,
                           sortField="fundnetassets", sortAsc=False)
        quotes = res.get("quotes", [])
        if not quotes:
            print(f"  Empty page at offset {offset} — end of screener.")
            break

        rows = [(q["symbol"], offset + i + 1, q.get("exchange", ""))
                for i, q in enumerate(quotes)]
        con.executemany("INSERT OR IGNORE INTO raw_tickers VALUES (?, ?, ?)", rows)
        batch_count += 1

        # Force WAL checkpoint every 10 pages (~2500 rows) so progress survives crashes
        if batch_count % 10 == 0:
            con.execute("CHECKPOINT")

        n_total = con.execute("SELECT COUNT(*) FROM raw_tickers").fetchone()[0]
        print(f"  offset {offset:5d}–{offset+len(quotes):5d}  db rows: {n_total:,}")

        if len(quotes) < PAGE_SIZE:
            print("  End of screener.")
            break
        time.sleep(CALL_DELAY)

    con.execute("CHECKPOINT")
    n_final = con.execute("SELECT COUNT(*) FROM raw_tickers").fetchone()[0]
    print(f"\nDiscover done: {n_final:,} tickers in raw_tickers.\n")


# ── Phase 2: Enrich ───────────────────────────────────────────────────────────

def enrich(con: duckdb.DuckDBPyConnection) -> None:
    pending = [r[0] for r in con.execute("""
        SELECT r.symbol FROM raw_tickers r
        LEFT JOIN ticker_info t ON r.symbol = t.symbol
        WHERE t.symbol IS NULL
        ORDER BY r.rank_pos
    """).fetchall()]

    if not pending:
        print("All tickers already enriched.")
        return

    print(f"Enriching {len(pending):,} tickers  (~{len(pending)*CALL_DELAY/60:.0f} min estimated)")
    print("Resumable: Ctrl+C or crash is safe — restart continues from here.\n")

    done = skipped = errors = 0
    t0 = time.time()

    for i, sym in enumerate(pending, 1):
        try:
            info  = yf.Ticker(sym).info
            ltype = (info.get("legalType") or "").lower()

            # Skip closed-end funds and ETFs
            if ltype and "open" not in ltype and "mutual" not in ltype:
                con.execute(
                    "INSERT OR IGNORE INTO ticker_info(symbol, legal_type, fetch_error) VALUES (?,?,?)",
                    [sym, info.get("legalType"), f"skip:{info.get('legalType')}"]
                )
                skipped += 1
                time.sleep(CALL_DELAY)
                continue

            aum = (info.get("totalAssets") or info.get("fundNetAssets")
                   or info.get("netAssets") or 0.0)

            inc_ts   = info.get("fundInceptionDate")
            inc_date = date.fromtimestamp(inc_ts) if inc_ts else None

            long_name = info.get("longName")
            summary   = info.get("longBusinessSummary")

            # Derive fields not directly in yfinance 1.4.1
            fund_family = _extract_fund_family(long_name)
            category    = _infer_category(long_name, summary)
            front, defer = _infer_load(long_name)

            # ER: annualReportExpenseRatio is decimal (0.0074); netExpenseRatio is pct (0.74)
            er = info.get("annualReportExpenseRatio")
            if er is None:
                net_er = info.get("netExpenseRatio")
                er = net_er / 100.0 if net_er else None

            con.execute("""
                INSERT OR IGNORE INTO ticker_info(
                    symbol, long_name, fund_family, category,
                    total_assets, inception_date, expense_ratio,
                    front_load, deferred_load,
                    ms_overall, ms_risk, ms_return,
                    legal_type, currency
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, [
                sym, long_name, fund_family, category,
                aum, inc_date, er,
                front, defer,
                info.get("morningStarOverallRating"),
                info.get("morningStarRiskRating"),
                info.get("morningStarReturnRating"),
                info.get("legalType"),
                info.get("currency"),
            ])
            done += 1

        except Exception as exc:
            con.execute(
                "INSERT OR IGNORE INTO ticker_info(symbol, fetch_error) VALUES (?,?)",
                [sym, str(exc)[:250]]
            )
            errors += 1

        if i % 200 == 0:
            con.execute("CHECKPOINT")
            elapsed   = time.time() - t0
            remaining = (len(pending) - i) / (i / elapsed) / 60
            print(f"  [{i:5d}/{len(pending):5d}]  ok={done}  skip={skipped}  err={errors}"
                  f"  ~{remaining:.0f} min left")

        time.sleep(CALL_DELAY)

    con.execute("CHECKPOINT")
    print(f"\nEnrich done: {done} ok, {skipped} skipped, {errors} errors.\n")


# ── Phase 3: Select ───────────────────────────────────────────────────────────

def select(con: duckdb.DuckDBPyConnection) -> None:
    df = con.execute("""
        SELECT symbol, long_name, fund_family, category,
               total_assets, inception_date, expense_ratio,
               front_load, deferred_load,
               ms_overall, ms_risk, legal_type
        FROM ticker_info
        WHERE fetch_error IS NULL
          AND total_assets >= ?
          AND long_name IS NOT NULL
        ORDER BY total_assets DESC
    """, [MIN_AUM]).df()

    print(f"Valid tickers above ${MIN_AUM/1e6:.0f}M AUM: {len(df):,}")
    if df.empty:
        print("Nothing to select — check enrich phase.")
        return

    today = date.today()

    # For tickers where enrich returned NULL fund_family (older rows), derive now
    mask = df["fund_family"].isna()
    df.loc[mask, "fund_family"] = df.loc[mask, "long_name"].apply(_extract_fund_family)

    mask2 = df["category"].isna()
    df.loc[mask2, "category"] = df.loc[mask2, "long_name"].apply(
        lambda n: _infer_category(n, None)
    )

    # Re-derive load flags from name for any nulls
    def _front(row):
        if pd.notna(row["front_load"]):
            return row["front_load"]
        f, _ = _infer_load(row["long_name"])
        return f

    def _defer(row):
        if pd.notna(row["deferred_load"]):
            return row["deferred_load"]
        _, d = _infer_load(row["long_name"])
        return d

    df["front_load"]    = df.apply(_front, axis=1)
    df["deferred_load"] = df.apply(_defer, axis=1)

    # Cost score: ER + load penalty (lower = better for investors)
    df["er"]         = df["expense_ratio"].fillna(0.02)
    df["load_total"] = df["front_load"].fillna(0) + df["deferred_load"].fillna(0)
    df["cost_score"] = df["er"] + df["load_total"] * 0.20   # load amortised over 5yr

    # History: days since inception
    df["history_days"] = df["inception_date"].apply(
        lambda d: (today - d.date()).days if pd.notna(d) and d is not None else 0
    )

    # Deduplication key: round AUM to nearest $10M — all share classes of the same
    # fund report identical total AUM, so this clusters them reliably.
    df["aum_bucket"] = (df["total_assets"] / 10e6).round().astype(int)

    # Sort so within each AUM bucket, oldest-inception rows come first, then cheapest
    df = df.sort_values(["aum_bucket", "history_days", "cost_score"],
                        ascending=[True, False, True])

    records: list[dict] = []

    for bucket, grp in df.groupby("aum_bucket", sort=False):
        grp = grp.reset_index(drop=True)
        aum  = grp.iloc[0]["total_assets"]
        cat  = grp.iloc[0]["category"]
        fam  = grp.iloc[0]["fund_family"]

        def make_row(row: pd.Series, role: str) -> dict:
            fl = row["front_load"]
            dl = row["deferred_load"]
            return {
                "ticker":             row["symbol"],
                "long_name":          row["long_name"],
                "fund_family":        fam,
                "category":           cat,
                "asset_class":        _infer_asset_class(cat, row["long_name"]),
                "aum_millions":       round(aum / 1e6, 1),
                "inception_date":     row["inception_date"] if pd.notna(row["inception_date"]) else None,
                "expense_ratio":      float(row["expense_ratio"]) if pd.notna(row["expense_ratio"]) else None,
                "has_front_load":     bool(fl > 0) if pd.notna(fl) else None,
                "has_deferred_load":  bool(dl > 0) if pd.notna(dl) else None,
                "ms_overall":         int(row["ms_overall"]) if pd.notna(row["ms_overall"]) else None,
                "ms_risk":            int(row["ms_risk"])    if pd.notna(row["ms_risk"])     else None,
                "is_active":          _infer_is_active(row["long_name"]),
                "share_class_role":   role,
            }

        primary = grp.iloc[0]
        records.append(make_row(primary, "primary"))

        # Secondary: meaningfully cheaper share class with ≥5yr history
        if len(grp) > 1:
            rest    = grp.iloc[1:]
            cheaper = rest[
                (rest["cost_score"] < primary["cost_score"] * 0.80) &
                (rest["history_days"] >= 365 * 5)
            ]
            if not cheaper.empty:
                records.append(make_row(cheaper.iloc[0], "secondary"))

    result = pd.DataFrame(records)
    n_primary   = (result["share_class_role"] == "primary").sum()
    n_secondary = (result["share_class_role"] == "secondary").sum()
    n_active    = result[
        (result["share_class_role"] == "primary") & result["is_active"]
    ].shape[0]

    print(f"Selected {n_primary:,} primary + {n_secondary:,} secondary share classes")
    print(f"  Active: {n_active:,} / {n_primary:,}  |  Passive: {n_primary - n_active:,}")

    # Write to fund_universe (drop and recreate to ensure schema matches)
    con.execute("DROP TABLE IF EXISTS fund_universe")
    con.execute("""
        CREATE TABLE fund_universe (
            ticker              VARCHAR PRIMARY KEY,
            long_name           VARCHAR,
            fund_family         VARCHAR,
            category            VARCHAR,
            asset_class         VARCHAR,
            aum_millions        DOUBLE,
            inception_date      DATE,
            expense_ratio       DOUBLE,
            has_front_load      BOOLEAN,
            has_deferred_load   BOOLEAN,
            ms_overall          INTEGER,
            ms_risk             INTEGER,
            is_active           BOOLEAN,
            share_class_role    VARCHAR,
            updated_at          TIMESTAMP DEFAULT now()
        )
    """)
    con.register("_sel", result)
    con.execute("""
        INSERT INTO fund_universe
        SELECT ticker, long_name, fund_family, category, asset_class,
               aum_millions, inception_date, expense_ratio,
               has_front_load, has_deferred_load,
               ms_overall, ms_risk, is_active, share_class_role, now()
        FROM _sel
    """)
    con.execute("CHECKPOINT")
    print(f"Written {len(result):,} rows to fund_universe in {DB_PATH.name}\n")

    print("Breakdown by asset class (primary share classes only):")
    print(con.execute("""
        SELECT asset_class,
               COUNT(*)                                       AS funds,
               SUM(CASE WHEN is_active THEN 1 ELSE 0 END)    AS active,
               ROUND(AVG(aum_millions), 0)                    AS avg_aum_m,
               ROUND(AVG(expense_ratio)*100, 2)               AS avg_er_pct,
               SUM(CASE WHEN has_front_load THEN 1 ELSE 0 END) AS front_load_n
        FROM fund_universe
        WHERE share_class_role = 'primary'
        GROUP BY asset_class ORDER BY funds DESC
    """).df().to_string(index=False))


# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(con: duckdb.DuckDBPyConnection) -> None:
    n = con.execute("SELECT COUNT(*) FROM fund_universe").fetchone()[0]
    if n == 0:
        print("fund_universe is empty — run select phase first.")
        return
    print(f"fund_universe: {n:,} rows\n")
    print(con.execute("""
        SELECT share_class_role, asset_class, COUNT(*) AS n,
               MIN(CAST(inception_date AS VARCHAR)) AS oldest,
               ROUND(MIN(aum_millions),0)  AS min_aum_m,
               ROUND(MAX(aum_millions),0)  AS max_aum_m,
               ROUND(AVG(expense_ratio)*100, 2) AS avg_er_pct
        FROM fund_universe
        GROUP BY share_class_role, asset_class
        ORDER BY share_class_role, n DESC
    """).df().to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["discover", "enrich", "select", "all", "stats"])
    args = parser.parse_args()

    con = get_db()
    try:
        if args.phase in ("discover", "all"):
            discover(con)
        if args.phase in ("enrich", "all"):
            enrich(con)
        if args.phase in ("select", "all"):
            select(con)
        if args.phase == "stats":
            stats(con)
    finally:
        con.execute("CHECKPOINT")
        con.close()


if __name__ == "__main__":
    main()
