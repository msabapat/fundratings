import duckdb
from pathlib import Path

DB = Path(__file__).parent / "fund_universe.duckdb"
con = duckdb.connect(str(DB))

print("=== raw_tickers ===")
print(con.execute("SELECT COUNT(*) FROM raw_tickers").fetchone()[0], "total")
print(con.execute("SELECT MAX(rank_pos) FROM raw_tickers").fetchone()[0], "max rank")

print()
print("=== ticker_info ===")
print(con.execute("SELECT COUNT(*) FROM ticker_info").fetchone()[0], "total")
print(con.execute("SELECT COUNT(*) FROM ticker_info WHERE fetch_error IS NULL").fetchone()[0], "no error")
print(con.execute("SELECT COUNT(*) FROM ticker_info WHERE total_assets >= 250e6").fetchone()[0], "above $250M")
print(con.execute("SELECT COUNT(*) FROM ticker_info WHERE fund_family IS NOT NULL").fetchone()[0], "with fund_family")
print(con.execute("SELECT COUNT(*) FROM ticker_info WHERE category IS NOT NULL").fetchone()[0], "with category")

print()
print("=== Remaining to enrich ===")
pending = con.execute("""
    SELECT COUNT(*) FROM raw_tickers r
    LEFT JOIN ticker_info t ON r.symbol = t.symbol
    WHERE t.symbol IS NULL
""").fetchone()[0]
print(pending, "tickers not yet in ticker_info")

print()
print("=== AUM distribution (enriched, no error) ===")
print(con.execute("""
    SELECT
        SUM(CASE WHEN total_assets IS NULL OR total_assets = 0 THEN 1 ELSE 0 END) as null_zero,
        SUM(CASE WHEN total_assets > 0 AND total_assets < 250e6  THEN 1 ELSE 0 END) as under_250m,
        SUM(CASE WHEN total_assets >= 250e6 AND total_assets < 1e9 THEN 1 ELSE 0 END) as btwn_250m_1b,
        SUM(CASE WHEN total_assets >= 1e9 THEN 1 ELSE 0 END) as above_1b
    FROM ticker_info WHERE fetch_error IS NULL
""").df().to_string(index=False))

print()
print("=== Sample with longName to verify fund_family extraction ===")
print(con.execute("""
    SELECT symbol, long_name, total_assets, inception_date, expense_ratio
    FROM ticker_info WHERE fetch_error IS NULL AND total_assets >= 250e6
    ORDER BY total_assets DESC LIMIT 15
""").df().to_string(index=False))

con.close()
