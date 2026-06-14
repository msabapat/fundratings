# fund_replication/run.py
# Analyse one or more active funds against the passive ETF universe.
#
# Usage examples:
#   py run.py                              # all configured funds, full report
#   py run.py --funds FCNTX OAKMX         # specific funds only
#   py run.py --funds ARKK --oos 2020-01-01
#   py run.py --funds FCNTX --rolling     # add rolling window chart
#   py run.py --lasso-only                # LASSO ETF selection only

import argparse
import warnings
import logging
from pathlib import Path

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Dropping ETFs")

HERE = Path(__file__).parent

import sys
sys.path.insert(0, str(HERE))

import config as cfg
from data     import load_etf_returns, load_fund_returns, align
from analysis import full_is_regression, is_oos_split, rolling_replication, lasso_selection


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ann_ret(monthly_series: pd.Series) -> float:
    return (1 + monthly_series.mean()) ** 12 - 1

def _fmt_pct(v) -> str:
    return f"{v*100:+.1f}%" if not (isinstance(v, float) and np.isnan(v)) else "  n/a"

def _fmt_pos(v) -> str:
    return f"{v*100:.1f}%" if not (isinstance(v, float) and np.isnan(v)) else "n/a"

def _divider(width=72) -> str:
    return "-" * width


def print_weights(weights: pd.Series, top_n: int = 10) -> None:
    w = weights[weights > 0.005].head(top_n)
    for ticker, wt in w.items():
        bar = "#" * int(wt * 40)
        desc = cfg.PASSIVE_ETFS.get(ticker, cfg.ACTIVE_ETFS.get(ticker, ""))
        short_desc = desc.split("(")[0].strip()[:28]
        print(f"  {ticker:<6} {wt*100:5.1f}%  {bar:<16}  {short_desc}")


def print_fund_report(
    ticker:     str,
    fund_ret:   pd.Series,
    etf_ret:    pd.DataFrame,
    oos_start:  str,
    do_rolling: bool,
    do_lasso:   bool,
) -> None:
    # Metadata
    if ticker in cfg.ACTIVE_MUTUAL_FUNDS:
        meta      = cfg.ACTIVE_MUTUAL_FUNDS[ticker]
        name      = meta.get("name", ticker)
        er        = meta.get("er")
        n_stars   = meta.get("stars")
        stars_str = ("*" * n_stars) if n_stars else "n/a"
        src       = "yfinance (mutual fund)"
    else:
        name      = cfg.ACTIVE_ETFS.get(ticker, ticker)
        er        = 0.0075
        stars_str = "see Morningstar"
        src       = "DuckDB (active ETF)"

    er_str = f"{er*100:.2f}%" if er is not None else "n/a"
    print()
    print(_divider())
    print(f"  {ticker}  —  {name}")
    print(f"  ER: {er_str}   Morningstar: {stars_str}   Source: {src}")
    print(f"  Period: {fund_ret.index[0].date()} to {fund_ret.index[-1].date()}"
          f"  ({len(fund_ret)} months)")
    ann = _ann_ret(fund_ret)
    vol = fund_ret.std() * np.sqrt(12)
    print(f"  Fund total return (ann): {_fmt_pos(ann)}   Volatility: {_fmt_pos(vol)}")
    print(_divider())

    # ── 1. Full IS regression ─────────────────────────────────────────────────
    print("\n[1] Full In-Sample Regression (RBSA)")
    res = full_is_regression(fund_ret, etf_ret)
    print(f"    R²: {res['r_squared']:.3f}   "
          f"Ann return — Fund: {_fmt_pos(res['ann_fund_ret'])}  "
          f"Replica: {_fmt_pos(res['ann_replica_ret'])}  "
          f"Diff: {_fmt_pct(res['ret_diff'])}")
    print(f"    Tracking error: {_fmt_pos(res['tracking_error'])}  "
          f"Info ratio: {res['info_ratio']:.2f}")
    print("    Top ETF weights:")
    print_weights(res["weights"])

    # ── 2. IS / OOS split ─────────────────────────────────────────────────────
    # Auto-shift OOS date if fund started after it (need >=24 IS months)
    effective_oos = oos_start
    if fund_ret.index[0] >= pd.Timestamp(oos_start):
        split_idx = len(fund_ret) // 2
        effective_oos = str(fund_ret.index[split_idx].date())
    print(f"\n[2] IS / OOS Split  (train < {effective_oos}, test >= {effective_oos})")
    split = is_oos_split(fund_ret, etf_ret, effective_oos)
    if not split:
        print("    Insufficient data for IS/OOS split.")
    else:
        im, om = split["is_metrics"], split["oos_metrics"]
        print(f"    {'':12}  {'Ann Ret':>8}  {'Replica':>8}  {'Diff':>7}  {'Tkg Err':>7}  {'R²':>6}")
        print(f"    {'IS  (%d mo)' % split['n_is_months']:<12}  "
              f"{_fmt_pos(im['ann_fund_ret']):>8}  "
              f"{_fmt_pos(im['ann_replica_ret']):>8}  "
              f"{_fmt_pct(im['ret_diff']):>7}  "
              f"{_fmt_pos(im['tracking_error']):>7}  "
              f"{im['r_squared']:>6.3f}")
        print(f"    {'OOS (%d mo)' % split['n_oos_months']:<12}  "
              f"{_fmt_pos(om['ann_fund_ret']):>8}  "
              f"{_fmt_pos(om['ann_replica_ret']):>8}  "
              f"{_fmt_pct(om['ret_diff']):>7}  "
              f"{_fmt_pos(om['tracking_error']):>7}  "
              f"{om['r_squared']:>6.3f}")
        print("    OOS weights (trained on IS data):")
        print_weights(split["weights"])

    # ── 3. LASSO sparse selection ─────────────────────────────────────────────
    if do_lasso:
        print(f"\n[3] LASSO Sparse ETF Selection  (alpha CV={cfg.LASSO_N_ALPHAS})")
        lass = lasso_selection(fund_ret, etf_ret, n_alphas=cfg.LASSO_N_ALPHAS)
        print(f"    Selected {lass['n_selected']} ETFs (alpha={lass['best_alpha']:.6f})")
        print(f"    R²: {lass['r_squared']:.3f}   "
              f"Ann return — Fund: {_fmt_pos(lass['ann_fund_ret'])}  "
              f"Replica: {_fmt_pos(lass['ann_replica_ret'])}  "
              f"Diff: {_fmt_pct(lass['ret_diff'])}")
        print(f"    Tracking error: {_fmt_pos(lass['tracking_error'])}  "
              f"Info ratio: {lass['info_ratio']:.2f}")
        print("    LASSO-selected weights:")
        print_weights(lass["weights"])

    # ── 4. Rolling window ─────────────────────────────────────────────────────
    if do_rolling:
        print(f"\n[4] Rolling {cfg.TRAIN_MONTHS}m window, "
              f"rebalance every {cfg.REBAL_MONTHS}m")
        try:
            roll_result = rolling_replication(
                fund_ret, etf_ret,
                train_months=cfg.TRAIN_MONTHS,
                rebal_months=cfg.REBAL_MONTHS,
            )
            roll      = roll_result["returns"]
            roll_fund = roll["fund"]
            roll_rep  = roll["replica"]
            te   = (roll_fund - roll_rep).std() * np.sqrt(12)
            ir   = ((roll_fund - roll_rep).mean() / (roll_fund - roll_rep).std()
                    * np.sqrt(12))
            print(f"    OOS period: {roll.index[0].date()} to {roll.index[-1].date()}"
                  f"  ({len(roll)} months)")
            print(f"    Ann return — Fund: {_fmt_pos(_ann_ret(roll_fund))}  "
                  f"Replica: {_fmt_pos(_ann_ret(roll_rep))}  "
                  f"Diff: {_fmt_pct(_ann_ret(roll_fund) - _ann_ret(roll_rep))}")
            print(f"    Tracking error: {_fmt_pos(te)}   Info ratio: {ir:.2f}")

            # Cumulative wealth
            cum_fund = (1 + roll_fund).cumprod()
            cum_rep  = (1 + roll_rep).cumprod()
            print(f"    Cumulative wealth — Fund: {cum_fund.iloc[-1]:.2f}x  "
                  f"Replica: {cum_rep.iloc[-1]:.2f}x")
        except ValueError as e:
            print(f"    Skipped: {e}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fund replication analysis")
    parser.add_argument("--funds",      nargs="+", default=None,
                        help="Ticker(s) to analyse (default: all configured)")
    parser.add_argument("--start",      default=cfg.DEFAULT_START)
    parser.add_argument("--end",        default=cfg.DEFAULT_END)
    parser.add_argument("--oos",        default=cfg.OOS_START,
                        help="IS/OOS split date (default %(default)s)")
    parser.add_argument("--rolling",    action="store_true",
                        help="Include rolling-window replication")
    parser.add_argument("--no-lasso",   action="store_true",
                        help="Skip LASSO sparse selection")
    parser.add_argument("--train-months", type=int, default=cfg.TRAIN_MONTHS)
    args = parser.parse_args()

    # Resolve fund list
    all_active_etfs = list(cfg.ACTIVE_ETFS.keys())
    all_mf          = list(cfg.ACTIVE_MUTUAL_FUNDS.keys())

    if args.funds:
        requested = [t.upper() for t in args.funds]
        etf_funds = [t for t in requested if t in all_active_etfs]
        mf_funds  = [t for t in requested if t in all_mf]
        unknown   = [t for t in requested if t not in all_active_etfs and t not in all_mf]
        if unknown:
            print(f"Warning: unknown tickers skipped: {unknown}")
    else:
        etf_funds = all_active_etfs
        mf_funds  = all_mf

    # Load passive ETF returns (shared benchmark universe)
    print(f"\nLoading {len(cfg.PASSIVE_ETFS)} passive ETF returns from DuckDB ...")
    etf_returns = load_etf_returns(
        list(cfg.PASSIVE_ETFS.keys()), start=args.start, end=args.end
    )
    print(f"  Loaded {etf_returns.shape[1]} ETFs × {etf_returns.shape[0]} months")

    # ── Active ETFs from DuckDB ───────────────────────────────────────────────
    if etf_funds:
        print(f"\nLoading {len(etf_funds)} active ETF returns from DuckDB ...")
        active_etf_returns = load_etf_returns(etf_funds, start=args.start, end=args.end)

        for ticker in etf_funds:
            if ticker not in active_etf_returns.columns:
                print(f"  {ticker}: not found in DB — skipping")
                continue
            fund_ret_raw  = active_etf_returns[ticker].dropna()
            fund_ret, etf_aligned = align(fund_ret_raw, etf_returns)
            print_fund_report(
                ticker, fund_ret, etf_aligned, args.oos,
                do_rolling=args.rolling, do_lasso=not args.no_lasso,
            )

    # ── Mutual funds from yfinance ─────────────────────────────────────────────
    if mf_funds:
        print(f"\nLoading {len(mf_funds)} mutual fund returns from yfinance ...")
        mf_returns = load_fund_returns(mf_funds, start=args.start, end=args.end)
        print(f"  Loaded {mf_returns.shape[1]} funds × {mf_returns.shape[0]} months")

        for ticker in mf_funds:
            if ticker not in mf_returns.columns:
                print(f"  {ticker}: not found in yfinance data — skipping")
                continue
            fund_ret_raw  = mf_returns[ticker].dropna()
            fund_ret, etf_aligned = align(fund_ret_raw, etf_returns)
            print_fund_report(
                ticker, fund_ret, etf_aligned, args.oos,
                do_rolling=args.rolling, do_lasso=not args.no_lasso,
            )

    print("Done.")


if __name__ == "__main__":
    main()
