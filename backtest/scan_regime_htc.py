"""
backtest/scan_regime_htc.py
============================
Tests whether a prior-month range filter improves the HTC strategy.
Bad months (May, Nov, Dec) cost ~$3,500. If we can identify and skip them
using only information available at the time, the strategy becomes viable.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time
import warnings
warnings.filterwarnings("ignore")

DATA_DIR  = Path("data/cache")
MGC_FILE  = DATA_DIR / "MGC_30min.parquet"
MGC_PT    = 10.0
MGC_COMM  = 0.80

FIRST_BAR_OPEN  = time(9, 0)
FIRST_BAR_CLOSE = time(9, 30)
SESSION_CLOSE   = time(16, 15)
STOP_POINTS     = 22.0

STARTING_BALANCE = 100_000.0
MLL_BUFFER       =   3_000.0
VOL_FILTER_RATIO = 2.5
MIN_HISTORY_DAYS = 20
N_CONTRACTS      = 2


def load():
    df = pd.read_parquet(MGC_FILE)
    df.index = pd.to_datetime(df.index, utc=False)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def build_daily(df):
    d = df.resample("1D").agg({"high":"max","low":"min"}).dropna()
    d["range"]     = d["high"] - d["low"]
    d["range_5d"]  = d["range"].rolling(5).mean()
    d["range_20d"] = d["range"].rolling(20).mean()
    d["vol_ratio"] = d["range_5d"] / d["range_20d"]
    return d


def build_monthly_range(df):
    """Prior month average daily range — available at start of each month."""
    daily_range = df.resample("1D").agg({"high":"max","low":"min"}).dropna()
    daily_range["range"] = daily_range["high"] - daily_range["low"]
    monthly = daily_range["range"].resample("ME").mean()
    monthly.index = monthly.index.to_period("M")
    return monthly


def run(df, daily, monthly_range, min_prior_range=None, max_prior_range=None):
    balance  = STARTING_BALANCE
    peak_eod = STARTING_BALANCE
    floor    = STARTING_BALANCE - MLL_BUFFER
    trades   = []

    for day_num, day in enumerate(sorted(set(df.index.date))):
        if balance < floor:
            break

        if day_num < MIN_HISTORY_DAYS:
            continue

        # Prior month range filter
        if min_prior_range is not None or max_prior_range is not None:
            current_month = pd.Period(day, freq="M")
            prior_month   = current_month - 1
            if prior_month in monthly_range.index:
                prior_rng = monthly_range[prior_month]
                if min_prior_range and prior_rng < min_prior_range:
                    continue
                if max_prior_range and prior_rng > max_prior_range:
                    continue

        mask = daily.index.date == day
        if not mask.any():
            continue
        info = daily[mask].iloc[0]
        if not np.isnan(info["vol_ratio"]) and info["vol_ratio"] > VOL_FILTER_RATIO:
            continue

        day_bars = df[df.index.date == day]
        if len(day_bars) < 2:
            continue

        fb = day_bars[
            (day_bars.index.time >= FIRST_BAR_OPEN) &
            (day_bars.index.time < FIRST_BAR_CLOSE)
        ]
        if len(fb) == 0:
            continue

        fb = fb.iloc[0]
        if fb["close"] > fb["open"]:     d = 1
        elif fb["close"] < fb["open"]:   d = -1
        else:                             continue

        entry      = fb["close"]
        stop_price = entry - d * STOP_POINTS
        exit_price = None
        exit_reason= None

        for _, bar in day_bars[day_bars.index.time >= FIRST_BAR_CLOSE].iterrows():
            if bar.name.time() >= SESSION_CLOSE:
                exit_price, exit_reason = bar["open"], "session_close"
                break
            if d == 1 and bar["low"] <= stop_price:
                exit_price, exit_reason = stop_price, "stop"
                break
            if d == -1 and bar["high"] >= stop_price:
                exit_price, exit_reason = stop_price, "stop"
                break

        if exit_price is None:
            exit_price, exit_reason = day_bars.iloc[-1]["close"], "eod"

        pnl = d * (exit_price - entry) * MGC_PT * N_CONTRACTS - MGC_COMM * N_CONTRACTS
        balance += pnl
        if balance > peak_eod:
            peak_eod = balance
        floor = min(peak_eod - MLL_BUFFER, STARTING_BALANCE)

        trades.append({
            "date": day,
            "pnl": pnl,
            "win": pnl > 0,
            "exit_reason": exit_reason,
        })

    if not trades:
        return None

    df_t   = pd.DataFrame(trades)
    n      = len(df_t)
    wins   = df_t[df_t["win"]]
    losses = df_t[~df_t["win"]]

    return {
        "trades":   n,
        "win_rate": df_t["win"].mean(),
        "total_pnl":df_t["pnl"].sum(),
        "pf":       (wins["pnl"].sum() / abs(losses["pnl"].sum())
                     if len(losses) and losses["pnl"].sum() != 0 else np.inf),
        "mll_safe": balance >= floor,
        "final_bal":balance,
        "skipped":  0,
    }


def main():
    df            = load()
    daily         = build_daily(df)
    monthly_range = build_monthly_range(df)

    print()
    print("Prior-month range filter test — 2 MGC contracts, 22pt stop")
    print("=" * 65)

    # Show monthly range context
    print("\nPrior month ranges:")
    import warnings; warnings.filterwarnings("ignore")
    monthly_range_copy = monthly_range.copy()
    for period, rng in monthly_range_copy.items():
        print(f"  {str(period):<8}  avg daily range = {rng:.1f} pts")

    print()
    print("Baseline (no monthly filter):")
    r = run(df, daily, monthly_range)
    if r:
        print(f"  Trades={r['trades']}  WR={r['win_rate']:.1%}  "
              f"PnL=${r['total_pnl']:,.0f}  PF={r['pf']:.2f}  "
              f"Safe={'YES' if r['mll_safe'] else 'NO'}")

    print()
    print("Prior-month minimum range filter (only trade if prior month was active):")
    print(f"{'MinRange':>10} {'Trades':>8} {'WinRate':>9} {'PnL':>11} "
          f"{'PF':>7} {'Safe':>6}")
    print("-" * 55)

    for min_rng in [0, 15, 20, 25, 28, 30, 32, 35, 40]:
        r = run(df, daily, monthly_range, min_prior_range=min_rng)
        if r:
            label = "baseline" if min_rng == 0 else f">= {min_rng}pt"
            safe  = "YES" if r["mll_safe"] else "NO"
            pf    = f"{r['pf']:.2f}" if r["pf"] != np.inf else "inf"
            print(f"  {label:>10}  {r['trades']:>6}  {r['win_rate']:>8.1%}  "
                  f"${r['total_pnl']:>9,.0f}  {pf:>6}  {safe:>5}")

    print()
    print("Prior-month maximum range filter (skip extreme vol months):")
    print(f"{'MaxRange':>10} {'Trades':>8} {'WinRate':>9} {'PnL':>11} "
          f"{'PF':>7} {'Safe':>6}")
    print("-" * 55)

    for max_rng in [9999, 80, 70, 60, 55, 50]:
        r = run(df, daily, monthly_range, max_prior_range=max_rng)
        if r:
            label = "baseline" if max_rng == 9999 else f"<= {max_rng}pt"
            safe  = "YES" if r["mll_safe"] else "NO"
            pf    = f"{r['pf']:.2f}" if r["pf"] != np.inf else "inf"
            print(f"  {label:>10}  {r['trades']:>6}  {r['win_rate']:>8.1%}  "
                  f"${r['total_pnl']:>9,.0f}  {pf:>6}  {safe:>5}")

    print()
    print("Combined filter (prior month range between min and max):")
    print(f"{'Filter':>16} {'Trades':>8} {'WinRate':>9} {'PnL':>11} "
          f"{'PF':>7} {'Safe':>6}")
    print("-" * 62)

    for min_rng, max_rng in [(25, 70), (25, 60), (28, 70), (28, 60),
                              (20, 70), (20, 60), (30, 70), (30, 60)]:
        r = run(df, daily, monthly_range,
                min_prior_range=min_rng, max_prior_range=max_rng)
        if r:
            label = f"{min_rng}-{max_rng}pt"
            safe  = "YES" if r["mll_safe"] else "NO"
            pf    = f"{r['pf']:.2f}" if r["pf"] != np.inf else "inf"
            print(f"  {label:>16}  {r['trades']:>6}  {r['win_rate']:>8.1%}  "
                  f"${r['total_pnl']:>9,.0f}  {pf:>6}  {safe:>5}")

    print()


if __name__ == "__main__":
    main()
