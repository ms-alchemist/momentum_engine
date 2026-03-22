"""
backtest/scan_stops_htc.py
==========================
Scans stop distances for the MGC hold-to-close strategy.
Tests stops from 10pts to 40pts to find the optimal level.
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

STARTING_BALANCE = 100_000.0
MLL_BUFFER       =   3_000.0
VOL_FILTER_RATIO = 2.5
MIN_HISTORY_DAYS = 20

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

def run(df, daily, stop_pts, n_contracts=1):
    balance  = STARTING_BALANCE
    peak_eod = STARTING_BALANCE
    floor    = STARTING_BALANCE - MLL_BUFFER
    trades   = []

    for day_num, day in enumerate(sorted(set(df.index.date))):
        if balance < floor:
            break

        if day_num < MIN_HISTORY_DAYS:
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
        if fb["close"] > fb["open"]:     direction = 1
        elif fb["close"] < fb["open"]:   direction = -1
        else:                             continue

        entry      = fb["close"]
        stop_price = entry - direction * stop_pts
        exit_price = None
        exit_reason= None

        for _, bar in day_bars[day_bars.index.time >= FIRST_BAR_CLOSE].iterrows():
            if bar.name.time() >= SESSION_CLOSE:
                exit_price, exit_reason = bar["open"], "session_close"
                break
            if direction == 1 and bar["low"] <= stop_price:
                exit_price, exit_reason = stop_price, "stop"
                break
            if direction == -1 and bar["high"] >= stop_price:
                exit_price, exit_reason = stop_price, "stop"
                break

        if exit_price is None:
            exit_price, exit_reason = day_bars.iloc[-1]["close"], "eod"

        pnl = direction * (exit_price - entry) * MGC_PT * n_contracts - MGC_COMM * n_contracts
        balance += pnl

        # Update MLL floor
        if balance > peak_eod:
            peak_eod = balance
        floor = min(peak_eod - MLL_BUFFER, STARTING_BALANCE)

        trades.append({
            "pnl": pnl, "win": pnl > 0,
            "exit_reason": exit_reason,
        })

    if not trades:
        return None

    df_t   = pd.DataFrame(trades)
    wins   = df_t[df_t["win"]]
    losses = df_t[~df_t["win"]]
    n      = len(df_t)
    stops  = df_t[df_t["exit_reason"] == "stop"]
    closes = df_t[df_t["exit_reason"].isin(["session_close","eod"])]

    return {
        "stop_pts":       stop_pts,
        "trades":         n,
        "win_rate":       df_t["win"].mean(),
        "total_pnl":      df_t["pnl"].sum(),
        "profit_factor":  (wins["pnl"].sum() / abs(losses["pnl"].sum())
                           if len(losses) and losses["pnl"].sum() != 0 else np.inf),
        "stop_pct":       len(stops) / n,
        "sc_pct":         len(closes) / n,
        "sc_wr":          closes["win"].mean() if len(closes) else 0,
        "mll_safe":       balance >= floor,
        "final_balance":  balance,
    }

def main():
    df    = load()
    daily = build_daily(df)

    stop_levels = [10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40]

    print()
    print("MGC Hold-To-Close — Stop Distance Scan (1 contract)")
    print("=" * 80)
    print(f"{'Stop':>6} {'Trades':>7} {'WinRate':>8} {'ProfFact':>9} "
          f"{'TotalPnL':>10} {'StopPct':>8} {'SC_WR':>7} {'Safe':>6}")
    print("-" * 80)

    results = []
    for stop in stop_levels:
        r = run(df, daily, stop)
        if r:
            results.append(r)
            safe_str = "YES" if r["mll_safe"] else "NO"
            pf_str   = f"{r['profit_factor']:.2f}" if r["profit_factor"] != np.inf else "inf"
            print(f"  {stop:>4}pt  {r['trades']:>6}  {r['win_rate']:>7.1%}  "
                  f"{pf_str:>9}  ${r['total_pnl']:>8,.0f}  "
                  f"{r['stop_pct']:>7.1%}  {r['sc_wr']:>6.1%}  {safe_str:>5}")

    print()
    safe = [r for r in results if r["mll_safe"]]
    if safe:
        best = max(safe, key=lambda r: r["total_pnl"])
        print(f"Best safe stop: {best['stop_pts']}pt  "
              f"P&L=${best['total_pnl']:,.0f}  "
              f"WR={best['win_rate']:.1%}  "
              f"PF={best['profit_factor']:.2f}")

    # Also show what happens with no stop (pure session close)
    print()
    print("No stop (session close only — for reference):")
    r_ns = run(df, daily, stop_pts=9999)
    if r_ns:
        print(f"  Trades={r_ns['trades']}  WR={r_ns['win_rate']:.1%}  "
              f"PnL=${r_ns['total_pnl']:,.0f}  "
              f"SC_WR={r_ns['sc_wr']:.1%}  "
              f"Safe={'YES' if r_ns['mll_safe'] else 'NO'}")
    print()

if __name__ == "__main__":
    main()
