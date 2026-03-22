"""
backtest/debug_loop.py
Traces exactly what happens in the first 30 days of the backtest.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time

df = pd.read_parquet("data/cache/MGC_30min.parquet")
df.index = pd.to_datetime(df.index)

daily = df.resample("1D").agg({"high":"max","low":"min","open":"first","close":"last"}).dropna()
daily["range"]     = daily["high"] - daily["low"]
daily["range_20d"] = daily["range"].rolling(20).mean()
daily["vol_ratio"] = daily["range"].rolling(5).mean() / daily["range_20d"]

SESSION_OPEN_ET  = time(9, 0)
SIGNAL_BAR_ET    = time(9, 30)
SESSION_CLOSE_ET = time(16, 30)

STARTING_BALANCE   = 100_000.0
MLL_BUFFER         =   3_000.0
MLL_LOCK_THRESHOLD = 100_000.0

balance   = STARTING_BALANCE
peak_eod  = STARTING_BALANCE
floor     = STARTING_BALANCE - MLL_BUFFER
locked    = False
trade_count = 0

print("Tracing first 30 trading days...\n")

for day in sorted(set(df.index.date))[:30]:
    # MLL check
    is_breached = balance < floor
    if is_breached:
        print(f"  {day}: MLL BREACHED — balance={balance:.0f} floor={floor:.0f} STOPPING")
        break

    day_mask = daily.index.date == day
    if not day_mask.any():
        print(f"  {day}: SKIP — no daily info")
        continue

    info = daily[day_mask].iloc[0]

    if np.isnan(info["range_20d"]):
        print(f"  {day}: SKIP — no 20d range history yet")
        # update eod with no change
        continue

    vol_ratio = info["vol_ratio"]
    if not np.isnan(vol_ratio) and vol_ratio > 2.5:
        print(f"  {day}: SKIP — vol filter (ratio={vol_ratio:.2f})")
        continue

    day_bars = df[df.index.date == day]
    first = day_bars[
        (day_bars.index.time >= SESSION_OPEN_ET) &
        (day_bars.index.time < SIGNAL_BAR_ET)
    ]
    if len(first) == 0:
        print(f"  {day}: SKIP — no first bar found")
        continue

    fb = first.iloc[0]
    direction = 1 if fb["close"] > fb["open"] else (-1 if fb["close"] < fb["open"] else 0)
    if direction == 0:
        print(f"  {day}: SKIP — doji first bar")
        continue

    dir_str   = "LONG" if direction == 1 else "SHORT"
    entry     = fb["close"]
    stop_pts  = 25
    tgt_pts   = 30
    stop_p    = entry - direction * stop_pts
    target_p  = entry + direction * tgt_pts

    # Simulate
    remaining = day_bars[day_bars.index.time >= SIGNAL_BAR_ET]
    exit_price  = None
    exit_reason = None

    for _, bar in remaining.iterrows():
        if bar.name.time() >= SESSION_CLOSE_ET:
            exit_price  = bar["open"]
            exit_reason = "session_close"
            break
        if direction == 1:
            if bar["low"] <= stop_p:
                exit_price  = stop_p
                exit_reason = "stop"
                break
            elif bar["high"] >= target_p:
                exit_price  = target_p
                exit_reason = "target"
                break
        else:
            if bar["high"] >= stop_p:
                exit_price  = stop_p
                exit_reason = "stop"
                break
            elif bar["low"] <= target_p:
                exit_price  = target_p
                exit_reason = "target"
                break

    if exit_price is None:
        exit_price  = day_bars.iloc[-1]["close"]
        exit_reason = "eod"

    pnl = direction * (exit_price - entry) * 10 - 0.80
    trade_count += 1

    # Update MLL
    new_balance = balance + pnl
    # Lock check — strictly greater than
    if not locked:
        if new_balance > MLL_LOCK_THRESHOLD:
            locked  = True
            floor   = MLL_LOCK_THRESHOLD
            peak_eod = new_balance
        else:
            if new_balance > peak_eod:
                peak_eod = new_balance
            floor = peak_eod - MLL_BUFFER
    balance = new_balance

    print(f"  {day}: TRADE #{trade_count} {dir_str} entry={entry:.1f} "
          f"exit={exit_price:.1f} reason={exit_reason} "
          f"pnl=${pnl:.0f} balance=${balance:.0f} floor=${floor:.0f} "
          f"buffer=${balance-floor:.0f} locked={locked}")

print(f"\nTotal trades in first 30 days: {trade_count}")
print(f"Final balance: {balance:.0f}  Floor: {floor:.0f}  Breached: {balance < floor}")
