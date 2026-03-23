"""
backtest/scan_contracts_3y.py
Quick scan of 2-4 contracts on full 3-year MGC baseline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time

bars = pd.read_parquet("data/cache/MGC_30min.parquet")
bars.index = pd.to_datetime(bars.index)
if bars.index.tz is None:
    bars.index = bars.index.tz_localize("America/New_York")

daily = bars.resample("1D").agg({"high":"max","low":"min"}).dropna()
daily["range"] = daily["high"] - daily["low"]
daily["r5"]    = daily["range"].rolling(5).mean()
daily["r20"]   = daily["range"].rolling(20).mean()
daily["vr"]    = daily["r5"] / daily["r20"]

FIRST_BAR_OPEN  = time(9, 0)
FIRST_BAR_CLOSE = time(9, 30)
SESSION_CLOSE   = time(16, 15)

print()
print("MGC Hold-To-Close — 3-Year Contract Size Scan")
print("=" * 60)
print(f"  {'Contracts':>10} {'Trades':>7} {'WinRate':>8} {'PnL':>10} "
      f"{'Sharpe':>8} {'MinBuf':>8} {'MLL':>8}")
print("  " + "-" * 58)

for n_ct in [1, 2, 3, 4]:
    bal    = 100_000.0
    peak   = 100_000.0
    floor  = 97_000.0
    trades = []
    min_buf = 3000.0

    for day_num, day in enumerate(sorted(set(bars.index.date))):
        if bal < floor:
            break
        if day_num < 20:
            continue

        mask = daily.index.date == day
        if not mask.any():
            continue
        info = daily[mask].iloc[0]
        if not np.isnan(info["vr"]) and info["vr"] > 2.5:
            continue

        db = bars[bars.index.date == day]
        if len(db) < 2:
            continue

        fb = db[
            (db.index.time >= FIRST_BAR_OPEN) &
            (db.index.time < FIRST_BAR_CLOSE)
        ]
        if len(fb) == 0:
            continue

        f = fb.iloc[0]
        if   f["close"] > f["open"]:  d = 1
        elif f["close"] < f["open"]:  d = -1
        else:                          continue

        entry = f["close"]
        sp    = entry - d * 22.0
        ep    = None

        for _, bar in db[db.index.time >= FIRST_BAR_CLOSE].iterrows():
            if bar.name.time() >= SESSION_CLOSE:
                ep = bar["open"]
                break
            if d == 1 and bar["low"] <= sp:
                ep = sp
                break
            if d == -1 and bar["high"] >= sp:
                ep = sp
                break
        if ep is None:
            ep = db.iloc[-1]["close"]

        pnl  = d * (ep - entry) * 10.0 * n_ct - 0.80 * n_ct
        bal += pnl
        if bal > peak:
            peak = bal
        floor   = min(peak - 3000.0, 100_000.0)
        buf     = bal - floor
        min_buf = min(min_buf, buf)
        trades.append(pnl)

    t       = pd.Series(trades)
    sharpe  = (t.mean() / t.std() * np.sqrt(252)) if len(t) > 1 and t.std() > 0 else 0
    breached = bal < floor
    mll_str  = "BREACH" if breached else "safe"

    print(f"  {n_ct:>10}   {len(t):>6}  {(t>0).mean():>7.1%}  "
          f"${t.sum():>8,.0f}  {sharpe:>7.2f}  "
          f"${min_buf:>6,.0f}  {mll_str:>7}")

print()
