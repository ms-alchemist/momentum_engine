"""
backtest/diagnose_mnq_entry.py
==============================
Tests the delayed entry hypothesis:
  - First bar (9:00-9:30) sets direction
  - Reversion dip occurs 9:30-10:30 AM (Lag 1 autocorr = -0.014)
  - Momentum resumes after 10:30 AM (Lag 2+ autocorr = +0.034)
  - Enter at 10:30 AM on days with strong first-bar confirmation

Tests:
  1. 10:30 AM entry on ALL days vs first-bar entry
  2. 10:30 AM entry filtered by first-bar strength (large vs small bar)
  3. 10:30 AM entry with reversion confirmation (price pulled back toward open)
  4. Combined: strong first bar + reversion dip confirmed
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time

CACHE_DIR = Path("data/cache")
PV        = 2.0
COMM      = 0.35

FIRST_BAR_OPEN  = time(9,  0)
FIRST_BAR_CLOSE = time(9, 30)
ENTRY_TIME      = time(10, 30)   # delayed entry
SESSION_CLOSE   = time(16, 15)

# First-bar strength: bar size relative to recent ATR
ATR_LOOKBACK = 20   # trading days
MIN_BAR_ATR_RATIO = 0.5   # first bar must be >= 50% of recent ATR to qualify


def main():
    bars = pd.read_parquet(CACHE_DIR / "MNQ_30min.parquet")
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    bars = bars.sort_index()

    # Daily ATR for first-bar strength filter
    daily = bars.resample("1D").agg(
        high=("high","max"), low=("low","min"),
        volume=("volume","sum")
    ).dropna()
    daily = daily[daily["volume"] > 0]
    daily["range"] = daily["high"] - daily["low"]
    daily["atr"]   = daily["range"].rolling(ATR_LOOKBACK).mean()

    print()
    print("MNQ Delayed Entry Diagnostic")
    print("=" * 65)
    print(f"  Hypothesis: enter at {ENTRY_TIME} after first-bar reversion dip")
    print()

    trading_days = sorted(set(bars.index.date))
    records = []

    for day in trading_days:
        day_bars = bars[bars.index.date == day]

        # First bar
        fb = day_bars[
            (day_bars.index.time >= FIRST_BAR_OPEN) &
            (day_bars.index.time <  FIRST_BAR_CLOSE)
        ]
        if len(fb) == 0:
            continue
        f   = fb.iloc[0]
        sig = np.sign(f["close"] - f["open"])
        if sig == 0:
            continue

        # First-bar metrics
        fb_size = abs(f["close"] - f["open"])   # bar body size in pts
        fb_range= f["high"] - f["low"]

        # Daily ATR for this day
        dmask = daily.index.date == day
        if not dmask.any():
            continue
        atr = daily[dmask].iloc[0]["atr"]
        if np.isnan(atr):
            continue

        # Bar strength ratio
        bar_strength = fb_size / atr if atr > 0 else 0

        # 10:30 AM entry bar
        entry_bars = day_bars[day_bars.index.time >= ENTRY_TIME]
        if len(entry_bars) == 0:
            continue
        entry_price = entry_bars.iloc[0]["open"]

        # Session close exit
        close_bars = day_bars[day_bars.index.time <= SESSION_CLOSE]
        if len(close_bars) == 0:
            continue
        exit_price = close_bars.iloc[-1]["close"]

        move = (exit_price - entry_price) * sig

        # Reversion check: did price retrace toward the first-bar open?
        # Between 9:30 and 10:30, check if price touched back toward open
        reversion_bars = day_bars[
            (day_bars.index.time >= FIRST_BAR_CLOSE) &
            (day_bars.index.time <  ENTRY_TIME)
        ]
        if len(reversion_bars) > 0:
            if sig == 1:  # long signal — look for low to dip below first-bar close
                min_price = reversion_bars["low"].min()
                reversion = min_price < f["close"]   # price pulled back
                retrace_pct = (f["close"] - min_price) / fb_range if fb_range > 0 else 0
            else:  # short signal
                max_price = reversion_bars["high"].max()
                reversion = max_price > f["close"]
                retrace_pct = (max_price - f["close"]) / fb_range if fb_range > 0 else 0
        else:
            reversion   = False
            retrace_pct = 0

        # Immediate 9:30 entry P&L for comparison
        post_fb = day_bars[day_bars.index.time >= FIRST_BAR_CLOSE]
        immediate_exit = close_bars.iloc[-1]["close"]
        immediate_move = (immediate_exit - f["close"]) * sig

        records.append({
            "date":          day,
            "sig":           sig,
            "fb_size":       fb_size,
            "bar_strength":  bar_strength,
            "reversion":     reversion,
            "retrace_pct":   retrace_pct,
            "move_delayed":  move,
            "move_immediate":immediate_move,
            "win_delayed":   move > 0,
            "win_immediate": immediate_move > 0,
        })

    df = pd.DataFrame(records)
    n  = len(df)
    print(f"  Total trading days analyzed: {n}")
    print()

    # --- Test 1: Delayed vs immediate entry, all days ---
    wr_d = df["win_delayed"].mean()
    wr_i = df["win_immediate"].mean()
    ev_d = df["move_delayed"].mean() * PV - COMM
    ev_i = df["move_immediate"].mean() * PV - COMM
    print("  TEST 1: All days — delayed (10:30) vs immediate (9:30) entry")
    print(f"  {'Entry':>12} {'N':>6} {'WR':>8} {'AvgMove':>10} {'EV/ct':>8}")
    print("  " + "-" * 50)
    print(f"  {'9:30 AM':>12} {n:>6} {wr_i:>7.1%} "
          f"{df['move_immediate'].mean():>9.1f}pt ${ev_i:>6.0f}")
    print(f"  {'10:30 AM':>12} {n:>6} {wr_d:>7.1%} "
          f"{df['move_delayed'].mean():>9.1f}pt ${ev_d:>6.0f}")
    print()

    # --- Test 2: Filter by first-bar strength ---
    print("  TEST 2: Delayed entry by first-bar strength")
    print(f"  {'Filter':>25} {'N':>6} {'WR':>8} {'AvgMove':>10} {'EV/ct':>8}")
    print("  " + "-" * 60)
    for thresh, label in [
        (0.0,  "All days"),
        (0.3,  "Bar >= 30% ATR"),
        (0.5,  "Bar >= 50% ATR"),
        (0.7,  "Bar >= 70% ATR"),
        (1.0,  "Bar >= 100% ATR"),
        (1.5,  "Bar >= 150% ATR"),
    ]:
        sub = df[df["bar_strength"] >= thresh]
        if len(sub) < 20:
            break
        wr  = sub["win_delayed"].mean()
        avg = sub["move_delayed"].mean()
        ev  = avg * PV - COMM
        print(f"  {label:>25} {len(sub):>6} {wr:>7.1%} "
              f"{avg:>9.1f}pt ${ev:>6.0f}")
    print()

    # --- Test 3: Filter by reversion confirmation ---
    sub_rev    = df[df["reversion"]]
    sub_no_rev = df[~df["reversion"]]
    print("  TEST 3: Delayed entry — reversion dip occurred vs not")
    print(f"  {'Filter':>30} {'N':>6} {'WR':>8} {'AvgMove':>10} {'EV/ct':>8}")
    print("  " + "-" * 65)
    for sub, label in [(sub_rev, "Reversion dip confirmed"),
                       (sub_no_rev, "No reversion dip")]:
        if len(sub) < 10:
            continue
        wr  = sub["win_delayed"].mean()
        avg = sub["move_delayed"].mean()
        ev  = avg * PV - COMM
        print(f"  {label:>30} {len(sub):>6} {wr:>7.1%} "
              f"{avg:>9.1f}pt ${ev:>6.0f}")
    print()

    # --- Test 4: Combined filter (strong bar + reversion confirmed) ---
    print("  TEST 4: Combined — strong bar + reversion dip")
    print(f"  {'Filter':>35} {'N':>6} {'WR':>8} {'AvgMove':>10} {'EV/ct':>8}")
    print("  " + "-" * 70)
    for bar_thresh, label in [(0.5, ">=50% ATR"), (0.7, ">=70% ATR"),
                              (1.0, ">=100% ATR")]:
        sub = df[(df["bar_strength"] >= bar_thresh) & df["reversion"]]
        if len(sub) < 15:
            break
        wr  = sub["win_delayed"].mean()
        avg = sub["move_delayed"].mean()
        ev  = avg * PV - COMM
        pct = len(sub)/n*100
        print(f"  Strong({label}) + reversion  {len(sub):>6} ({pct:.0f}%)  "
              f"{wr:>7.1%}  {avg:>9.1f}pt  ${ev:>6.0f}")

    # Also: strong bar alone for comparison
    for bar_thresh, label in [(0.5, ">=50% ATR"), (0.7, ">=70% ATR")]:
        sub = df[df["bar_strength"] >= bar_thresh]
        if len(sub) < 15:
            break
        wr  = sub["win_delayed"].mean()
        avg = sub["move_delayed"].mean()
        ev  = avg * PV - COMM
        pct = len(sub)/n*100
        print(f"  Strong({label}) only         {len(sub):>6} ({pct:.0f}%)  "
              f"{wr:>7.1%}  {avg:>9.1f}pt  ${ev:>6.0f}")
    print()

    # --- Test 5: Retrace depth ---
    print("  TEST 5: Does deeper reversion dip = better continuation?")
    print(f"  {'Retrace depth':>22} {'N':>6} {'WR':>8} {'AvgMove':>10} {'EV/ct':>8}")
    print("  " + "-" * 62)
    sub_rev2 = df[df["reversion"]].copy()
    for lo, hi, label in [
        (0.0, 0.25, "Shallow  (<25% of bar)"),
        (0.25,0.50, "Moderate (25-50%)"),
        (0.50,0.75, "Deep     (50-75%)"),
        (0.75,2.00, "Very deep (>75%)"),
    ]:
        sub = sub_rev2[
            (sub_rev2["retrace_pct"] >= lo) &
            (sub_rev2["retrace_pct"] <  hi)
        ]
        if len(sub) < 10:
            continue
        wr  = sub["win_delayed"].mean()
        avg = sub["move_delayed"].mean()
        ev  = avg * PV - COMM
        print(f"  {label:>22} {len(sub):>6} {wr:>7.1%} "
              f"{avg:>9.1f}pt ${ev:>6.0f}")
    print()

    # Key question
    best = df[(df["bar_strength"] >= 0.5) & df["reversion"]]
    if len(best) > 0:
        wr  = best["win_delayed"].mean()
        ev  = best["move_delayed"].mean() * PV - COMM
        pct = len(best)/n*100
        print(f"  KEY RESULT: Strong first bar + reversion confirmed")
        print(f"  → {len(best)} days ({pct:.0f}% of sessions), "
              f"WR={wr:.1%}, EV=${ev:.0f}/contract")
        if wr >= 0.58:
            print(f"  ✓ EDGE EXISTS — proceed to Direction 1 backtest")
        elif wr >= 0.53:
            print(f"  ~ Marginal improvement — needs additional filters")
        else:
            print(f"  ✗ No improvement — delayed entry doesn't help")
    print()


if __name__ == "__main__":
    main()
