"""
backtest/diagnose_breakout.py
==============================
Direction 3 — Volatility Breakout Diagnostic

Theory (Bühler 2014):
  Pseudomomentum accumulates in compressed price ranges.
  When accumulated energy exceeds a threshold, it releases directionally.
  The breakout bar is the release event — trade in its direction.

Tests every combination of:
  - Consolidation window: 2 bars (1hr) or 3 bars (1.5hr)
  - Consolidation threshold: range < N% of 20-day ATR (10%-60%)
  - Breakout confirmation: close outside range (vs. touch)
  - Stop: opposite end of consolidation range
  - Target: session close (hold-to-close like our baseline)

On both MGC and MNQ — compare side by side.

Key questions:
  1. Does breakout accuracy beat first-bar accuracy?
     MGC baseline: 62.9%  MNQ baseline: 55.3%
  2. On how many days does the signal fire? (frequency vs quality tradeoff)
  3. What consolidation threshold maximizes edge?
  4. Does the breakout signal add value vs pure first-bar?

Usage:
    python backtest/diagnose_breakout.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time

CACHE_DIR = Path("data/cache")

INSTRUMENTS = {
    "MGC": {"pv": 10.0, "comm": 0.80},
    "MNQ": {"pv":  2.0, "comm": 0.35},
}

FIRST_BAR_OPEN  = time(9,  0)
SESSION_CLOSE   = time(16, 15)

# Consolidation windows to test (number of 30-min bars)
WINDOWS = [2, 3]   # 2 = 9:00-10:00 AM, 3 = 9:00-10:30 AM

# ATR thresholds: consolidation range must be < thresh * ATR
ATR_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

ATR_LOOKBACK = 20  # days for rolling ATR


def load(symbol):
    path = CACHE_DIR / f"{symbol}_30min.parquet"
    df   = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def build_daily_atr(bars):
    """Rolling 20-day ATR from daily high-low range."""
    daily = bars.resample("1D").agg(
        high=("high", "max"),
        low=("low", "min"),
        volume=("volume", "sum")
    ).dropna()
    daily = daily[daily["volume"] > 0]
    daily["range"] = daily["high"] - daily["low"]
    daily["atr"]   = daily["range"].rolling(ATR_LOOKBACK).mean()
    return daily


def analyze_breakout(bars, daily_atr, symbol, n_consolidation_bars, atr_thresh):
    """
    For each trading day:
      1. Measure consolidation range over first n_consolidation_bars
      2. If range < atr_thresh * ATR → consolidation confirmed
      3. Watch for breakout: next bar closes outside the range
      4. Trade in breakout direction, stop at opposite end of range
      5. Exit at session close

    Returns DataFrame of trades.
    """
    spec   = INSTRUMENTS[symbol]
    pv     = spec["pv"]
    comm   = spec["comm"]
    trades = []

    for day in sorted(set(bars.index.date)):
        # Get ATR for this day
        dmask = daily_atr.index.date == day
        if not dmask.any():
            continue
        atr = daily_atr[dmask].iloc[0]["atr"]
        if np.isnan(atr) or atr <= 0:
            continue

        day_bars = bars[bars.index.date == day]
        session  = day_bars[day_bars.index.time >= FIRST_BAR_OPEN]
        if len(session) < n_consolidation_bars + 2:
            continue

        # Consolidation window
        consol = session.iloc[:n_consolidation_bars]
        c_high = consol["high"].max()
        c_low  = consol["low"].min()
        c_range= c_high - c_low

        # Check consolidation threshold
        if c_range >= atr_thresh * atr:
            continue   # too wide — not a consolidation day

        # Look for breakout in bars after consolidation
        post_consol = session.iloc[n_consolidation_bars:]
        if len(post_consol) == 0:
            continue

        entry_price  = None
        entry_time   = None
        breakout_dir = None
        stop_price   = None

        for ts, bar in post_consol.iterrows():
            if bar.name.time() >= SESSION_CLOSE:
                break

            # Breakout: bar closes outside consolidation range
            if bar["close"] > c_high and entry_price is None:
                breakout_dir = 1       # bullish breakout
                entry_price  = bar["close"]
                stop_price   = c_low   # stop at bottom of consolidation
                entry_time   = bar.name.time()
                break
            elif bar["close"] < c_low and entry_price is None:
                breakout_dir = -1      # bearish breakout
                entry_price  = bar["close"]
                stop_price   = c_high  # stop at top of consolidation
                entry_time   = bar.name.time()
                break

        if entry_price is None:
            continue   # no breakout today

        # Simulate hold-to-close with catastrophic stop
        remaining = post_consol[post_consol.index > pd.Timestamp(
            str(day) + " " + str(entry_time),
            tz="America/New_York"
        )]

        exit_price  = None
        exit_reason = None

        for ts, bar in remaining.iterrows():
            if bar.name.time() >= SESSION_CLOSE:
                exit_price  = bar["open"]
                exit_reason = "session_close"
                break
            # Stop hit
            if breakout_dir == 1 and bar["low"] <= stop_price:
                exit_price  = stop_price
                exit_reason = "stop"
                break
            if breakout_dir == -1 and bar["high"] >= stop_price:
                exit_price  = stop_price
                exit_reason = "stop"
                break

        if exit_price is None:
            if len(remaining) > 0:
                exit_price  = remaining.iloc[-1]["close"]
                exit_reason = "eod"
            else:
                exit_price  = entry_price
                exit_reason = "no_bars"

        pnl = breakout_dir * (exit_price - entry_price) * pv - comm
        stop_dist = abs(entry_price - stop_price)

        trades.append({
            "date":        day,
            "direction":   breakout_dir,
            "c_range":     c_range,
            "c_range_atr": c_range / atr,
            "atr":         atr,
            "stop_dist":   stop_dist,
            "entry":       entry_price,
            "exit":        exit_price,
            "exit_reason": exit_reason,
            "pnl":         pnl,
            "win":         pnl > 0,
        })

    return pd.DataFrame(trades)


def summarize(trades, n_days_total):
    if trades.empty:
        return None
    n    = len(trades)
    wr   = trades["win"].mean()
    tot  = trades["pnl"].sum()
    wins = trades[trades["win"]]["pnl"]
    loss = trades[~trades["win"]]["pnl"]
    pf   = wins.sum() / abs(loss.sum()) if len(loss) and loss.sum() != 0 else np.inf
    avg  = trades["pnl"].mean()
    freq = n / n_days_total * 100
    daily_pnl = trades.groupby("date")["pnl"].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)
    return {
        "n": n, "freq": freq, "wr": wr, "pf": pf,
        "avg": avg, "total": tot, "sharpe": sharpe,
    }


def main():
    print()
    print("Direction 3 — Volatility Breakout Diagnostic")
    print("=" * 70)
    print("  Theory: Bühler pseudomomentum accumulation → breakout release")
    print(f"  Testing: {list(INSTRUMENTS.keys())}  |  "
          f"Windows: {WINDOWS} bars  |  "
          f"ATR thresholds: {[f'{t:.0%}' for t in ATR_THRESHOLDS]}")
    print()

    for symbol in INSTRUMENTS:
        print(f"\n{'='*70}")
        print(f"  {symbol} — {INSTRUMENTS[symbol]}")
        print(f"{'='*70}")

        bars        = load(symbol)
        daily_atr   = build_daily_atr(bars)
        n_days      = len(set(bars.index.date))
        baseline_wr = 0.629 if symbol == "MGC" else 0.553

        print(f"\n  {n_days} trading days  |  "
              f"Baseline first-bar accuracy: {baseline_wr:.1%}")
        print()

        # Header
        print(f"  {'Win':>5}  {'Consol':>6}  {'ATR':>5}  {'N':>5}  "
              f"{'Freq':>6}  {'WR':>7}  {'PF':>6}  "
              f"{'Avg$':>8}  {'Total$':>9}  {'Sharpe':>7}")
        print("  " + "-" * 72)

        best_sharpe = -999
        best_params = None
        best_stats  = None

        for n_bars in WINDOWS:
            for thresh in ATR_THRESHOLDS:
                trades = analyze_breakout(
                    bars, daily_atr, symbol, n_bars, thresh
                )
                s = summarize(trades, n_days)
                if s is None:
                    continue

                # Flag if better than baseline
                flag = ""
                if s["wr"] > baseline_wr and s["n"] >= 30:
                    flag = " ✓"
                if s["sharpe"] > best_sharpe and s["n"] >= 30:
                    best_sharpe = s["sharpe"]
                    best_params = (n_bars, thresh)
                    best_stats  = s

                print(f"  {n_bars}bar  {thresh:>5.0%}ATR  "
                      f"{s['n']:>5}  {s['freq']:>5.1f}%  "
                      f"{s['wr']:>6.1%}  {s['pf']:>5.2f}  "
                      f"${s['avg']:>6.0f}  ${s['total']:>7,.0f}  "
                      f"{s['sharpe']:>6.2f}{flag}")

        if best_params:
            print()
            print(f"  BEST: window={best_params[0]}bar  "
                  f"threshold={best_params[1]:.0%}ATR")
            print(f"  → N={best_stats['n']}  "
                  f"WR={best_stats['wr']:.1%}  "
                  f"PF={best_stats['pf']:.2f}  "
                  f"Sharpe={best_stats['sharpe']:.2f}  "
                  f"Total=${best_stats['total']:,.0f}")

        # Also show exit reason breakdown for best params
        if best_params:
            print()
            trades_best = analyze_breakout(
                bars, daily_atr, symbol,
                best_params[0], best_params[1]
            )
            if not trades_best.empty:
                er = trades_best.groupby("exit_reason")["pnl"].agg(
                    n="count", wr=lambda x: (x>0).mean(), total="sum"
                )
                print(f"  Exit breakdown (best params):")
                print(f"  {'Exit':<16} {'N':>5} {'WR':>7} {'Total':>10}")
                print("  " + "-" * 42)
                for reason, row in er.iterrows():
                    print(f"  {reason:<16} {row['n']:>5.0f}  "
                          f"{row['wr']:>6.1%}  ${row['total']:>8,.0f}")

                # Monthly breakdown
                print()
                t2 = trades_best.copy()
                t2["date"] = pd.to_datetime(t2["date"])
                monthly = t2.groupby(
                    t2["date"].dt.to_period("M")
                )["pnl"].agg(n="count", total="sum",
                             wr=lambda x: (x>0).mean())
                print(f"  Monthly P&L (best params):")
                pos = (monthly["total"] > 0).sum()
                for period, row in monthly.iterrows():
                    sign = "+" if row["total"] >= 0 else ""
                    bar  = "█" * min(int(abs(row["total"])/100), 20)
                    print(f"    {str(period):<8}  n={row['n']:.0f}  "
                          f"WR={row['wr']:.0%}  "
                          f"{sign}${row['total']:>7,.0f}  {bar}")
                print(f"  Positive months: {pos}/{len(monthly)}")

    # Cross-instrument comparison
    print()
    print("=" * 70)
    print("  CROSS-INSTRUMENT COMPARISON — Best params per instrument")
    print("=" * 70)
    print()
    print("  Key question: does breakout accuracy beat first-bar baseline?")
    print("  MGC baseline: 62.9%  |  MNQ baseline: 55.3%")
    print()
    print("  Next step: build runner_breakout.py on the winning instrument/params")
    print()


if __name__ == "__main__":
    main()
