"""
backtest/regime_correlation.py
================================
Analyzes whether MGC's choppy months correspond to trending months
on MNQ, MYM. Identifies rotation opportunities.

Metric: monthly first-bar accuracy (signal audit methodology)
If MGC accuracy < 50% in a month where MNQ accuracy > 55%,
that's a rotation opportunity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time

CACHE_DIR = Path("data/cache")

INSTRUMENTS = {
    "MGC": {"point_value": 10.0, "commission": 0.80},
    "MNQ": {"point_value":  2.0, "commission": 0.35},
    "MYM": {"point_value":  0.5, "commission": 0.35},
}

SESSION_OPEN  = time(9, 0)
SIGNAL_BAR    = time(9, 30)
SESSION_CLOSE = time(16, 15)
STOP_POINTS   = {"MGC": 22.0, "MNQ": 80.0, "MYM": 200.0}


def load(symbol):
    path = CACHE_DIR / f"{symbol}_30min.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def monthly_first_bar_stats(df, symbol):
    """
    For each month: first-bar accuracy, avg winning move, avg losing move.
    Uses same methodology as signal_audit.py.
    """
    results = []
    for day in sorted(set(df.index.date)):
        day_bars = df[df.index.date == day]
        first = day_bars[
            (day_bars.index.time >= SESSION_OPEN) &
            (day_bars.index.time < SIGNAL_BAR)
        ]
        if len(first) == 0:
            continue
        fb = first.iloc[0]
        signal_dir = np.sign(fb["close"] - fb["open"])
        if signal_dir == 0:
            continue
        session_move = (day_bars.iloc[-1]["close"] - fb["open"]) * signal_dir
        results.append({
            "date":         pd.Timestamp(day),
            "signal_dir":   signal_dir,
            "session_move": session_move,
            "correct":      session_move > 0,
        })

    df_r = pd.DataFrame(results)
    if df_r.empty:
        return pd.DataFrame()

    monthly = df_r.groupby(df_r["date"].dt.to_period("M")).apply(
        lambda x: pd.Series({
            "days":       len(x),
            "accuracy":   x["correct"].mean(),
            "avg_win":    x[x["correct"]]["session_move"].mean()
                          if x["correct"].any() else 0,
            "avg_loss":   x[~x["correct"]]["session_move"].mean()
                          if (~x["correct"]).any() else 0,
            "avg_move":   x["session_move"].mean(),
        })
    )
    return monthly


def monthly_htc_pnl(df, symbol, n_contracts=1):
    """
    Simulate hold-to-close with catastrophic stop for each month.
    Returns monthly P&L in dollars.
    """
    stop = STOP_POINTS[symbol]
    pv   = INSTRUMENTS[symbol]["point_value"]
    comm = INSTRUMENTS[symbol]["commission"]

    daily_pnl = {}
    for day in sorted(set(df.index.date)):
        day_bars = df[df.index.date == day]
        first = day_bars[
            (day_bars.index.time >= SESSION_OPEN) &
            (day_bars.index.time < SIGNAL_BAR)
        ]
        if len(first) == 0:
            continue
        fb = first.iloc[0]
        direction = np.sign(fb["close"] - fb["open"])
        if direction == 0:
            continue

        entry      = fb["close"]
        stop_price = entry - direction * stop
        exit_price = None

        for _, bar in day_bars[day_bars.index.time >= SIGNAL_BAR].iterrows():
            if bar.name.time() >= SESSION_CLOSE:
                exit_price = bar["open"]
                break
            if direction == 1 and bar["low"] <= stop_price:
                exit_price = stop_price
                break
            if direction == -1 and bar["high"] >= stop_price:
                exit_price = stop_price
                break

        if exit_price is None:
            exit_price = day_bars.iloc[-1]["close"]

        pnl = direction * (exit_price - entry) * pv * n_contracts - comm * n_contracts
        daily_pnl[pd.Timestamp(day)] = pnl

    series = pd.Series(daily_pnl)
    if series.empty:
        return pd.Series(dtype=float)

    return series.resample("ME").sum().rename(
        lambda x: x.to_period("M")
    )


def main():
    print()
    print("=" * 70)
    print("  CROSS-INSTRUMENT REGIME CORRELATION ANALYSIS")
    print("  Goal: find instruments that trend when MGC chops")
    print("=" * 70)

    # Load all instruments
    data = {}
    for sym in INSTRUMENTS:
        df = load(sym)
        if df is not None:
            data[sym] = df
            print(f"  Loaded {sym}: {len(df)} bars "
                  f"({df.index[0].date()} to {df.index[-1].date()})")
        else:
            print(f"  {sym}: not found in cache")

    print()

    # Monthly first-bar accuracy for each instrument
    print("=" * 70)
    print("  MONTHLY FIRST-BAR ACCURACY")
    print("=" * 70)

    accuracy_data = {}
    for sym, df in data.items():
        accuracy_data[sym] = monthly_first_bar_stats(df, sym)

    # Find common months
    all_months = set()
    for sym in accuracy_data:
        if not accuracy_data[sym].empty:
            all_months.update(accuracy_data[sym].index.tolist())
    all_months = sorted(all_months)

    print(f"\n  {'Month':<10}", end="")
    for sym in INSTRUMENTS:
        if sym in accuracy_data:
            print(f"  {sym:>8}", end="")
    print()
    print("  " + "-" * (10 + len(data) * 10))

    rotation_months = []
    for month in all_months:
        print(f"  {str(month):<10}", end="")
        month_accs = {}
        for sym in INSTRUMENTS:
            if sym in accuracy_data and month in accuracy_data[sym].index:
                acc = accuracy_data[sym].loc[month, "accuracy"]
                month_accs[sym] = acc
                flag = "*" if acc < 0.50 else " "
                print(f"  {acc:>6.1%}{flag}", end="")
            else:
                print(f"  {'N/A':>7}", end="")
        print()

        # Check rotation opportunity
        if "MGC" in month_accs and month_accs["MGC"] < 0.50:
            better = {s: a for s, a in month_accs.items()
                      if s != "MGC" and a > 0.55}
            if better:
                rotation_months.append({
                    "month": month,
                    "mgc_acc": month_accs["MGC"],
                    "alternatives": better,
                })

    print()
    print("  (* = below 50% accuracy — choppy month)")

    # Monthly P&L comparison
    print()
    print("=" * 70)
    print("  MONTHLY P&L — HOLD TO CLOSE (1 contract each)")
    print("=" * 70)

    pnl_data = {}
    for sym, df in data.items():
        pnl_data[sym] = monthly_htc_pnl(df, sym)

    print(f"\n  {'Month':<10}", end="")
    for sym in data:
        print(f"  {sym:>10}", end="")
    print()
    print("  " + "-" * (10 + len(data) * 12))

    combined_baseline = 0.0
    combined_rotation = 0.0

    for month in all_months:
        print(f"  {str(month):<10}", end="")
        month_pnls = {}
        for sym in data:
            if month in pnl_data[sym].index:
                pnl = pnl_data[sym][month]
                month_pnls[sym] = pnl
                print(f"  ${pnl:>8,.0f}", end="")
            else:
                print(f"  {'N/A':>10}", end="")
        print()

    # Rotation opportunity summary
    print()
    print("=" * 70)
    print("  ROTATION OPPORTUNITIES")
    print("  (months where MGC < 50% accuracy AND alternative > 55%)")
    print("=" * 70)

    if rotation_months:
        for r in rotation_months:
            alts = ", ".join(f"{s}: {a:.1%}" for s, a in r["alternatives"].items())
            print(f"\n  {str(r['month'])}: MGC={r['mgc_acc']:.1%}  "
                  f"Better instruments: {alts}")
    else:
        print("\n  No clear rotation opportunities found in this dataset.")
        print("  Consider fetching MES data for broader analysis.")

    # Correlation between monthly accuracies
    print()
    print("=" * 70)
    print("  ACCURACY CORRELATIONS (monthly first-bar accuracy)")
    print("=" * 70)
    print()

    acc_df = pd.DataFrame({
        sym: accuracy_data[sym]["accuracy"]
        for sym in accuracy_data
        if not accuracy_data[sym].empty
    })
    acc_df = acc_df.dropna()

    if len(acc_df) > 3:
        corr = acc_df.corr()
        print("  Positive = instruments trend together (no diversification)")
        print("  Negative = instruments trend inversely (ideal for rotation)")
        print()
        print(corr.to_string())
    print()

    # MES data availability check
    print()
    print("  NOTE: MES (Micro E-mini S&P) data not yet fetched.")
    print("  MES would provide the most liquid equity index for rotation.")
    print("  Fetch with: python data/fetch_mnq_mym.py (add MES.c.0)")
    print()


if __name__ == "__main__":
    main()
