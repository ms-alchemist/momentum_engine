"""
backtest/signal_audit_mnq_mym.py
=================================
Replicates the MGC signal audit on MNQ and MYM.

Tests:
  1. First-bar accuracy: does 9:00-9:30 AM bar predict session direction?
  2. Autocorrelation at 30-min bars: does MNQ trend or mean-revert intraday?
  3. Monthly accuracy breakdown
  4. Stop distance survival: what % of winning trades survive various stops?
  5. Dollar comparison vs MGC baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path("data/cache")

INSTRUMENTS = {
    "MNQ": {"point_value": 2.0,  "commission": 0.35, "name": "Micro Nasdaq"},
    "MYM": {"point_value": 0.50, "commission": 0.35, "name": "Micro Dow"},
    "MGC": {"point_value": 10.0, "commission": 0.80, "name": "Micro Gold (baseline)"},
}


def load_bars(symbol):
    path = CACHE_DIR / f"{symbol}_30min.parquet"
    if not path.exists():
        print(f"  {symbol} not found — skipping")
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df


def first_bar_accuracy(df, symbol):
    """
    Test 1: Does the 9:00-9:30 AM bar direction predict session close direction?
    Session direction = sign(last_bar_close - first_bar_open)
    Signal direction  = sign(first_bar_close - first_bar_open)
    """
    results = []
    trading_days = sorted(set(df.index.date))

    for day in trading_days:
        day_bars = df[df.index.date == day]

        first = day_bars[
            (day_bars.index.time >= pd.Timestamp("09:00").time()) &
            (day_bars.index.time < pd.Timestamp("09:30").time())
        ]
        if len(first) == 0:
            continue

        fb = first.iloc[0]
        signal_dir = np.sign(fb["close"] - fb["open"])
        if signal_dir == 0:
            continue

        # Session move from first bar open to last bar close
        session_move = day_bars.iloc[-1]["close"] - fb["open"]
        session_dir  = np.sign(session_move)
        correct      = (signal_dir == session_dir)

        results.append({
            "date":         day,
            "signal_dir":   signal_dir,
            "session_move": session_move,
            "correct":      correct,
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    accuracy    = results_df["correct"].mean()
    avg_correct = results_df[results_df["correct"]]["session_move"].abs().mean()
    avg_wrong   = results_df[~results_df["correct"]]["session_move"].abs().mean()
    payoff      = avg_correct / avg_wrong if avg_wrong > 0 else 0

    spec = INSTRUMENTS[symbol]
    ev_pts = accuracy * avg_correct - (1 - accuracy) * avg_wrong
    ev_dollar = ev_pts * spec["point_value"] - spec["commission"]

    print(f"\n{symbol} ({spec['name']})")
    print(f"  First-bar accuracy:      {accuracy:.1%}  (need >55%)")
    print(f"  Avg session move correct: {avg_correct:.1f} pts  "
          f"(${avg_correct * spec['point_value']:.0f}/contract)")
    print(f"  Avg session move wrong:   {avg_wrong:.1f} pts  "
          f"(${avg_wrong * spec['point_value']:.0f}/contract)")
    print(f"  Payoff ratio:            {payoff:.2f}")
    print(f"  EV per trade:            {ev_pts:.1f} pts  "
          f"(${ev_dollar:.0f}/contract)")

    return results_df


def autocorrelation_test(df, symbol):
    """
    Test 2: Return autocorrelation at 30-min bars.
    Negative = mean-reverting (bad for momentum)
    Positive = trending (good for momentum)
    """
    returns = df["close"].pct_change().dropna()
    lags    = [1, 2, 4, 8, 16]

    print(f"\n{symbol} autocorrelation (30-min bars):")
    for lag in lags:
        ac = returns.autocorr(lag=lag)
        hours = lag * 0.5
        trend = "TREND    +" if ac > 0.02 else ("REVERT -" if ac < -0.02 else "NEUTRAL ~")
        print(f"  Lag {lag:>2} ({hours:.1f}h):  {ac:>+.4f}  [{trend}]")


def stop_survival_test(df, symbol):
    """
    Test 3: What % of winning trades survive various stop distances?
    A winning trade = session closes in signal direction.
    """
    spec     = INSTRUMENTS[symbol]
    results  = []
    trading_days = sorted(set(df.index.date))

    for day in trading_days:
        day_bars = df[df.index.date == day]

        first = day_bars[
            (day_bars.index.time >= pd.Timestamp("09:00").time()) &
            (day_bars.index.time < pd.Timestamp("09:30").time())
        ]
        if len(first) == 0:
            continue

        fb         = first.iloc[0]
        signal_dir = np.sign(fb["close"] - fb["open"])
        if signal_dir == 0:
            continue

        entry = fb["close"]
        remaining = day_bars[day_bars.index.time >= pd.Timestamp("09:30").time()]
        if len(remaining) == 0:
            continue

        # Session outcome
        session_move = (remaining.iloc[-1]["close"] - entry) * signal_dir
        is_winner    = session_move > 0

        # Worst adverse excursion (max move against us during session)
        if signal_dir == 1:
            mae = entry - remaining["low"].min()
        else:
            mae = remaining["high"].max() - entry

        results.append({
            "date":         day,
            "is_winner":    is_winner,
            "session_move": session_move,
            "mae":          mae,
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return

    winners = results_df[results_df["is_winner"]]
    if len(winners) == 0:
        return

    # Stop distances to test — in points
    # For MNQ: meaningful stops are 20-100 pts
    # For MYM: meaningful stops are 50-200 pts
    if symbol == "MNQ":
        stop_levels = [20, 30, 40, 50, 60, 80, 100]
    elif symbol == "MYM":
        stop_levels = [50, 75, 100, 125, 150, 200]
    else:
        stop_levels = [10, 15, 20, 25, 30]

    print(f"\n{symbol} stop survival (winning trades only, n={len(winners)}):")
    print(f"  {'Stop':>6}  {'Survival%':>10}  {'Dollar risk':>12}  {'EV/trade':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*10}")

    for stop in stop_levels:
        surviving = (winners["mae"] <= stop).mean()
        dollar_risk = stop * spec["point_value"]
        # EV = WR * avg_win - (1-WR) * stop - commission
        wr     = results_df["is_winner"].mean()
        avg_win_pts = winners["session_move"].mean()
        ev     = wr * avg_win_pts * spec["point_value"] - (1-wr) * dollar_risk - spec["commission"]
        print(f"  {stop:>6}pt  {surviving:>9.1%}  ${dollar_risk:>10.0f}  ${ev:>9.0f}")


def monthly_breakdown(df, symbol):
    """
    Test 4: Monthly accuracy — are there systematic bad months?
    """
    results = []
    trading_days = sorted(set(df.index.date))

    for day in trading_days:
        day_bars = df[df.index.date == day]
        first = day_bars[
            (day_bars.index.time >= pd.Timestamp("09:00").time()) &
            (day_bars.index.time < pd.Timestamp("09:30").time())
        ]
        if len(first) == 0:
            continue
        fb = first.iloc[0]
        signal_dir = np.sign(fb["close"] - fb["open"])
        if signal_dir == 0:
            continue
        session_move = (day_bars.iloc[-1]["close"] - fb["open"]) * signal_dir
        results.append({"date": pd.Timestamp(day), "correct": session_move > 0,
                        "move": session_move})

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return

    monthly = results_df.groupby(results_df["date"].dt.to_period("M")).apply(
        lambda x: pd.Series({
            "days":     len(x),
            "accuracy": x["correct"].mean(),
            "avg_move": x["move"].mean(),
        })
    )

    print(f"\n{symbol} monthly accuracy:")
    print(f"  {'Month':<10} {'Days':>5} {'Accuracy':>10} {'AvgMove':>10}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*10}")
    for period, row in monthly.iterrows():
        flag = " *** WEAK" if row["accuracy"] < 0.50 else ""
        print(f"  {str(period):<10} {row['days']:>5.0f}  {row['accuracy']:>9.1%}  "
              f"{row['avg_move']:>8.1f}pt{flag}")

    overall = results_df["correct"].mean()
    print(f"\n  Overall accuracy: {overall:.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 60)
print("  SIGNAL AUDIT — MNQ vs MYM vs MGC")
print("  First-bar momentum edge analysis")
print("=" * 60)

for symbol in ["MNQ", "MYM", "MGC"]:
    df = load_bars(symbol)
    if df is None:
        continue

    print()
    print("=" * 60)
    print(f"  {symbol} — {INSTRUMENTS[symbol]['name']}")
    print("=" * 60)

    first_bar_accuracy(df, symbol)
    autocorrelation_test(df, symbol)
    stop_survival_test(df, symbol)
    monthly_breakdown(df, symbol)

print()
print("=" * 60)
print("  SUMMARY — Dollar EV comparison")
print("=" * 60)
print()
print("  Run the above to see which instrument has the strongest")
print("  first-bar edge and cleanest autocorrelation structure.")
print()
