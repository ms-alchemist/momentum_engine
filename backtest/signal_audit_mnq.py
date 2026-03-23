"""
backtest/signal_audit_mnq.py
=============================
Session 6 — MNQ Signal Audit (3-year dataset)

Tests:
  1. First-bar accuracy (9:00-9:30 AM) vs session close direction
  2. Autocorrelation at 30-min bars — does MNQ trend intraday?
  3. Stop survival at MNQ-appropriate distances
  4. Monthly accuracy breakdown — systematic bad months?
  5. Time-of-day accuracy — does signal decay through session?
  6. Intraday autocorrelation decay — how long does momentum persist?
  7. Comparison vs MGC baseline

Usage:
    python backtest/signal_audit_mnq.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time

CACHE_DIR = Path("data/cache")

PV_MNQ = 2.0
PV_MGC = 10.0
COMM_MNQ = 0.35
COMM_MGC = 0.80

FIRST_BAR_OPEN  = time(9, 0)
FIRST_BAR_CLOSE = time(9, 30)
SESSION_CLOSE   = time(16, 15)


def load(symbol):
    path = CACHE_DIR / f"{symbol}_30min.parquet"
    if not path.exists():
        print(f"  {symbol}: not found")
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def first_bar_accuracy(df, symbol, pv, comm):
    """First-bar signal accuracy vs session close."""
    results = []
    for day in sorted(set(df.index.date)):
        day_bars = df[df.index.date == day]
        fb = day_bars[
            (day_bars.index.time >= FIRST_BAR_OPEN) &
            (day_bars.index.time <  FIRST_BAR_CLOSE)
        ]
        if len(fb) == 0:
            continue
        f = fb.iloc[0]
        sig = np.sign(f["close"] - f["open"])
        if sig == 0:
            continue
        session_move = day_bars.iloc[-1]["close"] - f["open"]
        correct = np.sign(session_move) == sig
        results.append({
            "date":    day,
            "sig":     sig,
            "move":    session_move * sig,   # positive = correct direction
            "correct": correct,
        })
    r = pd.DataFrame(results)
    if r.empty:
        return r

    acc       = r["correct"].mean()
    avg_win   = r[r["correct"]]["move"].mean()
    avg_loss  = r[~r["correct"]]["move"].abs().mean()
    payoff    = avg_win / avg_loss if avg_loss > 0 else 0
    ev_pts    = acc * avg_win - (1-acc) * avg_loss
    ev_dollar = ev_pts * pv - comm

    print(f"\n  {symbol} First-Bar Accuracy")
    print(f"  {'Days':>6} {'Accuracy':>10} {'AvgWin':>9} {'AvgLoss':>9} "
          f"{'Payoff':>8} {'EV/ct':>10}")
    print("  " + "-" * 60)
    print(f"  {len(r):>6} {acc:>9.1%} "
          f"{avg_win:>8.1f}pt  {avg_loss:>8.1f}pt  "
          f"{payoff:>7.2f}  ${ev_dollar:>8.0f}")
    return r


def autocorr_test(df, symbol):
    """Autocorrelation at 30-min bars — does it trend or revert intraday?"""
    rets = df["close"].pct_change().dropna()
    print(f"\n  {symbol} Autocorrelation (30-min bars)")
    print(f"  {'Lag':>5} {'Hours':>6} {'AutoCorr':>10} {'Regime':>12}")
    print("  " + "-" * 38)
    for lag in [1, 2, 4, 6, 8, 13]:
        ac  = rets.autocorr(lag=lag)
        hrs = lag * 0.5
        if   ac >  0.02: regime = "TREND   +"
        elif ac < -0.02: regime = "REVERT  -"
        else:            regime = "neutral ~"
        print(f"  {lag:>5}  {hrs:>5.1f}h  {ac:>+9.4f}  [{regime}]")


def stop_survival(df, symbol, pv, comm):
    """Stop survival on winning trades — what stop distance preserves edge?"""
    results = []
    for day in sorted(set(df.index.date)):
        day_bars = df[df.index.date == day]
        fb = day_bars[
            (day_bars.index.time >= FIRST_BAR_OPEN) &
            (day_bars.index.time <  FIRST_BAR_CLOSE)
        ]
        if len(fb) == 0:
            continue
        f    = fb.iloc[0]
        sig  = np.sign(f["close"] - f["open"])
        if sig == 0:
            continue
        entry = f["close"]
        post  = day_bars[day_bars.index.time >= FIRST_BAR_CLOSE]
        if len(post) == 0:
            continue
        session_move = (post.iloc[-1]["close"] - entry) * sig
        is_win       = session_move > 0
        # Max adverse excursion
        if sig == 1:
            mae = entry - post["low"].min()
        else:
            mae = post["high"].max() - entry
        results.append({"is_win": is_win, "session_move": session_move,
                        "mae": mae})

    r       = pd.DataFrame(results)
    winners = r[r["is_win"]]
    if len(winners) == 0:
        return

    # Stop levels scaled to instrument
    if symbol == "MNQ":
        stops = [20, 30, 40, 50, 60, 80, 100, 120]
    else:
        stops = [10, 15, 20, 25, 30, 35]

    wr      = r["is_win"].mean()
    avg_win = winners["session_move"].mean()

    print(f"\n  {symbol} Stop Survival (winners only, n={len(winners)})")
    print(f"  {'Stop':>6} {'Survival':>10} {'$Risk':>8} {'EV/ct':>9}")
    print("  " + "-" * 40)
    for s in stops:
        surv     = (winners["mae"] <= s).mean()
        risk_usd = s * pv
        ev       = wr * avg_win * pv - (1-wr) * risk_usd - comm
        print(f"  {s:>6}pt  {surv:>9.1%}  ${risk_usd:>6.0f}  ${ev:>8.0f}")


def monthly_breakdown(df, symbol):
    """Monthly accuracy — systematic weak months?"""
    results = []
    for day in sorted(set(df.index.date)):
        day_bars = df[df.index.date == day]
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
        move = (day_bars.iloc[-1]["close"] - f["open"]) * sig
        results.append({"date": pd.Timestamp(day), "correct": move > 0,
                        "move": move})

    r = pd.DataFrame(results)
    if r.empty:
        return

    monthly = r.groupby(r["date"].dt.to_period("M")).apply(
        lambda x: pd.Series({
            "days": len(x),
            "accuracy": x["correct"].mean(),
            "avg_move": x["move"].mean(),
        })
    )

    print(f"\n  {symbol} Monthly Accuracy")
    print(f"  {'Month':<10} {'Days':>5} {'Accuracy':>10} {'AvgMove':>10}")
    print("  " + "-" * 40)
    weak = 0
    for period, row in monthly.iterrows():
        flag = " *** WEAK" if row["accuracy"] < 0.50 else ""
        if row["accuracy"] < 0.50:
            weak += 1
        print(f"  {str(period):<10} {row['days']:>5.0f}  "
              f"{row['accuracy']:>9.1%}  {row['avg_move']:>9.1f}pt{flag}")
    print(f"\n  Overall: {r['correct'].mean():.1%}  "
          f"Weak months: {weak}/{len(monthly)}")


def time_of_day_accuracy(df, symbol):
    """Does first-bar signal accuracy hold across different entry times?"""
    # Test: enter at first bar close (9:30), 10:00, 10:30, 11:00
    entry_times = [
        (time(9, 30),  "9:30 AM  (first bar close)"),
        (time(10, 0),  "10:00 AM (2nd hour open)"),
        (time(10, 30), "10:30 AM (2nd hour mid)"),
        (time(11, 0),  "11:00 AM (3rd hour)"),
    ]
    # Use first-bar direction as the signal regardless of entry time
    print(f"\n  {symbol} Signal Decay by Entry Time")
    print(f"  {'Entry Time':<28} {'N':>5} {'Accuracy':>10} {'AvgMove':>10}")
    print("  " + "-" * 58)

    for entry_t, label in entry_times:
        results = []
        for day in sorted(set(df.index.date)):
            day_bars = df[df.index.date == day]
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
            # Find entry bar at entry_t
            entry_bars = day_bars[day_bars.index.time >= entry_t]
            if len(entry_bars) == 0:
                continue
            entry_price = entry_bars.iloc[0]["open"]
            exit_price  = day_bars[
                day_bars.index.time <= SESSION_CLOSE
            ].iloc[-1]["close"]
            move = (exit_price - entry_price) * sig
            results.append({"correct": move > 0, "move": move})

        if not results:
            continue
        r   = pd.DataFrame(results)
        acc = r["correct"].mean()
        avg = r["move"].mean()
        print(f"  {label:<28} {len(r):>5}  {acc:>9.1%}  {avg:>9.1f}pt")


def intraday_momentum_decay(df, symbol):
    """How many bars does intraday momentum persist after first bar?"""
    print(f"\n  {symbol} Intraday Momentum Persistence")
    print(f"  (First-bar signal accuracy measured N bars later)")
    print(f"  {'Bars later':>11} {'Hours':>6} {'N':>5} {'Accuracy':>10}")
    print("  " + "-" * 38)

    results_by_lag = {}
    for day in sorted(set(df.index.date)):
        day_bars = df[df.index.date == day]
        fb = day_bars[
            (day_bars.index.time >= FIRST_BAR_OPEN) &
            (day_bars.index.time <  FIRST_BAR_CLOSE)
        ]
        if len(fb) == 0:
            continue
        f    = fb.iloc[0]
        sig  = np.sign(f["close"] - f["open"])
        if sig == 0:
            continue
        entry  = f["close"]
        post   = day_bars[day_bars.index.time >= FIRST_BAR_CLOSE].reset_index()

        for lag in [1, 2, 4, 6, 8, 10, 13]:
            if lag >= len(post):
                continue
            exit_price = post.iloc[lag]["close"]
            move       = (exit_price - entry) * sig
            if lag not in results_by_lag:
                results_by_lag[lag] = []
            results_by_lag[lag].append(move > 0)

    for lag in sorted(results_by_lag):
        vals = results_by_lag[lag]
        acc  = np.mean(vals)
        hrs  = lag * 0.5
        print(f"  {lag:>11}  {hrs:>5.1f}h  {len(vals):>5}  {acc:>9.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 65)
    print("  MNQ Signal Audit — Session 6 (3-year dataset)")
    print("=" * 65)

    mnq = load("MNQ")
    mgc = load("MGC")

    if mnq is None:
        print("  ERROR: MNQ data not found. Run data/fetch_all_3y.py first.")
        return

    print(f"\n  MNQ: {len(mnq)} 30-min bars  "
          f"({mnq.index[0].date()} to {mnq.index[-1].date()})")
    if mgc is not None:
        print(f"  MGC: {len(mgc)} 30-min bars  "
              f"({mgc.index[0].date()} to {mgc.index[-1].date()})")

    # --- Test 1: First-bar accuracy ---
    print("\n" + "=" * 65)
    print("  TEST 1: First-Bar Accuracy")
    print("=" * 65)
    mnq_fb = first_bar_accuracy(mnq, "MNQ", PV_MNQ, COMM_MNQ)
    if mgc is not None:
        mgc_fb = first_bar_accuracy(mgc, "MGC", PV_MGC, COMM_MGC)

    # --- Test 2: Autocorrelation ---
    print("\n" + "=" * 65)
    print("  TEST 2: Autocorrelation (intraday trend vs revert)")
    print("=" * 65)
    autocorr_test(mnq, "MNQ")
    if mgc is not None:
        autocorr_test(mgc, "MGC")

    # --- Test 3: Stop survival ---
    print("\n" + "=" * 65)
    print("  TEST 3: Stop Survival (winning trades)")
    print("=" * 65)
    stop_survival(mnq, "MNQ", PV_MNQ, COMM_MNQ)

    # --- Test 4: Monthly breakdown ---
    print("\n" + "=" * 65)
    print("  TEST 4: Monthly Accuracy")
    print("=" * 65)
    monthly_breakdown(mnq, "MNQ")

    # --- Test 5: Time of day ---
    print("\n" + "=" * 65)
    print("  TEST 5: Signal Decay by Entry Time")
    print("=" * 65)
    time_of_day_accuracy(mnq, "MNQ")

    # --- Test 6: Momentum persistence ---
    print("\n" + "=" * 65)
    print("  TEST 6: Intraday Momentum Persistence")
    print("=" * 65)
    intraday_momentum_decay(mnq, "MNQ")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    if not mnq_fb.empty:
        acc = mnq_fb["correct"].mean()
        print(f"\n  MNQ first-bar accuracy: {acc:.1%}")
        if acc >= 0.57:
            print("  ✓ Strong edge — proceed to backtest")
        elif acc >= 0.53:
            print("  ~ Marginal edge — needs signal stack to add value")
        else:
            print("  ✗ Weak edge — below 53% threshold")

    print()
    print("  Next: python backtest/runner_mnq_htc.py")
    print()


if __name__ == "__main__":
    main()
