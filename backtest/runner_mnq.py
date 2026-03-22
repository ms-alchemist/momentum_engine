"""
backtest/runner_mnq.py
======================
MNQ Intraday Momentum Strategy — Full Four-Paper Signal Stack

Strategy:
  - Instrument: MNQ (Micro E-mini Nasdaq), $2/point
  - Signal: Six-layer stack (Moskowitz, Bühler, Chevyrev, Cont, Barroso, Di Graziano)
  - Entry window: 9:30 AM - 2:00 PM ET (signal fires on any bar in window)
  - Stop: 1.5× ATR of last 4 bars
  - Target: 2.5× ATR of last 4 bars (Di Graziano b/a = 1.67)
  - Hard close: 3:45 PM ET
  - Max trades per day: 2
  - Position sizing: Barroso vol targeting ($300/day target vol)

Multi-contract comparison: 1, 2, 3, 4 MNQ contracts
Correct Lucid MLL mechanics: floor = min(peak_EOD - $3,000, $100,000)

Usage:
    python backtest/runner_mnq.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import warnings
warnings.filterwarnings("ignore")

from signals.mnq_engine import MNQSignalEngine, MNQSignal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR     = Path("data/cache")
RESULTS_DIR  = Path("backtest/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MNQ_FILE_30M = DATA_DIR / "MNQ_30min.parquet"
MNQ_PNL_PT   = 2.0       # $2 per point per contract
MNQ_COMM     = 0.35       # $0.35 round-trip per contract

# Session timing — ET
ENTRY_START_ET   = time(9, 30)   # entries open after first bar
ENTRY_END_ET     = time(14, 0)   # no new entries after 2 PM
SESSION_CLOSE_ET = time(15, 45)  # hard close 3:45 PM

# Lucid prop firm
STARTING_BALANCE = 100_000.0
PROFIT_TARGET    =   6_000.0
MLL_BUFFER       =   3_000.0

# Barroso vol targeting
TARGET_DAILY_VOL = 300.0   # target $300/day vol at 1 contract
VOL_LOOKBACK     = 8       # bars for realized vol estimate

# Max trades per day
MAX_TRADES_PER_DAY = 2

CONTRACT_SIZES = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# MLL tracker — correct Lucid mechanics
# ---------------------------------------------------------------------------

class LucidMLL:
    """
    Floor = min(peak_EOD - $3,000, $100,000)
    Starts at $97,000. Only reaches $100,000 once peak_EOD >= $103,000.
    """
    def __init__(self):
        self.balance  = STARTING_BALANCE
        self.peak_eod = STARTING_BALANCE
        self.floor    = STARTING_BALANCE - MLL_BUFFER

    @property
    def buffer(self):
        return self.balance - self.floor

    @property
    def profit(self):
        return self.balance - STARTING_BALANCE

    def update_eod(self, eod_balance):
        self.balance = eod_balance
        if eod_balance > self.peak_eod:
            self.peak_eod = eod_balance
        self.floor = min(self.peak_eod - MLL_BUFFER, STARTING_BALANCE)

    def is_breached(self):
        return self.balance < self.floor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_parquet(MNQ_FILE_30M)
    df.index = pd.to_datetime(df.index, utc=False)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    df = df.sort_index()
    print(f"Loaded {len(df)} MNQ 30-min bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# Barroso vol-targeted position sizing
# ---------------------------------------------------------------------------

def vol_sized_contracts(bars: pd.DataFrame, n_max: int,
                        balance: float) -> int:
    """
    Scale contracts so expected daily dollar vol ≈ TARGET_DAILY_VOL.
    Barroso & Santa-Clara (2015): target constant realized vol.
    """
    closes  = bars["close"]
    returns = closes.pct_change().dropna()

    if len(returns) < VOL_LOOKBACK:
        return 1

    rv = returns.tail(VOL_LOOKBACK).std()
    if rv == 0 or np.isnan(rv):
        return 1

    price = float(closes.iloc[-1])
    dollar_vol_per_contract = rv * price * MNQ_PNL_PT

    if dollar_vol_per_contract == 0:
        return 1

    contracts = int(TARGET_DAILY_VOL / dollar_vol_per_contract)
    return max(1, min(contracts, n_max))


# ---------------------------------------------------------------------------
# Single backtest run
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, n_contracts: int):
    engine         = MNQSignalEngine()
    mll            = LucidMLL()
    trades         = []
    daily_results  = []
    cumulative_pnl = 0.0

    trading_days = sorted(set(df.index.date))

    for day in trading_days:
        if mll.is_breached():
            break
        if mll.profit >= PROFIT_TARGET:
            break

        day_bars     = df[df.index.date == day].copy()
        daily_trades = 0
        daily_pnl    = 0.0
        open_position = None

        for i, (ts, bar) in enumerate(day_bars.iterrows()):
            bar_time = ts.time()

            # Hard close — exit any open position
            if bar_time >= SESSION_CLOSE_ET:
                if open_position is not None:
                    exit_price  = float(bar["open"])
                    exit_reason = "session_close"
                    pnl = _close_position(
                        open_position, exit_price, exit_reason,
                        n_contracts, trades, ts
                    )
                    daily_pnl    += pnl
                    cumulative_pnl += pnl
                    open_position  = None
                break

            # Manage open position
            if open_position is not None:
                direction = open_position["direction"]
                stop_p    = open_position["stop_price"]
                target_p  = open_position["target_price"]
                exit_price  = None
                exit_reason = None

                if direction == 1:
                    if bar["low"] <= stop_p:
                        exit_price, exit_reason = stop_p, "stop"
                    elif bar["high"] >= target_p:
                        exit_price, exit_reason = target_p, "target"
                else:
                    if bar["high"] >= stop_p:
                        exit_price, exit_reason = stop_p, "stop"
                    elif bar["low"] <= target_p:
                        exit_price, exit_reason = target_p, "target"

                if exit_price is not None:
                    pnl = _close_position(
                        open_position, exit_price, exit_reason,
                        n_contracts, trades, ts
                    )
                    daily_pnl    += pnl
                    cumulative_pnl += pnl
                    daily_trades   += 1
                    open_position  = None

            # Generate signal — only in entry window, no open position
            if (open_position is None
                    and daily_trades < MAX_TRADES_PER_DAY
                    and ENTRY_START_ET <= bar_time < ENTRY_END_ET
                    and i >= 1):

                bars_to_here = day_bars.iloc[:i + 1]
                # Pad with prior day bars for coherence filter
                prior = df[df.index.date < day].tail(32)
                bars_for_signal = pd.concat([prior, bars_to_here])

                if len(bars_for_signal) < 20:
                    continue

                signal = engine.compute(bars_for_signal)

                if signal.is_tradeable:
                    entry_price = float(bar["close"])
                    stop_pts    = max(signal.stop_pts, 10.0)
                    target_pts  = max(signal.target_pts, 16.0)

                    if signal.direction == 1:
                        stop_p   = entry_price - stop_pts
                        target_p = entry_price + target_pts
                    else:
                        stop_p   = entry_price + stop_pts
                        target_p = entry_price - target_pts

                    open_position = {
                        "direction":   signal.direction,
                        "entry_price": entry_price,
                        "entry_time":  ts,
                        "stop_price":  stop_p,
                        "target_price": target_p,
                        "stop_pts":    stop_pts,
                        "target_pts":  target_pts,
                        "signal_score": signal.combined_score,
                        "regime":      signal.regime,
                    }

        # EOD — close any remaining position
        if open_position is not None:
            last_bar   = day_bars.iloc[-1]
            exit_price  = float(last_bar["close"])
            pnl = _close_position(
                open_position, exit_price, "eod",
                n_contracts, trades, day_bars.index[-1]
            )
            daily_pnl    += pnl
            cumulative_pnl += pnl

        mll.update_eod(mll.balance + daily_pnl)

        daily_results.append({
            "date":    day,
            "pnl":     daily_pnl,
            "trades":  daily_trades,
            "balance": mll.balance,
            "floor":   mll.floor,
            "buffer":  mll.buffer,
        })

    return pd.DataFrame(trades), pd.DataFrame(daily_results), mll


def _close_position(pos, exit_price, exit_reason,
                    n_contracts, trades, exit_time):
    direction  = pos["direction"]
    entry      = pos["entry_price"]
    price_diff = (exit_price - entry) * direction
    gross_pnl  = price_diff * MNQ_PNL_PT * n_contracts
    comm       = MNQ_COMM * n_contracts
    net_pnl    = gross_pnl - comm

    trades.append({
        "entry_time":   pos["entry_time"],
        "exit_time":    exit_time,
        "direction":    "LONG" if direction == 1 else "SHORT",
        "entry_price":  entry,
        "exit_price":   exit_price,
        "exit_reason":  exit_reason,
        "stop_pts":     pos["stop_pts"],
        "target_pts":   pos["target_pts"],
        "gross_pnl":    gross_pnl,
        "commission":   comm,
        "net_pnl":      net_pnl,
        "win":          net_pnl > 0,
        "signal_score": pos["signal_score"],
        "regime":       pos["regime"],
    })
    return net_pnl


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(trades_df, daily_df, mll, n_contracts):
    if trades_df.empty:
        return {"n_contracts": n_contracts, "total_trades": 0,
                "win_rate": 0, "total_pnl": 0, "sharpe": 0,
                "max_drawdown": 0, "final_balance": mll.balance,
                "mll_breached": mll.is_breached(), "target_hit": False,
                "days_to_target": None, "worst_losing_streak": 0,
                "min_mll_buffer": mll.buffer, "profit_factor": 0,
                "avg_win": 0, "avg_loss": 0, "n_consistency_flags": 0}

    wins   = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]
    n      = len(trades_df)

    win_rate      = len(wins) / n
    avg_win       = wins["net_pnl"].mean()   if len(wins)   else 0
    avg_loss      = losses["net_pnl"].mean() if len(losses) else 0
    total_pnl     = trades_df["net_pnl"].sum()
    gross_wins    = wins["net_pnl"].sum()
    gross_losses  = abs(losses["net_pnl"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses else np.inf

    sharpe = ((daily_df["pnl"].mean() / daily_df["pnl"].std()) * np.sqrt(252)
              if daily_df["pnl"].std() > 0 else 0.0)

    eq       = daily_df["balance"].dropna()
    roll_max = eq.cummax()
    max_dd   = (eq - roll_max).min() if len(eq) else 0.0

    target_hit = mll.profit >= PROFIT_TARGET

    days_to_target = None
    if target_hit:
        hit = daily_df[daily_df["balance"] >= STARTING_BALANCE + PROFIT_TARGET]
        if len(hit):
            days_to_target = int(hit.index[0]) + 1

    results    = trades_df["win"].astype(int).tolist()
    max_streak = cur = 0
    for r in results:
        cur = cur + 1 if r == 0 else 0
        max_streak = max(max_streak, cur)

    min_buffer = daily_df["buffer"].min() if "buffer" in daily_df.columns else 0

    # Consistency flags
    cumulative = 0.0
    flags = 0
    for _, row in daily_df.iterrows():
        if row["pnl"] > 0 and cumulative > 0:
            if row["pnl"] > 0.50 * cumulative:
                flags += 1
        cumulative += row["pnl"]

    return {
        "n_contracts":         n_contracts,
        "total_trades":        n,
        "win_rate":            win_rate,
        "avg_win":             avg_win,
        "avg_loss":            avg_loss,
        "profit_factor":       profit_factor,
        "total_pnl":           total_pnl,
        "sharpe":              sharpe,
        "max_drawdown":        max_dd,
        "final_balance":       mll.balance,
        "mll_breached":        mll.is_breached(),
        "target_hit":          target_hit,
        "days_to_target":      days_to_target,
        "worst_losing_streak": max_streak,
        "min_mll_buffer":      min_buffer,
        "n_consistency_flags": flags,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_stats):
    labels = [f"{s['n_contracts']} MNQ" for s in all_stats]
    w = 11

    print()
    print("=" * 76)
    print("  MNQ MOMENTUM  |  Full 4-Paper Signal Stack  |  Lucid 100K")
    print("=" * 76)
    print(f"  {'Metric':<32}" + "".join(f"{l:>{w}}" for l in labels))
    print("  " + "-" * 74)

    def row(label, key, fmt):
        vals = []
        for s in all_stats:
            v = s.get(key)
            if v is None:
                vals.append("N/A")
            else:
                try:
                    vals.append(fmt.format(v))
                except Exception:
                    vals.append(str(v))
        print(f"  {label:<32}" + "".join(f"{v:>{w}}" for v in vals))

    row("Total trades",          "total_trades",        "{:.0f}")
    row("Win rate",              "win_rate",            "{:.1%}")
    row("Avg win ($)",           "avg_win",             "${:.0f}")
    row("Avg loss ($)",          "avg_loss",            "${:.0f}")
    row("Profit factor",         "profit_factor",       "{:.2f}")
    row("Total P&L",             "total_pnl",           "${:,.0f}")
    row("Sharpe (annualised)",   "sharpe",              "{:.2f}")
    row("Max drawdown",          "max_drawdown",        "${:,.0f}")
    row("Worst losing streak",   "worst_losing_streak", "{:.0f} days")
    row("Min MLL buffer seen",   "min_mll_buffer",      "${:,.0f}")
    row("Final balance",         "final_balance",       "${:,.0f}")
    row("MLL breached",          "mll_breached",        "{}")
    row("Profit target hit",     "target_hit",          "{}")
    row("Days to target",        "days_to_target",      "{}")
    row("Consistency flags",     "n_consistency_flags", "{:.0f}")

    print("=" * 76)
    print()

    safe = [s for s in all_stats if not s["mll_breached"]]
    print("  RECOMMENDATION")
    print("  " + "-" * 55)
    if safe:
        best = max(safe, key=lambda s: s["total_pnl"])
        n    = best["n_contracts"]
        print(f"  Optimal size:  {n} MNQ contract(s)")
        if best["target_hit"]:
            d = best["days_to_target"]
            print(f"  Est. eval completion: {d} trading days (~{d//5} weeks)")
        print(f"  Est. monthly P&L: ${best['total_pnl']/12:,.0f}")
        print(f"  Min MLL buffer:   ${best['min_mll_buffer']:,.0f}")
    else:
        print("  All sizes breached MLL — tune signal thresholds.")
    print()
    print("=" * 76)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("MNQ Momentum — Full Four-Paper Signal Stack | Multi-Contract Backtest")
    print("=" * 70)

    df = load_data()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_stats = []

    for n in CONTRACT_SIZES:
        print(f"  Running {n} contract(s)...", end=" ", flush=True)
        trades_df, daily_df, mll = run_backtest(df, n)
        stats = compute_stats(trades_df, daily_df, mll, n)
        all_stats.append(stats)

        trades_df.to_csv(
            RESULTS_DIR / f"mnq_trades_{n}ct_{timestamp}.csv", index=False)
        daily_df.to_csv(
            RESULTS_DIR / f"mnq_daily_{n}ct_{timestamp}.csv",  index=False)

        breach = "MLL BREACH" if mll.is_breached() else "safe"
        print(f"trades={stats['total_trades']}  "
              f"wr={stats['win_rate']:.1%}  "
              f"P&L=${stats['total_pnl']:,.0f}  "
              f"MLL={breach}")

    print_report(all_stats)


if __name__ == "__main__":
    main()
