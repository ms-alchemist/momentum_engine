"""
backtest/runner_mnq_full.py
===========================
Same as runner_mnq.py but WITHOUT the profit target stopping condition.
Runs the full 12 months to see true strategy performance.
This tells us if the 78% win rate / 9.14 Sharpe is real or just
luck in the first 10 trading days.
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

from signals.mnq_engine import MNQSignalEngine

DATA_DIR     = Path("data/cache")
RESULTS_DIR  = Path("backtest/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MNQ_FILE_30M     = DATA_DIR / "MNQ_30min.parquet"
MNQ_PNL_PT       = 2.0
MNQ_COMM         = 0.35

ENTRY_START_ET   = time(9, 30)
ENTRY_END_ET     = time(14, 0)
SESSION_CLOSE_ET = time(15, 45)

STARTING_BALANCE = 100_000.0
MLL_BUFFER       =   3_000.0
MAX_TRADES_PER_DAY = 2

CONTRACT_SIZES = [1, 2, 3, 4]


class LucidMLL:
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


def load_data():
    df = pd.read_parquet(MNQ_FILE_30M)
    df.index = pd.to_datetime(df.index, utc=False)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def _close_position(pos, exit_price, exit_reason, n_contracts, trades, exit_time):
    direction  = pos["direction"]
    price_diff = (exit_price - pos["entry_price"]) * direction
    gross_pnl  = price_diff * MNQ_PNL_PT * n_contracts
    comm       = MNQ_COMM * n_contracts
    net_pnl    = gross_pnl - comm
    trades.append({
        "entry_time":   pos["entry_time"],
        "exit_time":    exit_time,
        "direction":    "LONG" if direction == 1 else "SHORT",
        "entry_price":  pos["entry_price"],
        "exit_price":   exit_price,
        "exit_reason":  exit_reason,
        "stop_pts":     pos["stop_pts"],
        "target_pts":   pos["target_pts"],
        "gross_pnl":    gross_pnl,
        "commission":   comm,
        "net_pnl":      net_pnl,
        "win":          net_pnl > 0,
        "signal_score": pos["signal_score"],
    })
    return net_pnl


def run_backtest(df, n_contracts):
    engine        = MNQSignalEngine()
    mll           = LucidMLL()
    trades        = []
    daily_results = []
    cumulative    = 0.0

    for day in sorted(set(df.index.date)):
        # Only stop on MLL breach — NO profit target stop
        if mll.is_breached():
            break

        day_bars      = df[df.index.date == day]
        prior         = df[df.index.date < day].tail(32)
        daily_trades  = 0
        daily_pnl     = 0.0
        open_position = None

        for i, (ts, bar) in enumerate(day_bars.iterrows()):
            bar_time = ts.time()

            if bar_time >= SESSION_CLOSE_ET:
                if open_position is not None:
                    pnl = _close_position(open_position, float(bar["open"]),
                                          "session_close", n_contracts, trades, ts)
                    daily_pnl += pnl
                    cumulative += pnl
                    open_position = None
                break

            if open_position is not None:
                d = open_position["direction"]
                s = open_position["stop_price"]
                t = open_position["target_price"]
                ep, er = None, None
                if d == 1:
                    if bar["low"] <= s:  ep, er = s, "stop"
                    elif bar["high"] >= t: ep, er = t, "target"
                else:
                    if bar["high"] >= s: ep, er = s, "stop"
                    elif bar["low"] <= t: ep, er = t, "target"
                if ep is not None:
                    pnl = _close_position(open_position, ep, er,
                                          n_contracts, trades, ts)
                    daily_pnl  += pnl
                    cumulative += pnl
                    daily_trades += 1
                    open_position = None

            if (open_position is None
                    and daily_trades < MAX_TRADES_PER_DAY
                    and ENTRY_START_ET <= bar_time < ENTRY_END_ET
                    and i >= 1):

                bars_sig = pd.concat([prior, day_bars.iloc[:i+1]])
                if len(bars_sig) < 20:
                    continue

                signal = engine.compute(bars_sig)
                if signal.is_tradeable:
                    entry = float(bar["close"])
                    sp    = max(signal.stop_pts, 10.0)
                    tp    = max(signal.target_pts, 16.0)
                    if signal.direction == 1:
                        sp_price = entry - sp
                        tp_price = entry + tp
                    else:
                        sp_price = entry + sp
                        tp_price = entry - tp
                    open_position = {
                        "direction":    signal.direction,
                        "entry_price":  entry,
                        "entry_time":   ts,
                        "stop_price":   sp_price,
                        "target_price": tp_price,
                        "stop_pts":     sp,
                        "target_pts":   tp,
                        "signal_score": signal.combined_score,
                    }

        if open_position is not None:
            last = day_bars.iloc[-1]
            pnl  = _close_position(open_position, float(last["close"]),
                                   "eod", n_contracts, trades, day_bars.index[-1])
            daily_pnl  += pnl
            cumulative += pnl

        mll.update_eod(mll.balance + daily_pnl)
        daily_results.append({
            "date":    day,
            "pnl":     daily_pnl,
            "balance": mll.balance,
            "floor":   mll.floor,
            "buffer":  mll.buffer,
        })

    return pd.DataFrame(trades), pd.DataFrame(daily_results), mll


def summarize(trades_df, daily_df, mll, n):
    if trades_df.empty:
        print(f"  {n} contract(s): NO TRADES")
        return

    wins   = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]
    nt     = len(trades_df)
    wr     = len(wins) / nt
    pf     = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
               if len(losses) and losses["net_pnl"].sum() != 0 else np.inf)
    sharpe = ((daily_df["pnl"].mean() / daily_df["pnl"].std()) * np.sqrt(252)
               if daily_df["pnl"].std() > 0 else 0)
    eq     = daily_df["balance"]
    max_dd = (eq - eq.cummax()).min()
    buf    = daily_df["buffer"].min()

    # Monthly breakdown
    daily_df2 = daily_df.copy()
    daily_df2["date"] = pd.to_datetime(daily_df2["date"])
    monthly = daily_df2.groupby(daily_df2["date"].dt.to_period("M"))["pnl"].sum()

    # Exit breakdown
    exits = trades_df["exit_reason"].value_counts()

    print(f"\n  {n} MNQ contract(s) — FULL 12 MONTHS (no profit target cap)")
    print(f"  Trades={nt}  WR={wr:.1%}  PF={pf:.2f}  "
          f"P&L=${trades_df['net_pnl'].sum():,.0f}  "
          f"Sharpe={sharpe:.2f}  MaxDD=${max_dd:,.0f}")
    print(f"  Min buffer=${buf:,.0f}  "
          f"MLL={'BREACHED' if mll.is_breached() else 'safe'}")
    print(f"  Avg win=${wins['net_pnl'].mean():.0f}  "
          f"Avg loss=${losses['net_pnl'].mean():.0f}" if len(losses) else
          f"  Avg win=${wins['net_pnl'].mean():.0f}  No losses!")
    print(f"  Exits: {dict(exits)}")
    print(f"  Monthly P&L:")
    for m, p in monthly.items():
        bar  = "+" * int(abs(p)/100) if p >= 0 else "-" * int(abs(p)/100)
        bar  = bar[:25]
        sign = "+" if p >= 0 else ""
        print(f"    {str(m):<8}  {sign}${p:>7,.0f}  {bar}")


def main():
    print()
    print("MNQ Full 12-Month Backtest — No Profit Target Cap")
    print("=" * 55)
    df = load_data()
    print(f"Loaded {len(df)} bars\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in CONTRACT_SIZES:
        print(f"Running {n} contract(s)...", end=" ", flush=True)
        trades_df, daily_df, mll = run_backtest(df, n)
        trades_df.to_csv(
            RESULTS_DIR / f"mnq_full_{n}ct_{timestamp}.csv", index=False)
        print("done")
        summarize(trades_df, daily_df, mll, n)

    print()
    print("=" * 55)


if __name__ == "__main__":
    main()
