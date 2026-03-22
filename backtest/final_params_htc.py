"""
backtest/final_params_htc.py
=============================
Tests the final strategy configuration:
  - 22pt stop
  - Hold to session close
  - Prior month max range <= 50pt filter (skips extreme vol months)
  - Tests 1, 2, 3 contracts with profit target stopping condition

This is the eval-ready configuration.
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
STOP_POINTS     = 22.0

STARTING_BALANCE  = 100_000.0
PROFIT_TARGET     =   6_000.0
MLL_BUFFER        =   3_000.0
VOL_FILTER_RATIO  = 2.5
MIN_HISTORY_DAYS  = 20
MAX_PRIOR_RANGE   = 50.0   # skip months after extreme vol


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


def build_monthly_range(df):
    daily = df.resample("1D").agg({"high":"max","low":"min"}).dropna()
    daily["range"] = daily["high"] - daily["low"]
    monthly = daily["range"].resample("ME").mean()
    monthly.index = monthly.index.to_period("M")
    return monthly


def run(df, daily, monthly_range, n_contracts, stop_at_target=True):
    balance       = STARTING_BALANCE
    peak_eod      = STARTING_BALANCE
    floor         = STARTING_BALANCE - MLL_BUFFER
    trades        = []
    daily_results = []
    cumulative    = 0.0

    for day_num, day in enumerate(sorted(set(df.index.date))):
        if balance < floor:
            break
        if stop_at_target and (balance - STARTING_BALANCE) >= PROFIT_TARGET:
            break

        if day_num < MIN_HISTORY_DAYS:
            daily_results.append({"date": day, "pnl": 0,
                                   "balance": balance, "floor": floor,
                                   "buffer": balance - floor})
            continue

        # Prior month extreme vol filter
        current_month = pd.Period(day, freq="M")
        prior_month   = current_month - 1
        if prior_month in monthly_range.index:
            if monthly_range[prior_month] > MAX_PRIOR_RANGE:
                daily_results.append({"date": day, "pnl": 0,
                                       "balance": balance, "floor": floor,
                                       "buffer": balance - floor})
                continue

        # Daily vol filter
        mask = daily.index.date == day
        if not mask.any():
            continue
        info = daily[mask].iloc[0]
        if not np.isnan(info["vol_ratio"]) and info["vol_ratio"] > VOL_FILTER_RATIO:
            daily_results.append({"date": day, "pnl": 0,
                                   "balance": balance, "floor": floor,
                                   "buffer": balance - floor})
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
        stop_price = entry - direction * STOP_POINTS
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
        balance    += pnl
        cumulative += pnl

        if balance > peak_eod:
            peak_eod = balance
        floor = min(peak_eod - MLL_BUFFER, STARTING_BALANCE)

        consistency_flag = (cumulative > 0 and pnl > 0 and
                            pnl > 0.50 * cumulative)

        trades.append({
            "date": day, "pnl": pnl, "win": pnl > 0,
            "exit_reason": exit_reason,
            "consistency_flag": consistency_flag,
        })
        daily_results.append({"date": day, "pnl": pnl,
                               "balance": balance, "floor": floor,
                               "buffer": balance - floor})

    return pd.DataFrame(trades), pd.DataFrame(daily_results), balance, floor


def summarize(trades_df, daily_df, balance, floor, n, label=""):
    if trades_df.empty:
        print(f"  {n}ct: NO TRADES")
        return

    wins   = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]
    n_t    = len(trades_df)
    wr     = trades_df["win"].mean()
    pf     = (wins["pnl"].sum() / abs(losses["pnl"].sum())
              if len(losses) and losses["pnl"].sum() != 0 else np.inf)
    total  = trades_df["pnl"].sum()

    active = daily_df[daily_df["pnl"] != 0]
    sharpe = ((active["pnl"].mean() / active["pnl"].std()) * np.sqrt(252)
              if len(active) > 1 and active["pnl"].std() > 0 else 0)

    eq     = daily_df["balance"]
    max_dd = (eq - eq.cummax()).min()
    min_buf = daily_df["buffer"].min()
    safe   = balance >= floor

    stops  = trades_df[trades_df["exit_reason"] == "stop"]
    closes = trades_df[trades_df["exit_reason"].isin(["session_close","eod"])]
    flags  = int(trades_df["consistency_flag"].sum())

    target_hit = (balance - STARTING_BALANCE) >= PROFIT_TARGET
    if target_hit:
        hit_idx = daily_df[daily_df["balance"] >=
                           STARTING_BALANCE + PROFIT_TARGET].index
        days_to = int(hit_idx[0]) + 1 if len(hit_idx) else "?"
    else:
        days_to = "N/A"

    print(f"\n  {n} contract(s) {label}")
    print(f"  Trades={n_t}  WR={wr:.1%}  PF={pf:.2f}  "
          f"P&L=${total:,.0f}  Sharpe={sharpe:.2f}")
    print(f"  MaxDD=${max_dd:,.0f}  MinBuf=${min_buf:,.0f}  "
          f"MLL={'SAFE' if safe else 'BREACH'}  "
          f"Target={'HIT' if target_hit else 'miss'}  "
          f"DaysToTarget={days_to}")
    print(f"  Stops={len(stops)/n_t:.0%}  SC={len(closes)/n_t:.0%}  "
          f"SC_WR={closes['win'].mean():.1%}  "
          f"ConsistencyFlags={flags}")

    # Monthly
    trades_df2 = trades_df.copy()
    trades_df2["date"] = pd.to_datetime(trades_df2["date"])
    monthly = trades_df2.groupby(
        trades_df2["date"].dt.to_period("M")
    )["pnl"].agg(["sum","count",lambda x:(x>0).mean()])
    monthly.columns = ["pnl","trades","wr"]

    print(f"  Monthly P&L:")
    for period, row in monthly.iterrows():
        flag = " ***" if row["wr"] < 0.45 else ""
        bar  = ("+" * min(int(abs(row["pnl"])/100), 20)
                if row["pnl"] >= 0
                else "-" * min(int(abs(row["pnl"])/100), 20))
        print(f"    {str(period):<8}  {row['trades']:>2.0f}d  "
              f"{row['wr']:>5.0%}  ${row['pnl']:>7,.0f}  {bar}{flag}")


def main():
    df            = load()
    daily         = build_daily(df)
    monthly_range = build_monthly_range(df)

    print()
    print("MGC Hold-To-Close — Final Configuration")
    print(f"  Stop: {STOP_POINTS:.0f}pt  |  Filter: prior month <= {MAX_PRIOR_RANGE:.0f}pt  "
          f"|  Vol ratio <= {VOL_FILTER_RATIO}x")
    print("=" * 60)

    print("\n--- WITH PROFIT TARGET (eval simulation) ---")
    for n in [1, 2, 3]:
        trades_df, daily_df, balance, floor = run(
            df, daily, monthly_range, n, stop_at_target=True)
        summarize(trades_df, daily_df, balance, floor, n, "— eval mode")

    print()
    print("\n--- FULL 12 MONTHS (no profit target cap) ---")
    for n in [1, 2, 3]:
        trades_df, daily_df, balance, floor = run(
            df, daily, monthly_range, n, stop_at_target=False)
        summarize(trades_df, daily_df, balance, floor, n, "— full year")

    print()


if __name__ == "__main__":
    main()
