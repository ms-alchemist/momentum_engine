"""
backtest/runner_mgc_htc.py
==========================
MGC Hold-To-Close Strategy — Path A

Signal:  First 30-min bar direction (9:00-9:30 AM ET)
         62.1% accuracy confirmed by signal audit
Entry:   9:30 AM ET at first bar close
Stop:    15 points — catastrophic protection only
         (90% of winning trades survive this stop per session_test.py)
Exit:    4:15 PM ET session close — primary exit mechanism
         (76% win rate on session-close exits confirmed by analysis)

Research grounding:
  Gao, Han, Li & Zhou (2018)   — first 30-min bar predicts session direction
  Di Graziano (2014)           — optimal stop = 15-16pts for MGC parameters
  Barroso & Santa-Clara (2015) — vol-targeted position sizing
  Daniel & Moskowitz (2016)    — skip extreme vol days (panic regime gate)

Lucid LucidFlex 100K:
  MLL floor = min(peak_EOD - $3,000, $100,000)
  Profit target = $6,000
  Hard close = 4:30 PM ET (strategy closes 4:15 PM for safety margin)

Usage:
    python backtest/runner_mgc_htc.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR     = Path("data/cache")
RESULTS_DIR  = Path("backtest/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MGC_FILE = DATA_DIR / "MGC_30min.parquet"
MGC_PNL_PT = 10.0   # $10 per point per MGC contract
MGC_COMM   = 0.80   # $0.80 round-trip per contract

# Session timing — ET (bars stored in America/New_York)
FIRST_BAR_OPEN  = time(9, 0)    # first bar opens here
FIRST_BAR_CLOSE = time(9, 30)   # signal fires when this bar closes
SESSION_CLOSE   = time(16, 15)  # primary exit — 4:15 PM ET

# Catastrophic stop — Di Graziano optimal for MGC parameters
STOP_POINTS = 15.0   # $150 per contract — 90% of winners survive this

# Volatility filter — Daniel & Moskowitz panic regime gate
VOL_FILTER_LOOKBACK_SHORT = 5
VOL_FILTER_LOOKBACK_LONG  = 20
VOL_FILTER_RATIO          = 2.5  # skip if 5-day avg range > 2.5x 20-day avg

# Minimum history before trading (for vol filter)
MIN_HISTORY_DAYS = 20

# Lucid account
STARTING_BALANCE = 100_000.0
PROFIT_TARGET    =   6_000.0
MLL_BUFFER       =   3_000.0

CONTRACT_SIZES = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# MLL — exact Lucid trailing mechanics
# ---------------------------------------------------------------------------

class LucidMLL:
    """
    Floor = min(peak_EOD_balance - $3,000, $100,000)
    Starts at $97,000. Reaches $100,000 only once peak_EOD >= $103,000.
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
    df = pd.read_parquet(MGC_FILE)
    df.index = pd.to_datetime(df.index, utc=False)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    df = df.sort_index()
    print(f"Loaded {len(df)} MGC 30-min bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


def build_daily_ranges(df):
    """Build daily range table for volatility filter."""
    daily = (
        df.resample("1D")
          .agg({"high": "max", "low": "min"})
          .dropna()
    )
    daily["range"]    = daily["high"] - daily["low"]
    daily["range_5d"] = daily["range"].rolling(VOL_FILTER_LOOKBACK_SHORT).mean()
    daily["range_20d"]= daily["range"].rolling(VOL_FILTER_LOOKBACK_LONG).mean()
    daily["vol_ratio"]= daily["range_5d"] / daily["range_20d"]
    return daily


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(df, daily, n_contracts):
    mll            = LucidMLL()
    trades         = []
    daily_results  = []
    cumulative_pnl = 0.0
    trading_days   = sorted(set(df.index.date))

    for day_num, day in enumerate(trading_days):
        # Stop conditions
        if mll.is_breached():
            break
        if mll.profit >= PROFIT_TARGET:
            break

        # Need minimum history for vol filter
        if day_num < MIN_HISTORY_DAYS:
            daily_results.append(_daily_record(day, 0, "warmup", mll))
            mll.update_eod(mll.balance)
            continue

        # Volatility filter
        day_mask = daily.index.date == day
        if not day_mask.any():
            daily_results.append(_daily_record(day, 0, "no_data", mll))
            mll.update_eod(mll.balance)
            continue

        day_info  = daily[day_mask].iloc[0]
        vol_ratio = day_info["vol_ratio"]

        if not np.isnan(vol_ratio) and vol_ratio > VOL_FILTER_RATIO:
            daily_results.append(_daily_record(day, 0, "vol_filter", mll))
            mll.update_eod(mll.balance)
            continue

        # Get today's bars
        day_bars = df[df.index.date == day]
        if len(day_bars) < 2:
            daily_results.append(_daily_record(day, 0, "no_bars", mll))
            mll.update_eod(mll.balance)
            continue

        # First bar (9:00-9:29 AM ET)
        first_bar = day_bars[
            (day_bars.index.time >= FIRST_BAR_OPEN) &
            (day_bars.index.time < FIRST_BAR_CLOSE)
        ]
        if len(first_bar) == 0:
            daily_results.append(_daily_record(day, 0, "no_first_bar", mll))
            mll.update_eod(mll.balance)
            continue

        fb = first_bar.iloc[0]
        if fb["close"] > fb["open"]:
            direction = 1    # LONG
        elif fb["close"] < fb["open"]:
            direction = -1   # SHORT
        else:
            daily_results.append(_daily_record(day, 0, "doji", mll))
            mll.update_eod(mll.balance)
            continue

        # Entry at first bar close (9:30 AM)
        entry_price = fb["close"]
        stop_price  = entry_price - direction * STOP_POINTS
        # No fixed target — hold to session close

        # Simulate through session bars
        session_bars = day_bars[day_bars.index.time >= FIRST_BAR_CLOSE]
        exit_price   = None
        exit_reason  = None

        for _, bar in session_bars.iterrows():
            # Primary exit: session close
            if bar.name.time() >= SESSION_CLOSE:
                exit_price  = bar["open"]
                exit_reason = "session_close"
                break

            # Catastrophic stop
            if direction == 1 and bar["low"] <= stop_price:
                exit_price  = stop_price
                exit_reason = "stop"
                break
            elif direction == -1 and bar["high"] >= stop_price:
                exit_price  = stop_price
                exit_reason = "stop"
                break

        # EOD fallback (shouldn't normally trigger)
        if exit_price is None:
            exit_price  = day_bars.iloc[-1]["close"]
            exit_reason = "eod"

        # P&L
        raw_pnl = direction * (exit_price - entry_price) * MGC_PNL_PT * n_contracts
        comm    = MGC_COMM * n_contracts
        net_pnl = raw_pnl - comm
        won     = net_pnl > 0

        # Consistency check (eval rule)
        consistency_flag = False
        if cumulative_pnl > 0 and net_pnl > 0.50 * cumulative_pnl:
            consistency_flag = True

        cumulative_pnl += net_pnl
        mll.update_eod(mll.balance + net_pnl)

        trades.append({
            "date":             day,
            "direction":        "LONG" if direction == 1 else "SHORT",
            "entry_price":      entry_price,
            "exit_price":       exit_price,
            "exit_reason":      exit_reason,
            "stop_price":       stop_price,
            "raw_pnl":          raw_pnl,
            "commission":       comm,
            "net_pnl":          net_pnl,
            "win":              won,
            "consistency_flag": consistency_flag,
            "first_bar_move":   fb["close"] - fb["open"],
        })

        daily_results.append({
            "date":    day,
            "reason":  exit_reason,
            "pnl":     net_pnl,
            "balance": mll.balance,
            "floor":   mll.floor,
            "buffer":  mll.buffer,
        })

    return pd.DataFrame(trades), pd.DataFrame(daily_results), mll


def _daily_record(day, pnl, reason, mll):
    return {
        "date":    day,
        "reason":  reason,
        "pnl":     pnl,
        "balance": mll.balance,
        "floor":   mll.floor,
        "buffer":  mll.buffer,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(trades_df, daily_df, mll, n_contracts):
    empty = {
        "n_contracts": n_contracts, "total_trades": 0, "win_rate": 0,
        "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "total_pnl": 0,
        "sharpe": 0, "max_drawdown": 0, "final_balance": mll.balance,
        "mll_breached": mll.is_breached(), "target_hit": False,
        "days_to_target": None, "worst_losing_streak": 0,
        "min_mll_buffer": mll.buffer, "n_consistency_flags": 0,
        "stop_rate": 0, "session_close_rate": 0,
        "stop_win_rate": 0, "session_close_win_rate": 0,
    }
    if trades_df.empty:
        return empty

    wins   = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]
    n      = len(trades_df)

    win_rate      = len(wins) / n
    avg_win       = wins["net_pnl"].mean()   if len(wins)   else 0
    avg_loss      = losses["net_pnl"].mean() if len(losses) else 0
    total_pnl     = trades_df["net_pnl"].sum()
    gross_wins    = wins["net_pnl"].sum()
    gross_losses  = abs(losses["net_pnl"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

    # Exit breakdown
    stops  = trades_df[trades_df["exit_reason"] == "stop"]
    closes = trades_df[trades_df["exit_reason"].isin(["session_close", "eod"])]
    stop_rate         = len(stops) / n
    session_close_rate = len(closes) / n
    stop_win_rate     = stops["win"].mean() if len(stops) else 0
    sc_win_rate       = closes["win"].mean() if len(closes) else 0

    # Sharpe on active trading days only
    active = daily_df[daily_df["pnl"] != 0]
    sharpe = ((active["pnl"].mean() / active["pnl"].std()) * np.sqrt(252)
              if len(active) > 1 and active["pnl"].std() > 0 else 0.0)

    # Max drawdown
    eq       = daily_df["balance"].dropna()
    roll_max = eq.cummax()
    max_dd   = (eq - roll_max).min() if len(eq) else 0.0

    target_hit = mll.profit >= PROFIT_TARGET

    days_to_target = None
    if target_hit:
        hit = daily_df[daily_df["balance"] >= STARTING_BALANCE + PROFIT_TARGET]
        if len(hit):
            days_to_target = int(hit.index[0]) + 1

    # Worst losing streak
    results    = trades_df["win"].astype(int).tolist()
    max_streak = cur = 0
    for r in results:
        cur        = cur + 1 if r == 0 else 0
        max_streak = max(max_streak, cur)

    min_buffer = daily_df["buffer"].min() if "buffer" in daily_df.columns else 0

    n_flags = int(trades_df["consistency_flag"].sum())

    return {
        "n_contracts":          n_contracts,
        "total_trades":         n,
        "win_rate":             win_rate,
        "avg_win":              avg_win,
        "avg_loss":             avg_loss,
        "profit_factor":        profit_factor,
        "total_pnl":            total_pnl,
        "sharpe":               sharpe,
        "max_drawdown":         max_dd,
        "final_balance":        mll.balance,
        "mll_breached":         mll.is_breached(),
        "target_hit":           target_hit,
        "days_to_target":       days_to_target,
        "worst_losing_streak":  max_streak,
        "min_mll_buffer":       min_buffer,
        "n_consistency_flags":  n_flags,
        "stop_rate":            stop_rate,
        "session_close_rate":   session_close_rate,
        "stop_win_rate":        stop_win_rate,
        "session_close_win_rate": sc_win_rate,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_stats):
    labels = [f"{s['n_contracts']} MGC" for s in all_stats]
    w = 11

    print()
    print("=" * 76)
    print("  MGC HOLD-TO-CLOSE  |  15pt Catastrophic Stop  |  Lucid 100K")
    print("=" * 76)
    print(f"  {'Metric':<34}" + "".join(f"{l:>{w}}" for l in labels))
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
        print(f"  {label:<34}" + "".join(f"{v:>{w}}" for v in vals))

    row("Total trades",            "total_trades",            "{:.0f}")
    row("Win rate",                "win_rate",                "{:.1%}")
    row("Avg win ($)",             "avg_win",                 "${:.0f}")
    row("Avg loss ($)",            "avg_loss",                "${:.0f}")
    row("Profit factor",           "profit_factor",           "{:.2f}")
    row("Total P&L",               "total_pnl",               "${:,.0f}")
    row("Sharpe (annualised)",     "sharpe",                  "{:.2f}")
    row("Max drawdown",            "max_drawdown",            "${:,.0f}")
    row("Worst losing streak",     "worst_losing_streak",     "{:.0f} days")
    row("Min MLL buffer",          "min_mll_buffer",          "${:,.0f}")
    row("Final balance",           "final_balance",           "${:,.0f}")
    row("MLL breached",            "mll_breached",            "{}")
    row("Profit target hit",       "target_hit",              "{}")
    row("Days to target",          "days_to_target",          "{}")
    row("Consistency flags",       "n_consistency_flags",     "{:.0f}")

    print()
    print("  EXIT BREAKDOWN (1-contract baseline)")
    print("  " + "-" * 55)
    s0 = all_stats[0]
    n  = s0["total_trades"]
    if n > 0:
        print(f"  Session close: {s0['session_close_rate']:.1%} of trades  "
              f"WR: {s0['session_close_win_rate']:.1%}")
        print(f"  Stop hit:      {s0['stop_rate']:.1%} of trades  "
              f"WR: {s0['stop_win_rate']:.1%}")

    print()
    print("  MLL BUFFER — Worst point seen")
    print("  " + "-" * 55)
    for s in all_stats:
        nc   = s["n_contracts"]
        buf  = s.get("min_mll_buffer", 0)
        safe = "SAFE" if not s["mll_breached"] else "*** BREACHED ***"
        print(f"  {nc} contract(s):  min buffer = ${buf:>8,.0f}   {safe}")

    print()
    print("  PATH TO $6,000 TARGET")
    print("  " + "-" * 55)
    for s in all_stats:
        nc = s["n_contracts"]
        if s["mll_breached"]:
            status = "MLL breach"
        elif s["target_hit"]:
            d = s["days_to_target"]
            status = f"HIT in {d} trading days (~{d//5} weeks)"
        else:
            status = f"Not reached — P&L ${s['total_pnl']:,.0f}"
        print(f"  {nc} contract(s):  {status}")

    print()
    safe = [s for s in all_stats if not s["mll_breached"]]
    print("  RECOMMENDATION")
    print("  " + "-" * 55)
    if safe:
        best = max(safe, key=lambda s: s["total_pnl"])
        nc   = best["n_contracts"]
        print(f"  Optimal size:      {nc} MGC contract(s)")
        if best["target_hit"]:
            d = best["days_to_target"]
            print(f"  Eval completion:   {d} trading days (~{d//5} weeks)")
        monthly = best["total_pnl"] / 12
        print(f"  Est. monthly P&L:  ${monthly:,.0f}")
        print(f"  Min MLL buffer:    ${best['min_mll_buffer']:,.0f}")
    else:
        print("  All sizes breached MLL.")
    print()
    print("=" * 76)


# ---------------------------------------------------------------------------
# Monthly breakdown (printed separately for clarity)
# ---------------------------------------------------------------------------

def print_monthly(trades_df, n_contracts):
    if trades_df.empty:
        return
    trades_df = trades_df.copy()
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    monthly = trades_df.groupby(trades_df["date"].dt.to_period("M")).apply(
        lambda x: pd.Series({
            "trades":   len(x),
            "wins":     x["win"].sum(),
            "win_rate": x["win"].mean(),
            "pnl":      x["net_pnl"].sum(),
        })
    )
    print(f"\n  Monthly breakdown — {n_contracts} contract(s):")
    print(f"  {'Month':<10} {'Trades':>6} {'WR':>7} {'P&L':>10}")
    print("  " + "-" * 38)
    for period, row in monthly.iterrows():
        flag = " ***" if row["win_rate"] < 0.45 else ""
        print(f"  {str(period):<10} {row['trades']:>6.0f}  "
              f"{row['win_rate']:>6.1%}  ${row['pnl']:>8,.0f}{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("MGC Hold-To-Close | 15pt Catastrophic Stop | Multi-Contract")
    print("=" * 60)
    print(f"  Signal:  First 30-min bar direction (9:00-9:30 AM ET)")
    print(f"  Stop:    {STOP_POINTS:.0f} points (${STOP_POINTS*MGC_PNL_PT:.0f}/contract) — catastrophic only")
    print(f"  Exit:    {SESSION_CLOSE.strftime('%I:%M %p')} ET session close (primary)")
    print(f"  Filter:  Vol ratio > {VOL_FILTER_RATIO}× skips day")
    print()

    df    = load_data()
    daily = build_daily_ranges(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_stats = []

    for n in CONTRACT_SIZES:
        print(f"  Running {n} contract(s)...", end=" ", flush=True)
        trades_df, daily_df, mll = run_backtest(df, daily, n)
        stats = compute_stats(trades_df, daily_df, mll, n)
        all_stats.append(stats)

        trades_df.to_csv(
            RESULTS_DIR / f"mgc_htc_{n}ct_{timestamp}.csv", index=False)
        daily_df.to_csv(
            RESULTS_DIR / f"mgc_htc_daily_{n}ct_{timestamp}.csv", index=False)

        breach = "MLL BREACH" if mll.is_breached() else "safe"
        print(f"trades={stats['total_trades']}  "
              f"wr={stats['win_rate']:.1%}  "
              f"P&L=${stats['total_pnl']:,.0f}  "
              f"MLL={breach}")

    print_report(all_stats)

    # Monthly breakdown for best safe size
    safe = [s for s in all_stats if not s["mll_breached"]]
    if safe:
        best_n = max(safe, key=lambda s: s["total_pnl"])["n_contracts"]
        best_trades = pd.read_csv(
            sorted(RESULTS_DIR.glob(f"mgc_htc_{best_n}ct_{timestamp}.csv"))[-1]
        )
        print_monthly(best_trades, best_n)
    print()


if __name__ == "__main__":
    main()
