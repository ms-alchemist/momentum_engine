"""
backtest/runner_v4.py
=====================
MGC Intraday Momentum Strategy — v4 (Di Graziano Calibrated)

Signal: First 30-min bar direction (9:00-9:30 AM ET)
Edge:   62.1% first-bar accuracy on MGC (signal audit 2025-03 to 2026-03)

Stops derived from survival analysis:
  Low-vol  (<20pt range):   stop=15pt, target=18pt
  Normal   (20-50pt range): stop=25pt, target=30pt
  High-vol (>50pt range):   stop=30pt, target=36pt

MLL: Lucid trailing mechanics correctly modelled.
  Floor = min(peak_EOD - $3,000, $100,000)
  Floor starts at $97,000 and only reaches $100,000 once peak_EOD >= $103,000.
  No premature locking on first winning trade.

Usage:
    python backtest/runner_v4.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time
import warnings
warnings.filterwarnings("ignore")

DATA_DIR     = Path("data/cache")
RESULTS_DIR  = Path("backtest/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MGC_FILE_30M = DATA_DIR / "MGC_30min.parquet"
MGC_PNL_PT   = 10.0
MGC_COMM     = 0.80

STOPS = {
    "low_vol":  {"stop": 15, "target": 18},
    "normal":   {"stop": 25, "target": 30},
    "high_vol": {"stop": 30, "target": 36},
}

VOL_FILTER_RATIO   = 2.5
VOL_LOOKBACK_SHORT = 5
VOL_LOOKBACK_LONG  = 20

SESSION_OPEN_ET  = time(9, 0)
SIGNAL_BAR_ET    = time(9, 30)
SESSION_CLOSE_ET = time(16, 30)

STARTING_BALANCE = 100_000.0
PROFIT_TARGET    =   6_000.0
MLL_BUFFER       =   3_000.0

CONTRACT_SIZES = [1, 2, 3, 4]


class LucidMLL:
    """
    Lucid trailing EOD drawdown limit — correctly modelled.

    Floor = min(peak_EOD_balance - $3,000, $100,000)

    Examples:
      Start:            peak=$100,000  floor=$97,000   buffer=$3,000
      After +$1,000:    peak=$101,000  floor=$98,000   buffer=$3,000
      After +$3,000:    peak=$103,000  floor=$100,000  buffer=$3,000 (max floor)
      After +$10,000:   peak=$110,000  floor=$100,000  buffer=$10,000 (grows)

    The floor never exceeds $100,000, so it only reaches that level
    once peak EOD hits $103,000 — not on the first $1 of profit.
    """
    def __init__(self):
        self.balance  = STARTING_BALANCE
        self.peak_eod = STARTING_BALANCE
        self.floor    = STARTING_BALANCE - MLL_BUFFER  # starts at $97,000

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
        # Floor trails peak but is capped at STARTING_BALANCE
        self.floor = min(self.peak_eod - MLL_BUFFER, STARTING_BALANCE)

    def is_breached(self):
        return self.balance < self.floor


def load_data():
    df = pd.read_parquet(MGC_FILE_30M)
    df.index = pd.to_datetime(df.index, utc=False)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    df = df.sort_index()
    print(f"Loaded {len(df)} MGC 30-min bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


def build_daily_ranges(df):
    daily = (
        df.resample("1D")
          .agg({"high": "max", "low": "min", "open": "first", "close": "last"})
          .dropna()
    )
    daily["range"]     = daily["high"] - daily["low"]
    daily["range_5d"]  = daily["range"].rolling(VOL_LOOKBACK_SHORT).mean()
    daily["range_20d"] = daily["range"].rolling(VOL_LOOKBACK_LONG).mean()
    daily["vol_ratio"] = daily["range_5d"] / daily["range_20d"]
    return daily


def get_regime(rng):
    if rng < 20:
        return "low_vol"
    elif rng < 50:
        return "normal"
    else:
        return "high_vol"


def run_backtest(df, daily, n_contracts):
    mll            = LucidMLL()
    trades         = []
    daily_results  = []
    cumulative_pnl = 0.0
    best_day_pnl   = 0.0

    trading_days = sorted(set(df.index.date))

    for day in trading_days:
        if mll.is_breached():
            break
        if mll.profit >= PROFIT_TARGET:
            break

        day_mask = daily.index.date == day
        if not day_mask.any():
            continue
        day_info   = daily[day_mask].iloc[0]
        vol_ratio  = day_info["vol_ratio"]
        prev_range = day_info["range"]

        if np.isnan(day_info["range_20d"]):
            mll.update_eod(mll.balance)
            continue

        if not np.isnan(vol_ratio) and vol_ratio > VOL_FILTER_RATIO:
            daily_results.append({
                "date": day, "filtered": True, "reason": "vol_filter",
                "pnl": 0.0, "trades": 0, "regime": "",
                "balance": mll.balance, "floor": mll.floor,
                "buffer": mll.buffer,
            })
            mll.update_eod(mll.balance)
            continue

        regime     = get_regime(prev_range)
        stop_pts   = STOPS[regime]["stop"]
        target_pts = STOPS[regime]["target"]

        day_bars = df[df.index.date == day].copy()
        if len(day_bars) < 2:
            mll.update_eod(mll.balance)
            continue

        first_bar = day_bars[
            (day_bars.index.time >= SESSION_OPEN_ET) &
            (day_bars.index.time < SIGNAL_BAR_ET)
        ]
        if len(first_bar) == 0:
            mll.update_eod(mll.balance)
            continue

        fb          = first_bar.iloc[0]
        first_open  = fb["open"]
        first_close = fb["close"]

        if first_close > first_open:
            direction = 1
        elif first_close < first_open:
            direction = -1
        else:
            mll.update_eod(mll.balance)
            continue

        entry_price  = first_close
        stop_price   = entry_price - direction * stop_pts
        target_price = entry_price + direction * target_pts

        remaining = day_bars[day_bars.index.time >= SIGNAL_BAR_ET]
        exit_price  = None
        exit_reason = None

        for _, bar in remaining.iterrows():
            if bar.name.time() >= SESSION_CLOSE_ET:
                exit_price  = bar["open"]
                exit_reason = "session_close"
                break
            if direction == 1:
                if bar["low"] <= stop_price:
                    exit_price  = stop_price
                    exit_reason = "stop"
                    break
                elif bar["high"] >= target_price:
                    exit_price  = target_price
                    exit_reason = "target"
                    break
            else:
                if bar["high"] >= stop_price:
                    exit_price  = stop_price
                    exit_reason = "stop"
                    break
                elif bar["low"] <= target_price:
                    exit_price  = target_price
                    exit_reason = "target"
                    break

        if exit_price is None:
            exit_price  = day_bars.iloc[-1]["close"]
            exit_reason = "eod"

        raw_pnl = direction * (exit_price - entry_price) * MGC_PNL_PT * n_contracts
        comm    = MGC_COMM * n_contracts
        net_pnl = raw_pnl - comm

        consistency_flag = False
        if cumulative_pnl > 0 and net_pnl > 0:
            if net_pnl > 0.50 * cumulative_pnl:
                consistency_flag = True

        cumulative_pnl += net_pnl
        if net_pnl > best_day_pnl:
            best_day_pnl = net_pnl

        mll.update_eod(mll.balance + net_pnl)

        trades.append({
            "date":             day,
            "direction":        "LONG" if direction == 1 else "SHORT",
            "entry_price":      entry_price,
            "exit_price":       exit_price,
            "exit_reason":      exit_reason,
            "stop_pts":         stop_pts,
            "target_pts":       target_pts,
            "regime":           regime,
            "raw_pnl":          raw_pnl,
            "commission":       comm,
            "net_pnl":          net_pnl,
            "win":              net_pnl > 0,
            "consistency_flag": consistency_flag,
        })

        daily_results.append({
            "date":        day,
            "filtered":    False,
            "reason":      "",
            "pnl":         net_pnl,
            "trades":      1,
            "regime":      regime,
            "exit_reason": exit_reason,
            "balance":     mll.balance,
            "floor":       mll.floor,
            "buffer":      mll.buffer,
        })

    return pd.DataFrame(trades), pd.DataFrame(daily_results), mll


def compute_stats(trades_df, daily_df, mll, n_contracts):
    if trades_df.empty:
        return {"n_contracts": n_contracts, "total_trades": 0,
                "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                "profit_factor": 0, "total_pnl": 0, "sharpe": 0,
                "max_drawdown": 0, "final_balance": mll.balance,
                "mll_breached": mll.is_breached(), "target_hit": False,
                "days_to_target": None, "n_consistency_flags": 0,
                "worst_losing_streak": 0, "min_mll_buffer": mll.buffer,
                "regime_stats": pd.DataFrame()}

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

    active = daily_df[~daily_df["filtered"]]
    sharpe = ((active["pnl"].mean() / active["pnl"].std()) * np.sqrt(252)
              if len(active) > 1 and active["pnl"].std() > 0 else 0.0)

    eq       = daily_df["balance"].dropna()
    roll_max = eq.cummax()
    max_dd   = (eq - roll_max).min() if len(eq) else 0.0

    target_hit = mll.profit >= PROFIT_TARGET

    days_to_target = None
    if target_hit:
        hit_rows = daily_df[daily_df["balance"] >= STARTING_BALANCE + PROFIT_TARGET]
        if len(hit_rows):
            days_to_target = int(hit_rows.index[0]) + 1

    regime_stats = (
        trades_df.groupby("regime")["net_pnl"]
                 .agg(count="count", total="sum", avg="mean")
    )

    results = trades_df["win"].astype(int).tolist()
    max_streak = cur = 0
    for r in results:
        cur = cur + 1 if r == 0 else 0
        max_streak = max(max_streak, cur)

    min_buffer = daily_df["buffer"].min() if "buffer" in daily_df.columns else 0

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
        "n_consistency_flags": int(trades_df["consistency_flag"].sum()),
        "worst_losing_streak": max_streak,
        "min_mll_buffer":      min_buffer,
        "regime_stats":        regime_stats,
    }


def print_report(all_stats, all_daily):
    labels = [f"{s['n_contracts']} MGC" for s in all_stats]
    w = 11

    print()
    print("=" * 76)
    print("  MGC MOMENTUM v4  |  Di Graziano Stops  |  Lucid LucidFlex 100K")
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

    s0 = all_stats[0]
    if "regime_stats" in s0 and not s0["regime_stats"].empty:
        print("  REGIME BREAKDOWN (1-contract baseline)")
        print("  " + "-" * 55)
        print(f"  {'Regime':<14} {'Trades':>8} {'Total P&L':>12} {'Avg P&L':>12}")
        for regime, r in s0["regime_stats"].iterrows():
            print(f"  {regime:<14} {r['count']:>8.0f} "
                  f"  ${r['total']:>9.0f}   ${r['avg']:>9.0f}")
    print()

    print("  MLL BUFFER — Worst point seen during backtest")
    print("  " + "-" * 55)
    for s in all_stats:
        n    = s["n_contracts"]
        buf  = s.get("min_mll_buffer", 0)
        safe = "SAFE" if not s["mll_breached"] else "*** BREACHED ***"
        print(f"  {n} contract(s):  min buffer = ${buf:>8,.0f}   {safe}")
    print()

    print("  PATH TO $6,000 PROFIT TARGET")
    print("  " + "-" * 55)
    for s in all_stats:
        n = s["n_contracts"]
        if s["mll_breached"]:
            status = "MLL breach — account blown"
        elif s["target_hit"]:
            status = f"TARGET HIT  —  {s['days_to_target']} trading days"
        else:
            status = f"Not reached — final P&L ${s['total_pnl']:,.0f}"
        print(f"  {n} contract(s):  {status}")
    print()

    safe_sizes = [s for s in all_stats if not s["mll_breached"]]
    print("  RECOMMENDATION")
    print("  " + "-" * 55)
    if safe_sizes:
        best = max(safe_sizes, key=lambda s: s["total_pnl"])
        n    = best["n_contracts"]
        print(f"  Optimal size:  {n} MGC contract(s)")
        if best["target_hit"]:
            print(f"  Est. eval completion: {best['days_to_target']} trading days "
                  f"(~{best['days_to_target']//5} calendar weeks)")
        monthly = best["total_pnl"] / 12
        print(f"  Est. monthly P&L at {n}x: ${monthly:,.0f}")
        print(f"  Min MLL buffer at {n}x:   ${best['min_mll_buffer']:,.0f}")
    else:
        print("  All sizes breached MLL — strategy needs adjustment.")
    print()
    print("=" * 76)


def main():
    print()
    print("MGC Momentum v4 — Di Graziano Calibrated | Multi-Contract Backtest")
    print("=" * 68)

    df    = load_data()
    daily = build_daily_ranges(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_stats = []
    all_daily = []

    for n in CONTRACT_SIZES:
        print(f"  Running {n} contract(s)...", end=" ")
        trades_df, daily_df, mll = run_backtest(df, daily, n)
        stats = compute_stats(trades_df, daily_df, mll, n)
        all_stats.append(stats)
        all_daily.append(daily_df)

        trades_df.to_csv(RESULTS_DIR / f"v4_trades_{n}ct_{timestamp}.csv", index=False)
        daily_df.to_csv(RESULTS_DIR  / f"v4_daily_{n}ct_{timestamp}.csv",  index=False)

        breach = "MLL BREACH" if mll.is_breached() else "safe"
        print(f"trades={stats['total_trades']}  "
              f"wr={stats['win_rate']:.1%}  "
              f"P&L=${stats['total_pnl']:,.0f}  "
              f"MLL={breach}")

    print_report(all_stats, all_daily)


if __name__ == "__main__":
    main()
