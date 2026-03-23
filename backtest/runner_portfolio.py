"""
backtest/runner_portfolio.py
============================
Portfolio Backtest — MGC HTC + Wasserstein HMM Regime Overlay

Strategy logic (data-driven from Session 4 analysis):
  1. MGC hold-to-close (22pt stop) at FULL size always — primary edge
  2. ADD MNQ alongside MGC when HMM says trending_equity
  3. REDUCE MGC to 1 contract when HMM says volatile
  4. Choppy regime: trade MGC only at full size (rotation to MNQ hurts)

This is deliberately simple — the data showed that adding MNQ
in choppy months reduces P&L vs pure MGC.

Instruments:
  MGC: $10/pt, 22pt stop, session close exit, 9:30 AM ET
  MNQ: $2/pt,  80pt stop, session close exit, 9:30 AM ET

MLL: single unified account across both instruments
     floor = min(peak_EOD - $3,000, $100,000)

Usage:
    python backtest/runner_portfolio.py
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

DATA_DIR    = Path("data/cache")
RESULTS_DIR = Path("backtest/results")
RESULTS_DIR.mkdir(exist_ok=True)

# Instrument specs
SPECS = {
    "MGC": {"pv": 10.0, "comm": 0.80, "stop": 22.0},
    "MNQ": {"pv":  2.0, "comm": 0.35, "stop": 80.0},
}

# Session timing — ET
FIRST_BAR_OPEN  = time(9, 0)
FIRST_BAR_CLOSE = time(9, 30)
SESSION_CLOSE   = time(16, 15)

# Vol filter
VOL_FILTER_RATIO   = 2.5
VOL_LOOKBACK_SHORT = 5
VOL_LOOKBACK_LONG  = 20
MIN_HISTORY_DAYS   = 20

# Lucid account
STARTING_BALANCE = 100_000.0
PROFIT_TARGET    =   6_000.0
MLL_BUFFER       =   3_000.0

# Contract allocation by regime — data-driven from Session 4
# choppy: MGC 2ct only (rotation to MNQ hurts in our best months)
# trending_equity: MGC 2ct + MNQ 1ct (Nov/Dec 2024 equity rally)
# trending_commodity: MGC 2ct only (already our best months)
# volatile: MGC 1ct only (May 2025 Liberation Day — reduce exposure)
REGIME_CONTRACTS = {
    "trending_commodity": {"MGC": 2, "MNQ": 0},
    "trending_equity":    {"MGC": 2, "MNQ": 1},
    "volatile":           {"MGC": 1, "MNQ": 0},
    "choppy":             {"MGC": 2, "MNQ": 0},
    "insufficient_data":  {"MGC": 2, "MNQ": 0},
    "error":              {"MGC": 2, "MNQ": 0},
}


# ---------------------------------------------------------------------------
# MLL tracker
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars(symbol):
    path = DATA_DIR / f"{symbol}_30min.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=False)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def build_daily_ranges(df):
    daily = df.resample("1D").agg({"high":"max","low":"min"}).dropna()
    daily["range"]     = daily["high"] - daily["low"]
    daily["range_5d"]  = daily["range"].rolling(VOL_LOOKBACK_SHORT).mean()
    daily["range_20d"] = daily["range"].rolling(VOL_LOOKBACK_LONG).mean()
    daily["vol_ratio"] = daily["range_5d"] / daily["range_20d"]
    return daily


def load_regimes():
    path = RESULTS_DIR / "wasserstein_regimes.csv"
    if not path.exists():
        print("  WARNING: wasserstein_regimes.csv not found.")
        print("  Run: python signals/wasserstein_hmm.py first")
        return pd.Series(dtype=str)
    regimes = pd.read_csv(path, index_col=0, parse_dates=True)
    regimes.index = pd.to_datetime(regimes.index)
    if hasattr(regimes, 'iloc'):
        regimes = regimes.iloc[:, 0]
    regimes.name = "regime"
    print(f"  Loaded {len(regimes)} regime labels "
          f"({regimes.index[0].date()} to {regimes.index[-1].date()})")
    return regimes


# ---------------------------------------------------------------------------
# Single instrument trade simulation
# ---------------------------------------------------------------------------

def simulate_trade(day_bars, direction, stop_pts):
    """
    Simulate hold-to-close trade with catastrophic stop.
    Returns (exit_price, exit_reason).
    """
    first = day_bars[
        (day_bars.index.time >= FIRST_BAR_OPEN) &
        (day_bars.index.time < FIRST_BAR_CLOSE)
    ]
    if len(first) == 0:
        return None, None

    fb          = first.iloc[0]
    entry_price = fb["close"]
    stop_price  = entry_price - direction * stop_pts
    exit_price  = None
    exit_reason = None

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
        exit_price  = day_bars.iloc[-1]["close"]
        exit_reason = "eod"

    return entry_price, exit_price, exit_reason


def get_signal_direction(day_bars):
    """Get first-bar signal direction. Returns 1, -1, or 0."""
    first = day_bars[
        (day_bars.index.time >= FIRST_BAR_OPEN) &
        (day_bars.index.time < FIRST_BAR_CLOSE)
    ]
    if len(first) == 0:
        return 0
    fb = first.iloc[0]
    if fb["close"] > fb["open"]:   return 1
    if fb["close"] < fb["open"]:   return -1
    return 0


# ---------------------------------------------------------------------------
# Portfolio backtest
# ---------------------------------------------------------------------------

def run_portfolio_backtest(bars_mgc, bars_mnq, daily_mgc, regimes):
    mll           = LucidMLL()
    trades        = []
    daily_results = []
    cumulative    = 0.0

    trading_days = sorted(set(bars_mgc.index.date))

    for day_num, day in enumerate(trading_days):
        if mll.is_breached():
            break
        if mll.profit >= PROFIT_TARGET:
            break

        # Warm-up
        if day_num < MIN_HISTORY_DAYS:
            daily_results.append(_daily_rec(day, 0, mll, "warmup"))
            mll.update_eod(mll.balance)
            continue

        # Vol filter on MGC
        mask = daily_mgc.index.date == day
        if not mask.any():
            daily_results.append(_daily_rec(day, 0, mll, "no_daily"))
            mll.update_eod(mll.balance)
            continue

        info = daily_mgc[mask].iloc[0]
        if not np.isnan(info["vol_ratio"]) and info["vol_ratio"] > VOL_FILTER_RATIO:
            daily_results.append(_daily_rec(day, 0, mll, "vol_filter"))
            mll.update_eod(mll.balance)
            continue

        # Get regime — use previous trading day's label (strict causality)
        ts_day  = pd.Timestamp(day)
        regime  = "choppy"  # default
        if len(regimes) > 0:
            past = regimes[regimes.index < ts_day]
            if len(past) > 0:
                regime = str(past.iloc[-1])

        contracts = REGIME_CONTRACTS.get(regime, {"MGC": 2, "MNQ": 0})
        mgc_ct    = contracts["MGC"]
        mnq_ct    = contracts["MNQ"]

        # MGC bars
        mgc_day = bars_mgc[bars_mgc.index.date == day]
        if len(mgc_day) < 2:
            daily_results.append(_daily_rec(day, 0, mll, "no_bars"))
            mll.update_eod(mll.balance)
            continue

        # MGC signal and trade
        mgc_dir = get_signal_direction(mgc_day)
        daily_pnl = 0.0

        if mgc_dir != 0 and mgc_ct > 0:
            result = simulate_trade(mgc_day, mgc_dir, SPECS["MGC"]["stop"])
            if result[0] is not None:
                entry, exit_p, exit_r = result
                pnl = mgc_dir * (exit_p - entry) * SPECS["MGC"]["pv"] * mgc_ct
                pnl -= SPECS["MGC"]["comm"] * mgc_ct
                daily_pnl += pnl
                trades.append({
                    "date":        day,
                    "instrument":  "MGC",
                    "contracts":   mgc_ct,
                    "regime":      regime,
                    "direction":   "LONG" if mgc_dir == 1 else "SHORT",
                    "entry":       entry,
                    "exit":        exit_p,
                    "exit_reason": exit_r,
                    "net_pnl":     pnl,
                    "win":         pnl > 0,
                })

        # MNQ trade (only in trending_equity regime)
        if mnq_ct > 0 and day in set(bars_mnq.index.date):
            mnq_day = bars_mnq[bars_mnq.index.date == day]
            if len(mnq_day) >= 2:
                mnq_dir = get_signal_direction(mnq_day)
                if mnq_dir != 0:
                    result = simulate_trade(mnq_day, mnq_dir, SPECS["MNQ"]["stop"])
                    if result[0] is not None:
                        entry, exit_p, exit_r = result
                        pnl = mnq_dir * (exit_p - entry) * SPECS["MNQ"]["pv"] * mnq_ct
                        pnl -= SPECS["MNQ"]["comm"] * mnq_ct
                        daily_pnl += pnl
                        trades.append({
                            "date":        day,
                            "instrument":  "MNQ",
                            "contracts":   mnq_ct,
                            "regime":      regime,
                            "direction":   "LONG" if mnq_dir == 1 else "SHORT",
                            "entry":       entry,
                            "exit":        exit_p,
                            "exit_reason": exit_r,
                            "net_pnl":     pnl,
                            "win":         pnl > 0,
                        })

        cumulative += daily_pnl
        mll.update_eod(mll.balance + daily_pnl)

        daily_results.append({
            "date":    day,
            "reason":  "traded",
            "regime":  regime,
            "pnl":     daily_pnl,
            "balance": mll.balance,
            "floor":   mll.floor,
            "buffer":  mll.buffer,
            "mgc_ct":  mgc_ct,
            "mnq_ct":  mnq_ct,
        })

    return pd.DataFrame(trades), pd.DataFrame(daily_results), mll


def _daily_rec(day, pnl, mll, reason):
    return {"date": day, "reason": reason, "regime": "", "pnl": pnl,
            "balance": mll.balance, "floor": mll.floor,
            "buffer": mll.buffer, "mgc_ct": 0, "mnq_ct": 0}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(trades_df, daily_df, mll, label="Portfolio"):
    if trades_df.empty:
        print(f"  {label}: NO TRADES")
        return {}

    wins   = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]
    n      = len(trades_df)
    wr     = wins["win"].sum() / n
    total  = trades_df["net_pnl"].sum()
    pf     = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
              if len(losses) and losses["net_pnl"].sum() != 0 else np.inf)

    active = daily_df[daily_df["pnl"] != 0]
    sharpe = ((active["pnl"].mean() / active["pnl"].std()) * np.sqrt(252)
              if len(active) > 1 and active["pnl"].std() > 0 else 0)

    eq     = daily_df["balance"].dropna()
    max_dd = (eq - eq.cummax()).min()
    min_buf= daily_df["buffer"].min()
    target_hit = mll.profit >= PROFIT_TARGET

    print()
    print("=" * 65)
    print(f"  {label}")
    print("=" * 65)
    print(f"  Trades={n}  WR={wr:.1%}  PF={pf:.2f}  "
          f"P&L=${total:,.0f}  Sharpe={sharpe:.2f}")
    print(f"  MaxDD=${max_dd:,.0f}  MinBuf=${min_buf:,.0f}  "
          f"MLL={'SAFE' if not mll.is_breached() else 'BREACH'}  "
          f"Target={'HIT' if target_hit else 'miss'}")

    # By instrument
    print()
    if "instrument" not in trades_df.columns:
        return {"total_pnl": total, "sharpe": sharpe, "win_rate": wr,
                "profit_factor": pf, "mll_breached": mll.is_breached(),
                "target_hit": target_hit, "max_dd": max_dd}
    for inst in ["MGC", "MNQ"]:
        sub = trades_df[trades_df["instrument"] == inst]
        if len(sub) == 0:
            continue
        iw = sub[sub["win"]]
        il = sub[~sub["win"]]
        ipf = (iw["net_pnl"].sum() / abs(il["net_pnl"].sum())
               if len(il) and il["net_pnl"].sum() != 0 else np.inf)
        print(f"  {inst}: {len(sub)} trades  WR={sub['win'].mean():.1%}  "
              f"PF={ipf:.2f}  P&L=${sub['net_pnl'].sum():,.0f}")

    # By regime
    print()
    regime_stats = trades_df.groupby("regime")["net_pnl"].agg(
        trades="count", total="sum", avg="mean", wr=lambda x: (x>0).mean()
    )
    print(f"  {'Regime':<22} {'Trades':>7} {'WinRate':>8} {'Total':>10} {'Avg':>8}")
    print("  " + "-" * 58)
    for regime, row in regime_stats.iterrows():
        print(f"  {regime:<22} {row['trades']:>7.0f}  "
              f"{row['wr']:>7.1%}  ${row['total']:>8,.0f}  ${row['avg']:>6.0f}")

    # Monthly
    trades_df2 = trades_df.copy()
    trades_df2["date"] = pd.to_datetime(trades_df2["date"])
    monthly = trades_df2.groupby(
        trades_df2["date"].dt.to_period("M")
    )["net_pnl"].sum()

    print()
    print(f"  Monthly P&L:")
    for period, pnl in monthly.items():
        bar  = ("+" * min(int(abs(pnl)/100), 20)
                if pnl >= 0 else "-" * min(int(abs(pnl)/100), 20))
        sign = "+" if pnl >= 0 else ""
        print(f"    {str(period):<8}  {sign}${pnl:>8,.0f}  {bar}")

    return {"total_pnl": total, "sharpe": sharpe, "win_rate": wr,
            "profit_factor": pf, "mll_breached": mll.is_breached(),
            "target_hit": target_hit, "max_dd": max_dd}


# ---------------------------------------------------------------------------
# Baseline comparison — pure MGC 2ct
# ---------------------------------------------------------------------------

def run_mgc_baseline(bars_mgc, daily_mgc):
    """Pure MGC 2-contract baseline for comparison."""
    mll     = LucidMLL()
    trades  = []
    daily_r = []
    cum     = 0.0

    for day_num, day in enumerate(sorted(set(bars_mgc.index.date))):
        if mll.is_breached():
            break
        if mll.profit >= PROFIT_TARGET:
            break
        if day_num < MIN_HISTORY_DAYS:
            mll.update_eod(mll.balance)
            continue

        mask = daily_mgc.index.date == day
        if not mask.any():
            mll.update_eod(mll.balance)
            continue
        info = daily_mgc[mask].iloc[0]
        if not np.isnan(info["vol_ratio"]) and info["vol_ratio"] > VOL_FILTER_RATIO:
            daily_r.append({"date":day,"pnl":0,"balance":mll.balance,
                            "floor":mll.floor,"buffer":mll.buffer})
            mll.update_eod(mll.balance)
            continue

        day_bars = bars_mgc[bars_mgc.index.date == day]
        if len(day_bars) < 2:
            mll.update_eod(mll.balance)
            continue

        direction = get_signal_direction(day_bars)
        if direction == 0:
            mll.update_eod(mll.balance)
            continue

        result = simulate_trade(day_bars, direction, 22.0)
        if result[0] is None:
            mll.update_eod(mll.balance)
            continue

        entry, exit_p, exit_r = result
        pnl = direction * (exit_p - entry) * 10.0 * 2 - 0.80 * 2
        cum += pnl
        mll.update_eod(mll.balance + pnl)

        trades.append({"date":day,"net_pnl":pnl,"win":pnl>0,"exit_reason":exit_r})
        daily_r.append({"date":day,"pnl":pnl,"balance":mll.balance,
                        "floor":mll.floor,"buffer":mll.buffer})

    return pd.DataFrame(trades), pd.DataFrame(daily_r), mll


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("Portfolio Backtest — MGC HTC + Wasserstein HMM Regime Overlay")
    print("=" * 65)

    # Load data
    print("\nLoading data...")
    bars_mgc  = load_bars("MGC")
    bars_mnq  = load_bars("MNQ")
    daily_mgc = build_daily_ranges(bars_mgc)
    regimes   = load_regimes()

    print(f"  MGC: {len(bars_mgc)} 30-min bars")
    print(f"  MNQ: {len(bars_mnq)} 30-min bars")

    # Restrict to overlapping period for fair comparison
    # MGC starts 2024-03-22 (3yr fetch), MNQ also starts 2024-03-22
    start_date = max(bars_mgc.index[0].date(), bars_mnq.index[0].date())
    bars_mgc   = bars_mgc[bars_mgc.index.date >= start_date]
    bars_mnq   = bars_mnq[bars_mnq.index.date >= start_date]
    daily_mgc  = daily_mgc[daily_mgc.index.date >= start_date]
    print(f"  Common period: {start_date} to {bars_mgc.index[-1].date()}")

    # Run portfolio backtest
    print("\nRunning portfolio backtest (MGC + HMM regime overlay)...")
    trades_p, daily_p, mll_p = run_portfolio_backtest(
        bars_mgc, bars_mnq, daily_mgc, regimes)

    # Run pure MGC baseline
    print("Running pure MGC baseline (2 contracts, no regime filter)...")
    trades_b, daily_b, mll_b = run_mgc_baseline(bars_mgc, daily_mgc)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_p.to_csv(RESULTS_DIR/f"portfolio_{timestamp}.csv", index=False)
    trades_b.to_csv(RESULTS_DIR/f"baseline_{timestamp}.csv",  index=False)

    # Print comparison
    stats_p = print_report(trades_p, daily_p, mll_p,
                           "PORTFOLIO — MGC + HMM Regime Overlay")
    stats_b = print_report(trades_b, daily_b, mll_b,
                           "BASELINE — Pure MGC 2ct (no regime)")

    # Summary comparison
    print()
    print("=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    metrics = [("Total P&L", "total_pnl", "${:,.0f}"),
               ("Sharpe", "sharpe", "{:.2f}"),
               ("Win rate", "win_rate", "{:.1%}"),
               ("Profit factor", "profit_factor", "{:.2f}"),
               ("Max drawdown", "max_dd", "${:,.0f}")]

    print(f"  {'Metric':<20} {'Baseline':>12} {'Portfolio':>12} {'Delta':>10}")
    print("  " + "-" * 56)
    for label, key, fmt in metrics:
        b = stats_b.get(key, 0)
        p = stats_p.get(key, 0)
        try:
            delta = p - b
            print(f"  {label:<20} {fmt.format(b):>12} {fmt.format(p):>12} "
                  f"{'+' if delta>=0 else ''}{fmt.format(delta):>9}")
        except Exception:
            print(f"  {label:<20} {str(b):>12} {str(p):>12}")

    print()
    improvement = stats_p.get("sharpe", 0) - stats_b.get("sharpe", 0)
    if improvement > 0.1:
        print(f"  ✓ HMM overlay improves Sharpe by {improvement:+.2f}")
    elif improvement > 0:
        print(f"  ~ Marginal improvement: {improvement:+.2f} Sharpe")
    else:
        print(f"  ✗ HMM overlay does not improve performance ({improvement:+.2f})")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
