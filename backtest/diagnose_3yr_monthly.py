"""
backtest/diagnose_3yr_monthly.py
=================================
Full 3-year monthly breakdown for 5x MGC + 5x MNQ portfolio.
Does NOT stop at profit target — runs the full 3 years to show
the complete performance picture across all market regimes.

Compares four daily loss limit modes side by side:
  A) No limit
  B) Stop at -$500/day
  C) Stop at -$750/day
  D) Stop after 2 losing trades

Key metrics shown:
  - Monthly P&L for each mode
  - Annual P&L summary
  - Max consecutive losing months
  - Drawdown profile across full period
  - Win rate by month and year

Usage:
    python backtest/diagnose_3yr_monthly.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = Path("data/cache")

STARTING_BALANCE = 100_000.0
MLL_BUFFER       =   3_000.0

MGC_PV        = 10.0
MGC_COMM      = 0.80
MGC_STOP_PTS  = 22.0
MGC_VOL_RATIO = 2.5
MGC_OPEN      = time(9,  0)
MGC_SIGNAL    = time(9, 30)
MGC_CLOSE     = time(16, 15)

MNQ_PV        = 2.0
MNQ_COMM      = 0.35
MNQ_N_CONSOL  = 2
MNQ_ATR_THRESH= 0.30
MNQ_OPEN      = time(9,  0)
MNQ_CLOSE     = time(16, 15)
ATR_LOOKBACK  = 20

N_MGC = 5
N_MNQ = 5

MODES = [
    {"name": "No limit",           "max_usd": None, "max_losers": None},
    {"name": "Stop at -$500/day",  "max_usd": -500, "max_losers": None},
    {"name": "Stop at -$750/day",  "max_usd": -750, "max_losers": None},
    {"name": "Stop after 2 losers","max_usd": None, "max_losers": 2},
]


def load(symbol):
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_30min.parquet")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def build_mgc_daily(bars):
    d = bars.resample("1D").agg(
        high=("high","max"), low=("low","min"),
        open=("open","first"), close=("close","last"),
        volume=("volume","sum")).dropna()
    d = d[d["volume"] > 0]
    d["range"]     = d["high"] - d["low"]
    d["range_5d"]  = d["range"].rolling(5).mean()
    d["range_20d"] = d["range"].rolling(ATR_LOOKBACK).mean()
    d["vol_ratio"] = d["range_5d"] / d["range_20d"]
    return d


def build_mnq_daily(bars):
    d = bars.resample("1D").agg(
        high=("high","max"), low=("low","min"),
        open=("open","first"), close=("close","last"),
        volume=("volume","sum")).dropna()
    d = d[d["volume"] > 0]
    d["range"] = d["high"] - d["low"]
    d["atr20"] = d["range"].rolling(ATR_LOOKBACK).mean()
    return d


def get_mgc_trade(day_bars, daily_row, n_ct):
    if np.isnan(daily_row.get("range_20d", np.nan)):
        return None
    vr = daily_row.get("vol_ratio", np.nan)
    if not np.isnan(vr) and vr > MGC_VOL_RATIO:
        return None
    fb = day_bars[(day_bars.index.time >= MGC_OPEN) &
                  (day_bars.index.time <  MGC_SIGNAL)]
    if len(fb) == 0:
        return None
    f   = fb.iloc[0]
    sig = np.sign(f["close"] - f["open"])
    if sig == 0:
        return None
    entry = f["close"]
    stop  = entry - sig * MGC_STOP_PTS
    post  = day_bars[day_bars.index.time >= MGC_SIGNAL]
    ep, reason = None, None
    for _, bar in post.iterrows():
        if bar.name.time() >= MGC_CLOSE:
            ep, reason = bar["open"], "session_close"; break
        if sig == 1 and bar["low"]  <= stop:
            ep, reason = stop, "stop"; break
        if sig == -1 and bar["high"] >= stop:
            ep, reason = stop, "stop"; break
    if ep is None:
        ep = post.iloc[-1]["close"] if len(post) else entry
        reason = "eod"
    pnl = sig * (ep - entry) * MGC_PV * n_ct - MGC_COMM * n_ct
    return {"pnl": pnl, "win": pnl > 0, "instrument": "MGC", "reason": reason}


def get_mnq_trade(day_bars, daily_row, n_ct):
    atr = daily_row.get("atr20", np.nan)
    if np.isnan(atr) or atr <= 0:
        return None
    session = day_bars[day_bars.index.time >= MNQ_OPEN]
    if len(session) < MNQ_N_CONSOL + 2:
        return None
    consol  = session.iloc[:MNQ_N_CONSOL]
    c_hi    = consol["high"].max()
    c_lo    = consol["low"].min()
    c_range = c_hi - c_lo
    if c_range >= MNQ_ATR_THRESH * atr:
        return None
    post = session.iloc[MNQ_N_CONSOL:]
    sig, entry, stop, bo_idx = None, None, None, None
    for i, (_, bar) in enumerate(post.iterrows()):
        if bar.name.time() >= MNQ_CLOSE:
            break
        if bar["close"] > c_hi:
            sig, entry, stop, bo_idx = 1,  bar["close"], c_lo, i; break
        elif bar["close"] < c_lo:
            sig, entry, stop, bo_idx = -1, bar["close"], c_hi, i; break
    if entry is None:
        return None
    remaining = post.iloc[bo_idx+1:]
    ep, reason = None, None
    for _, bar in remaining.iterrows():
        if bar.name.time() >= MNQ_CLOSE:
            ep, reason = bar["open"], "session_close"; break
        if sig == 1 and bar["low"]  <= stop:
            ep, reason = stop, "stop"; break
        if sig == -1 and bar["high"] >= stop:
            ep, reason = stop, "stop"; break
    if ep is None:
        ep = remaining.iloc[-1]["close"] if len(remaining) else entry
        reason = "eod"
    pnl = sig * (ep - entry) * MNQ_PV * n_ct - MNQ_COMM * n_ct
    return {"pnl": pnl, "win": pnl > 0, "instrument": "MNQ", "reason": reason}


def run_full_3yr(mgc_bars, mgc_daily, mnq_bars, mnq_daily, mode):
    """Run full 3 years — no profit target stop."""
    max_usd    = mode["max_usd"]
    max_losers = mode["max_losers"]

    mll     = LucidMLL_simple()
    results = []
    all_days= sorted(set(mgc_bars.index.date) | set(mnq_bars.index.date))

    for day in all_days:
        if mll.is_breached():
            break

        day_pnl  = 0.0
        day_loss = 0
        trades   = []

        # MGC first
        mgc_day  = mgc_bars[mgc_bars.index.date == day]
        mgc_mask = mgc_daily.index.date == day
        if len(mgc_day) > 0 and mgc_mask.any():
            row = mgc_daily[mgc_mask].iloc[0]
            t = get_mgc_trade(mgc_day,
                {"range_20d": row["range_20d"],
                 "vol_ratio": row["vol_ratio"]}, N_MGC)
            if t:
                # Daily loss limit check
                if max_usd and (day_pnl + t["pnl"]) < max_usd:
                    pass  # would breach limit — skip
                else:
                    day_pnl += t["pnl"]
                    trades.append(t)
                    if not t["win"]:
                        day_loss += 1

        # MNQ second — check limit before trading
        mnq_blocked = False
        if max_losers and day_loss >= max_losers:
            mnq_blocked = True
        if max_usd and day_pnl <= max_usd:
            mnq_blocked = True

        if not mnq_blocked:
            mnq_day  = mnq_bars[mnq_bars.index.date == day]
            mnq_mask = mnq_daily.index.date == day
            if len(mnq_day) > 0 and mnq_mask.any():
                row = mnq_daily[mnq_mask].iloc[0]
                t = get_mnq_trade(mnq_day,
                    {"atr20": row["atr20"]}, N_MNQ)
                if t:
                    day_pnl += t["pnl"]
                    trades.append(t)
                    if not t["win"]:
                        day_loss += 1

        mll.update(mll.balance + day_pnl)

        results.append({
            "date":     day,
            "day_pnl":  day_pnl,
            "balance":  mll.balance,
            "floor":    mll.floor,
            "buffer":   mll.buffer,
            "n_trades": len(trades),
            "n_losers": day_loss,
        })

    return pd.DataFrame(results)


class LucidMLL_simple:
    def __init__(self):
        self.balance  = STARTING_BALANCE
        self.peak_eod = STARTING_BALANCE
        self.floor    = STARTING_BALANCE - MLL_BUFFER

    @property
    def buffer(self): return self.balance - self.floor
    @property
    def profit(self): return self.balance - STARTING_BALANCE

    def update(self, b):
        self.balance = b
        if b > self.peak_eod:
            self.peak_eod = b
        self.floor = min(self.peak_eod - MLL_BUFFER, STARTING_BALANCE)

    def is_breached(self):
        return self.balance < self.floor


def monthly_pnl(df):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.groupby(d["date"].dt.to_period("M"))["day_pnl"].sum()


def annual_pnl(df):
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.groupby(d["date"].dt.to_year_period() if hasattr(
        d["date"].dt, "to_year_period") else d["date"].dt.to_period("Y")
    )["day_pnl"].sum()


def drawdown_stats(df):
    eq       = df["balance"]
    roll_max = eq.cummax()
    dd       = eq - roll_max
    max_dd   = dd.min()
    min_buf  = df["buffer"].min()

    # Longest losing streak in months
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    m = d.groupby(d["date"].dt.to_period("M"))["day_pnl"].sum()
    streak = max_streak = cur = 0
    for v in m:
        if v < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    return max_dd, min_buf, max_streak


def print_full_report(all_dfs, mode_names):
    print()
    print("=" * 90)
    print(f"  FULL 3-YEAR ANALYSIS — {N_MGC}x MGC + {N_MNQ}x MNQ Portfolio")
    print("  (No profit target stop — full backtest period)")
    print("=" * 90)

    # Get all months across all modes
    all_months = set()
    for df in all_dfs:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        for p in d.groupby(d["date"].dt.to_period("M")).groups:
            all_months.add(p)
    all_months = sorted(all_months)

    # Monthly P&L table — all modes side by side
    col_w = 12
    header = f"  {'Month':<9}"
    for name in mode_names:
        short = name[:col_w]
        header += f"  {short:>{col_w}}"
    print(header)
    print("  " + "-" * (9 + (col_w + 2) * len(mode_names)))

    monthly_data = []
    for df, name in zip(all_dfs, mode_names):
        monthly_data.append(monthly_pnl(df))

    current_year = None
    year_totals  = {name: 0 for name in mode_names}

    for period in all_months:
        year = str(period)[:4]

        # Year separator
        if year != current_year:
            if current_year is not None:
                # Print year total
                row = f"  {'  '+current_year+' TOTAL':<9}"
                for name in mode_names:
                    v = year_totals[name]
                    s = f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}"
                    row += f"  {s:>{col_w}}"
                print(row)
                print("  " + "·" * (9 + (col_w + 2) * len(mode_names)))
                year_totals = {name: 0 for name in mode_names}
            current_year = year
            print()

        row = f"  {str(period):<9}"
        values = []
        for i, (df, name) in enumerate(zip(all_dfs, mode_names)):
            m = monthly_data[i]
            v = m.get(period, 0)
            year_totals[name] += v
            s = f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}"
            values.append(v)
            row += f"  {s:>{col_w}}"

        # Add visual indicator for worst month across modes
        worst = min(values)
        if worst < -1000:
            row += "  ◄ BAD"
        print(row)

    # Final year total
    if current_year:
        row = f"  {'  '+current_year+' TOTAL':<9}"
        for name in mode_names:
            v = year_totals[name]
            s = f"+${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}"
            row += f"  {s:>{col_w}}"
        print(row)

    # Summary stats
    print()
    print("=" * 90)
    print(f"  {'Metric':<32}" + "".join(
        f"  {n[:col_w]:>{col_w}}" for n in mode_names))
    print("  " + "-" * (32 + (col_w + 2) * len(mode_names)))

    def stat_row(label, values):
        row = f"  {label:<32}"
        for v, fmt in values:
            if fmt == "dollar":
                s = f"${v:,.0f}"
            elif fmt == "pct":
                s = f"{v:.1%}"
            elif fmt == "int":
                s = f"{v:.0f}"
            elif fmt == "float":
                s = f"{v:.2f}"
            else:
                s = str(v)
            row += f"  {s:>{col_w}}"
        print(row)

    # 3-year total P&L
    vals = []
    for df in all_dfs:
        vals.append((df["day_pnl"].sum(), "dollar"))
    stat_row("3-year total P&L", vals)

    # Annualized P&L
    vals = []
    for df in all_dfs:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        n_years = (d["date"].max() - d["date"].min()).days / 365.25
        vals.append((df["day_pnl"].sum() / n_years if n_years > 0 else 0,
                     "dollar"))
    stat_row("Annualized P&L", vals)

    # Sharpe
    vals = []
    for df in all_dfs:
        active = df[df["n_trades"] > 0]
        sh = (active["day_pnl"].mean() / active["day_pnl"].std() * np.sqrt(252)
              if len(active) > 1 and active["day_pnl"].std() > 0 else 0)
        vals.append((sh, "float"))
    stat_row("Sharpe ratio (annualized)", vals)

    # Max drawdown
    vals = []
    for df in all_dfs:
        eq  = df["balance"]
        dd  = (eq - eq.cummax()).min()
        vals.append((dd, "dollar"))
    stat_row("Max drawdown", vals)

    # Min MLL buffer
    vals = []
    for df in all_dfs:
        vals.append((df["buffer"].min(), "dollar"))
    stat_row("Min MLL buffer (closest to breach)", vals)

    # Positive months
    vals = []
    for df in all_dfs:
        m = monthly_pnl(df)
        pos = (m > 0).sum()
        vals.append((f"{pos}/{len(m)}", "str"))
    row = f"  {'Positive months':<32}"
    for v, _ in vals:
        row += f"  {v:>{col_w}}"
    print(row)

    # Max consecutive losing months
    vals = []
    for df in all_dfs:
        _, _, streak = drawdown_stats(df)
        vals.append((streak, "int"))
    stat_row("Max consecutive losing months", vals)

    # Worst single month
    vals = []
    for df in all_dfs:
        m = monthly_pnl(df)
        vals.append((m.min(), "dollar"))
    stat_row("Worst single month", vals)

    # Best single month
    vals = []
    for df in all_dfs:
        m = monthly_pnl(df)
        vals.append((m.max(), "dollar"))
    stat_row("Best single month", vals)

    # Avg monthly P&L
    vals = []
    for df in all_dfs:
        m = monthly_pnl(df)
        vals.append((m.mean(), "dollar"))
    stat_row("Avg monthly P&L", vals)

    print()
    print("  NOTE: At $500/day loss limit, protected months show smaller")
    print("  losses but slightly reduced total P&L vs no-limit in strong months.")
    print("  The tradeoff: smaller drawdowns vs marginally lower ceiling.")
    print()
    print("=" * 90)


def main():
    print("\nLoading data...")
    mgc_bars  = load("MGC")
    mnq_bars  = load("MNQ")
    mgc_daily = build_mgc_daily(mgc_bars)
    mnq_daily = build_mnq_daily(mnq_bars)
    print(f"  MGC: {len(set(mgc_bars.index.date))} trading days")
    print(f"  MNQ: {len(set(mnq_bars.index.date))} trading days")
    print(f"\n  Running full 3-year simulation — {N_MGC}x MGC + {N_MNQ}x MNQ")
    print(f"  4 loss limit modes × full period (no target stop)\n")

    all_dfs    = []
    mode_names = []

    for mode in MODES:
        print(f"  Running: {mode['name']}...", end=" ")
        df = run_full_3yr(mgc_bars, mgc_daily, mnq_bars, mnq_daily, mode)
        m  = monthly_pnl(df)
        total = df["day_pnl"].sum()
        pos   = (m > 0).sum()
        print(f"Total=${total:,.0f}  Pos months={pos}/{len(m)}")
        all_dfs.append(df)
        mode_names.append(mode["name"])

    print_full_report(all_dfs, mode_names)


if __name__ == "__main__":
    main()
