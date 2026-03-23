"""
backtest/runner_portfolio.py
=============================
Option A — Portfolio Combination
  MGC Hold-to-Close  +  MNQ Volatility Breakout
  Both running simultaneously on one Lucid LucidFlex 100K account.

MGC HTC params (LOCKED from Session 3):
  Signal: first 30-min bar direction
  Stop: 22pt, hold-to-close
  Vol filter: 5-day range > 2.5x 20-day avg → skip

MNQ Breakout params (best from diagnose_breakout.py):
  Consolidation: first 2 bars, range < 30% ATR
  Breakout: close outside range → enter, stop at opposite end

Scans: MGC 1-3 contracts × MNQ 1-3 contracts = 9 combinations

Key outputs per combination:
  - Total P&L over 3 years
  - Max drawdown
  - Min MLL buffer (closest to breach)
  - Days to hit $6,000 target
  - Consistency flags

Usage:
    python backtest/runner_portfolio.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time, datetime
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR   = Path("data/cache")
RESULTS_DIR = Path("backtest/results")
RESULTS_DIR.mkdir(exist_ok=True)

STARTING_BALANCE = 100_000.0
PROFIT_TARGET    =   6_000.0
MLL_BUFFER       =   3_000.0

MGC_PV            = 10.0
MGC_COMM          = 0.80
MGC_STOP_PTS      = 22.0
MGC_VOL_RATIO     = 2.5
MGC_OPEN          = time(9,  0)
MGC_SIGNAL        = time(9, 30)
MGC_CLOSE         = time(16, 15)

MNQ_PV            = 2.0
MNQ_COMM          = 0.35
MNQ_N_CONSOL      = 2
MNQ_ATR_THRESH    = 0.30
MNQ_OPEN          = time(9,  0)
MNQ_CLOSE         = time(16, 15)
ATR_LOOKBACK      = 20


class LucidMLL:
    def __init__(self):
        self.balance  = STARTING_BALANCE
        self.peak_eod = STARTING_BALANCE
        self.floor    = STARTING_BALANCE - MLL_BUFFER

    @property
    def buffer(self):  return self.balance - self.floor
    @property
    def profit(self):  return self.balance - STARTING_BALANCE

    def update_eod(self, b):
        self.balance = b
        if b > self.peak_eod:
            self.peak_eod = b
        self.floor = min(self.peak_eod - MLL_BUFFER, STARTING_BALANCE)

    def is_breached(self):
        return self.balance < self.floor


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


def get_mgc_pnl(day_bars, daily_row, n_ct):
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
            ep, reason = bar["open"], "session_close"
            break
        if sig == 1 and bar["low"]  <= stop:
            ep, reason = stop, "stop"; break
        if sig == -1 and bar["high"] >= stop:
            ep, reason = stop, "stop"; break

    if ep is None:
        ep, reason = post.iloc[-1]["close"] if len(post) else entry, "eod"

    pnl = sig * (ep - entry) * MGC_PV * n_ct - MGC_COMM * n_ct
    return {"pnl": pnl, "reason": reason, "instrument": "MGC"}


def get_mnq_pnl(day_bars, daily_row, n_ct):
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
            sig, entry, stop, bo_idx = 1, bar["close"], c_lo, i; break
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
    return {"pnl": pnl, "reason": reason, "instrument": "MNQ"}


def run_portfolio(mgc_bars, mgc_daily, mnq_bars, mnq_daily, n_mgc, n_mnq):
    mll     = LucidMLL()
    results = []
    trades  = []
    cum_pnl = 0.0

    all_days = sorted(set(mgc_bars.index.date) | set(mnq_bars.index.date))

    for day in all_days:
        if mll.is_breached() or mll.profit >= PROFIT_TARGET:
            break

        day_pnl = 0.0
        day_trades = []

        # MGC
        mgc_day  = mgc_bars[mgc_bars.index.date == day]
        mgc_mask = mgc_daily.index.date == day
        if len(mgc_day) > 0 and mgc_mask.any():
            row = mgc_daily[mgc_mask].iloc[0]
            di  = {"range_20d": row["range_20d"], "vol_ratio": row["vol_ratio"]}
            t   = get_mgc_pnl(mgc_day, di, n_mgc)
            if t:
                day_pnl += t["pnl"]
                day_trades.append(t)

        # MNQ
        mnq_day  = mnq_bars[mnq_bars.index.date == day]
        mnq_mask = mnq_daily.index.date == day
        if len(mnq_day) > 0 and mnq_mask.any():
            row = mnq_daily[mnq_mask].iloc[0]
            di  = {"atr20": row["atr20"]}
            t   = get_mnq_pnl(mnq_day, di, n_mnq)
            if t:
                day_pnl += t["pnl"]
                day_trades.append(t)

        cum_pnl += day_pnl
        mll.update_eod(mll.balance + day_pnl)

        cf = (cum_pnl > 0 and day_pnl > 0 and
              day_pnl > 0.50 * cum_pnl)

        results.append({
            "date": day, "day_pnl": day_pnl,
            "balance": mll.balance, "floor": mll.floor,
            "buffer": mll.buffer,
            "n_trades": len(day_trades),
            "consistency_flag": cf,
        })
        for t in day_trades:
            t.update({"date": day})
            trades.append(t)

    return pd.DataFrame(results), pd.DataFrame(trades), mll


def compute_stats(daily, trades, mll, n_mgc, n_mnq):
    eq       = daily["balance"]
    roll_max = eq.cummax()
    max_dd   = (eq - roll_max).min()

    target_hit     = mll.profit >= PROFIT_TARGET
    days_to_target = None
    if target_hit:
        hit = daily[daily["balance"] >= STARTING_BALANCE + PROFIT_TARGET]
        if len(hit):
            days_to_target = int(hit.index[0]) + 1

    active = daily[daily["n_trades"] > 0]
    sharpe = 0.0
    if len(active) > 1 and active["day_pnl"].std() > 0:
        sharpe = active["day_pnl"].mean() / active["day_pnl"].std() * np.sqrt(252)

    mgc_pnl = trades[trades["instrument"]=="MGC"]["pnl"].sum() if len(trades) else 0
    mnq_pnl = trades[trades["instrument"]=="MNQ"]["pnl"].sum() if len(trades) else 0

    return {
        "n_mgc": n_mgc, "n_mnq": n_mnq,
        "total_pnl":      mll.profit,
        "mgc_pnl":        mgc_pnl,
        "mnq_pnl":        mnq_pnl,
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
        "min_buffer":     daily["buffer"].min(),
        "mll_breached":   mll.is_breached(),
        "target_hit":     target_hit,
        "days_to_target": days_to_target,
        "n_consistency":  int(daily["consistency_flag"].sum()),
        "final_balance":  mll.balance,
    }


def print_report(all_stats, all_daily):
    print()
    print("=" * 80)
    print("  PORTFOLIO BACKTEST: MGC Hold-to-Close + MNQ Volatility Breakout")
    print("  Lucid LucidFlex 100K  |  $6,000 Target  |  $3,000 MLL Buffer")
    print("=" * 80)
    print(f"\n  {'Sizing':<14} {'P&L':>9} {'Sharpe':>7} "
          f"{'MaxDD':>9} {'MinBuf':>8} {'Target':>7} "
          f"{'Weeks':>6} {'CFlags':>7} {'MLL':>7}")
    print("  " + "-" * 80)

    all_stats.sort(key=lambda s: (not s["mll_breached"], s.get("days_to_target") or 9999))

    for s in all_stats:
        sizing  = f"{s['n_mgc']}xMGC+{s['n_mnq']}xMNQ"
        target  = "HIT" if s["target_hit"] else "miss"
        days    = s["days_to_target"] or 0
        weeks   = f"{days//5}wk" if days else "---"
        breach  = "BREACH" if s["mll_breached"] else "safe"
        flag    = "⚠" if s["mll_breached"] else " "
        print(f"  {flag}{sizing:<13} ${s['total_pnl']:>8,.0f}  "
              f"{s['sharpe']:>6.2f}  "
              f"${s['max_drawdown']:>7,.0f}  "
              f"${s['min_buffer']:>6,.0f}  "
              f"{target:>7}  {weeks:>6}  "
              f"{s['n_consistency']:>5}flg  {breach}")

    print()
    safe_hit = [s for s in all_stats
                if not s["mll_breached"] and s["target_hit"]]

    if safe_hit:
        print("  ─── PASSING COMBINATIONS ───────────────────────────────────────────")
        for s in safe_hit:
            wks = (s["days_to_target"] or 0) // 5
            print(f"\n  {s['n_mgc']}x MGC + {s['n_mnq']}x MNQ")
            print(f"    P&L total:      ${s['total_pnl']:,.0f}  "
                  f"(MGC ${s['mgc_pnl']:,.0f}  +  MNQ ${s['mnq_pnl']:,.0f})")
            print(f"    Sharpe:         {s['sharpe']:.2f}")
            print(f"    Max drawdown:   ${s['max_drawdown']:,.0f}")
            print(f"    Min MLL buffer: ${s['min_buffer']:,.0f}  "
                  f"({'SAFE' if s['min_buffer'] > 500 else 'TIGHT'})")
            print(f"    Days to target: {s['days_to_target']} trading days "
                  f"(~{wks} calendar weeks / ~{wks//4} months)")
            print(f"    Consistency:    {s['n_consistency']} flags")

    # Monthly P&L for best combo
    best = safe_hit[0] if safe_hit else next(
        (s for s in all_stats if not s["mll_breached"]), None)

    if best:
        idx = next(i for i, s in enumerate(all_stats)
                   if s["n_mgc"]==best["n_mgc"] and s["n_mnq"]==best["n_mnq"])
        df  = all_daily[idx].copy()
        df["date"] = pd.to_datetime(df["date"])
        monthly = df.groupby(df["date"].dt.to_period("M"))["day_pnl"].agg(
            total="sum", active=lambda x: (x!=0).sum()
        )
        pos = (monthly["total"] > 0).sum()
        print(f"\n  Monthly P&L — {best['n_mgc']}x MGC + {best['n_mnq']}x MNQ:")
        for period, row in monthly.iterrows():
            sign = "+" if row["total"] >= 0 else ""
            bar  = "█" * min(int(abs(row["total"])/300), 22)
            print(f"    {str(period):<8}  "
                  f"{sign}${row['total']:>8,.0f}  {bar}")
        print(f"  Positive months: {pos}/{len(monthly)}")

    print()
    print("=" * 80)


def main():
    print("\nLoading data...")
    mgc_bars  = load("MGC")
    mnq_bars  = load("MNQ")
    mgc_daily = build_mgc_daily(mgc_bars)
    mnq_daily = build_mnq_daily(mnq_bars)
    print(f"  MGC: {len(set(mgc_bars.index.date))} days  "
          f"MNQ: {len(set(mnq_bars.index.date))} days")

    all_stats = []
    all_daily = []

    print("\nRunning 9 combinations...\n")
    for n_mgc in [1, 2, 3]:
        for n_mnq in [1, 2, 3]:
            daily, trades, mll = run_portfolio(
                mgc_bars, mgc_daily, mnq_bars, mnq_daily, n_mgc, n_mnq)
            s = compute_stats(daily, trades, mll, n_mgc, n_mnq)
            all_stats.append(s)
            all_daily.append(daily)
            status = "BREACH" if s["mll_breached"] else (
                f"target in {s['days_to_target']}d" if s["target_hit"]
                else f"P&L=${s['total_pnl']:,.0f}")
            print(f"  {n_mgc}xMGC+{n_mnq}xMNQ:  "
                  f"Sh={s['sharpe']:.2f}  "
                  f"MinBuf=${s['min_buffer']:,.0f}  "
                  f"MaxDD=${s['max_drawdown']:,.0f}  "
                  f"{status}")

    print_report(all_stats, all_daily)


if __name__ == "__main__":
    main()
