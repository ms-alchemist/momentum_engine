"""
backtest/runner_portfolio_v2.py
================================
Option A v2 — Extended sizing + Daily Loss Limits

Changes from v1:
  - Contract sizes extended to 5 (MGC and MNQ)
  - Daily loss limit modes:
      A) No daily limit (baseline)
      B) Stop after $500 loss on the day
      C) Stop after $750 loss on the day
      D) Stop after 2 losing trades on the day

Lucid max: 60 micro contracts total. 5 MGC + 5 MNQ = 10 contracts. Well within limits.

Note on Lucid consistency rule (eval only):
  Best single day P&L must stay <= 50% of total cumulative P&L.
  Flags tracked but not enforced — shown for awareness.

Usage:
    python backtest/runner_portfolio_v2.py
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

# Contract combos to test
CONTRACT_COMBOS = [
    (2, 2), (3, 3), (4, 4), (5, 5),
    (3, 5), (5, 3), (4, 5), (5, 4),
]

# Daily loss limit modes
LOSS_LIMIT_MODES = [
    {"name": "No limit",          "max_loss_usd": None, "max_losers": None},
    {"name": "Stop at -$500/day", "max_loss_usd": -500, "max_losers": None},
    {"name": "Stop at -$750/day", "max_loss_usd": -750, "max_losers": None},
    {"name": "Stop after 2 losers","max_loss_usd": None, "max_losers": 2},
]


class LucidMLL:
    def __init__(self):
        self.balance  = STARTING_BALANCE
        self.peak_eod = STARTING_BALANCE
        self.floor    = STARTING_BALANCE - MLL_BUFFER

    @property
    def buffer(self): return self.balance - self.floor
    @property
    def profit(self): return self.balance - STARTING_BALANCE

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
    return {"pnl": pnl, "reason": reason, "instrument": "MGC", "win": pnl > 0}


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
    return {"pnl": pnl, "reason": reason, "instrument": "MNQ", "win": pnl > 0}


def run_portfolio(mgc_bars, mgc_daily, mnq_bars, mnq_daily,
                  n_mgc, n_mnq, loss_mode):
    """
    Run portfolio with a specific daily loss limit mode.
    MGC fires first (opens at 9:30), MNQ fires later (after 10:00 consolidation).
    Daily loss limit applies across both instruments combined.
    """
    max_loss_usd = loss_mode["max_loss_usd"]
    max_losers   = loss_mode["max_losers"]

    mll      = LucidMLL()
    results  = []
    trades   = []
    cum_pnl  = 0.0

    all_days = sorted(set(mgc_bars.index.date) | set(mnq_bars.index.date))

    for day in all_days:
        if mll.is_breached() or mll.profit >= PROFIT_TARGET:
            break

        day_pnl    = 0.0
        day_trades = []
        day_losers = 0
        day_blocked= False

        # --- MGC fires first (signal at 9:30 AM) ---
        mgc_day  = mgc_bars[mgc_bars.index.date == day]
        mgc_mask = mgc_daily.index.date == day
        if len(mgc_day) > 0 and mgc_mask.any():
            row = mgc_daily[mgc_mask].iloc[0]
            di  = {"range_20d": row["range_20d"], "vol_ratio": row["vol_ratio"]}
            t   = get_mgc_trade(mgc_day, di, n_mgc)
            if t:
                # Check daily loss limit before adding
                projected_loss = day_pnl + t["pnl"]
                if (max_loss_usd and projected_loss < max_loss_usd):
                    day_blocked = True   # would breach — skip
                else:
                    day_pnl += t["pnl"]
                    day_trades.append(t)
                    if not t["win"]:
                        day_losers += 1

        # --- MNQ fires later (after 10:00 AM consolidation) ---
        # Apply daily loss limit: don't trade MNQ if already at limit
        mnq_day  = mnq_bars[mnq_bars.index.date == day]
        mnq_mask = mnq_daily.index.date == day

        mnq_blocked = False
        if max_losers and day_losers >= max_losers:
            mnq_blocked = True
        if max_loss_usd and day_pnl <= max_loss_usd:
            mnq_blocked = True

        if not mnq_blocked and len(mnq_day) > 0 and mnq_mask.any():
            row = mnq_daily[mnq_mask].iloc[0]
            di  = {"atr20": row["atr20"]}
            t   = get_mnq_trade(mnq_day, di, n_mnq)
            if t:
                day_pnl += t["pnl"]
                day_trades.append(t)
                if not t["win"]:
                    day_losers += 1

        cum_pnl += day_pnl
        mll.update_eod(mll.balance + day_pnl)

        cf = (cum_pnl > 0 and day_pnl > 0 and
              day_pnl > 0.50 * cum_pnl)

        results.append({
            "date": day, "day_pnl": day_pnl,
            "balance": mll.balance, "floor": mll.floor,
            "buffer": mll.buffer,
            "n_trades": len(day_trades),
            "n_losers": day_losers,
            "consistency_flag": cf,
            "blocked": day_blocked or mnq_blocked,
        })
        for t in day_trades:
            t["date"] = day
            trades.append(t)

    return pd.DataFrame(results), pd.DataFrame(trades), mll


def stats(daily, trades, mll, n_mgc, n_mnq, mode_name):
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
    n_flags = int(daily["consistency_flag"].sum())

    return {
        "n_mgc": n_mgc, "n_mnq": n_mnq,
        "mode": mode_name,
        "total_pnl":      mll.profit,
        "mgc_pnl":        mgc_pnl,
        "mnq_pnl":        mnq_pnl,
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
        "min_buffer":     daily["buffer"].min(),
        "mll_breached":   mll.is_breached(),
        "target_hit":     target_hit,
        "days_to_target": days_to_target,
        "n_consistency":  n_flags,
        "final_balance":  mll.balance,
    }


def print_section(all_results, title, filter_fn):
    subset = [r for r in all_results if filter_fn(r)]
    if not subset:
        return
    print(f"\n  ── {title} ──────────────────────────────────────────────────")
    print(f"  {'Sizing':<14} {'Mode':<22} {'P&L':>9} {'Sh':>6} "
          f"{'MaxDD':>9} {'MinBuf':>8} {'Days':>6} "
          f"{'Wks':>5} {'CF':>4}")
    print("  " + "-" * 90)
    for s in sorted(subset, key=lambda x: x.get("days_to_target") or 9999):
        sizing = f"{s['n_mgc']}MGC+{s['n_mnq']}MNQ"
        days   = s["days_to_target"] or 0
        weeks  = f"{days//5}" if days else "---"
        breach = " ⚠" if s["mll_breached"] else ""
        print(f"  {sizing:<14} {s['mode']:<22} "
              f"${s['total_pnl']:>8,.0f}  "
              f"{s['sharpe']:>5.2f}  "
              f"${s['max_drawdown']:>7,.0f}  "
              f"${s['min_buffer']:>6,.0f}  "
              f"{days:>6}d  {weeks:>4}wk  "
              f"{s['n_consistency']:>3}cf{breach}")


def main():
    print("\nLoading data...")
    mgc_bars  = load("MGC")
    mnq_bars  = load("MNQ")
    mgc_daily = build_mgc_daily(mgc_bars)
    mnq_daily = build_mnq_daily(mnq_bars)
    print(f"  MGC: {len(set(mgc_bars.index.date))} days  "
          f"MNQ: {len(set(mnq_bars.index.date))} days")

    all_results = []
    total_runs  = len(CONTRACT_COMBOS) * len(LOSS_LIMIT_MODES)
    run_n       = 0

    print(f"\nRunning {total_runs} combinations "
          f"({len(CONTRACT_COMBOS)} sizings × "
          f"{len(LOSS_LIMIT_MODES)} loss modes)...\n")

    for n_mgc, n_mnq in CONTRACT_COMBOS:
        for mode in LOSS_LIMIT_MODES:
            run_n += 1
            daily, trades, mll = run_portfolio(
                mgc_bars, mgc_daily, mnq_bars, mnq_daily,
                n_mgc, n_mnq, mode)
            s = stats(daily, trades, mll, n_mgc, n_mnq, mode["name"])
            all_results.append((s, daily))

            status = ("BREACH" if s["mll_breached"] else
                      f"{s['days_to_target']}d" if s["target_hit"]
                      else f"${s['total_pnl']:,.0f}")
            print(f"  [{run_n:>2}/{total_runs}] "
                  f"{n_mgc}MGC+{n_mnq}MNQ  {mode['name']:<22}  "
                  f"Sh={s['sharpe']:.2f}  "
                  f"MinBuf=${s['min_buffer']:,.0f}  "
                  f"MaxDD=${s['max_drawdown']:,.0f}  "
                  f"{status}")

    all_stats = [r[0] for r in all_results]
    all_daily = [r[1] for r in all_results]

    print()
    print("=" * 92)
    print("  PORTFOLIO v2: MGC Hold-to-Close + MNQ Volatility Breakout")
    print("  Extended sizing (up to 5x5) + Daily Loss Limits")
    print("=" * 92)

    # Safe + target hit
    safe_hit = [s for s in all_stats
                if not s["mll_breached"] and s["target_hit"]]

    print_section(all_stats, "ALL SAFE + TARGET HIT — sorted by days to target",
                  lambda s: not s["mll_breached"] and s["target_hit"])

    # Highlight fastest eval passes
    fastest = sorted(safe_hit, key=lambda s: s["days_to_target"] or 9999)
    if fastest:
        print()
        print("  ── TOP 5 FASTEST EVAL PASSES ─────────────────────────────────────────")
        for s in fastest[:5]:
            days = s["days_to_target"]
            wks  = days // 5
            mths = wks  // 4
            print(f"\n  {s['n_mgc']}x MGC + {s['n_mnq']}x MNQ  |  {s['mode']}")
            print(f"    Days to target:  {days} ({wks} calendar weeks / ~{mths} months)")
            print(f"    Total P&L:       ${s['total_pnl']:,.0f}  "
                  f"(MGC ${s['mgc_pnl']:,.0f} + MNQ ${s['mnq_pnl']:,.0f})")
            print(f"    Sharpe:          {s['sharpe']:.2f}")
            print(f"    Max drawdown:    ${s['max_drawdown']:,.0f}")
            print(f"    Min MLL buffer:  ${s['min_buffer']:,.0f}  "
                  f"({'comfortable' if s['min_buffer']>1000 else 'TIGHT'})")
            print(f"    Consistency flags: {s['n_consistency']}")

    # MLL breaches summary
    breached = [s for s in all_stats if s["mll_breached"]]
    if breached:
        print(f"\n  ⚠ {len(breached)} combinations breached MLL — "
              f"sizing too large or insufficient loss protection")
        for s in breached[:5]:
            print(f"    {s['n_mgc']}MGC+{s['n_mnq']}MNQ  {s['mode']:<22}  "
                  f"MinBuf=${s['min_buffer']:,.0f}")

    # Best combo monthly breakdown
    if fastest:
        best = fastest[0]
        best_idx = next(i for i, s in enumerate(all_stats)
                       if s["n_mgc"]==best["n_mgc"]
                       and s["n_mnq"]==best["n_mnq"]
                       and s["mode"]==best["mode"])
        df = all_daily[best_idx].copy()
        df["date"] = pd.to_datetime(df["date"])
        monthly = df.groupby(df["date"].dt.to_period("M"))["day_pnl"].agg(
            total="sum", active=lambda x: (x!=0).sum()
        )
        pos = (monthly["total"] > 0).sum()
        print(f"\n  Monthly P&L — {best['n_mgc']}x MGC + {best['n_mnq']}x MNQ"
              f"  [{best['mode']}]:")
        print(f"  {'Month':<10} {'P&L':>10}  Chart")
        print("  " + "-" * 50)
        for period, row in monthly.iterrows():
            sign = "+" if row["total"] >= 0 else ""
            bar  = "█" * min(int(abs(row["total"])/300), 24)
            neg  = "░" * min(int(abs(row["total"])/300), 24) if row["total"] < 0 else ""
            print(f"  {str(period):<10} {sign}${row['total']:>8,.0f}  "
                  f"{bar if row['total']>=0 else neg}")
        print(f"  Positive months: {pos}/{len(monthly)}")

    print()
    print("=" * 92)
    print()
    print("  KEY QUESTION: Does a daily loss limit improve eval speed?")
    print("  Compare 'No limit' vs best loss-limited version for each sizing.")
    print()

    # Side-by-side: no limit vs best limited version per combo
    print(f"  {'Sizing':<14} {'No limit':>10}  {'Best w/ limit':>14}  "
          f"{'Improvement':>12}")
    print("  " + "-" * 56)
    for n_mgc, n_mnq in CONTRACT_COMBOS:
        base = next((s for s in all_stats
                     if s["n_mgc"]==n_mgc and s["n_mnq"]==n_mnq
                     and s["mode"]=="No limit"), None)
        limited = [s for s in all_stats
                   if s["n_mgc"]==n_mgc and s["n_mnq"]==n_mnq
                   and s["mode"]!="No limit"
                   and not s["mll_breached"]
                   and s["target_hit"]]
        if not base or not limited:
            continue
        best_lim = min(limited, key=lambda s: s["days_to_target"] or 9999)
        base_days= base["days_to_target"] or 0
        lim_days = best_lim["days_to_target"] or 0
        diff     = base_days - lim_days
        sizing   = f"{n_mgc}MGC+{n_mnq}MNQ"
        base_str = f"{base_days}d" if base["target_hit"] else "miss"
        lim_str  = f"{lim_days}d [{best_lim['mode'][:12]}]"
        diff_str = f"-{diff}d faster" if diff > 0 else (
                   f"+{-diff}d slower" if diff < 0 else "same")
        print(f"  {sizing:<14} {base_str:>10}  {lim_str:>22}  {diff_str:>12}")


if __name__ == "__main__":
    main()
