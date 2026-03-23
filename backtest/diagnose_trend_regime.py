"""
backtest/diagnose_trend_regime.py
==================================
Diagnostic: Are MGC stop days clustered in sustained trends?

For every day MGC hits its 22-point stop, measure the prior
trend context using multiple lookbacks:
  - 5-day return  (1 week)
  - 10-day return (2 weeks)
  - 20-day return (1 month)
  - 60-day return (3 months)
  - Price vs 20-day SMA (above/below)
  - Price vs 50-day SMA (above/below)
  - ADX-style trend strength (directional movement)

Then test: does skipping MGC when prior trend is strong
(trending against the first-bar signal) reduce drawdown
without significantly hurting total P&L?

Tests:
  1. Profile of stop days vs winning days by trend context
  2. Simple trend filter: skip MGC when 20-day return > threshold
     AND first-bar signal opposes the trend
  3. Full backtest with trend filter applied
  4. Compare: no filter vs trend filter on full 3 years

Usage:
    python backtest/diagnose_trend_regime.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time
import warnings
warnings.filterwarnings("ignore")

CACHE_DIR = Path("data/cache")

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
STARTING_BALANCE = 100_000.0
MLL_BUFFER       =   3_000.0

N_MGC = 5
N_MNQ = 5


def load(symbol):
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_30min.parquet")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def build_mgc_context(bars):
    """Build daily context table with trend indicators."""
    d = bars.resample("1D").agg(
        high=("high","max"), low=("low","min"),
        open=("open","first"), close=("close","last"),
        volume=("volume","sum")).dropna()
    d = d[d["volume"] > 0]

    d["range"]     = d["high"] - d["low"]
    d["range_5d"]  = d["range"].rolling(5).mean()
    d["range_20d"] = d["range"].rolling(ATR_LOOKBACK).mean()
    d["vol_ratio"] = d["range_5d"] / d["range_20d"]
    d["atr20"]     = d["range"].rolling(ATR_LOOKBACK).mean()

    # Trend lookbacks (returns)
    d["ret_5d"]  = d["close"].pct_change(5)
    d["ret_10d"] = d["close"].pct_change(10)
    d["ret_20d"] = d["close"].pct_change(20)
    d["ret_60d"] = d["close"].pct_change(60)

    # Price vs moving averages
    d["sma20"]       = d["close"].rolling(20).mean()
    d["sma50"]       = d["close"].rolling(50).mean()
    d["above_sma20"] = d["close"] > d["sma20"]
    d["above_sma50"] = d["close"] > d["sma50"]

    # Trend strength: abs(20d return) / ATR ratio
    d["trend_strength"] = d["ret_20d"].abs() / (d["atr20"] / d["close"])

    # Consecutive up/down days
    d["daily_ret"] = d["close"].pct_change()
    d["up_day"]    = (d["daily_ret"] > 0).astype(int)
    # Rolling sum of last 5 days direction
    d["consec_bias"] = d["up_day"].rolling(5).sum()  # 5=all up, 0=all down

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


def get_mgc_trade(day_bars, ctx, n_ct):
    """Returns trade dict or None. ctx is daily context row."""
    if np.isnan(ctx.get("range_20d", np.nan)):
        return None
    vr = ctx.get("vol_ratio", np.nan)
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
    return {
        "pnl": pnl, "win": pnl > 0, "reason": reason,
        "sig": sig, "entry": entry, "exit": ep,
    }


def get_mnq_trade(day_bars, ctx, n_ct):
    atr = ctx.get("atr20", np.nan)
    if np.isnan(atr) or atr <= 0:
        return None
    session = day_bars[day_bars.index.time >= MNQ_OPEN]
    if len(session) < MNQ_N_CONSOL + 2:
        return None
    consol  = session.iloc[:MNQ_N_CONSOL]
    c_hi    = consol["high"].max()
    c_lo    = consol["low"].min()
    if (c_hi - c_lo) >= MNQ_ATR_THRESH * atr:
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
    return {"pnl": pnl, "win": pnl > 0, "reason": reason}


# ---------------------------------------------------------------------------
# PART 1 — Profile stop days vs winning days
# ---------------------------------------------------------------------------

def profile_stop_days(mgc_bars, mgc_ctx):
    """
    For every MGC trade, record outcome and trend context.
    Then compare: what does the context look like on stop days vs wins?
    """
    records = []
    for day in sorted(set(mgc_bars.index.date)):
        day_bars = mgc_bars[mgc_bars.index.date == day]
        dmask    = mgc_ctx.index.date == day
        if not dmask.any() or len(day_bars) == 0:
            continue
        ctx = mgc_ctx[dmask].iloc[0].to_dict()
        t   = get_mgc_trade(day_bars, ctx, 1)
        if t is None:
            continue

        # Is first-bar signal WITH or AGAINST prior trend?
        ret20     = ctx.get("ret_20d", 0) or 0
        trend_dir = np.sign(ret20)           # +1 = uptrend, -1 = downtrend
        signal_vs_trend = (
            "WITH"    if t["sig"] == trend_dir  else
            "AGAINST" if t["sig"] != trend_dir  else
            "NEUTRAL"
        )

        records.append({
            "date":              day,
            "outcome":           "stop" if t["reason"] == "stop" else "win",
            "pnl":               t["pnl"],
            "win":               t["win"],
            "sig":               t["sig"],
            "ret_5d":            ctx.get("ret_5d",  0) or 0,
            "ret_10d":           ctx.get("ret_10d", 0) or 0,
            "ret_20d":           ctx.get("ret_20d", 0) or 0,
            "ret_60d":           ctx.get("ret_60d", 0) or 0,
            "above_sma20":       ctx.get("above_sma20", False),
            "above_sma50":       ctx.get("above_sma50", False),
            "trend_strength":    ctx.get("trend_strength", 0) or 0,
            "consec_bias":       ctx.get("consec_bias", 2.5) or 2.5,
            "signal_vs_trend":   signal_vs_trend,
            "reason":            t["reason"],
        })

    return pd.DataFrame(records)


def print_profile(df):
    print()
    print("=" * 72)
    print("  PART 1 — MGC STOP DAY PROFILE")
    print("  What market context characterises days where MGC hits its stop?")
    print("=" * 72)

    stops  = df[df["reason"] == "stop"]
    wins   = df[df["win"]]
    losses = df[~df["win"]]

    print(f"\n  Total MGC trades: {len(df)}")
    print(f"  Stops:   {len(stops)} ({len(stops)/len(df):.1%})")
    print(f"  Wins:    {len(wins)}  ({len(wins)/len(df):.1%})")
    print(f"  Losses (non-stop): {len(losses)-len(stops)}")

    # Signal vs trend breakdown
    print(f"\n  Signal direction vs prior 20-day trend:")
    print(f"  {'Direction':<12} {'N':>5} {'StopRate':>10} {'WinRate':>9} {'AvgPnL':>9}")
    print("  " + "-" * 50)
    for direction in ["WITH", "AGAINST", "NEUTRAL"]:
        sub = df[df["signal_vs_trend"] == direction]
        if len(sub) == 0:
            continue
        stop_rate = (sub["reason"] == "stop").mean()
        win_rate  = sub["win"].mean()
        avg_pnl   = sub["pnl"].mean()
        print(f"  {direction:<12} {len(sub):>5}  {stop_rate:>9.1%}  "
              f"{win_rate:>8.1%}  ${avg_pnl:>7.0f}")

    # 20-day return context on stop days
    print(f"\n  Prior 20-day return on STOP days vs WIN days:")
    print(f"  {'Metric':<25} {'Stop days':>12} {'Win days':>12}")
    print("  " + "-" * 52)
    for metric, label in [
        ("ret_20d",        "Avg 20d return"),
        ("ret_60d",        "Avg 60d return"),
        ("trend_strength", "Trend strength"),
        ("consec_bias",    "5d up-day bias (0-5)"),
    ]:
        s_val = stops[metric].mean()  if len(stops) else 0
        w_val = wins[metric].mean()   if len(wins)  else 0
        if metric in ("ret_20d", "ret_60d"):
            print(f"  {label:<25} {s_val:>+11.2%}  {w_val:>+11.2%}")
        else:
            print(f"  {label:<25} {s_val:>11.2f}  {w_val:>11.2f}")

    # Bucket by 20d return magnitude
    print(f"\n  Stop rate by prior 20-day return magnitude:")
    print(f"  {'20d Return Range':<22} {'N':>5} {'StopRate':>10} "
          f"{'WinRate':>9} {'AvgPnL':>9}")
    print("  " + "-" * 58)
    buckets = [
        (-1.0, -0.05, "< -5%  (strong down)"),
        (-0.05,-0.02, "-5% to -2%"),
        (-0.02, 0.02, "-2% to +2% (flat)"),
        ( 0.02, 0.05, "+2% to +5%"),
        ( 0.05, 1.00, "> +5%  (strong up)"),
    ]
    for lo, hi, label in buckets:
        sub = df[(df["ret_20d"] >= lo) & (df["ret_20d"] < hi)]
        if len(sub) < 5:
            continue
        stop_r = (sub["reason"] == "stop").mean()
        win_r  = sub["win"].mean()
        avg_p  = sub["pnl"].mean()
        print(f"  {label:<22} {len(sub):>5}  {stop_r:>9.1%}  "
              f"{win_r:>8.1%}  ${avg_p:>7.0f}")

    # Signal-against-trend by magnitude
    print(f"\n  When signal OPPOSES trend — stop rate by trend strength:")
    print(f"  {'20d Return':<22} {'N':>5} {'StopRate':>10} "
          f"{'WinRate':>9} {'AvgPnL':>9}")
    print("  " + "-" * 58)
    against = df[df["signal_vs_trend"] == "AGAINST"]
    for lo, hi, label in buckets:
        sub = against[(against["ret_20d"] >= lo) & (against["ret_20d"] < hi)]
        if len(sub) < 3:
            continue
        stop_r = (sub["reason"] == "stop").mean()
        win_r  = sub["win"].mean()
        avg_p  = sub["pnl"].mean()
        print(f"  {label:<22} {len(sub):>5}  {stop_r:>9.1%}  "
              f"{win_r:>8.1%}  ${avg_p:>7.0f}")

    # SMA context
    print(f"\n  Stop rate by SMA context:")
    print(f"  {'Context':<30} {'N':>5} {'StopRate':>10} {'WinRate':>9}")
    print("  " + "-" * 58)
    for above20 in [True, False]:
        for above50 in [True, False]:
            sub = df[(df["above_sma20"] == above20) &
                     (df["above_sma50"] == above50)]
            if len(sub) < 5:
                continue
            label = (f"Price {'above' if above20 else 'below'} SMA20, "
                     f"{'above' if above50 else 'below'} SMA50")
            stop_r = (sub["reason"] == "stop").mean()
            win_r  = sub["win"].mean()
            print(f"  {label:<30} {len(sub):>5}  "
                  f"{stop_r:>9.1%}  {win_r:>8.1%}")


# ---------------------------------------------------------------------------
# PART 2 — Test trend filters
# ---------------------------------------------------------------------------

def run_with_filter(mgc_bars, mgc_ctx, mnq_bars, mnq_daily,
                    filter_fn, filter_name):
    """
    Run full portfolio with MGC trend filter applied.
    filter_fn(ctx) returns True if MGC trade should be SKIPPED.
    """
    balance  = STARTING_BALANCE
    peak_eod = STARTING_BALANCE
    floor    = STARTING_BALANCE - MLL_BUFFER
    results  = []
    skipped  = 0
    all_days = sorted(set(mgc_bars.index.date) | set(mnq_bars.index.date))

    for day in all_days:
        if balance < floor:
            break

        day_pnl = 0.0

        # MGC — apply filter
        mgc_day  = mgc_bars[mgc_bars.index.date == day]
        mgc_mask = mgc_ctx.index.date == day
        if len(mgc_day) > 0 and mgc_mask.any():
            ctx = mgc_ctx[mgc_mask].iloc[0].to_dict()
            if filter_fn(ctx, mgc_day):
                skipped += 1
            else:
                t = get_mgc_trade(mgc_day, ctx, N_MGC)
                if t:
                    day_pnl += t["pnl"]

        # MNQ — always trade (no filter on MNQ)
        mnq_day  = mnq_bars[mnq_bars.index.date == day]
        mnq_mask = mnq_daily.index.date == day
        if len(mnq_day) > 0 and mnq_mask.any():
            ctx = mnq_daily[mnq_mask].iloc[0].to_dict()
            t   = get_mnq_trade(mnq_day, ctx, N_MNQ)
            if t:
                day_pnl += t["pnl"]

        balance += day_pnl
        if balance > peak_eod:
            peak_eod = balance
        floor = min(peak_eod - MLL_BUFFER, STARTING_BALANCE)

        results.append({
            "date":    day,
            "day_pnl": day_pnl,
            "balance": balance,
            "floor":   floor,
            "buffer":  balance - floor,
        })

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])

    total  = df["day_pnl"].sum()
    eq     = df["balance"]
    max_dd = (eq - eq.cummax()).min()
    min_buf= df["buffer"].min()

    active = df[df["day_pnl"] != 0]
    sharpe = 0.0
    if len(active) > 1 and active["day_pnl"].std() > 0:
        sharpe = active["day_pnl"].mean() / active["day_pnl"].std() * np.sqrt(252)

    monthly  = df.groupby(df["date"].dt.to_period("M"))["day_pnl"].sum()
    pos_m    = (monthly > 0).sum()
    worst_m  = monthly.min()

    # Consecutive losing months
    streak = max_streak = cur = 0
    for v in monthly:
        if v < 0:
            cur += 1; max_streak = max(max_streak, cur)
        else:
            cur = 0

    breached = balance < floor

    return {
        "name":         filter_name,
        "total":        total,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "min_buf":      min_buf,
        "pos_months":   f"{pos_m}/{len(monthly)}",
        "worst_month":  worst_m,
        "skipped_mgc":  skipped,
        "max_consec_loss_months": max_streak,
        "breached":     breached,
        "monthly":      monthly,
        "df":           df,
    }


def no_filter(ctx, day_bars):
    return False


def filter_against_strong_trend(threshold):
    """Skip MGC when signal opposes a strong 20d trend."""
    def fn(ctx, day_bars):
        ret20 = ctx.get("ret_20d", 0) or 0
        if np.isnan(ret20):
            return False
        # Get first bar direction
        fb = day_bars[(day_bars.index.time >= MGC_OPEN) &
                      (day_bars.index.time <  MGC_SIGNAL)]
        if len(fb) == 0:
            return False
        f   = fb.iloc[0]
        sig = np.sign(f["close"] - f["open"])
        trend_dir = np.sign(ret20)
        # Skip if signal opposes trend AND trend is strong
        opposing = (sig != trend_dir and trend_dir != 0 and sig != 0)
        strong   = abs(ret20) > threshold
        return opposing and strong
    return fn


def filter_below_sma(use_sma50=False):
    """Skip MGC longs when price below SMA; skip shorts when above."""
    sma_key = "above_sma50" if use_sma50 else "above_sma20"
    def fn(ctx, day_bars):
        above = ctx.get(sma_key, True)
        fb = day_bars[(day_bars.index.time >= MGC_OPEN) &
                      (day_bars.index.time <  MGC_SIGNAL)]
        if len(fb) == 0:
            return False
        sig = np.sign(fb.iloc[0]["close"] - fb.iloc[0]["open"])
        # Skip going long when price below SMA (trend is down)
        # Skip going short when price above SMA (trend is up)
        return (sig == 1 and not above) or (sig == -1 and above)
    return fn


def filter_combined(ret_thresh, use_sma50=False):
    """Skip when signal opposes strong trend AND SMA confirms trend."""
    sma_key = "above_sma50" if use_sma50 else "above_sma20"
    def fn(ctx, day_bars):
        ret20 = ctx.get("ret_20d", 0) or 0
        above = ctx.get(sma_key, True)
        if np.isnan(ret20):
            return False
        fb = day_bars[(day_bars.index.time >= MGC_OPEN) &
                      (day_bars.index.time <  MGC_SIGNAL)]
        if len(fb) == 0:
            return False
        sig       = np.sign(fb.iloc[0]["close"] - fb.iloc[0]["open"])
        trend_dir = np.sign(ret20)
        opposing  = (sig != trend_dir and trend_dir != 0 and sig != 0)
        strong    = abs(ret20) > ret_thresh
        sma_confirms = (trend_dir == 1 and above) or (trend_dir == -1 and not above)
        return opposing and strong and sma_confirms
    return fn


def print_filter_comparison(results):
    print()
    print("=" * 80)
    print("  PART 2 — TREND FILTER COMPARISON")
    print(f"  5x MGC + 5x MNQ  |  Full 3-year period  |  No target stop")
    print("=" * 80)

    print(f"\n  {'Filter':<35} {'P&L':>9} {'Sharpe':>7} "
          f"{'MaxDD':>9} {'MinBuf':>8} {'PosM':>7} "
          f"{'WorstM':>9} {'Skip':>5} {'Breach':>7}")
    print("  " + "-" * 95)

    for r in results:
        breach = "YES ⚠" if r["breached"] else "safe"
        print(f"  {r['name']:<35} "
              f"${r['total']:>8,.0f}  "
              f"{r['sharpe']:>6.2f}  "
              f"${r['max_dd']:>7,.0f}  "
              f"${r['min_buf']:>6,.0f}  "
              f"{r['pos_months']:>7}  "
              f"${r['worst_month']:>7,.0f}  "
              f"{r['skipped_mgc']:>5}  "
              f"{breach}")

    # Best filter analysis
    safe    = [r for r in results if not r["breached"]]
    if safe:
        best    = max(safe, key=lambda r: r["sharpe"])
        base    = results[0]  # no filter

        print(f"\n  BEST FILTER: {best['name']}")
        print(f"  {'Metric':<25} {'No filter':>12} {'Best filter':>12} {'Change':>10}")
        print("  " + "-" * 62)
        metrics = [
            ("Total P&L",       "total",       "$"),
            ("Sharpe",          "sharpe",      "f"),
            ("Max drawdown",    "max_dd",       "$"),
            ("Min MLL buffer",  "min_buf",      "$"),
            ("Worst month",     "worst_month",  "$"),
        ]
        for label, key, fmt in metrics:
            bv = base[key]
            fv = best[key]
            if fmt == "$":
                diff = fv - bv
                sign = "+" if diff >= 0 else ""
                print(f"  {label:<25} ${bv:>10,.0f}  ${fv:>10,.0f}  "
                      f"{sign}${diff:>7,.0f}")
            else:
                diff = fv - bv
                sign = "+" if diff >= 0 else ""
                print(f"  {label:<25} {bv:>11.2f}  {fv:>11.2f}  "
                      f"{sign}{diff:>8.2f}")
        print(f"\n  MGC trades skipped: {best['skipped_mgc']} "
              f"({best['skipped_mgc']/base['skipped_mgc']*100 if base['skipped_mgc'] else 0:.0f}% "
              f"vs base {base['skipped_mgc']})")

    # Monthly comparison: no filter vs best filter
    if safe:
        best  = max(safe, key=lambda r: r["sharpe"])
        base  = results[0]
        bm    = base["monthly"]
        fm    = best["monthly"]
        all_m = sorted(set(bm.index) | set(fm.index))

        print(f"\n  Monthly P&L — No filter vs {best['name']}:")
        print(f"  {'Month':<10} {'No filter':>12} {'W/ filter':>12} "
              f"{'Diff':>10} {'Filter helped?':>15}")
        print("  " + "-" * 64)

        helped = hurt = neutral = 0
        for period in all_m:
            v_base = bm.get(period, 0)
            v_filt = fm.get(period, 0)
            diff   = v_filt - v_base
            if diff > 50:
                verdict = "YES ✓"; helped += 1
            elif diff < -50:
                verdict = "NO  ✗"; hurt += 1
            else:
                verdict = "neutral"; neutral += 1
            sb = "+" if v_base >= 0 else ""
            sf = "+" if v_filt >= 0 else ""
            sd = "+" if diff   >= 0 else ""
            print(f"  {str(period):<10} "
                  f"{sb}${v_base:>9,.0f}  "
                  f"{sf}${v_filt:>9,.0f}  "
                  f"{sd}${diff:>7,.0f}  "
                  f"{verdict:>15}")

        print(f"\n  Filter helped: {helped}  |  hurt: {hurt}  |  neutral: {neutral}")

    print()
    print("=" * 80)
    print()

    # Recommendation
    safe_sorted = sorted(safe, key=lambda r: r["sharpe"], reverse=True)
    if safe_sorted:
        best = safe_sorted[0]
        print("  RECOMMENDATION:")
        print(f"  Best filter: {best['name']}")
        print(f"  P&L:        ${best['total']:,.0f}  "
              f"({'better' if best['total'] > results[0]['total'] else 'lower'} "
              f"than no-filter ${results[0]['total']:,.0f})")
        print(f"  Sharpe:      {best['sharpe']:.2f}  "
              f"(vs no-filter {results[0]['sharpe']:.2f})")
        print(f"  Max DD:     ${best['max_dd']:,.0f}  "
              f"(vs no-filter ${results[0]['max_dd']:,.0f})")
        print(f"  Worst month:${best['worst_month']:,.0f}  "
              f"(vs no-filter ${results[0]['worst_month']:,.0f})")
        print()
        dd_improvement = results[0]["max_dd"] - best["max_dd"]
        pnl_cost       = results[0]["total"]  - best["total"]
        if dd_improvement > 0:
            ratio = dd_improvement / pnl_cost if pnl_cost > 0 else float('inf')
            print(f"  Drawdown reduced by ${dd_improvement:,.0f}  "
                  f"at a cost of ${pnl_cost:,.0f} in total P&L")
            if ratio > 2:
                print(f"  ✓ Excellent tradeoff — saves ${ratio:.1f} of drawdown per $1 of P&L")
            elif ratio > 1:
                print(f"  ~ Good tradeoff — saves ${ratio:.1f} of drawdown per $1 of P&L")
            else:
                print(f"  ✗ Poor tradeoff — drawdown reduction costs more P&L than it saves")
    print()


def main():
    print("\nLoading data...")
    mgc_bars  = load("MGC")
    mnq_bars  = load("MNQ")
    mgc_ctx   = build_mgc_context(mgc_bars)
    mnq_daily = build_mnq_daily(mnq_bars)
    print(f"  MGC: {len(set(mgc_bars.index.date))} days  "
          f"MNQ: {len(set(mnq_bars.index.date))} days")

    # -------------------------------------------------------------------
    # PART 1 — Profile
    # -------------------------------------------------------------------
    print("\nPart 1: Profiling MGC stop days vs win days...")
    profile_df = profile_stop_days(mgc_bars, mgc_ctx)
    print_profile(profile_df)

    # -------------------------------------------------------------------
    # PART 2 — Filter tests
    # -------------------------------------------------------------------
    print("\nPart 2: Testing trend filters...")

    filters = [
        (no_filter,                             "No filter (baseline)"),
        (filter_against_strong_trend(0.02),     "Skip if opposing >2% trend"),
        (filter_against_strong_trend(0.03),     "Skip if opposing >3% trend"),
        (filter_against_strong_trend(0.05),     "Skip if opposing >5% trend"),
        (filter_below_sma(False),               "Skip counter-SMA20 trades"),
        (filter_below_sma(True),                "Skip counter-SMA50 trades"),
        (filter_combined(0.02, False),          "Combined: >2% trend + SMA20"),
        (filter_combined(0.03, False),          "Combined: >3% trend + SMA20"),
        (filter_combined(0.02, True),           "Combined: >2% trend + SMA50"),
        (filter_combined(0.03, True),           "Combined: >3% trend + SMA50"),
        (filter_combined(0.05, True),           "Combined: >5% trend + SMA50"),
    ]

    results = []
    for fn, name in filters:
        print(f"  {name}...", end=" ")
        r = run_with_filter(mgc_bars, mgc_ctx, mnq_bars, mnq_daily, fn, name)
        print(f"P&L=${r['total']:,.0f}  Sh={r['sharpe']:.2f}  "
              f"MaxDD=${r['max_dd']:,.0f}  Skip={r['skipped_mgc']}")
        results.append(r)

    print_filter_comparison(results)


if __name__ == "__main__":
    main()
