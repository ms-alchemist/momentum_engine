"""
backtest/diagnose_filter_oos.py
================================
Out-of-sample validation of the top MGC trend filters.

Split:
  IN-SAMPLE:      2023-03 to 2024-09 (18 months) — where filters were evaluated
  OUT-OF-SAMPLE:  2024-10 to 2026-03 (18 months) — blind validation

Filters tested:
  1. No filter (baseline)
  2. Skip counter-SMA20 trades         (best IS Sharpe: 3.19)
  3. Skip if opposing >2% trend        (best IS P&L balance: $69K)
  4. Skip counter-SMA50 trades         (best IS max DD: -$8,271)

For each filter, IS vs OOS comparison shows:
  - Does the improvement persist on unseen data?
  - Does the Sharpe improvement retain at least 50%?
  - Does the drawdown reduction hold OOS?

This is the final validation gate. If a filter passes:
  → Lock in strategy parameters and build the final runner.
If a filter fails:
  → Fall back to next best or use no filter with wider stop.

Usage:
    python backtest/diagnose_filter_oos.py
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

ATR_LOOKBACK     = 20
STARTING_BALANCE = 100_000.0
MLL_BUFFER       =   3_000.0
N_MGC = 5
N_MNQ = 5

IS_START  = pd.Timestamp("2023-03-01")
OOS_START = pd.Timestamp("2024-10-01")
OOS_END   = pd.Timestamp("2026-04-01")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(symbol):
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_30min.parquet")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def build_mgc_ctx(bars):
    d = bars.resample("1D").agg(
        high=("high","max"), low=("low","min"),
        open=("open","first"), close=("close","last"),
        volume=("volume","sum")).dropna()
    d = d[d["volume"] > 0]
    d["range"]     = d["high"] - d["low"]
    d["range_5d"]  = d["range"].rolling(5).mean()
    d["range_20d"] = d["range"].rolling(ATR_LOOKBACK).mean()
    d["vol_ratio"] = d["range_5d"] / d["range_20d"]
    d["ret_20d"]   = d["close"].pct_change(20)
    d["sma20"]     = d["close"].rolling(20).mean()
    d["sma50"]     = d["close"].rolling(50).mean()
    d["above_sma20"] = (d["close"] > d["sma20"]).astype(float)
    d["above_sma50"] = (d["close"] > d["sma50"]).astype(float)
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


# ---------------------------------------------------------------------------
# Trade logic
# ---------------------------------------------------------------------------

def get_first_bar_sig(day_bars):
    fb = day_bars[(day_bars.index.time >= MGC_OPEN) &
                  (day_bars.index.time <  MGC_SIGNAL)]
    if len(fb) == 0:
        return 0
    f = fb.iloc[0]
    return int(np.sign(f["close"] - f["open"]))


def get_mgc_trade(day_bars, ctx, n_ct):
    if np.isnan(ctx.get("range_20d", np.nan)):
        return None
    vr = ctx.get("vol_ratio", np.nan)
    if not np.isnan(vr) and vr > MGC_VOL_RATIO:
        return None
    sig = get_first_bar_sig(day_bars)
    if sig == 0:
        return None
    fb    = day_bars[(day_bars.index.time >= MGC_OPEN) &
                     (day_bars.index.time <  MGC_SIGNAL)]
    entry = fb.iloc[0]["close"]
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
    return {"pnl": pnl, "win": pnl > 0, "reason": reason, "sig": sig}


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
# Filter definitions
# ---------------------------------------------------------------------------

def make_no_filter():
    def fn(ctx, sig): return False
    fn.name = "No filter (baseline)"
    return fn


def make_sma20_filter():
    """Skip MGC when signal opposes SMA20 trend direction."""
    def fn(ctx, sig):
        above = ctx.get("above_sma20", np.nan)
        if np.isnan(above):
            return False
        # Price above SMA20 = uptrend → skip longs (mean reversion: fade the up)
        # Price below SMA20 = downtrend → skip shorts (fade the down)
        # In other words: only trade in the mean-reverting direction
        # Long when price is above SMA20 (expect it to pull back to SMA)
        # Short when price is below SMA20 (expect it to rally back to SMA)
        # WAIT — re-read the filter:
        # "counter-SMA20" = skip trades that go AGAINST the SMA direction
        # i.e. skip going short when price is ABOVE SMA20 (trend is up, short is counter)
        # and skip going long when price is BELOW SMA20 (trend is down, long is counter)
        return (sig == -1 and above == 1.0) or (sig == 1 and above == 0.0)
    fn.name = "Skip counter-SMA20 trades"
    return fn


def make_trend_filter(threshold):
    """Skip when first-bar signal opposes prior 20d trend by threshold."""
    def fn(ctx, sig):
        ret20 = ctx.get("ret_20d", 0) or 0
        if np.isnan(ret20):
            return False
        trend_dir = np.sign(ret20)
        opposing  = (sig != trend_dir and trend_dir != 0 and sig != 0)
        strong    = abs(ret20) > threshold
        return opposing and strong
    fn.name = f"Skip opposing >{threshold*100:.0f}% trend"
    return fn


def make_sma50_filter():
    """Skip counter-SMA50 trades."""
    def fn(ctx, sig):
        above = ctx.get("above_sma50", np.nan)
        if np.isnan(above):
            return False
        return (sig == -1 and above == 1.0) or (sig == 1 and above == 0.0)
    fn.name = "Skip counter-SMA50 trades"
    return fn


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_period(mgc_bars, mgc_ctx, mnq_bars, mnq_daily,
               start, end, filter_fn):
    days = sorted(
        d for d in (set(mgc_bars.index.date) | set(mnq_bars.index.date))
        if pd.Timestamp(d) >= start and pd.Timestamp(d) < end
    )

    balance  = STARTING_BALANCE
    peak_eod = STARTING_BALANCE
    floor    = STARTING_BALANCE - MLL_BUFFER
    results  = []

    for day in days:
        if balance < floor:
            break
        day_pnl = 0.0

        # MGC
        mgc_day  = mgc_bars[mgc_bars.index.date == day]
        mgc_mask = mgc_ctx.index.date == day
        if len(mgc_day) > 0 and mgc_mask.any():
            ctx = mgc_ctx[mgc_mask].iloc[0].to_dict()
            sig = get_first_bar_sig(mgc_day)
            if sig != 0 and not filter_fn(ctx, sig):
                t = get_mgc_trade(mgc_day, ctx, N_MGC)
                if t:
                    day_pnl += t["pnl"]

        # MNQ — no filter
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
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"])
    total    = df["day_pnl"].sum()
    eq       = df["balance"]
    max_dd   = (eq - eq.cummax()).min()
    min_buf  = df["buffer"].min()
    breached = balance < floor

    active = df[df["day_pnl"] != 0]
    sharpe = 0.0
    if len(active) > 1 and active["day_pnl"].std() > 0:
        sharpe = active["day_pnl"].mean() / active["day_pnl"].std() * np.sqrt(252)

    n_days  = (end - start).days
    n_years = n_days / 365.25
    ann     = total / n_years if n_years > 0 else 0

    monthly = df.groupby(df["date"].dt.to_period("M"))["day_pnl"].sum()
    pos_m   = (monthly > 0).sum()

    # Worst consecutive losing months
    streak = max_s = cur = 0
    for v in monthly:
        if v < 0:
            cur += 1; max_s = max(max_s, cur)
        else:
            cur = 0

    return {
        "total":       total,
        "ann":         ann,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "min_buf":     min_buf,
        "pos_months":  f"{pos_m}/{len(monthly)}",
        "worst_month": monthly.min() if len(monthly) else 0,
        "best_month":  monthly.max() if len(monthly) else 0,
        "avg_month":   monthly.mean() if len(monthly) else 0,
        "max_consec_loss": max_s,
        "breached":    breached,
        "monthly":     monthly,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_oos_report(filters, is_results, oos_results):
    print()
    print("=" * 85)
    print("  FILTER OUT-OF-SAMPLE VALIDATION")
    print(f"  IN-SAMPLE:     {IS_START.date()} → {OOS_START.date()} (18 months)")
    print(f"  OUT-OF-SAMPLE: {OOS_START.date()} → {OOS_END.date()} (18 months)")
    print(f"  5x MGC (mean-reversion) + 5x MNQ (breakout momentum)")
    print("=" * 85)

    # Summary table — IS
    print(f"\n  ── IN-SAMPLE RESULTS ────────────────────────────────────────────────")
    _print_table(filters, is_results)

    # Summary table — OOS
    print(f"\n  ── OUT-OF-SAMPLE RESULTS ────────────────────────────────────────────")
    _print_table(filters, oos_results)

    # Retention analysis
    print(f"\n  ── SHARPE RETENTION (IS → OOS) ──────────────────────────────────────")
    print(f"  {'Filter':<32} {'IS Sharpe':>10} {'OOS Sharpe':>11} "
          f"{'Retained':>10} {'Verdict':>12}")
    print("  " + "-" * 78)

    base_is  = is_results[0]["sharpe"]
    base_oos = oos_results[0]["sharpe"]

    for i, (filt, is_r, oos_r) in enumerate(
            zip(filters, is_results, oos_results)):
        is_sh  = is_r["sharpe"]
        oos_sh = oos_r["sharpe"]
        is_imp  = is_sh  - base_is
        oos_imp = oos_sh - base_oos
        retention = (oos_imp / is_imp * 100) if abs(is_imp) > 0.01 else 0

        if i == 0:
            verdict = "baseline"
        elif oos_r.get("breached"):
            verdict = "BREACH ⚠"
        elif retention >= 70:
            verdict = "✓ ROBUST"
        elif retention >= 40:
            verdict = "~ PARTIAL"
        elif retention >= 0:
            verdict = "✗ WEAK"
        else:
            verdict = "✗ REVERSED"

        print(f"  {filt.name:<32} {is_sh:>9.2f}  {oos_sh:>10.2f}  "
              f"{retention:>9.0f}%  {verdict:>12}")

    # Drawdown retention
    print(f"\n  ── DRAWDOWN COMPARISON ──────────────────────────────────────────────")
    print(f"  {'Filter':<32} {'IS MaxDD':>10} {'OOS MaxDD':>11} "
          f"{'IS WorstM':>10} {'OOS WorstM':>11}")
    print("  " + "-" * 78)
    for filt, is_r, oos_r in zip(filters, is_results, oos_results):
        print(f"  {filt.name:<32} "
              f"${is_r['max_dd']:>8,.0f}  "
              f"${oos_r['max_dd']:>9,.0f}  "
              f"${is_r['worst_month']:>8,.0f}  "
              f"${oos_r['worst_month']:>9,.0f}")

    # Month-by-month OOS for best passing filter
    passing = [(f, isr, oosr) for f, isr, oosr in
               zip(filters, is_results, oos_results)
               if not oosr.get("breached") and f.name != "No filter (baseline)"]

    if passing:
        # Best by OOS Sharpe
        best_f, best_is, best_oos = max(
            passing, key=lambda x: x[2]["sharpe"])

        base_oos_m = oos_results[0]["monthly"]
        filt_oos_m = best_oos["monthly"]
        all_months = sorted(set(base_oos_m.index) | set(filt_oos_m.index))

        print(f"\n  ── OOS MONTHLY: No filter vs {best_f.name} ─────────────────────────")
        print(f"  {'Month':<10} {'No filter':>12} {'W/ filter':>12} "
              f"{'Diff':>10} {'Filter helped?':>15}")
        print("  " + "-" * 64)

        helped = hurt = neutral = 0
        for period in all_months:
            v_base = base_oos_m.get(period, 0)
            v_filt = filt_oos_m.get(period, 0)
            diff   = v_filt - v_base
            if diff > 100:
                verdict = "YES ✓"; helped += 1
            elif diff < -100:
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

        print(f"\n  OOS: helped {helped} months | hurt {hurt} | neutral {neutral}")

    # Final verdict
    print()
    print("  ═══ FINAL VERDICT ═══════════════════════════════════════════════════")
    print()

    robust = [(f, isr, oosr) for f, isr, oosr in
              zip(filters, is_results, oos_results)
              if not oosr.get("breached")
              and f.name != "No filter (baseline)"
              and oosr["sharpe"] > oos_results[0]["sharpe"]]

    if robust:
        best_f, best_is, best_oos = max(robust, key=lambda x: x[2]["sharpe"])
        base_is  = is_results[0]
        base_oos = oos_results[0]

        print(f"  RECOMMENDED FILTER: {best_f.name}")
        print()
        print(f"  {'Metric':<28} {'IS baseline':>12} {'IS w/filter':>12} "
              f"{'OOS baseline':>13} {'OOS w/filter':>13}")
        print("  " + "-" * 82)

        for label, key, fmt in [
            ("Total P&L",       "total",       "$"),
            ("Annualized P&L",  "ann",         "$"),
            ("Sharpe",          "sharpe",      "f"),
            ("Max drawdown",    "max_dd",      "$"),
            ("Worst month",     "worst_month", "$"),
            ("Positive months", "pos_months",  "s"),
        ]:
            biv = base_is[key]
            fiv = best_is[key]
            bov = base_oos[key]
            fov = best_oos[key]
            if fmt == "$":
                print(f"  {label:<28} ${biv:>10,.0f}  ${fiv:>10,.0f}  "
                      f"${bov:>11,.0f}  ${fov:>11,.0f}")
            elif fmt == "f":
                print(f"  {label:<28} {biv:>11.2f}  {fiv:>11.2f}  "
                      f"{bov:>12.2f}  {fov:>12.2f}")
            else:
                print(f"  {label:<28} {str(biv):>12}  {str(fiv):>12}  "
                      f"{str(bov):>13}  {str(fov):>13}")

        print()
        # P&L verdict
        oos_better = best_oos["total"] > base_oos["total"]
        sh_better  = best_oos["sharpe"] > base_oos["sharpe"]
        dd_better  = best_oos["max_dd"] > base_oos["max_dd"]

        if oos_better and sh_better:
            print("  ✓ FILTER VALIDATED — improves both P&L and Sharpe OOS")
            print("  ✓ Proceed to final strategy build with this filter")
        elif sh_better and dd_better:
            print("  ✓ FILTER VALIDATED — improves Sharpe and drawdown OOS")
            print("    P&L slightly lower but risk-adjusted return is better")
            print("  ✓ Proceed to final strategy build with this filter")
        elif sh_better:
            print("  ~ FILTER PARTIALLY VALIDATED — improves Sharpe OOS")
            print("    Consider using this filter with awareness that P&L")
            print("    improvement may be smaller than IS suggested")
        else:
            print("  ✗ FILTER NOT VALIDATED — does not improve OOS")
            print("    Recommend using no filter or re-evaluating parameters")

        print()
        print("  STRATEGY SUMMARY (if filter validated):")
        print(f"    Instrument 1: MGC — mean-reversion first-bar signal")
        print(f"    Filter:       {best_f.name}")
        print(f"    Instrument 2: MNQ — volatility breakout momentum")
        print(f"    Sizing:       5x MGC + 5x MNQ")
        print(f"    OOS Sharpe:   {best_oos['sharpe']:.2f}")
        print(f"    OOS Ann P&L:  ${best_oos['ann']:,.0f}/year")
        print(f"    OOS Max DD:   ${best_oos['max_dd']:,.0f}")
        ann_per_month = best_oos['ann'] / 12
        print(f"    OOS Monthly:  ~${ann_per_month:,.0f}/month avg")
    else:
        print("  No filter clears the OOS validation bar.")
        print("  Recommend building final runner with no filter.")
        print(f"  Baseline OOS: P&L=${oos_results[0]['total']:,.0f}  "
              f"Sharpe={oos_results[0]['sharpe']:.2f}")

    print()
    print("=" * 85)


def _print_table(filters, results):
    print(f"  {'Filter':<32} {'P&L':>10} {'Ann':>10} {'Sharpe':>7} "
          f"{'MaxDD':>9} {'MinBuf':>8} {'PosM':>6} {'WorstM':>9}")
    print("  " + "-" * 95)
    for filt, r in zip(filters, results):
        breach = " ⚠" if r.get("breached") else ""
        print(f"  {filt.name:<32} "
              f"${r['total']:>8,.0f}  "
              f"${r['ann']:>8,.0f}  "
              f"{r['sharpe']:>6.2f}  "
              f"${r['max_dd']:>7,.0f}  "
              f"${r['min_buf']:>6,.0f}  "
              f"{r['pos_months']:>6}  "
              f"${r['worst_month']:>7,.0f}{breach}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nLoading data...")
    mgc_bars  = load("MGC")
    mnq_bars  = load("MNQ")
    mgc_ctx   = build_mgc_ctx(mgc_bars)
    mnq_daily = build_mnq_daily(mnq_bars)
    print(f"  MGC: {len(set(mgc_bars.index.date))} days  "
          f"MNQ: {len(set(mnq_bars.index.date))} days")

    filters = [
        make_no_filter(),
        make_sma20_filter(),
        make_trend_filter(0.02),
        make_sma50_filter(),
    ]

    is_results  = []
    oos_results = []

    print(f"\n  Running IS  ({IS_START.date()} → {OOS_START.date()})...")
    for filt in filters:
        r = run_period(mgc_bars, mgc_ctx, mnq_bars, mnq_daily,
                       IS_START, OOS_START, filt)
        is_results.append(r)
        print(f"    {filt.name:<35} "
              f"P&L=${r['total']:,.0f}  Sh={r['sharpe']:.2f}  "
              f"MaxDD=${r['max_dd']:,.0f}")

    print(f"\n  Running OOS ({OOS_START.date()} → {OOS_END.date()})...")
    for filt in filters:
        r = run_period(mgc_bars, mgc_ctx, mnq_bars, mnq_daily,
                       OOS_START, OOS_END, filt)
        oos_results.append(r)
        breach = " ⚠ BREACH" if r.get("breached") else ""
        print(f"    {filt.name:<35} "
              f"P&L=${r['total']:,.0f}  Sh={r['sharpe']:.2f}  "
              f"MaxDD=${r['max_dd']:,.0f}{breach}")

    print_oos_report(filters, is_results, oos_results)


if __name__ == "__main__":
    main()
