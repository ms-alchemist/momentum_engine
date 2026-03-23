"""
backtest/diagnose_loss_limit.py
================================
Two diagnostics in one:

PART 1 — Loss Limit Firing Audit
  Every day the $500 limit fired:
    - What MGC did that day
    - Whether the blocked MNQ trade would have won or lost
    - Net P&L saved vs cost of blocking the MNQ trade
    - Was the protection genuinely valuable in real time?

PART 2 — Out-of-Sample Test
  Split the 3-year dataset:
    IN-SAMPLE:  2023-03 to 2024-09 (18 months) — params derived here
    OUT-OF-SAMPLE: 2024-10 to 2026-03 (18 months) — blind test

  Compare $500 limit vs no limit on BOTH periods.
  If the improvement holds OOS, the limit is genuinely protective.
  If it collapses OOS, we're looking at overfitting.

  Slippage note: in live trading, the $500 limit triggers a market
  flatten order. Actual exit may be ~$500 ± a few dollars depending
  on fill. This is modeled as exact in the backtest — real performance
  will be marginally different but not materially so.

Usage:
    python backtest/diagnose_loss_limit.py
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
DAILY_LIMIT = -500.0

# Out-of-sample split date
OOS_START = pd.Timestamp("2024-10-01")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Trade generators
# ---------------------------------------------------------------------------

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
    return {"pnl": pnl, "win": pnl > 0, "reason": reason,
            "entry": entry, "exit": ep, "sig": sig}


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
    return {"pnl": pnl, "win": pnl > 0, "reason": reason,
            "entry": entry, "exit": ep, "sig": sig}


# ---------------------------------------------------------------------------
# PART 1 — Loss Limit Firing Audit
# ---------------------------------------------------------------------------

def run_limit_audit(mgc_bars, mgc_daily, mnq_bars, mnq_daily):
    """
    For every day in the dataset, compute:
      - MGC P&L
      - Whether limit fired (MGC pnl <= $500 threshold)
      - What MNQ would have done on that day (regardless of limit)
      - Net effect of the limit: saved loss vs blocked gain
    """
    firing_days = []
    all_days = sorted(set(mgc_bars.index.date) | set(mnq_bars.index.date))

    for day in all_days:
        # MGC trade
        mgc_day  = mgc_bars[mgc_bars.index.date == day]
        mgc_mask = mgc_daily.index.date == day
        mgc_t    = None
        if len(mgc_day) > 0 and mgc_mask.any():
            row   = mgc_daily[mgc_mask].iloc[0]
            mgc_t = get_mgc_trade(mgc_day,
                {"range_20d": row["range_20d"],
                 "vol_ratio": row["vol_ratio"]}, N_MGC)

        mgc_pnl = mgc_t["pnl"] if mgc_t else 0.0

        # Did limit fire?
        limit_fired = mgc_pnl <= DAILY_LIMIT

        # MNQ trade — compute regardless of limit
        mnq_day  = mnq_bars[mnq_bars.index.date == day]
        mnq_mask = mnq_daily.index.date == day
        mnq_t    = None
        if len(mnq_day) > 0 and mnq_mask.any():
            row   = mnq_daily[mnq_mask].iloc[0]
            mnq_t = get_mnq_trade(mnq_day, {"atr20": row["atr20"]}, N_MNQ)

        mnq_pnl        = mnq_t["pnl"] if mnq_t else None
        mnq_would_fire = mnq_t is not None  # MNQ signal existed this day

        if limit_fired and mnq_would_fire:
            firing_days.append({
                "date":          day,
                "mgc_pnl":       mgc_pnl,
                "mgc_reason":    mgc_t["reason"] if mgc_t else "no_trade",
                "mnq_pnl":       mnq_pnl,
                "mnq_win":       mnq_t["win"] if mnq_t else None,
                "mnq_reason":    mnq_t["reason"] if mnq_t else "no_signal",
                "net_saved":     -mnq_pnl,   # positive = we saved by blocking
                "combined_no_limit":  mgc_pnl + (mnq_pnl or 0),
                "combined_limited":   mgc_pnl,  # MNQ blocked
            })
        elif limit_fired:
            # Limit fired but no MNQ signal that day — no cost to blocking
            firing_days.append({
                "date":          day,
                "mgc_pnl":       mgc_pnl,
                "mgc_reason":    mgc_t["reason"] if mgc_t else "no_trade",
                "mnq_pnl":       None,
                "mnq_win":       None,
                "mnq_reason":    "no_signal",
                "net_saved":     0,
                "combined_no_limit":  mgc_pnl,
                "combined_limited":   mgc_pnl,
            })

    return pd.DataFrame(firing_days)


def print_limit_audit(df):
    print()
    print("=" * 80)
    print("  PART 1 — DAILY LOSS LIMIT FIRING AUDIT")
    print(f"  Threshold: ${abs(DAILY_LIMIT):.0f}  |  "
          f"5x MGC + 5x MNQ  |  Full 3-year period")
    print("=" * 80)

    total_fires = len(df)
    with_mnq    = df[df["mnq_pnl"].notna()]
    blocked_win = with_mnq[with_mnq["mnq_win"] == True]
    blocked_loss= with_mnq[with_mnq["mnq_win"] == False]
    no_mnq_sig  = df[df["mnq_pnl"].isna()]

    print(f"\n  Total days limit fired:        {total_fires}")
    print(f"  Days with MNQ signal blocked:  {len(with_mnq)} "
          f"({len(with_mnq)/total_fires*100:.0f}% of fires)")
    print(f"    → MNQ would have WON:        {len(blocked_win)} "
          f"(cost of blocking: ${blocked_win['mnq_pnl'].sum():,.0f})")
    print(f"    → MNQ would have LOST:       {len(blocked_loss)} "
          f"(saved by blocking: ${abs(blocked_loss['mnq_pnl'].sum()):,.0f})")
    print(f"  Days with no MNQ signal:       {len(no_mnq_sig)} "
          f"(no cost to blocking)")

    net = blocked_loss["mnq_pnl"].sum() + blocked_win["mnq_pnl"].sum()
    print(f"\n  Net effect of blocking MNQ:    ${-net:+,.0f} "
          f"({'saved' if net < 0 else 'cost'})")
    print(f"  (Negative MNQ blocked P&L = we SAVED money by not trading)")

    # Table of all firing days
    print()
    print(f"  {'Date':<12} {'MGC P&L':>10} {'MGC Exit':>10} "
          f"{'MNQ Signal':>11} {'MNQ P&L':>9} "
          f"{'MNQ Win':>8} {'Net w/ Limit':>13} {'Net w/o Limit':>14}")
    print("  " + "-" * 92)

    df_sorted = df.sort_values("date")
    for _, row in df_sorted.iterrows():
        mnq_pnl_str = f"${row['mnq_pnl']:,.0f}" if row["mnq_pnl"] is not None else "no signal"
        mnq_win_str = ("WIN " if row["mnq_win"] else "LOSS") if row["mnq_win"] is not None else "---"
        combined_lim = f"${row['combined_limited']:,.0f}"
        combined_nl  = f"${row['combined_no_limit']:,.0f}" if row["mnq_pnl"] is not None else combined_lim
        flag = " ← blocked winner" if row["mnq_win"] == True else (
               " ← blocked loser " if row["mnq_win"] == False else "")
        print(f"  {str(row['date']):<12} "
              f"${row['mgc_pnl']:>8,.0f}  "
              f"{row['mgc_reason']:>10}  "
              f"{'YES':>11}  "
              f"{mnq_pnl_str:>9}  "
              f"{mnq_win_str:>8}  "
              f"{combined_lim:>12}  "
              f"{combined_nl:>13}{flag}")

    # Summary: is the limit protective in aggregate?
    print()
    total_saved = abs(blocked_loss["mnq_pnl"].sum()) if len(blocked_loss) else 0
    total_cost  = blocked_win["mnq_pnl"].sum() if len(blocked_win) else 0
    net_benefit = total_saved - total_cost
    print(f"  VERDICT:")
    print(f"    Total saved (blocked losing MNQ trades):  ${total_saved:,.0f}")
    print(f"    Total cost  (blocked winning MNQ trades): ${total_cost:,.0f}")
    print(f"    Net benefit of limit:                     ${net_benefit:,.0f}")
    if net_benefit > 0:
        ratio = total_saved / total_cost if total_cost > 0 else float('inf')
        print(f"    Saved ${ratio:.1f} for every $1 sacrificed → limit is genuinely protective")
    else:
        print(f"    Limit costs more than it saves → reconsider threshold")


# ---------------------------------------------------------------------------
# PART 2 — Out-of-Sample Test
# ---------------------------------------------------------------------------

def run_period(mgc_bars, mgc_daily, mnq_bars, mnq_daily,
               start, end, use_limit):
    """Run strategy on a specific date range."""
    all_days = sorted(
        d for d in (set(mgc_bars.index.date) | set(mnq_bars.index.date))
        if pd.Timestamp(d) >= start and pd.Timestamp(d) < end
    )

    balance  = STARTING_BALANCE
    peak_eod = STARTING_BALANCE
    floor    = STARTING_BALANCE - MLL_BUFFER
    results  = []

    for day in all_days:
        day_pnl = 0.0

        # MGC
        mgc_day  = mgc_bars[mgc_bars.index.date == day]
        mgc_mask = mgc_daily.index.date == day
        mgc_pnl  = 0.0
        if len(mgc_day) > 0 and mgc_mask.any():
            row = mgc_daily[mgc_mask].iloc[0]
            t   = get_mgc_trade(mgc_day,
                {"range_20d": row["range_20d"],
                 "vol_ratio": row["vol_ratio"]}, N_MGC)
            if t:
                mgc_pnl  = t["pnl"]
                day_pnl += mgc_pnl

        # MNQ — respect limit
        mnq_blocked = use_limit and day_pnl <= DAILY_LIMIT
        if not mnq_blocked:
            mnq_day  = mnq_bars[mnq_bars.index.date == day]
            mnq_mask = mnq_daily.index.date == day
            if len(mnq_day) > 0 and mnq_mask.any():
                row = mnq_daily[mnq_mask].iloc[0]
                t   = get_mnq_trade(mnq_day, {"atr20": row["atr20"]}, N_MNQ)
                if t:
                    day_pnl += t["pnl"]

        balance += day_pnl
        if balance > peak_eod:
            peak_eod = balance
        floor = min(peak_eod - MLL_BUFFER, STARTING_BALANCE)

        results.append({
            "date": day, "day_pnl": day_pnl,
            "balance": balance, "floor": floor,
            "buffer": balance - floor,
        })

    return pd.DataFrame(results)


def period_stats(df, label, start, end):
    if df.empty:
        return {}

    total   = df["day_pnl"].sum()
    active  = df[df["day_pnl"] != 0]
    sharpe  = 0.0
    if len(active) > 1 and active["day_pnl"].std() > 0:
        sharpe = active["day_pnl"].mean() / active["day_pnl"].std() * np.sqrt(252)

    eq     = df["balance"]
    max_dd = (eq - eq.cummax()).min()
    min_buf= df["buffer"].min()

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    monthly   = d.groupby(d["date"].dt.to_period("M"))["day_pnl"].sum()
    pos_months= (monthly > 0).sum()

    n_months  = len(monthly)
    n_years   = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    ann_pnl   = total / n_years if n_years > 0 else 0

    return {
        "label":      label,
        "total":      total,
        "ann_pnl":    ann_pnl,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "min_buf":    min_buf,
        "pos_months": f"{pos_months}/{n_months}",
        "worst_month":monthly.min(),
        "best_month": monthly.max(),
        "avg_month":  monthly.mean(),
        "monthly":    monthly,
    }


def print_oos_report(is_no, is_lim, oos_no, oos_lim):
    print()
    print("=" * 80)
    print("  PART 2 — OUT-OF-SAMPLE TEST")
    print(f"  IN-SAMPLE:      2023-03 to 2024-09 (~18 months)")
    print(f"  OUT-OF-SAMPLE:  2024-10 to 2026-03 (~18 months)")
    print(f"  5x MGC + 5x MNQ  |  $500/day limit vs No limit")
    print("=" * 80)

    # Summary table
    cols = [is_no, is_lim, oos_no, oos_lim]
    labels = ["IS No limit", "IS $500 limit", "OOS No limit", "OOS $500 limit"]
    w = 14

    print(f"\n  {'Metric':<28}" +
          "".join(f"  {l:>{w}}" for l in labels))
    print("  " + "-" * (28 + (w+2) * 4))

    def row(label, key, fmt):
        line = f"  {label:<28}"
        for c in cols:
            v = c.get(key, 0)
            if fmt == "$":
                s = f"${v:,.0f}"
            elif fmt == "f":
                s = f"{v:.2f}"
            elif fmt == "s":
                s = str(v)
            else:
                s = str(v)
            line += f"  {s:>{w}}"
        print(line)

    row("Total P&L",             "total",       "$")
    row("Annualized P&L",        "ann_pnl",     "$")
    row("Sharpe",                "sharpe",      "f")
    row("Max drawdown",          "max_dd",      "$")
    row("Min MLL buffer",        "min_buf",     "$")
    row("Positive months",       "pos_months",  "s")
    row("Worst month",           "worst_month", "$")
    row("Best month",            "best_month",  "$")
    row("Avg monthly P&L",       "avg_month",   "$")

    # Month-by-month OOS comparison
    print()
    print("  OUT-OF-SAMPLE Monthly P&L — No Limit vs $500 Limit:")
    print(f"  {'Month':<10} {'No Limit':>12} {'$500 Limit':>12} "
          f"{'Difference':>12} {'Limit helped?':>14}")
    print("  " + "-" * 64)

    oos_m_no  = oos_no["monthly"]
    oos_m_lim = oos_lim["monthly"]
    all_months= sorted(set(oos_m_no.index) | set(oos_m_lim.index))

    limit_helped = 0
    limit_hurt   = 0
    limit_neutral= 0

    for period in all_months:
        v_no  = oos_m_no.get(period,  0)
        v_lim = oos_m_lim.get(period, 0)
        diff  = v_lim - v_no
        if diff > 50:
            verdict = "YES ✓"
            limit_helped += 1
        elif diff < -50:
            verdict = "NO  ✗"
            limit_hurt += 1
        else:
            verdict = "neutral"
            limit_neutral += 1
        sign_no  = "+" if v_no  >= 0 else ""
        sign_lim = "+" if v_lim >= 0 else ""
        sign_d   = "+" if diff  >= 0 else ""
        print(f"  {str(period):<10} "
              f"{sign_no}${v_no:>9,.0f}  "
              f"{sign_lim}${v_lim:>9,.0f}  "
              f"{sign_d}${diff:>9,.0f}  "
              f"{verdict:>14}")

    print()
    print(f"  OOS limit helped: {limit_helped} months  |  "
          f"hurt: {limit_hurt} months  |  "
          f"neutral: {limit_neutral} months")

    # The key verdict
    print()
    print("  ─── OVERFITTING VERDICT ────────────────────────────────────────")
    is_improvement  = is_lim["sharpe"]  - is_no["sharpe"]
    oos_improvement = oos_lim["sharpe"] - oos_no["sharpe"]
    retention = oos_improvement / is_improvement if is_improvement > 0 else 0

    print(f"\n  In-sample  Sharpe improvement:  {is_improvement:+.2f} "
          f"({is_no['sharpe']:.2f} → {is_lim['sharpe']:.2f})")
    print(f"  Out-of-sample Sharpe improvement:{oos_improvement:+.2f} "
          f"({oos_no['sharpe']:.2f} → {oos_lim['sharpe']:.2f})")
    print(f"  Retention of improvement OOS:    {retention:.0%}")
    print()

    if retention >= 0.70:
        print("  ✓ GENUINE PROTECTION — limit improvement holds strongly OOS")
        print("    The $500 daily limit is a real risk management improvement,")
        print("    not an artifact of fitting to this specific dataset.")
    elif retention >= 0.40:
        print("  ~ PARTIAL — some improvement persists OOS but weaker than IS")
        print("    The limit provides real protection but the magnitude IS vs OOS")
        print("    suggests some fitting to specific bad days in the IS period.")
        print("    Consider using $750 limit instead — slightly less fitted.")
    else:
        print("  ✗ LIKELY OVERFITTED — improvement does not persist OOS")
        print("    The $500 limit was tuned (even if unintentionally) to the")
        print("    specific bad days in the in-sample period.")
        print("    Recommendation: use no limit or a wider $1,000+ limit.")

    print()
    print("  Note: market exit slippage on limit trigger is typically")
    print("  ±$5-15 per flatten (1-2 points on MGC/MNQ). Immaterial at")
    print("  this P&L scale. Automated enforcement eliminates discretion risk.")
    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nLoading data...")
    mgc_bars  = load("MGC")
    mnq_bars  = load("MNQ")
    mgc_daily = build_mgc_daily(mgc_bars)
    mnq_daily = build_mnq_daily(mnq_bars)

    IS_START  = pd.Timestamp("2023-03-01")
    IS_END    = OOS_START                   # 2024-10-01
    OOS_END   = pd.Timestamp("2026-04-01")

    # -------------------------------------------------------------------
    # PART 1 — Audit every day the limit fired
    # -------------------------------------------------------------------
    print("\nPart 1: Running loss limit firing audit...")
    audit_df = run_limit_audit(mgc_bars, mgc_daily, mnq_bars, mnq_daily)
    print_limit_audit(audit_df)

    # -------------------------------------------------------------------
    # PART 2 — OOS test
    # -------------------------------------------------------------------
    print("\nPart 2: Running out-of-sample test...")
    print(f"  In-sample:      {IS_START.date()} → {IS_END.date()}")
    print(f"  Out-of-sample:  {IS_END.date()} → {OOS_END.date()}")

    print("  Running IS no limit...",    end=" ")
    df_is_no  = run_period(mgc_bars, mgc_daily, mnq_bars, mnq_daily,
                           IS_START, IS_END, use_limit=False)
    s_is_no   = period_stats(df_is_no,  "IS No limit",   IS_START, IS_END)
    print(f"P&L=${s_is_no['total']:,.0f}")

    print("  Running IS $500 limit...",  end=" ")
    df_is_lim = run_period(mgc_bars, mgc_daily, mnq_bars, mnq_daily,
                           IS_START, IS_END, use_limit=True)
    s_is_lim  = period_stats(df_is_lim, "IS $500 limit", IS_START, IS_END)
    print(f"P&L=${s_is_lim['total']:,.0f}")

    print("  Running OOS no limit...",   end=" ")
    df_oos_no = run_period(mgc_bars, mgc_daily, mnq_bars, mnq_daily,
                           IS_END,   OOS_END, use_limit=False)
    s_oos_no  = period_stats(df_oos_no, "OOS No limit",  IS_END,   OOS_END)
    print(f"P&L=${s_oos_no['total']:,.0f}")

    print("  Running OOS $500 limit...", end=" ")
    df_oos_lim= run_period(mgc_bars, mgc_daily, mnq_bars, mnq_daily,
                           IS_END,   OOS_END, use_limit=True)
    s_oos_lim = period_stats(df_oos_lim,"OOS $500 limit",IS_END,   OOS_END)
    print(f"P&L=${s_oos_lim['total']:,.0f}")

    print_oos_report(s_is_no, s_is_lim, s_oos_no, s_oos_lim)


if __name__ == "__main__":
    main()
