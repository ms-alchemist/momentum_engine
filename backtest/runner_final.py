"""
backtest/runner_final.py
=========================
FINAL PRODUCTION RUNNER
MGC Mean-Reversion + MNQ Volatility Breakout Portfolio

Strategy summary:
  MGC: First-bar mean-reversion signal
       Filter: Skip counter-SMA20 trades AND test TWAP filter
       Stop: 22pt hard stop
       Exit: Hold to session close (4:15 PM ET)

  MNQ: Volatility breakout momentum
       Signal: 2-bar consolidation < 30% ATR → breakout close
       Stop: Opposite end of consolidation range
       Exit: Hold to session close

Two modes per run:
  MODE A — Lucid Eval Simulation
    Starts at $100,000, stops when $6,000 profit target hit.
    Tracks MLL trailing floor exactly.
    Reports: days to target, max drawdown, consistency flags.

  MODE B — 3-Year Income Projection
    Runs full 3-year period with no target stop.
    Reports: annual P&L, monthly breakdown, drawdown profile.

Filters compared:
  1. No filter (baseline)
  2. Skip counter-SMA20 (validated OOS — Sharpe 3.08)
  3. TWAP filter (prior-session VWAP as intraday reference)

Sizing compared:
  4x MGC + 4x MNQ  (conservative)
  5x MGC + 5x MNQ  (standard)

Usage:
    python backtest/runner_final.py
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

# Lucid LucidFlex 100K
STARTING_BALANCE = 100_000.0
PROFIT_TARGET    =   6_000.0
MLL_BUFFER       =   3_000.0

# MGC
MGC_PV        = 10.0
MGC_COMM      = 0.80
MGC_STOP_PTS  = 22.0
MGC_VOL_RATIO = 2.5
MGC_OPEN      = time(9,  0)
MGC_SIGNAL    = time(9, 30)
MGC_CLOSE     = time(16, 15)

# MNQ
MNQ_PV        = 2.0
MNQ_COMM      = 0.35
MNQ_N_CONSOL  = 2
MNQ_ATR_THRESH= 0.30
MNQ_OPEN      = time(9,  0)
MNQ_CLOSE     = time(16, 15)

ATR_LOOKBACK  = 20

SIZING_COMBOS = [(4, 4), (5, 5)]

FILTER_NAMES = [
    "No filter",
    "SMA20 filter",
    "TWAP filter",
]


# ---------------------------------------------------------------------------
# MLL Tracker
# ---------------------------------------------------------------------------

class LucidMLL:
    """
    Exact Lucid trailing MLL mechanics.
    Floor = min(peak_EOD - $3,000, $100,000)
    Starts at $97,000. Only reaches $100,000 when peak >= $103,000.
    """
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

    def is_breached(self): return self.balance < self.floor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(symbol):
    df = pd.read_parquet(CACHE_DIR / f"{symbol}_30min.parquet")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    return df.sort_index()


def build_mgc_ctx(bars):
    """Daily context for MGC including SMA and VWAP."""
    d = bars.resample("1D").agg(
        high=("high","max"), low=("low","min"),
        open=("open","first"), close=("close","last"),
        volume=("volume","sum")).dropna()
    d = d[d["volume"] > 0]

    d["range"]     = d["high"] - d["low"]
    d["range_5d"]  = d["range"].rolling(5).mean()
    d["range_20d"] = d["range"].rolling(ATR_LOOKBACK).mean()
    d["vol_ratio"] = d["range_5d"] / d["range_20d"]

    # SMA20 filter
    d["sma20"]       = d["close"].rolling(20).mean()
    d["above_sma20"] = (d["close"] > d["sma20"]).astype(float)

    # VWAP filter: prior session VWAP
    # Compute daily VWAP = sum(price * volume) / sum(volume)
    # Use 30-min bars to compute intraday VWAP, then shift by 1 day
    daily_vwap = (
        bars.assign(pv=lambda df: ((df["high"] + df["low"] + df["close"]) / 3)
                    * df["volume"])
        .resample("1D")
        .agg(pv_sum=("pv","sum"), vol_sum=("volume","sum"))
    )
    daily_vwap["vwap"] = daily_vwap["pv_sum"] / daily_vwap["vol_sum"]
    daily_vwap = daily_vwap[daily_vwap["vol_sum"] > 0]

    # Shift VWAP by 1 day — we use PRIOR session VWAP as reference
    d["prior_vwap"] = daily_vwap["vwap"].shift(1).reindex(d.index)

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
# Filter functions
# ---------------------------------------------------------------------------

def apply_no_filter(ctx, sig, day_bars):
    """No filter — always trade."""
    return False  # False = don't skip


def apply_sma20_filter(ctx, sig, day_bars):
    """
    MGC mean-reversion filter:
    Price above SMA20 → market in uptrend → fade longs, take shorts
    Price below SMA20 → market in downtrend → fade shorts, take longs
    Skip if signal goes WITH the SMA trend (that's momentum, not reversion).
    Trade only when signal goes AGAINST the SMA trend (mean reversion).
    i.e. Skip longs when price below SMA20, skip shorts when price above SMA20.
    """
    above = ctx.get("above_sma20", np.nan)
    if np.isnan(above):
        return False
    # Skip: going short when above SMA20 (trend is up, mean reversion = long)
    # Skip: going long when below SMA20 (trend is down, mean reversion = short)
    return (sig == -1 and above == 1.0) or (sig == 1 and above == 0.0)


def apply_twap_filter(ctx, sig, day_bars):
    """
    TWAP/VWAP filter using prior session VWAP as reference level.
    Institutional logic: prior VWAP is the fair value reference.

    If today's open is ABOVE prior VWAP → market opened strong → demand
      → long signals are WITH order flow → take longs, skip shorts
    If today's open is BELOW prior VWAP → market opened weak → supply
      → short signals are WITH order flow → take shorts, skip longs

    Mean-reversion interpretation:
      → Skip trades that go against the VWAP-anchored order flow direction
      → This is the Bühler Lagrangian-mean signal: VWAP tracks the
         particle trajectory of order flow, not just price levels
    """
    prior_vwap = ctx.get("prior_vwap", np.nan)
    if np.isnan(prior_vwap) or prior_vwap <= 0:
        return False

    # Get today's opening price
    open_bars = day_bars[day_bars.index.time >= MGC_OPEN]
    if len(open_bars) == 0:
        return False
    today_open = open_bars.iloc[0]["open"]

    # Market opened above prior VWAP = demand → longs favored
    # Skip shorts when opened above VWAP (going against demand)
    # Skip longs when opened below VWAP (going against supply)
    opened_above_vwap = today_open > prior_vwap
    return (sig == -1 and opened_above_vwap) or (sig == 1 and not opened_above_vwap)


FILTERS = {
    "No filter":   apply_no_filter,
    "SMA20 filter": apply_sma20_filter,
    "TWAP filter": apply_twap_filter,
}


# ---------------------------------------------------------------------------
# Trade logic
# ---------------------------------------------------------------------------

def get_mgc_trade(day_bars, ctx, n_ct, filter_fn):
    """Returns trade dict or None. Applies filter before trading."""
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
    sig = int(np.sign(f["close"] - f["open"]))
    if sig == 0:
        return None

    # Apply filter
    if filter_fn(ctx, sig, day_bars):
        return None  # filtered out

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
        "sig": sig, "instrument": "MGC",
        "entry": entry, "exit": ep,
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
    return {"pnl": pnl, "win": pnl > 0, "reason": reason,
            "instrument": "MNQ", "entry": entry, "exit": ep}


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(mgc_bars, mgc_ctx, mnq_bars, mnq_daily,
                 n_mgc, n_mnq, filter_fn, stop_at_target=True):
    """
    Run portfolio backtest.
    stop_at_target=True  → Lucid eval mode (stop at $6,000)
    stop_at_target=False → Income projection mode (run full period)
    """
    mll     = LucidMLL()
    results = []
    trades  = []
    cum_pnl = 0.0

    all_days = sorted(set(mgc_bars.index.date) | set(mnq_bars.index.date))

    for day in all_days:
        if mll.is_breached():
            break
        if stop_at_target and mll.profit >= PROFIT_TARGET:
            break

        day_pnl    = 0.0
        day_trades = []

        # MGC
        mgc_day  = mgc_bars[mgc_bars.index.date == day]
        mgc_mask = mgc_ctx.index.date == day
        if len(mgc_day) > 0 and mgc_mask.any():
            ctx = mgc_ctx[mgc_mask].iloc[0].to_dict()
            t   = get_mgc_trade(mgc_day, ctx, n_mgc, filter_fn)
            if t:
                day_pnl += t["pnl"]
                day_trades.append(t)

        # MNQ
        mnq_day  = mnq_bars[mnq_bars.index.date == day]
        mnq_mask = mnq_daily.index.date == day
        if len(mnq_day) > 0 and mnq_mask.any():
            ctx = mnq_daily[mnq_mask].iloc[0].to_dict()
            t   = get_mnq_trade(mnq_day, ctx, n_mnq)
            if t:
                day_pnl += t["pnl"]
                day_trades.append(t)

        cum_pnl += day_pnl
        mll.update_eod(mll.balance + day_pnl)

        cf = (cum_pnl > 0 and day_pnl > 0 and
              day_pnl > 0.50 * cum_pnl)

        results.append({
            "date":             day,
            "day_pnl":          day_pnl,
            "balance":          mll.balance,
            "floor":            mll.floor,
            "buffer":           mll.buffer,
            "n_trades":         len(day_trades),
            "consistency_flag": cf,
        })
        for t in day_trades:
            t["date"] = day
            trades.append(t)

    return pd.DataFrame(results), pd.DataFrame(trades), mll


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(daily, trades, mll, n_mgc, n_mnq,
                  filter_name, mode):
    eq       = daily["balance"]
    roll_max = eq.cummax()
    max_dd   = (eq - roll_max).min()
    min_buf  = daily["buffer"].min()

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

    daily2 = daily.copy()
    daily2["date"] = pd.to_datetime(daily2["date"])
    monthly = daily2.groupby(daily2["date"].dt.to_period("M"))["day_pnl"].sum()
    pos_m   = (monthly > 0).sum()

    streak = max_s = cur = 0
    for v in monthly:
        if v < 0:
            cur += 1; max_s = max(max_s, cur)
        else:
            cur = 0

    n_years = len(monthly) / 12
    ann_pnl = mll.profit / n_years if n_years > 0 else 0

    return {
        "filter":          filter_name,
        "mode":            mode,
        "n_mgc":           n_mgc,
        "n_mnq":           n_mnq,
        "total_pnl":       mll.profit,
        "ann_pnl":         ann_pnl,
        "mgc_pnl":         mgc_pnl,
        "mnq_pnl":         mnq_pnl,
        "sharpe":          sharpe,
        "max_dd":          max_dd,
        "min_buf":         min_buf,
        "mll_breached":    mll.is_breached(),
        "target_hit":      target_hit,
        "days_to_target":  days_to_target,
        "n_consistency":   n_flags,
        "pos_months":      f"{pos_m}/{len(monthly)}",
        "worst_month":     monthly.min() if len(monthly) else 0,
        "best_month":      monthly.max() if len(monthly) else 0,
        "avg_month":       monthly.mean() if len(monthly) else 0,
        "max_consec_loss": max_s,
        "monthly":         monthly,
        "final_balance":   mll.balance,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_eval_report(all_stats):
    print()
    print("=" * 88)
    print("  MODE A — LUCID EVAL SIMULATION")
    print("  Target: $6,000 profit  |  MLL: $3,000 trailing  |  Start: $100,000")
    print("=" * 88)
    print(f"\n  {'Sizing':<10} {'Filter':<20} {'P&L':>8} {'Sharpe':>7} "
          f"{'MaxDD':>9} {'MinBuf':>8} {'Days':>6} {'Wks':>5} "
          f"{'CF':>4} {'Breach':>7}")
    print("  " + "-" * 88)

    for s in all_stats:
        days   = s["days_to_target"] or 0
        wks    = days // 5
        breach = "YES ⚠" if s["mll_breached"] else "safe"
        target = f"{days}d" if s["target_hit"] else "MISS"
        print(f"  {s['n_mgc']}M+{s['n_mnq']}N    "
              f"{s['filter']:<20} "
              f"${s['total_pnl']:>7,.0f}  "
              f"{s['sharpe']:>6.2f}  "
              f"${s['max_dd']:>7,.0f}  "
              f"${s['min_buf']:>6,.0f}  "
              f"{target:>6}  {wks:>4}wk  "
              f"{s['n_consistency']:>3}cf  {breach}")

    print()
    safe = [s for s in all_stats
            if not s["mll_breached"] and s["target_hit"]]
    if safe:
        fastest = min(safe, key=lambda s: s["days_to_target"] or 9999)
        print(f"  Fastest eval pass: "
              f"{fastest['n_mgc']}x MGC + {fastest['n_mnq']}x MNQ  "
              f"| {fastest['filter']}  "
              f"| {fastest['days_to_target']} days "
              f"(~{fastest['days_to_target']//5} calendar weeks)")


def print_income_report(all_stats, all_daily):
    print()
    print("=" * 88)
    print("  MODE B — 3-YEAR INCOME PROJECTION")
    print("  Full period, no target stop  |  5x MGC + 5x MNQ")
    print("=" * 88)
    print(f"\n  {'Sizing':<10} {'Filter':<20} {'3yr P&L':>9} {'Ann P&L':>9} "
          f"{'Sharpe':>7} {'MaxDD':>9} {'PosM':>7} "
          f"{'WorstM':>9} {'AvgM':>8}")
    print("  " + "-" * 95)

    for s in all_stats:
        breach = " ⚠" if s["mll_breached"] else ""
        print(f"  {s['n_mgc']}M+{s['n_mnq']}N    "
              f"{s['filter']:<20} "
              f"${s['total_pnl']:>8,.0f}  "
              f"${s['ann_pnl']:>8,.0f}  "
              f"{s['sharpe']:>6.2f}  "
              f"${s['max_dd']:>7,.0f}  "
              f"{s['pos_months']:>7}  "
              f"${s['worst_month']:>7,.0f}  "
              f"${s['avg_month']:>6,.0f}{breach}")

    # Monthly detail for best filter at each sizing
    print()
    shown = set()
    for s, df in zip(all_stats, all_daily):
        key = (s["n_mgc"], s["n_mnq"], s["filter"])
        if key in shown:
            continue
        shown.add(key)

        # Only show best filter per sizing
        if s["filter"] not in ("SMA20 filter", "TWAP filter"):
            continue

        print(f"\n  Monthly P&L — {s['n_mgc']}x MGC + {s['n_mnq']}x MNQ"
              f"  [{s['filter']}]:")
        print(f"  {'Month':<10} {'P&L':>10}  Chart")
        print("  " + "-" * 55)

        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"])
        monthly = df2.groupby(df2["date"].dt.to_period("M"))["day_pnl"].sum()
        pos = (monthly > 0).sum()

        for period, v in monthly.items():
            sign = "+" if v >= 0 else ""
            bar  = ("█" if v >= 0 else "░") * min(int(abs(v)/500), 24)
            print(f"  {str(period):<10} {sign}${v:>8,.0f}  {bar}")

        print(f"  Positive months: {pos}/{len(monthly)}  |  "
              f"Max consec loss: {s['max_consec_loss']} months  |  "
              f"Avg: ${s['avg_month']:,.0f}/month")


def print_final_summary(eval_stats, income_stats):
    print()
    print("=" * 88)
    print("  STRATEGY DECISION SUMMARY")
    print("=" * 88)

    # Find best eval combo
    safe_eval = [s for s in eval_stats
                 if not s["mll_breached"] and s["target_hit"]]
    best_eval = min(safe_eval, key=lambda s: s["days_to_target"] or 9999) \
                if safe_eval else None

    # Find best income combo (by Sharpe, safe only)
    safe_inc = [s for s in income_stats if not s["mll_breached"]]
    best_inc = max(safe_inc, key=lambda s: s["sharpe"]) if safe_inc else None

    if best_eval:
        print(f"\n  EVAL RECOMMENDATION:")
        print(f"    Sizing:  {best_eval['n_mgc']}x MGC + {best_eval['n_mnq']}x MNQ")
        print(f"    Filter:  {best_eval['filter']}")
        print(f"    Target in: {best_eval['days_to_target']} trading days "
              f"(~{best_eval['days_to_target']//5} weeks)")
        print(f"    Max DD during eval: ${best_eval['max_dd']:,.0f}")
        print(f"    Min MLL buffer: ${best_eval['min_buf']:,.0f}")
        print(f"    Consistency flags: {best_eval['n_consistency']}")

    if best_inc:
        print(f"\n  FUNDED ACCOUNT (income) RECOMMENDATION:")
        print(f"    Sizing:  {best_inc['n_mgc']}x MGC + {best_inc['n_mnq']}x MNQ")
        print(f"    Filter:  {best_inc['filter']}")
        print(f"    Ann P&L: ${best_inc['ann_pnl']:,.0f}/year  "
              f"(${best_inc['ann_pnl']/12:,.0f}/month)")
        print(f"    Sharpe:  {best_inc['sharpe']:.2f}")
        print(f"    Max DD:  ${best_inc['max_dd']:,.0f}")
        print(f"    Worst month: ${best_inc['worst_month']:,.0f}")
        print(f"    Positive months: {best_inc['pos_months']}")
        print(f"    Max consec losing months: {best_inc['max_consec_loss']}")

    print()
    print("  THEORETICAL BASIS:")
    print("    MGC: Mean-reversion to SMA20 (Ornstein-Uhlenbeck process,")
    print("         Tsekrekos 2010 + Micaletti 2023 variance ratio filter)")
    print("    MNQ: Pseudomomentum breakout (Bühler 2014 wave accumulation)")
    print("    Filter: SMA20 identifies OU equilibrium — trade only when")
    print("         first-bar signal is consistent with mean-reversion")
    print("         back toward the 20-day moving average")
    print("    TWAP: Prior-session VWAP as Lagrangian-mean reference")
    print("         (Bühler particle-following average vs Eulerian price)")
    print()
    print("  AUTOMATION NOTES:")
    print("    MGC stop: hard limit order at entry - 22pts (placed immediately)")
    print("    MNQ stop: hard limit order at consolidation range boundary")
    print("    Exit: MOC (Market-on-Close) order submitted by 4:10 PM ET")
    print("    SMA20: computed from prior close, known before market open")
    print("    TWAP:  computed from prior session, known before market open")
    print("    All signals fire once per day — minimal execution complexity")
    print()
    print("=" * 88)


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
          f"({mgc_bars.index[0].date()} to {mgc_bars.index[-1].date()})")
    print(f"  MNQ: {len(set(mnq_bars.index.date))} days  "
          f"({mnq_bars.index[0].date()} to {mnq_bars.index[-1].date()})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_stats   = []
    income_stats = []
    income_daily = []

    total = len(SIZING_COMBOS) * len(FILTER_NAMES) * 2
    run_n = 0

    print(f"\nRunning {total} combinations "
          f"({len(SIZING_COMBOS)} sizings × "
          f"{len(FILTER_NAMES)} filters × 2 modes)...\n")

    for n_mgc, n_mnq in SIZING_COMBOS:
        for fname in FILTER_NAMES:
            fn = FILTERS[fname]

            # MODE A — Eval simulation
            run_n += 1
            print(f"  [{run_n:>2}/{total}] EVAL  "
                  f"{n_mgc}M+{n_mnq}N  {fname}...", end=" ")
            daily, trades, mll = run_backtest(
                mgc_bars, mgc_ctx, mnq_bars, mnq_daily,
                n_mgc, n_mnq, fn, stop_at_target=True)
            s = compute_stats(daily, trades, mll,
                              n_mgc, n_mnq, fname, "eval")
            eval_stats.append(s)
            status = (f"target in {s['days_to_target']}d"
                      if s["target_hit"] else "MISS")
            breach = " BREACH ⚠" if s["mll_breached"] else ""
            print(f"Sh={s['sharpe']:.2f}  "
                  f"DD=${s['max_dd']:,.0f}  "
                  f"{status}{breach}")

            # Save eval trade log
            if len(trades) > 0:
                trades.to_csv(
                    RESULTS_DIR /
                    f"final_eval_{n_mgc}mgc_{n_mnq}mnq_"
                    f"{fname.replace(' ','_')}_{ts}.csv",
                    index=False)

            # MODE B — Income projection
            run_n += 1
            print(f"  [{run_n:>2}/{total}] INCOME "
                  f"{n_mgc}M+{n_mnq}N  {fname}...", end=" ")
            daily2, trades2, mll2 = run_backtest(
                mgc_bars, mgc_ctx, mnq_bars, mnq_daily,
                n_mgc, n_mnq, fn, stop_at_target=False)
            s2 = compute_stats(daily2, trades2, mll2,
                               n_mgc, n_mnq, fname, "income")
            income_stats.append(s2)
            income_daily.append(daily2)
            breach2 = " BREACH ⚠" if s2["mll_breached"] else ""
            print(f"Sh={s2['sharpe']:.2f}  "
                  f"Ann=${s2['ann_pnl']:,.0f}  "
                  f"DD=${s2['max_dd']:,.0f}{breach2}")

    print_eval_report(eval_stats)
    print_income_report(income_stats, income_daily)
    print_final_summary(eval_stats, income_stats)

    print(f"  Trade logs saved to backtest/results/")


if __name__ == "__main__":
    main()
