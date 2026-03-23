"""
backtest/runner_mgc_mr.py
=========================
MGC Intraday Mean-Reversion Backtest — Direction 2

Strategy logic (data-driven from Session 4-5 research):

FADE trades (new):
  Entry:  9:30 AM, opposite to first-bar direction
  Signal: MRSI < 15 (close near HIGH) or MRSI > 85 (close near LOW)
          confirmed by MTSI sign
  Filter: OU half-life <= 8 bars (mean-reversion regime)
  Target: dynamic θ (session mean = 20-day rolling open average)
  T-stop: exit at 12:30 PM if not yet at θ (1 half-life)
  C-stop: 15pt catastrophic price stop (1 σ_stat)

FOLLOW trades (baseline, retained):
  Entry:  9:30 AM, same direction as first bar
  Signal: MRSI 35-65 (close near middle of bar)
  Target: 4:15 PM session close
  C-stop: 22pt (Di Graziano — momentum strategy stop)

SKIP:
  OU half-life > 8 bars (trending day)
  Vol filter: 5-day range > 2.5x 20-day avg
  MRSI in ambiguous zone (15-35 or 65-85)

Sizing: 2 contracts always (vol-scaling is Phase 2 enhancement)

Comparison:
  - Direction 2 (fade + follow hybrid)
  - Baseline (pure follow, 2ct, 22pt stop)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time, datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = Path("data/cache")
RESULTS_DIR = Path("backtest/results")
RESULTS_DIR.mkdir(exist_ok=True)

# Instrument
PV          = 10.0   # $/pt MGC
COMM        = 0.80   # $ per contract RT
CONTRACTS   = 2

# Stops
FADE_STOP_PTS   = 15.0   # catastrophic — 1 σ_stat of OU process
FOLLOW_STOP_PTS = 22.0   # Di Graziano momentum stop

# Time stop: exit fade after 1 half-life if not at target
# Mean HL = 5.7 bars → use 6 bars = 3 hours
TIME_STOP_BARS  = 6

# OU regime filter
MAX_HL_BARS     = 8.0   # skip fade if HL > 8 bars (trending)

# Vol filter
VOL_RATIO_MAX   = 2.5
VOL_SHORT       = 5
VOL_LONG        = 20

# Theta window
THETA_WINDOW    = 20   # days for session mean

# MRSI thresholds
FADE_STRONG_LO  = 15
FADE_STRONG_HI  = 85
FADE_WEAK_LO    = 25
FADE_WEAK_HI    = 75
FOLLOW_LO       = 35
FOLLOW_HI       = 65

# Lucid account
STARTING_BALANCE = 100_000.0
PROFIT_TARGET    =   6_000.0
MLL_BUFFER       =   3_000.0

# Session timing
FIRST_BAR_OPEN   = time(9, 0)
FIRST_BAR_CLOSE  = time(9, 30)
SESSION_CLOSE    = time(16, 15)
TIME_STOP_TIME   = time(12, 30)   # 6 bars after 9:30 AM

MIN_WARMUP_DAYS  = 22


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

    def update(self, pnl):
        self.balance += pnl
        if self.balance > self.peak_eod:
            self.peak_eod = self.balance
        self.floor = min(self.peak_eod - MLL_BUFFER, STARTING_BALANCE)

    def breached(self):
        return self.balance < self.floor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars():
    bars = pd.read_parquet(DATA_DIR / "MGC_30min.parquet")
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    return bars.sort_index()


def load_ou():
    path = RESULTS_DIR / "ou_calibration.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).date
    return df


def load_mrsi():
    path = RESULTS_DIR / "mrsi_signals.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).date
    return df


# ---------------------------------------------------------------------------
# Indicators (computed inline if saved files not available)
# ---------------------------------------------------------------------------

def wilder_ma_series(values, period):
    n, out = len(values), np.full(len(values), np.nan)
    if n < period:
        return out
    out[period-1] = np.mean(values[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = out[i-1] + alpha * (values[i] - out[i-1])
    return out


def compute_mrsi_series(H, L, C, M=2, N=2):
    H = np.maximum(H, C)
    L = np.minimum(L, C)
    log_hc = np.log(H / C)
    log_cl = np.log(C / L)
    U  = wilder_ma_series(log_hc, M)
    D  = wilder_ma_series(log_cl, M)
    Un = np.power(np.maximum(U, 0), N)
    Dn = np.power(np.maximum(D, 0), N)
    denom = Un + Dn
    return np.where(denom > 1e-12, 100.0 * Dn / denom, 50.0)


def compute_mtsi_series(C, H, L, M=3, N=2):
    twap  = (H + L + C) / 3.0
    ratio = np.log(np.where(twap > 0, C / twap, 1.0))
    def ema(v, p):
        a, out = 2/(p+1), np.zeros(len(v))
        out[0] = v[0]
        for i in range(1, len(v)):
            out[i] = a*v[i] + (1-a)*out[i-1]
        return out
    num = ema(ema(ratio, M), N)
    den = ema(ema(np.abs(ratio), M), N)
    return np.where(den > 1e-12, 100.0 * num / den, 0.0)


# ---------------------------------------------------------------------------
# Per-day trade simulation
# ---------------------------------------------------------------------------

def simulate_fade(day_bars, direction, fade_stop_pts, theta, entry_price):
    """
    Simulate a FADE trade:
      - Go OPPOSITE to direction
      - Target: theta (session mean)
      - Time stop: 12:30 PM
      - Catastrophic stop: 15pts
    direction: original first-bar direction (1=up, -1=down)
    fade_dir:  -direction (we fade)
    """
    fade_dir    = -direction
    stop_price  = entry_price + fade_dir * fade_stop_pts   # note: fade_dir
    # Actually: if we're going short (fade_dir=-1), stop is ABOVE entry
    stop_price  = entry_price - fade_dir * fade_stop_pts

    exit_price  = None
    exit_reason = None

    post_entry  = day_bars[day_bars.index.time >= FIRST_BAR_CLOSE]

    for ts, bar in post_entry.iterrows():
        bar_time = ts.time()

        # Time stop: 12:30 PM
        if bar_time >= TIME_STOP_TIME and exit_price is None:
            exit_price  = bar["open"]
            exit_reason = "time_stop"
            break

        # Session close fallback
        if bar_time >= SESSION_CLOSE:
            exit_price  = bar["open"]
            exit_reason = "session_close"
            break

        # Catastrophic stop (fade_dir = -1 means short: stop if price goes UP)
        if fade_dir == -1 and bar["high"] >= stop_price:
            exit_price  = stop_price
            exit_reason = "cstop"
            break
        if fade_dir == 1 and bar["low"] <= stop_price:
            exit_price  = stop_price
            exit_reason = "cstop"
            break

        # Target: theta (dynamic session mean)
        if theta is not None:
            if fade_dir == -1 and bar["low"] <= theta:
                exit_price  = theta
                exit_reason = "target"
                break
            if fade_dir == 1 and bar["high"] >= theta:
                exit_price  = theta
                exit_reason = "target"
                break

    if exit_price is None:
        exit_price  = post_entry.iloc[-1]["close"] if len(post_entry) > 0 else entry_price
        exit_reason = "eod"

    pnl = fade_dir * (exit_price - entry_price) * PV * CONTRACTS - COMM * CONTRACTS
    return entry_price, exit_price, exit_reason, pnl


def simulate_follow(day_bars, direction, follow_stop_pts):
    """
    Simulate a FOLLOW trade (baseline momentum):
      - Go SAME direction as first bar
      - Target: 4:15 PM session close
      - Stop: 22pt
    """
    first = day_bars[
        (day_bars.index.time >= FIRST_BAR_OPEN) &
        (day_bars.index.time <  FIRST_BAR_CLOSE)
    ]
    if len(first) == 0:
        return None, None, None, 0.0

    entry_price = first.iloc[0]["close"]
    stop_price  = entry_price - direction * follow_stop_pts
    exit_price  = None
    exit_reason = None

    post = day_bars[day_bars.index.time >= FIRST_BAR_CLOSE]
    for ts, bar in post.iterrows():
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
        exit_price  = post.iloc[-1]["close"] if len(post) > 0 else entry_price
        exit_reason = "eod"

    pnl = direction * (exit_price - entry_price) * PV * CONTRACTS - COMM * CONTRACTS
    return entry_price, exit_price, exit_reason, pnl


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(bars, ou_df, mrsi_df, label="Direction 2"):
    mll     = LucidMLL()
    trades  = []
    daily_r = []

    # Build daily vol filter
    daily = bars.resample("1D").agg({"high":"max","low":"min","volume":"sum"})
    daily = daily[daily["volume"] > 0].copy()
    daily["range"]  = daily["high"] - daily["low"]
    daily["r5"]     = daily["range"].rolling(VOL_SHORT).mean()
    daily["r20"]    = daily["range"].rolling(VOL_LONG).mean()
    daily["vr"]     = daily["r5"] / daily["r20"]

    # Build rolling theta (20-day session open mean)
    session_opens = (
        bars[bars.index.time == FIRST_BAR_OPEN]
        .resample("1D")["open"].first()
        .dropna()
    )
    theta_series = session_opens.rolling(THETA_WINDOW).mean()

    # Build first-bar MRSI on the fly (or load from file)
    fb = bars[
        (bars.index.time >= FIRST_BAR_OPEN) &
        (bars.index.time <  FIRST_BAR_CLOSE)
    ].groupby(bars[
        (bars.index.time >= FIRST_BAR_OPEN) &
        (bars.index.time <  FIRST_BAR_CLOSE)
    ].index.date).first()

    H_all = fb["high"].values
    L_all = fb["low"].values
    C_all = fb["close"].values

    mrsi_vals = compute_mrsi_series(H_all, L_all, C_all)
    mtsi_vals = compute_mtsi_series(C_all, H_all, L_all)
    fb_dates  = list(fb.index)

    mrsi_by_date = {d: (mrsi_vals[i], mtsi_vals[i])
                    for i, d in enumerate(fb_dates)}

    trading_days = sorted(set(bars.index.date))

    for day_num, day in enumerate(trading_days):
        if mll.breached():
            break
        if mll.profit >= PROFIT_TARGET:
            break
        if day_num < MIN_WARMUP_DAYS:
            daily_r.append({"date": day, "pnl": 0,
                            "balance": mll.balance, "reason": "warmup"})
            continue

        # Vol filter
        dmask = daily.index.date == day
        if not dmask.any():
            continue
        di = daily[dmask].iloc[0]
        if not np.isnan(di["vr"]) and di["vr"] > VOL_RATIO_MAX:
            daily_r.append({"date": day, "pnl": 0,
                            "balance": mll.balance, "reason": "vol_filter"})
            continue

        # Day bars
        day_bars = bars[bars.index.date == day]
        if len(day_bars) < 3:
            continue

        # First bar
        first = day_bars[
            (day_bars.index.time >= FIRST_BAR_OPEN) &
            (day_bars.index.time <  FIRST_BAR_CLOSE)
        ]
        if len(first) == 0:
            continue
        fb_bar     = first.iloc[0]
        entry_price= fb_bar["close"]

        if   fb_bar["close"] > fb_bar["open"]: fb_dir = 1
        elif fb_bar["close"] < fb_bar["open"]: fb_dir = -1
        else:
            daily_r.append({"date": day, "pnl": 0,
                            "balance": mll.balance, "reason": "doji"})
            continue

        # MRSI signal
        mrsi_val, mtsi_val = mrsi_by_date.get(day, (np.nan, np.nan))

        # OU half-life filter
        hl_bars = np.nan
        if ou_df is not None and day in ou_df.index:
            hl_bars = ou_df.loc[day, "hl_bars"]

        # Theta (session mean target)
        ts_day = pd.Timestamp(day).tz_localize("America/New_York")
        theta  = None
        if ts_day in theta_series.index:
            theta = float(theta_series.loc[ts_day])
        elif pd.Timestamp(day) in theta_series.index:
            theta = float(theta_series.loc[pd.Timestamp(day)])

        # --- Signal classification ---
        trade_type = "no_trade"

        if not np.isnan(mrsi_val):
            # Fade signal
            fade_ok = (np.isnan(hl_bars) or hl_bars <= MAX_HL_BARS)
            if fade_ok:
                if mrsi_val < FADE_STRONG_LO and mtsi_val > 5:
                    trade_type = "fade"    # close near HIGH → fade short
                elif mrsi_val > FADE_STRONG_HI and mtsi_val < -5:
                    trade_type = "fade"    # close near LOW  → fade long
                elif mrsi_val < FADE_WEAK_LO and mtsi_val > 2:
                    trade_type = "fade_weak"
                elif mrsi_val > FADE_WEAK_HI and mtsi_val < -2:
                    trade_type = "fade_weak"

            # Follow signal (irrespective of OU filter)
            if trade_type == "no_trade":
                if FOLLOW_LO <= mrsi_val <= FOLLOW_HI:
                    trade_type = "follow"

        if trade_type == "no_trade":
            daily_r.append({"date": day, "pnl": 0,
                            "balance": mll.balance, "reason": "no_signal"})
            continue

        # --- Execute trade ---
        if trade_type in ("fade", "fade_weak"):
            entry, exit_p, exit_r, pnl = simulate_fade(
                day_bars, fb_dir, FADE_STOP_PTS, theta, entry_price
            )
            actual_dir = -fb_dir
        else:
            entry, exit_p, exit_r, pnl = simulate_follow(
                day_bars, fb_dir, FOLLOW_STOP_PTS
            )
            actual_dir = fb_dir
            if entry is None:
                continue

        mll.update(pnl)

        trades.append({
            "date":        day,
            "type":        trade_type,
            "direction":   "LONG" if actual_dir == 1 else "SHORT",
            "fb_dir":      "UP" if fb_dir == 1 else "DOWN",
            "entry":       entry,
            "exit":        exit_p,
            "exit_reason": exit_r,
            "mrsi":        mrsi_val,
            "mtsi":        mtsi_val,
            "hl_bars":     hl_bars,
            "theta":       theta,
            "net_pnl":     pnl,
            "win":         pnl > 0,
        })

        daily_r.append({
            "date":    day,
            "pnl":     pnl,
            "balance": mll.balance,
            "reason":  trade_type,
            "type":    trade_type,
        })

    return pd.DataFrame(trades), pd.DataFrame(daily_r), mll


# ---------------------------------------------------------------------------
# Baseline (pure follow, 22pt stop)
# ---------------------------------------------------------------------------

def run_baseline(bars):
    mll     = LucidMLL()
    trades  = []
    daily_r = []

    daily = bars.resample("1D").agg({"high":"max","low":"min","volume":"sum"})
    daily = daily[daily["volume"] > 0].copy()
    daily["range"] = daily["high"] - daily["low"]
    daily["r5"]    = daily["range"].rolling(VOL_SHORT).mean()
    daily["r20"]   = daily["range"].rolling(VOL_LONG).mean()
    daily["vr"]    = daily["r5"] / daily["r20"]

    for day_num, day in enumerate(sorted(set(bars.index.date))):
        if mll.breached() or mll.profit >= PROFIT_TARGET:
            break
        if day_num < MIN_WARMUP_DAYS:
            continue

        dmask = daily.index.date == day
        if not dmask.any():
            continue
        di = daily[dmask].iloc[0]
        if not np.isnan(di["vr"]) and di["vr"] > VOL_RATIO_MAX:
            continue

        day_bars = bars[bars.index.date == day]
        if len(day_bars) < 3:
            continue

        first = day_bars[
            (day_bars.index.time >= FIRST_BAR_OPEN) &
            (day_bars.index.time <  FIRST_BAR_CLOSE)
        ]
        if len(first) == 0:
            continue
        fb = first.iloc[0]
        if   fb["close"] > fb["open"]: d = 1
        elif fb["close"] < fb["open"]: d = -1
        else: continue

        entry, exit_p, exit_r, pnl = simulate_follow(day_bars, d, 22.0)
        if entry is None:
            continue

        mll.update(pnl)
        trades.append({"date":day,"net_pnl":pnl,"win":pnl>0,
                       "exit_reason":exit_r})
        daily_r.append({"date":day,"pnl":pnl,"balance":mll.balance})

    return pd.DataFrame(trades), pd.DataFrame(daily_r), mll


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(trades_df, daily_df, mll, label):
    if trades_df.empty:
        print(f"  {label}: NO TRADES")
        return {}

    n   = len(trades_df)
    wr  = trades_df["win"].mean()
    tot = trades_df["net_pnl"].sum()
    wins   = trades_df[trades_df["win"]]["net_pnl"]
    losses = trades_df[~trades_df["win"]]["net_pnl"]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.inf

    active = daily_df[daily_df["pnl"] != 0]
    sharpe = ((active["pnl"].mean() / active["pnl"].std()) * np.sqrt(252)
              if len(active) > 1 and active["pnl"].std() > 0 else 0)

    eq     = daily_df["balance"].dropna()
    max_dd = (eq - eq.cummax()).min() if len(eq) > 0 else 0
    min_buf= mll.buffer

    print()
    print("=" * 65)
    print(f"  {label}")
    print("=" * 65)
    print(f"  Trades={n}  WR={wr:.1%}  PF={pf:.2f}  "
          f"P&L=${tot:,.0f}  Sharpe={sharpe:.2f}")
    print(f"  MaxDD=${max_dd:,.0f}  MinBuf=${min_buf:,.0f}  "
          f"MLL={'SAFE' if not mll.breached() else 'BREACH'}  "
          f"Target={'HIT' if mll.profit >= PROFIT_TARGET else 'miss'}")

    # By trade type
    if "type" in trades_df.columns:
        print()
        for ttype in ["fade", "fade_weak", "follow"]:
            sub = trades_df[trades_df["type"] == ttype]
            if len(sub) == 0:
                continue
            sw = sub[sub["win"]]
            sl = sub[~sub["win"]]
            spf = (sw["net_pnl"].sum() / abs(sl["net_pnl"].sum())
                   if len(sl) and sl["net_pnl"].sum() != 0 else np.inf)
            print(f"  {ttype:<12} {len(sub):>4} trades  "
                  f"WR={sub['win'].mean():.1%}  "
                  f"PF={spf:.2f}  "
                  f"P&L=${sub['net_pnl'].sum():>8,.0f}  "
                  f"avg=${sub['net_pnl'].mean():>6.0f}")

    # By exit reason
    if "exit_reason" in trades_df.columns:
        print()
        er = trades_df.groupby("exit_reason")["net_pnl"].agg(
            n="count", total="sum",
            wr=lambda x: (x > 0).mean()
        )
        print(f"  {'Exit':<15} {'N':>5} {'WR':>7} {'Total':>10}")
        print("  " + "-" * 42)
        for reason, row in er.iterrows():
            print(f"  {reason:<15} {row['n']:>5.0f}  "
                  f"{row['wr']:>6.1%}  ${row['total']:>8,.0f}")

    # Monthly P&L
    t2 = trades_df.copy()
    t2["date"] = pd.to_datetime(t2["date"])
    monthly = t2.groupby(t2["date"].dt.to_period("M"))["net_pnl"].sum()
    print()
    print(f"  Monthly P&L:")
    pos = (monthly > 0).sum()
    neg = (monthly <= 0).sum()
    for period, pnl in monthly.items():
        sign = "+" if pnl >= 0 else ""
        bar  = ("█" * min(int(abs(pnl)/150), 20))
        print(f"    {str(period):<8}  {sign}${pnl:>8,.0f}  {bar}")
    print(f"  Positive months: {pos}/{pos+neg}")

    return {"total_pnl": tot, "sharpe": sharpe, "win_rate": wr,
            "profit_factor": pf, "max_dd": max_dd,
            "target_hit": mll.profit >= PROFIT_TARGET}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("MGC Mean-Reversion Backtest — Direction 2")
    print("=" * 65)
    print(f"  Fade stop:   {FADE_STOP_PTS}pt  |  Follow stop: {FOLLOW_STOP_PTS}pt")
    print(f"  Time stop:   {TIME_STOP_TIME}  |  Target: dynamic θ")
    print(f"  OU HL gate:  ≤ {MAX_HL_BARS} bars  |  Contracts: {CONTRACTS}")
    print()

    bars   = load_bars()
    ou_df  = load_ou()
    mrsi_df= load_mrsi()

    if ou_df is None:
        print("  WARNING: ou_calibration.csv not found. Run ou_calibrator.py first.")
    if mrsi_df is None:
        print("  WARNING: mrsi_signals.csv not found. Run mrsi.py first.")

    print(f"  MGC: {len(bars)} 30-min bars  "
          f"({bars.index[0].date()} to {bars.index[-1].date()})")
    print()

    # Run Direction 2
    print("  Running Direction 2 (fade + follow hybrid)...")
    trades_d2, daily_d2, mll_d2 = run_backtest(bars, ou_df, mrsi_df,
                                                 "Direction 2")

    # Run baseline
    print("  Running baseline (pure follow, 22pt stop)...")
    trades_bl, daily_bl, mll_bl = run_baseline(bars)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_d2.to_csv(RESULTS_DIR / f"mr_trades_{ts}.csv", index=False)
    trades_bl.to_csv(RESULTS_DIR / f"bl_trades_{ts}.csv", index=False)

    # Report
    stats_d2 = report(trades_d2, daily_d2, mll_d2,
                      "DIRECTION 2 — Fade + Follow Hybrid")
    stats_bl = report(trades_bl, daily_bl, mll_bl,
                      "BASELINE — Pure Follow 2ct 22pt stop")

    # Comparison
    print()
    print("=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<20} {'Baseline':>12} {'Dir2':>12} {'Delta':>10}")
    print("  " + "-" * 56)
    for label, key, fmt in [
        ("Total P&L",     "total_pnl",     "${:,.0f}"),
        ("Sharpe",        "sharpe",        "{:.2f}"),
        ("Win rate",      "win_rate",      "{:.1%}"),
        ("Profit factor", "profit_factor", "{:.2f}"),
        ("Max drawdown",  "max_dd",        "${:,.0f}"),
    ]:
        b = stats_bl.get(key, 0)
        d = stats_d2.get(key, 0)
        try:
            delta = d - b
            sign  = "+" if delta >= 0 else ""
            print(f"  {label:<20} {fmt.format(b):>12} "
                  f"{fmt.format(d):>12} "
                  f"{sign}{fmt.format(delta):>9}")
        except Exception:
            pass

    print()
    sharpe_d  = stats_d2.get("sharpe", 0)
    sharpe_b  = stats_bl.get("sharpe", 0)
    improvement = sharpe_d - sharpe_b
    target_gap  = 2.0 - sharpe_d

    if sharpe_d >= 2.0:
        print(f"  ✓ TARGET ACHIEVED: Sharpe {sharpe_d:.2f} ≥ 2.0")
    elif improvement > 0.1:
        print(f"  ~ Improved: Sharpe {sharpe_b:.2f} → {sharpe_d:.2f} "
              f"(+{improvement:.2f}), gap to target: {target_gap:.2f}")
    else:
        print(f"  ✗ No meaningful improvement ({improvement:+.2f}). "
              f"Diagnose signal quality.")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
