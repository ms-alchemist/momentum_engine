"""
signals/mrsi.py
===============
MRSI, MTSI, and VWMRSI indicators for MGC intraday mean-reversion.
Theory: Micaletti (2023)

Fix v2:
  - MTSI confirmation threshold loosened (was ±5, now ±1)
  - MTSI diagnostic added to understand actual value distribution
  - classify_signal now uses MRSI as primary, MTSI as directional confirm only
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/cache")
RESULTS_DIR = Path("backtest/results")
RESULTS_DIR.mkdir(exist_ok=True)

MRSI_M = 2
MRSI_N = 2
MTSI_M = 3
MTSI_N = 2
VWMA_K = 21

# Thresholds — MRSI is primary signal
FADE_STRONG_LO = 15
FADE_STRONG_HI = 85
FADE_WEAK_LO   = 25
FADE_WEAK_HI   = 75
FOLLOW_LO      = 35
FOLLOW_HI      = 65

# MTSI confirmation — just checks SIGN agreement, not magnitude
# MTSI > 0 means close above TWAP (stretched up → confirms fade short)
# MTSI < 0 means close below TWAP (stretched down → confirms fade long)
MTSI_CONFIRM_THRESHOLD = 0.0   # any positive/negative value confirms

FIRST_BAR_OPEN  = time(9, 0)
FIRST_BAR_CLOSE = time(9, 30)
SESSION_END     = time(16, 15)


def wilder_ma_series(values, period):
    n, out = len(values), np.full(len(values), np.nan)
    if n < period:
        return out
    out[period-1] = np.mean(values[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = out[i-1] + alpha * (values[i] - out[i-1])
    return out


def ema_series(values, period):
    a, out = 2.0/(period+1), np.zeros(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = a*values[i] + (1-a)*out[i-1]
    return out


def compute_mrsi(H, L, C, M=MRSI_M, N=MRSI_N):
    H = np.maximum(H, C)
    L = np.minimum(L, C)
    U  = wilder_ma_series(np.log(H/C), M)
    D  = wilder_ma_series(np.log(C/L), M)
    Un = np.power(np.maximum(U, 0), N)
    Dn = np.power(np.maximum(D, 0), N)
    den = Un + Dn
    return np.where(den > 1e-12, 100.0*Dn/den, 50.0)


def compute_mtsi(C, H, L, M=MTSI_M, N=MTSI_N):
    twap  = (H + L + C) / 3.0
    ratio = np.log(np.where(twap > 0, C/twap, 1.0))
    num   = ema_series(ema_series(ratio,         M), N)
    den   = ema_series(ema_series(np.abs(ratio), M), N)
    return np.where(den > 1e-12, 100.0*num/den, 0.0)


def classify_signal(mrsi_val, mtsi_val):
    """
    MRSI is primary. MTSI confirms direction only (sign check).
    This fixes the zero-signal problem from the overly strict MTSI threshold.
    """
    if np.isnan(mrsi_val):
        return "no_trade"

    mtsi_ok_short = np.isnan(mtsi_val) or mtsi_val >= MTSI_CONFIRM_THRESHOLD
    mtsi_ok_long  = np.isnan(mtsi_val) or mtsi_val <= MTSI_CONFIRM_THRESHOLD

    # Strong fade
    if mrsi_val < FADE_STRONG_LO and mtsi_ok_short:
        return "fade_short"
    if mrsi_val > FADE_STRONG_HI and mtsi_ok_long:
        return "fade_long"

    # Weak fade
    if mrsi_val < FADE_WEAK_LO and mtsi_ok_short:
        return "fade_short_weak"
    if mrsi_val > FADE_WEAK_HI and mtsi_ok_long:
        return "fade_long_weak"

    # Follow
    if FOLLOW_LO <= mrsi_val <= FOLLOW_HI:
        return "follow"

    return "no_trade"


def build_signal_series(bars):
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")

    fb = bars[
        (bars.index.time >= FIRST_BAR_OPEN) &
        (bars.index.time <  FIRST_BAR_CLOSE)
    ].groupby(bars[
        (bars.index.time >= FIRST_BAR_OPEN) &
        (bars.index.time <  FIRST_BAR_CLOSE)
    ].index.date).first()

    H = fb["high"].values
    L = fb["low"].values
    C = fb["close"].values
    V = fb["volume"].values

    mrsi   = compute_mrsi(H, L, C)
    mtsi   = compute_mtsi(C, H, L)
    bar_pos= (C - L) / np.where(H-L > 0, H-L, 1.0)

    result = pd.DataFrame({
        "open": fb["open"].values,
        "high": H, "low": L, "close": C, "volume": V,
        "mrsi": mrsi, "mtsi": mtsi, "bar_pos": bar_pos,
    }, index=pd.to_datetime(fb.index))

    result["signal"] = [
        classify_signal(r["mrsi"], r["mtsi"])
        for _, r in result.iterrows()
    ]
    return result


def main():
    print()
    print("MRSI/MTSI Signal Diagnostic — MGC First-Bar Analysis")
    print("=" * 62)

    bars    = pd.read_parquet(DATA_DIR / "MGC_30min.parquet")
    signals = build_signal_series(bars)
    valid   = signals.dropna(subset=["mrsi"])
    n       = len(valid)

    print(f"  First bars: {n}  "
          f"({valid.index[0].date()} to {valid.index[-1].date()})")
    print()

    # MTSI distribution — diagnostic
    print("  MTSI VALUE DISTRIBUTION (was this killing signals?)")
    print("  " + "-" * 45)
    mtsi_vals = valid["mtsi"].dropna()
    for lo, hi, label in [
        (-100, -10, "strongly below TWAP"),
        (-10,   -1, "slightly below TWAP"),
        (-1,     0, "just below TWAP"),
        (0,      1, "just above TWAP"),
        (1,     10, "slightly above TWAP"),
        (10,   100, "strongly above TWAP"),
    ]:
        c   = ((mtsi_vals >= lo) & (mtsi_vals < hi)).sum()
        pct = c/n*100
        print(f"  MTSI {lo:>5} to {hi:<5}  {label:<24}  "
              f"{c:>4}  {pct:>5.1f}%")
    print()

    # Signal distribution
    counts = valid["signal"].value_counts()
    total_fade = (counts.get("fade_short",0) + counts.get("fade_long",0) +
                  counts.get("fade_short_weak",0) + counts.get("fade_long_weak",0))
    print("  SIGNAL DISTRIBUTION")
    print("  " + "-" * 50)
    for sig in ["fade_short","fade_short_weak","fade_long","fade_long_weak",
                "follow","no_trade"]:
        c   = counts.get(sig, 0)
        pct = c/n*100
        bar = "█" * int(pct/2)
        print(f"  {sig:<20} {c:>4} days  {pct:>5.1f}%  {bar}")
    print(f"  {'TOTAL FADE':<20} {total_fade:>4} days  "
          f"{total_fade/n*100:>5.1f}%")
    print()

    # MRSI distribution
    print("  MRSI DISTRIBUTION")
    print("  " + "-" * 50)
    for lo, hi, label in [
        (0,  15, "fade_short STRONG"),
        (15, 25, "fade_short weak"),
        (25, 35, "no_trade zone"),
        (35, 65, "follow"),
        (65, 75, "no_trade zone"),
        (75, 85, "fade_long weak"),
        (85, 100,"fade_long STRONG"),
    ]:
        mask = (valid["mrsi"] >= lo) & (valid["mrsi"] < hi)
        c    = mask.sum()
        pct  = c/n*100
        print(f"  MRSI {lo:>3}-{hi:<3}  {label:<22}  {c:>4}  {pct:>5.1f}%")
    print()

    # Monthly breakdown
    print("  MONTHLY SIGNAL BREAKDOWN")
    print(f"  {'Month':<8} {'fade_s':>7} {'fade_l':>7} "
          f"{'follow':>7} {'no_trd':>7} {'total':>6}")
    print("  " + "-" * 50)
    valid2 = valid.copy()
    valid2["month"] = valid2.index.to_period("M")
    for m, g in valid2.groupby("month"):
        vc = g["signal"].value_counts()
        fs = vc.get("fade_short",0) + vc.get("fade_short_weak",0)
        fl = vc.get("fade_long", 0) + vc.get("fade_long_weak", 0)
        print(f"  {str(m):<8} {fs:>7} {fl:>7} "
              f"{vc.get('follow',0):>7} "
              f"{vc.get('no_trade',0):>7} {len(g):>6}")
    print()

    out = RESULTS_DIR / "mrsi_signals.csv"
    signals.to_csv(out)
    print(f"  Saved → {out}")
    print()
    print("  Next: python backtest/runner_mgc_mr.py")
    print()


if __name__ == "__main__":
    main()
