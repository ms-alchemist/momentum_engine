"""
backtest/diagnose_mr.py
=======================
Diagnostic: test the fade signal in pure isolation.
No follow trades. No MRSI filtering of follow.
Just: does fading the first bar work when MRSI says extreme?

Three questions:
1. When MRSI < 15 (close near HIGH), does price revert to theta by 12:30 PM?
2. When MRSI > 85 (close near LOW), does price revert to theta by 12:30 PM?
3. What is the win rate / avg P&L purely from the fade signal?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time
import warnings
warnings.filterwarnings("ignore")

DATA_DIR    = Path("data/cache")
RESULTS_DIR = Path("backtest/results")

FADE_STOP_PTS  = 15.0
PV             = 10.0
COMM           = 0.80
CONTRACTS      = 2
THETA_WINDOW   = 20

FIRST_BAR_OPEN  = time(9,  0)
FIRST_BAR_CLOSE = time(9, 30)
TIME_STOP_TIME  = time(12, 30)
SESSION_CLOSE   = time(16, 15)

MRSI_M = 2
MRSI_N = 2


def wilder_ma(values, period):
    n, out = len(values), np.full(len(values), np.nan)
    if n < period: return out
    out[period-1] = np.mean(values[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = out[i-1] + alpha*(values[i]-out[i-1])
    return out


def compute_mrsi(H, L, C, M=MRSI_M, N=MRSI_N):
    H = np.maximum(H, C); L = np.minimum(L, C)
    U  = wilder_ma(np.log(H/C), M)
    D  = wilder_ma(np.log(C/L), M)
    Un = np.power(np.maximum(U,0), N)
    Dn = np.power(np.maximum(D,0), N)
    den = Un + Dn
    return np.where(den > 1e-12, 100.0*Dn/den, 50.0)


def main():
    print()
    print("Fade Signal Isolation Diagnostic")
    print("=" * 62)

    bars = pd.read_parquet(DATA_DIR / "MGC_30min.parquet")
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")

    # First bars
    fb_mask = ((bars.index.time >= FIRST_BAR_OPEN) &
               (bars.index.time <  FIRST_BAR_CLOSE))
    fb = bars[fb_mask].groupby(bars[fb_mask].index.date).first()

    H_all = fb["high"].values
    L_all = fb["low"].values
    C_all = fb["close"].values
    O_all = fb["open"].values
    dates = list(fb.index)

    mrsi_all = compute_mrsi(H_all, L_all, C_all)

    # Rolling theta
    session_opens = pd.Series(O_all, index=pd.to_datetime(dates))
    theta_series  = session_opens.rolling(THETA_WINDOW).mean()

    # Build daily vol filter
    daily = bars.resample("1D").agg({"high":"max","low":"min","volume":"sum"})
    daily = daily[daily["volume"]>0].copy()
    daily["range"] = daily["high"] - daily["low"]
    daily["r5"]    = daily["range"].rolling(5).mean()
    daily["r20"]   = daily["range"].rolling(20).mean()
    daily["vr"]    = daily["r5"] / daily["r20"]

    # Simulate pure fade trades across all MRSI thresholds
    results = []

    for thresh_lo, thresh_hi, label in [
        (0,  10, "MRSI 0-10  (extreme)"),
        (10, 20, "MRSI 10-20 (strong)"),
        (20, 30, "MRSI 20-30 (moderate)"),
        (80, 90, "MRSI 80-90 (moderate)"),
        (90,100, "MRSI 90-100 (extreme)"),
        (0,  20, "MRSI 0-20  (strong fade)"),
        (80,100, "MRSI 80-100 (strong fade)"),
        (0,  15, "MRSI 0-15  (Micaletti threshold)"),
        (85,100, "MRSI 85-100 (Micaletti threshold)"),
    ]:
        trades = []
        for i, day in enumerate(dates):
            if i < THETA_WINDOW:
                continue
            mrsi_val = mrsi_all[i]
            if np.isnan(mrsi_val):
                continue

            # Check if this day falls in our MRSI band
            fade_short = (mrsi_val >= thresh_lo and mrsi_val < thresh_hi
                          and thresh_lo < 50)   # short side
            fade_long  = (mrsi_val >= thresh_lo and mrsi_val < thresh_hi
                          and thresh_lo >= 50)   # long side

            if not (fade_short or fade_long):
                continue

            # Vol filter
            dmask = daily.index.date == day
            if not dmask.any(): continue
            di = daily[dmask].iloc[0]
            if not np.isnan(di["vr"]) and di["vr"] > 2.5: continue

            # First bar direction
            o, c = O_all[i], C_all[i]
            if c > o:   fb_dir = 1
            elif c < o: fb_dir = -1
            else: continue

            # Theta
            ts = pd.Timestamp(day)
            if ts not in theta_series.index: continue
            theta = float(theta_series.loc[ts])
            if np.isnan(theta): continue

            # Entry price = first bar close
            entry = c
            fade_dir = -fb_dir   # we fade

            # Stop price
            stop = entry - fade_dir * FADE_STOP_PTS

            # Simulate
            day_bars  = bars[bars.index.date == day]
            post      = day_bars[day_bars.index.time >= FIRST_BAR_CLOSE]
            exit_p, exit_r = None, None

            for ts_bar, bar in post.iterrows():
                bt = ts_bar.time()
                if bt >= TIME_STOP_TIME and exit_p is None:
                    exit_p, exit_r = bar["open"], "time_stop"
                    break
                if bt >= SESSION_CLOSE:
                    exit_p, exit_r = bar["open"], "session_close"
                    break
                # Catastrophic stop
                if fade_dir == -1 and bar["high"] >= stop:
                    exit_p, exit_r = stop, "cstop"
                    break
                if fade_dir == 1 and bar["low"] <= stop:
                    exit_p, exit_r = stop, "cstop"
                    break
                # Target: theta
                if fade_dir == -1 and bar["low"] <= theta:
                    exit_p, exit_r = theta, "target"
                    break
                if fade_dir == 1 and bar["high"] >= theta:
                    exit_p, exit_r = theta, "target"
                    break

            if exit_p is None:
                exit_p = post.iloc[-1]["close"] if len(post) > 0 else entry
                exit_r = "eod"

            pnl = fade_dir*(exit_p-entry)*PV*CONTRACTS - COMM*CONTRACTS
            trades.append({
                "pnl": pnl, "win": pnl > 0,
                "exit_reason": exit_r,
                "mrsi": mrsi_val,
                "entry_deviation": abs(entry - theta),
            })

        if not trades:
            results.append((label, 0, 0, 0, 0, {}))
            continue

        t   = pd.DataFrame(trades)
        wr  = t["win"].mean()
        avg = t["net_pnl"].mean() if "net_pnl" in t else t["pnl"].mean()
        tot = t["pnl"].sum()
        n   = len(t)
        er  = t["exit_reason"].value_counts().to_dict()
        results.append((label, n, wr, avg, tot, er))

    # Print results
    print(f"  {'MRSI Band':<35} {'N':>5} {'WR':>7} {'Avg$':>8} "
          f"{'Total$':>9} {'target%':>8} {'cstop%':>7}")
    print("  " + "-" * 82)

    for label, n, wr, avg, tot, er in results:
        if n == 0:
            print(f"  {label:<35} {'0':>5}")
            continue
        t_pct  = er.get("target",    0) / n * 100
        cs_pct = er.get("cstop",     0) / n * 100
        ts_pct = er.get("time_stop", 0) / n * 100
        print(f"  {label:<35} {n:>5}  {wr:>6.1%}  "
              f"${avg:>6.0f}  ${tot:>7,.0f}  "
              f"{t_pct:>7.1f}%  {cs_pct:>6.1f}%")

    print()

    # Also test: pure unconditional fade (fade every extreme first bar
    # regardless of MRSI — just bar_position based)
    print("  BAR POSITION FADE TEST (no MRSI filter)")
    print(f"  {'Bar Position':<35} {'N':>5} {'WR':>7} {'Avg$':>8} "
          f"{'Total$':>9} {'target%':>8}")
    print("  " + "-" * 75)

    for pos_lo, pos_hi, label in [
        (0.0, 0.1, "bar_pos 0.0-0.1 (extreme low)"),
        (0.0, 0.2, "bar_pos 0.0-0.2 (near low)"),
        (0.8, 1.0, "bar_pos 0.8-1.0 (near high)"),
        (0.9, 1.0, "bar_pos 0.9-1.0 (extreme high)"),
    ]:
        bar_pos_all = (C_all - L_all) / np.where(H_all-L_all > 0, H_all-L_all, 1.0)
        trades = []
        for i, day in enumerate(dates):
            if i < THETA_WINDOW: continue
            bp = bar_pos_all[i]
            if not (pos_lo <= bp <= pos_hi): continue

            dmask = daily.index.date == day
            if not dmask.any(): continue
            di = daily[dmask].iloc[0]
            if not np.isnan(di["vr"]) and di["vr"] > 2.5: continue

            o, c = O_all[i], C_all[i]
            if c > o:   fb_dir = 1
            elif c < o: fb_dir = -1
            else: continue

            ts = pd.Timestamp(day)
            if ts not in theta_series.index: continue
            theta = float(theta_series.loc[ts])
            if np.isnan(theta): continue

            fade_dir = -fb_dir
            stop     = c - fade_dir * FADE_STOP_PTS
            entry    = c

            day_bars = bars[bars.index.date == day]
            post     = day_bars[day_bars.index.time >= FIRST_BAR_CLOSE]
            exit_p, exit_r = None, None

            for ts_bar, bar in post.iterrows():
                bt = ts_bar.time()
                if bt >= TIME_STOP_TIME and exit_p is None:
                    exit_p, exit_r = bar["open"], "time_stop"
                    break
                if bt >= SESSION_CLOSE:
                    exit_p, exit_r = bar["open"], "session_close"
                    break
                if fade_dir == -1 and bar["high"] >= stop:
                    exit_p, exit_r = stop, "cstop"
                    break
                if fade_dir == 1 and bar["low"] <= stop:
                    exit_p, exit_r = stop, "cstop"
                    break
                if fade_dir == -1 and bar["low"] <= theta:
                    exit_p, exit_r = theta, "target"
                    break
                if fade_dir == 1 and bar["high"] >= theta:
                    exit_p, exit_r = theta, "target"
                    break

            if exit_p is None:
                exit_p = post.iloc[-1]["close"] if len(post) > 0 else entry
                exit_r = "eod"

            pnl = fade_dir*(exit_p-entry)*PV*CONTRACTS - COMM*CONTRACTS
            trades.append({"pnl":pnl,"win":pnl>0,"exit_reason":exit_r})

        if not trades:
            print(f"  {label:<35}  0 trades")
            continue
        t   = pd.DataFrame(trades)
        wr  = t["win"].mean()
        avg = t["pnl"].mean()
        tot = t["pnl"].sum()
        n   = len(t)
        t_pct = (t["exit_reason"]=="target").sum()/n*100
        print(f"  {label:<35} {n:>5}  {wr:>6.1%}  "
              f"${avg:>6.0f}  ${tot:>7,.0f}  {t_pct:>7.1f}%")

    print()
    print("  Key question: does target% > 50% for any band?")
    print("  If yes → fade edge exists, refine parameters")
    print("  If no  → fade doesn't work on MGC first bar → abandon Direction 2")
    print()


if __name__ == "__main__":
    main()
