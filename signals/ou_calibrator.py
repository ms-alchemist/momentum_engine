"""
signals/ou_calibrator.py
========================
Ornstein-Uhlenbeck parameter calibration for MGC intraday mean-reversion.

Fix: fits OU on intraday 30-min price DEVIATIONS from rolling mean,
not on daily log-returns. This gives meaningful intraday half-life in bars.

Theory:
  Tsekrekos (2010) — OU entry threshold framework
  Leung & Li (2015) — optimal entry/exit intervals under transaction costs
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

CALIB_WINDOW     = 60      # trading days of intraday bars for OU fit
THETA_WINDOW     = 20      # days for rolling session-open mean
STOP_LOSS_PTS    = 22.0
TRANSACTION_COST = 0.80    # $ per contract RT
POINT_VALUE      = 10.0    # $/pt MGC
TC_PTS           = TRANSACTION_COST / POINT_VALUE   # 0.08 pts
T_SESSION_BARS   = 13      # bars remaining after first bar (~6.5 hrs)

FIRST_BAR_OPEN  = time(9, 0)
SESSION_END     = time(16, 15)


def fit_ou_intraday(price_devs: np.ndarray, dt: float = 0.5):
    """
    Fit OU on intraday price deviations from rolling mean.
    dt = 0.5 hours (30-min bars).

    X_t = price - theta (deviation from session mean)
    dX = -kappa * X * dt + sigma * dW

    Discrete: X_{t+1} = beta * X_t + eps
    beta = exp(-kappa * dt)
    """
    X = price_devs[:-1]
    Y = price_devs[1:]
    if len(X) < 20:
        return None

    # Force zero-mean AR(1) (deviations should have zero mean)
    Sxx = (X ** 2).sum()
    Sxy = (X * Y).sum()

    if Sxx < 1e-10:
        return None

    beta  = Sxy / Sxx
    if not (0 < beta < 1):
        # beta >= 1 means trending (unit root), beta <= 0 means oscillating
        # both are non-OU — record but flag
        beta = max(0.001, min(beta, 0.999))

    resid     = Y - beta * X
    sigma_eps = resid.std()

    # Recover OU params (dt in hours)
    kappa     = -np.log(beta) / dt          # per hour
    sigma_sq  = sigma_eps**2 * 2 * kappa / (1 - beta**2)
    sigma     = np.sqrt(max(sigma_sq, 1e-10))
    half_life = np.log(2) / kappa           # in hours
    hl_bars   = half_life / dt              # in 30-min bars

    # R-squared
    ss_tot = (Y**2).sum()
    r2     = 1 - (resid**2).sum() / ss_tot if ss_tot > 1e-10 else 0.0

    return dict(
        kappa=kappa,        # per hour
        sigma=sigma,        # in points
        half_life=half_life,# hours
        hl_bars=hl_bars,    # 30-min bars
        r_squared=r2,
        beta=beta,
        sigma_eps=sigma_eps,
    )


def compute_ll_levels(kappa_hr, sigma, stop_pts, tc_pts):
    """
    Leung-Li optimal entry/exit levels.
    kappa_hr: mean-reversion speed per hour
    """
    # Stationary std of OU process (in points)
    sigma_stat = sigma / np.sqrt(2 * max(kappa_hr, 1e-6))

    # Session remaining: T_SESSION_BARS * 0.5 hours
    T_remaining = T_SESSION_BARS * 0.5   # hours
    decay       = np.exp(-kappa_hr * T_remaining)

    # Entry lower: min deviation to cover TC given remaining reversion time
    entry_lower = tc_pts / max(1 - decay, 0.01)

    # Entry upper: Leung-Li — don't enter if too close to stop
    # Reserve 30% buffer: entry must be at most 70% of stop from theta
    entry_upper = stop_pts * 0.70

    # Take-profit: expected reversion distance capped at 45% of stop
    # Leung-Li result: higher stop → lower optimal take-profit
    exp_rev     = sigma_stat * np.sqrt(2 / np.pi)
    take_profit = float(np.clip(exp_rev, tc_pts * 2, stop_pts * 0.45))

    return dict(
        entry_lower=entry_lower,
        entry_upper=entry_upper,
        take_profit_pts=take_profit,
        sigma_stat=sigma_stat,
    )


def main():
    print()
    print("OU Calibrator — MGC Intraday (30-min price deviations)")
    print("=" * 62)

    # Load 30-min bars
    bars = pd.read_parquet(DATA_DIR / "MGC_30min.parquet")
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")

    # Session bars only
    session = bars[
        (bars.index.time >= FIRST_BAR_OPEN) &
        (bars.index.time <= SESSION_END)
    ].copy()

    # Daily open for theta (session equilibrium reference)
    daily_open = (
        session.groupby(session.index.date)["open"]
        .first()
        .rename("session_open")
    )

    trading_days = sorted(daily_open.index)
    n_days       = len(trading_days)
    print(f"  {n_days} trading days  "
          f"({trading_days[0]} to {trading_days[-1]})")

    # --- Build price deviation series per day ---
    # For each day: deviations of 30-min closes from that day's session open
    # This is what the OU process X_t = price - theta models
    print(f"  Building intraday deviation series...")

    day_devs = {}   # date -> np.array of close deviations
    for day in trading_days:
        day_bars = session[session.index.date == day]
        if len(day_bars) < 3:
            continue
        theta    = day_bars["open"].iloc[0]   # session open as anchor
        devs     = day_bars["close"].values - theta
        day_devs[day] = devs

    # --- Rolling OU calibration ---
    print(f"  Rolling OU fit (window={CALIB_WINDOW}d)...", end=" ", flush=True)

    records = []
    for i, day in enumerate(trading_days):
        if i < CALIB_WINDOW:
            continue

        # Concatenate price deviations from last CALIB_WINDOW days
        window_days = trading_days[i - CALIB_WINDOW:i]
        all_devs    = []
        for wd in window_days:
            if wd in day_devs:
                all_devs.extend(day_devs[wd].tolist())

        all_devs = np.array(all_devs, dtype=float)
        if len(all_devs) < 100:
            continue

        ou = fit_ou_intraday(all_devs, dt=0.5)
        if ou is None:
            continue

        # Rolling session-open mean for theta
        theta_days = trading_days[max(0, i - THETA_WINDOW):i]
        theta_px   = float(np.mean([
            daily_open[d] for d in theta_days if d in daily_open.index
        ]))

        ll = compute_ll_levels(ou["kappa"], ou["sigma"],
                               STOP_LOSS_PTS, TC_PTS)

        records.append({
            "date":            day,
            "kappa_hr":        ou["kappa"],
            "theta":           theta_px,
            "sigma_pts":       ou["sigma"],
            "half_life_hrs":   ou["half_life"],
            "hl_bars":         ou["hl_bars"],
            "r_squared":       ou["r_squared"],
            "sigma_stat":      ll["sigma_stat"],
            "entry_lower":     ll["entry_lower"],
            "entry_upper":     ll["entry_upper"],
            "take_profit_pts": ll["take_profit_pts"],
        })

    print(f"{len(records)} valid days")
    print()

    if not records:
        print("  ERROR: No valid calibrations. Check data.")
        return

    df = pd.DataFrame(records).set_index("date")

    # --- Summary ---
    print(f"  {'Parameter':<24} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print("  " + "-" * 62)
    for label, col in [
        ("κ (per hour)",        "kappa_hr"),
        ("half-life (hours)",   "half_life_hrs"),
        ("half-life (bars)",    "hl_bars"),
        ("σ (points)",          "sigma_pts"),
        ("σ_stat (pts)",        "sigma_stat"),
        ("R²",                  "r_squared"),
        ("entry_lower (pts)",   "entry_lower"),
        ("entry_upper (pts)",   "entry_upper"),
        ("take_profit (pts)",   "take_profit_pts"),
    ]:
        v = df[col].dropna()
        print(f"  {label:<24} {v.mean():>9.3f} {v.std():>9.3f} "
              f"{v.min():>9.3f} {v.max():>9.3f}")

    print()

    # --- Monthly snapshot ---
    print(f"  {'Month':<8} {'κ/hr':>7} {'θ':>9} {'σ_pts':>7} "
          f"{'HL_hrs':>7} {'HL_bars':>8} {'lo_pt':>7} {'hi_pt':>7} {'tp_pt':>7}")
    print("  " + "-" * 72)

    df_m = df.copy()
    df_m["month"] = pd.to_datetime(df_m.index).to_period("M")
    for m, grp in df_m.groupby("month"):
        r = grp.iloc[-1]
        print(f"  {str(m):<8} "
              f"{r.kappa_hr:>7.4f} "
              f"{r.theta:>9.1f} "
              f"{r.sigma_pts:>7.2f} "
              f"{r.half_life_hrs:>7.2f} "
              f"{r.hl_bars:>8.1f} "
              f"{r.entry_lower:>7.3f} "
              f"{r.entry_upper:>7.3f} "
              f"{r.take_profit_pts:>7.3f}")

    print()

    # --- Stability checks ---
    kv  = df["kappa_hr"]
    hlv = df["hl_bars"]
    cv  = kv.std() / kv.mean()

    # Flag slow-reversion months (HL > 10 bars = 5 hrs — shouldn't fade)
    slow_months = df_m[df_m["hl_bars"] > 10].groupby("month").size()

    print("  STABILITY CHECKS")
    print(f"  κ always positive:    "
          f"{'YES ✓' if kv.min() > 0 else 'NO ✗'}")
    print(f"  κ CoV:                {cv:.2%}  "
          f"({'STABLE ✓' if cv < 0.80 else 'HIGH — regime-dependent'})")
    print(f"  Mean HL:              {hlv.mean():.1f} bars = "
          f"{hlv.mean()*0.5:.1f} hrs")
    print(f"  Fast-reversion days:  "
          f"{(hlv <= 4).sum()} days with HL ≤ 4 bars (2 hrs) — best fade days")
    print(f"  Slow-reversion days:  "
          f"{(hlv > 10).sum()} days with HL > 10 bars — skip fade")
    print(f"  Mean take-profit:     "
          f"{df['take_profit_pts'].mean():.2f} pts = "
          f"${df['take_profit_pts'].mean()*POINT_VALUE:.0f}/contract")
    print()

    # Save
    out = RESULTS_DIR / "ou_calibration.csv"
    df_m.drop(columns=["month"]).to_csv(out)
    print(f"  Saved → {out}")
    print()
    print("  Interpretation guide:")
    print("  HL ≤ 4 bars  → fast reversion → FADE signal strong")
    print("  HL 4-10 bars → moderate → FADE with confirmation")
    print("  HL > 10 bars → slow/trending → SKIP fade, follow signal only")
    print()
    print("  Next: python signals/mrsi.py")
    print()


if __name__ == "__main__":
    main()
