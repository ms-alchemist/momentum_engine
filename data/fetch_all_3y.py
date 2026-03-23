"""
data/fetch_all_3y.py
====================
Fetches 3-year history for all 6 instruments in one pass.
Overwrites any existing cache files.

Universe:
  MGC  — Micro Gold          $10/pt   safe-haven commodity
  MCL  — Micro Crude Oil     $100/pt  risk-on commodity
  MNQ  — Micro Nasdaq        $2/pt    growth equity
  MES  — Micro S&P 500       $5/pt    broad equity
  MYM  — Micro Dow           $0.50/pt value equity
  M2K  — Micro Russell 2000  $5/pt    small-cap equity

3 years (~750 daily bars) gives the Wasserstein HMM:
  - Enough history for reliable 4-5 state regime detection
  - Multiple full market cycles (2024 bull, 2025 volatility, Liberation Day)
  - Stable Wasserstein template tracking across regime transitions
"""

import os
import databento as db
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

API_KEY   = os.getenv("DATABENTO_API_KEY")
DATASET   = "GLBX.MDP3"
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(exist_ok=True)

END_DATE   = datetime(2026, 3, 22, tzinfo=timezone.utc).date()
START_DATE = (datetime(2026, 3, 22, tzinfo=timezone.utc) - timedelta(days=1095)).date()

SESSION_START = "09:00"
SESSION_END   = "16:30"

UNIVERSE = {
    "MGC.c.0": {"name": "MGC", "pv": 10.0,  "desc": "Micro Gold"},
    "MCL.c.0": {"name": "MCL", "pv": 100.0, "desc": "Micro Crude Oil"},
    "MNQ.c.0": {"name": "MNQ", "pv": 2.0,   "desc": "Micro Nasdaq"},
    "MES.c.0": {"name": "MES", "pv": 5.0,   "desc": "Micro S&P 500"},
    "MYM.c.0": {"name": "MYM", "pv": 0.5,   "desc": "Micro Dow"},
    "M2K.c.0": {"name": "M2K", "pv": 5.0,   "desc": "Micro Russell 2000"},
}


def fetch_and_cache(client, db_sym, info):
    clean = info["name"]
    pv    = info["pv"]
    print(f"Fetching {clean} ({info['desc']})...", end=" ", flush=True)

    data = client.timeseries.get_range(
        dataset=DATASET, symbols=[db_sym], stype_in="continuous",
        schema="ohlcv-1m", start=str(START_DATE), end=str(END_DATE),
    )
    df_raw = data.to_df()

    df_raw.index = pd.to_datetime(df_raw.index, utc=True).tz_convert("America/New_York")
    df = df_raw[["open","high","low","close","volume"]].copy().dropna()
    df = df.between_time(SESSION_START, SESSION_END)

    direction        = (df["close"]-df["open"]).apply(
                           lambda x: 1 if x>0 else (-1 if x<0 else 0))
    buy_frac         = direction.map({1:0.65, -1:0.35, 0:0.50})
    df["buy_volume"] = (df["volume"]*buy_frac).astype(int)
    df["sell_volume"]= df["volume"]-df["buy_volume"]

    # 1-min
    df.to_parquet(CACHE_DIR/f"{clean}_1min.parquet", compression="snappy")

    # 30-min
    agg = {"open":"first","high":"max","low":"min","close":"last",
           "volume":"sum","buy_volume":"sum","sell_volume":"sum"}
    df30 = df.resample("30min", label="left", closed="left").agg(agg)
    df30 = df30[df30["volume"]>0].dropna(subset=["open","close"])
    df30.to_parquet(CACHE_DIR/f"{clean}_30min.parquet", compression="snappy")

    # Daily
    agg_d = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    dfd   = df.resample("1D").agg(agg_d)
    dfd   = dfd[dfd["volume"]>0].dropna(subset=["open","close"])
    dfd["range"] = dfd["high"] - dfd["low"]
    dfd.to_parquet(CACHE_DIR/f"{clean}_daily.parquet", compression="snappy")

    avg_rng = dfd["range"].mean()
    print(f"{len(df_raw):,} bars → "
          f"{len(dfd)} daily  "
          f"avg_range={avg_rng:.1f}pt  "
          f"${avg_rng*pv:.0f}/contract")

    return len(dfd)


def main():
    print()
    print("3-Year Full Universe Fetch")
    print("=" * 60)
    print(f"  Start: {START_DATE}  End: {END_DATE}")
    print(f"  Instruments: {', '.join(v['name'] for v in UNIVERSE.values())}")
    print()

    client = db.Historical(API_KEY)
    total_days = {}

    for db_sym, info in UNIVERSE.items():
        try:
            n_days = fetch_and_cache(client, db_sym, info)
            total_days[info["name"]] = n_days
        except Exception as e:
            print(f"  ERROR on {info['name']}: {e}")

    print()
    print("=" * 60)
    print("  CACHE SUMMARY — Full Universe")
    print("=" * 60)
    print(f"  {'Sym':<5} {'Days':>6} {'PV':>8}  Description")
    print("  " + "-" * 45)
    for db_sym, info in UNIVERSE.items():
        sym  = info["name"]
        days = total_days.get(sym, 0)
        pv   = info["pv"]
        print(f"  {sym:<5} {days:>6}  ${pv:>6.1f}/pt  {info['desc']}")

    print()
    common = min(total_days.values()) if total_days else 0
    print(f"  Common trading days (approx): ~{common}")
    print(f"  HMM rolling window will be:   {common//2} days")
    print(f"  Regime history available:     ~{common//2} trading days")
    print()
    print("Next step: python signals/wasserstein_hmm.py")
    print("  (update INSTRUMENTS dict in wasserstein_hmm.py to include all 6)")
    print()


if __name__ == "__main__":
    main()
