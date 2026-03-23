"""
data/fetch_universe.py
======================
Fetches 2-year history for MCL, MYM, M2K to complete the cross-asset
universe for the Wasserstein HMM.

Full universe after this fetch:
  MGC  — Micro Gold          (commodity, safe-haven)
  MCL  — Micro Crude Oil     (commodity, risk-on)
  MNQ  — Micro Nasdaq        (equity, growth)
  MES  — Micro S&P 500       (equity, broad)
  MYM  — Micro Dow           (equity, value)
  M2K  — Micro Russell 2000  (equity, small-cap)

This diversification gives the HMM genuine cross-asset variation:
  - Gold vs crude oil: sometimes inversely correlated
  - Growth (NQ) vs value (Dow) vs small-cap (Russell): diverge in regime shifts
  - Commodities vs equities: decorrelated at monthly timescale
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

END_DATE   = datetime(2026, 3, 22, tzinfo=timezone.utc).date()
START_DATE = (datetime(2026, 3, 22, tzinfo=timezone.utc) - timedelta(days=730)).date()

SESSION_START = "09:00"
SESSION_END   = "16:30"

SYMBOLS = {
    "MCL.c.0": {"name": "MCL", "pv": 100.0, "desc": "Micro Crude Oil"},
    "MYM.c.0": {"name": "MYM", "pv":   0.5, "desc": "Micro Dow"},
    "M2K.c.0": {"name": "M2K", "pv":   5.0, "desc": "Micro Russell 2000"},
}


def fetch_and_cache(client, db_sym, info):
    clean = info["name"]
    print(f"Fetching {clean} ({info['desc']}) {START_DATE} to {END_DATE}...")

    data = client.timeseries.get_range(
        dataset=DATASET, symbols=[db_sym], stype_in="continuous",
        schema="ohlcv-1m", start=str(START_DATE), end=str(END_DATE),
    )
    df_raw = data.to_df()
    print(f"  {len(df_raw):,} raw 1-min bars")

    df_raw.index = pd.to_datetime(df_raw.index, utc=True).tz_convert("America/New_York")
    df = df_raw[["open","high","low","close","volume"]].copy().dropna()
    df = df.between_time(SESSION_START, SESSION_END)

    direction        = (df["close"]-df["open"]).apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    buy_frac         = direction.map({1:0.65, -1:0.35, 0:0.50})
    df["buy_volume"] = (df["volume"]*buy_frac).astype(int)
    df["sell_volume"]= df["volume"]-df["buy_volume"]

    # 1-min
    df.to_parquet(CACHE_DIR/f"{clean}_1min.parquet", compression="snappy")

    # 30-min
    agg = {"open":"first","high":"max","low":"min","close":"last",
           "volume":"sum","buy_volume":"sum","sell_volume":"sum"}
    df30 = df.resample("30min",label="left",closed="left").agg(agg)
    df30 = df30[df30["volume"]>0].dropna(subset=["open","close"])
    df30.to_parquet(CACHE_DIR/f"{clean}_30min.parquet", compression="snappy")

    # Daily
    agg_d = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    dfd   = df.resample("1D").agg(agg_d)
    dfd   = dfd[dfd["volume"]>0].dropna(subset=["open","close"])
    dfd["range"] = dfd["high"]-dfd["low"]
    dfd.to_parquet(CACHE_DIR/f"{clean}_daily.parquet", compression="snappy")

    pv = info["pv"]
    print(f"  30-min: {len(df30):,} bars  daily: {len(dfd)} bars  "
          f"range: {df30.index[0].date()} to {df30.index[-1].date()}")
    print(f"  Avg daily range: {dfd['range'].mean():.1f} pts  "
          f"(${dfd['range'].mean()*pv:.0f}/contract)")
    print()


def main():
    print("\nFetching cross-asset universe — MCL, MYM, M2K")
    print("=" * 50)
    print(f"Date range: {START_DATE} to {END_DATE}\n")

    client = db.Historical(API_KEY)

    for db_sym, info in SYMBOLS.items():
        fetch_and_cache(client, db_sym, info)

    # Summary of full universe
    universe = {
        "MGC": 10.0, "MCL": 100.0,
        "MNQ": 2.0,  "MES": 5.0,
        "MYM": 0.5,  "M2K": 5.0,
    }

    print("=" * 60)
    print("  COMPLETE UNIVERSE CACHE SUMMARY")
    print("=" * 60)
    for sym, pv in universe.items():
        p30    = CACHE_DIR/f"{sym}_30min.parquet"
        p_daily= CACHE_DIR/f"{sym}_daily.parquet"
        if p30.exists() and p_daily.exists():
            df30 = pd.read_parquet(p30)
            dfd  = pd.read_parquet(p_daily)
            dfd.index = pd.to_datetime(dfd.index)
            avg_rng = dfd["range"].mean()
            print(f"  {sym:<5}  daily={len(dfd):>3}  "
                  f"range={avg_rng:>7.1f}pt  "
                  f"${avg_rng*pv:>7.0f}/contract  "
                  f"({dfd.index[0].date()} to {dfd.index[-1].date()})")
        else:
            print(f"  {sym:<5}  NOT IN CACHE")

    print()
    print("Next: python signals/wasserstein_hmm.py")
    print("      (update INSTRUMENTS in wasserstein_hmm.py to include MCL, MYM, M2K)")


if __name__ == "__main__":
    main()
