"""
data/fetch_mgc_2y.py
====================
Extends MGC cache to 2 years for better Wasserstein HMM calibration.
With 515 common days (matching MNQ/MES), the 252-day rolling window
gives reliable 4-state regime detection.
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


def main():
    print(f"\nFetching MGC 2-year history ({START_DATE} to {END_DATE})...")

    client = db.Historical(API_KEY)

    # Cost check first
    cost = client.metadata.get_cost(
        dataset=DATASET, symbols=["MGC.c.0"], stype_in="continuous",
        schema="ohlcv-1m", start=str(START_DATE), end=str(END_DATE),
    )
    print(f"Cost: ${cost:.4f}")

    # Fetch
    data = client.timeseries.get_range(
        dataset=DATASET, symbols=["MGC.c.0"], stype_in="continuous",
        schema="ohlcv-1m", start=str(START_DATE), end=str(END_DATE),
    )
    df_raw = data.to_df()
    print(f"Retrieved {len(df_raw):,} raw 1-min bars")

    df_raw.index = pd.to_datetime(df_raw.index, utc=True).tz_convert("America/New_York")
    df = df_raw[["open","high","low","close","volume"]].copy().dropna()
    df = df.between_time(SESSION_START, SESSION_END)

    direction        = (df["close"]-df["open"]).apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    buy_frac         = direction.map({1:0.65,-1:0.35,0:0.50})
    df["buy_volume"] = (df["volume"]*buy_frac).astype(int)
    df["sell_volume"]= df["volume"]-df["buy_volume"]

    # 1-min
    df.to_parquet(CACHE_DIR/"MGC_1min.parquet", compression="snappy")

    # 30-min
    agg = {"open":"first","high":"max","low":"min","close":"last",
           "volume":"sum","buy_volume":"sum","sell_volume":"sum"}
    df30 = df.resample("30min",label="left",closed="left").agg(agg)
    df30 = df30[df30["volume"]>0].dropna(subset=["open","close"])
    df30.to_parquet(CACHE_DIR/"MGC_30min.parquet", compression="snappy")

    # Daily
    agg_d = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    dfd   = df.resample("1D").agg(agg_d)
    dfd   = dfd[dfd["volume"]>0].dropna(subset=["open","close"])
    dfd["range"] = dfd["high"]-dfd["low"]
    dfd.to_parquet(CACHE_DIR/"MGC_daily.parquet", compression="snappy")

    print(f"MGC 30-min: {len(df30):,} bars  daily: {len(dfd)} bars")
    print(f"Range: {df30.index[0].date()} to {df30.index[-1].date()}")
    print(f"Avg daily range: {dfd['range'].mean():.1f} pts")
    print(f"\nNow run: python signals/wasserstein_hmm.py")


if __name__ == "__main__":
    main()
