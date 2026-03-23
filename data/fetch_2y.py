"""
data/fetch_2y.py
================
Fetches 2-year history for MNQ and MES.
Free on our Databento plan. Needed for Wasserstein HMM calibration —
more history = better regime estimation.

Overwrites existing 1-year cache files.
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
STYPE_IN  = "continuous"
CACHE_DIR = Path("data/cache")

END_DATE   = datetime.now(tz=timezone.utc).date()
START_DATE = (datetime.now(tz=timezone.utc) - timedelta(days=730)).date()

SESSION_START = "09:00"
SESSION_END   = "16:30"

SYMBOLS = {"MNQ.c.0": "MNQ", "MES.c.0": "MES"}


def fetch_and_cache(client, db_sym, clean_sym):
    print(f"Fetching {clean_sym} 2-year history ({START_DATE} to {END_DATE})...")
    data = client.timeseries.get_range(
        dataset=DATASET, symbols=[db_sym], stype_in=STYPE_IN,
        schema="ohlcv-1m", start=str(START_DATE), end=str(END_DATE),
    )
    df_raw = data.to_df()
    print(f"  {len(df_raw):,} raw 1-min bars")

    df_raw.index = pd.to_datetime(df_raw.index, utc=True).tz_convert("America/New_York")
    df = df_raw[["open","high","low","close","volume"]].copy().dropna()
    df = df.between_time(SESSION_START, SESSION_END)

    direction        = (df["close"] - df["open"]).apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    buy_frac         = direction.map({1:0.65, -1:0.35, 0:0.50})
    df["buy_volume"] = (df["volume"] * buy_frac).astype(int)
    df["sell_volume"]= df["volume"] - df["buy_volume"]

    # 1-min
    p1 = CACHE_DIR / f"{clean_sym}_1min.parquet"
    df.to_parquet(p1, compression="snappy")

    # 30-min
    agg = {"open":"first","high":"max","low":"min","close":"last",
           "volume":"sum","buy_volume":"sum","sell_volume":"sum"}
    df30 = df.resample("30min",label="left",closed="left").agg(agg)
    df30 = df30[df30["volume"]>0].dropna(subset=["open","close"])
    p30  = CACHE_DIR / f"{clean_sym}_30min.parquet"
    df30.to_parquet(p30, compression="snappy")

    # Daily
    agg_d = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    dfd   = df.resample("1D").agg(agg_d)
    dfd   = dfd[dfd["volume"]>0].dropna(subset=["open","close"])
    dfd["range"] = dfd["high"] - dfd["low"]
    pd_   = CACHE_DIR / f"{clean_sym}_daily.parquet"
    dfd.to_parquet(pd_, compression="snappy")

    print(f"  30-min: {len(df30):,} bars  daily: {len(dfd)} bars  "
          f"range: {df30.index[0].date()} to {df30.index[-1].date()}")
    print(f"  Avg daily range: {dfd['range'].mean():.1f} pts")
    print()


def main():
    print("\n2-Year Data Fetch — MNQ + MES\n" + "="*40)
    client = db.Historical(API_KEY)
    for db_sym, clean_sym in SYMBOLS.items():
        fetch_and_cache(client, db_sym, clean_sym)
    print("Done. Daily bars ready for Wasserstein HMM calibration.")


if __name__ == "__main__":
    main()
