"""
data/fetch_mnq_mym.py
=====================
Fetches 12 months of MNQ and MYM 1-min bars from Databento,
aggregates to 30-min, and saves to data/cache/.
"""

import os
import databento as db
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY   = os.getenv("DATABENTO_API_KEY")
DATASET   = "GLBX.MDP3"
SYMBOLS   = ["MNQ.c.0", "MYM.c.0"]
STYPE_IN  = "continuous"

END_DATE   = datetime.now(tz=timezone.utc).date()
START_DATE = (datetime.now(tz=timezone.utc) - timedelta(days=365)).date()

CACHE_DIR  = Path("data/cache")
CACHE_DIR.mkdir(exist_ok=True)

SYMBOL_MAP = {"MNQ.c.0": "MNQ", "MYM.c.0": "MYM"}

SESSION_START = "09:00"
SESSION_END   = "16:30"


def fetch_and_cache():
    client = db.Historical(API_KEY)

    print(f"Fetching MNQ + MYM 1-min bars: {START_DATE} to {END_DATE}")
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=SYMBOLS,
        stype_in=STYPE_IN,
        schema="ohlcv-1m",
        start=str(START_DATE),
        end=str(END_DATE),
    )

    df_raw = data.to_df()
    print(f"Retrieved {len(df_raw):,} raw 1-min bars")
    print()

    for db_symbol, clean_symbol in SYMBOL_MAP.items():
        if "symbol" in df_raw.columns:
            df = df_raw[df_raw["symbol"] == db_symbol].copy()
        else:
            df = df_raw.copy()

        # Timezone
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")

        # Standard columns
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Session filter
        df = df.between_time(SESSION_START, SESSION_END)

        # Buy/sell volume approximation
        direction = (df["close"] - df["open"]).apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        buy_frac = direction.map({1: 0.65, -1: 0.35, 0: 0.50})
        df["buy_volume"]  = (df["volume"] * buy_frac).astype(int)
        df["sell_volume"] = df["volume"] - df["buy_volume"]

        # Save 1-min cache
        path_1min = CACHE_DIR / f"{clean_symbol}_1min.parquet"
        df.to_parquet(path_1min, compression="snappy")
        size_kb = path_1min.stat().st_size / 1024
        print(f"{clean_symbol} 1-min: {len(df):,} bars saved ({size_kb:.0f} KB)")

        # Aggregate to 30-min
        agg = {
            "open":         "first",
            "high":         "max",
            "low":          "min",
            "close":        "last",
            "volume":       "sum",
            "buy_volume":   "sum",
            "sell_volume":  "sum",
        }
        df_30 = df.resample("30min", label="left", closed="left").agg(agg)
        df_30 = df_30[df_30["volume"] > 0].dropna(subset=["open", "close"])

        path_30 = CACHE_DIR / f"{clean_symbol}_30min.parquet"
        df_30.to_parquet(path_30, compression="snappy")
        size_kb = path_30.stat().st_size / 1024
        print(f"{clean_symbol} 30-min: {len(df_30):,} bars saved ({size_kb:.0f} KB)")

        print(f"  Date range:  {df_30.index[0].date()} to {df_30.index[-1].date()}")
        print(f"  Price range: {df_30['low'].min():.1f} to {df_30['high'].max():.1f}")
        daily_range = df_30.resample("1D").agg({"high": "max", "low": "min"}).dropna()
        avg_range = (daily_range["high"] - daily_range["low"]).mean()
        print(f"  Avg daily range: {avg_range:.1f} points")
        print()


if __name__ == "__main__":
    print("=" * 55)
    print("  Databento Fetcher — MNQ + MYM")
    print(f"  {START_DATE} to {END_DATE}")
    print("=" * 55)
    print()
    fetch_and_cache()
    print("Done. Next step: python backtest/signal_audit_mnq_mym.py")
