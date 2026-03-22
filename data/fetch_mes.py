"""
data/fetch_mes.py
=================
Fetches MES (Micro E-mini S&P 500) 1-min bars from Databento.
Also checks cost for extended 2-year history on MNQ for better HMM calibration.

MES specs:
  - Symbol: MES.c.0 (continuous front-month)
  - Multiplier: $5/point
  - Commission: ~$0.35 RT
  - Avg daily range: ~60-80 pts = $300-400/contract
  - Most liquid micro equity index (tighter spreads than MNQ)

Usage:
    python data/fetch_mes.py
"""

import os
import databento as db
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("DATABENTO_API_KEY")
DATASET  = "GLBX.MDP3"
STYPE_IN = "continuous"
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(exist_ok=True)

# Date ranges
END_DATE      = datetime.now(tz=timezone.utc).date()
START_1Y      = (datetime.now(tz=timezone.utc) - timedelta(days=365)).date()
START_2Y      = (datetime.now(tz=timezone.utc) - timedelta(days=730)).date()

SESSION_START = "09:00"
SESSION_END   = "16:30"

SYMBOLS = {
    "MES.c.0": "MES",   # Micro E-mini S&P 500 — primary new fetch
}


def check_costs(client):
    print("=" * 55)
    print("  COST ESTIMATES")
    print("=" * 55)

    checks = [
        ("MES 1-min 1 year",  ["MES.c.0"], str(START_1Y), str(END_DATE)),
        ("MES 1-min 2 years", ["MES.c.0"], str(START_2Y), str(END_DATE)),
        ("MNQ 1-min 2 years", ["MNQ.c.0"], str(START_2Y), str(END_DATE)),
    ]

    costs = {}
    for label, symbols, start, end in checks:
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols=symbols,
                stype_in=STYPE_IN,
                schema="ohlcv-1m",
                start=start,
                end=end,
            )
            print(f"  {label:<30} ${cost:.4f}")
            costs[label] = cost
        except Exception as e:
            print(f"  {label:<30} ERROR: {str(e)[:50]}")
            costs[label] = None

    print()
    return costs


def fetch_symbol(client, db_symbol, clean_symbol, start_date):
    print(f"Fetching {clean_symbol} ({db_symbol}) from {start_date}...")

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[db_symbol],
        stype_in=STYPE_IN,
        schema="ohlcv-1m",
        start=str(start_date),
        end=str(END_DATE),
    )

    df_raw = data.to_df()
    print(f"  Retrieved {len(df_raw):,} raw 1-min bars")

    # Timezone
    df_raw.index = pd.to_datetime(df_raw.index, utc=True).tz_convert("America/New_York")

    # Standard columns
    df = df_raw[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Session filter
    df = df.between_time(SESSION_START, SESSION_END)

    # Buy/sell volume approximation
    direction = (df["close"] - df["open"]).apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    buy_frac          = direction.map({1: 0.65, -1: 0.35, 0: 0.50})
    df["buy_volume"]  = (df["volume"] * buy_frac).astype(int)
    df["sell_volume"] = df["volume"] - df["buy_volume"]

    # Save 1-min
    path_1min = CACHE_DIR / f"{clean_symbol}_1min.parquet"
    df.to_parquet(path_1min, compression="snappy")
    print(f"  {clean_symbol} 1-min: {len(df):,} bars → {path_1min.name} "
          f"({path_1min.stat().st_size/1024:.0f} KB)")

    # Aggregate to 30-min
    agg = {
        "open":        "first",
        "high":        "max",
        "low":         "min",
        "close":       "last",
        "volume":      "sum",
        "buy_volume":  "sum",
        "sell_volume": "sum",
    }
    df_30 = df.resample("30min", label="left", closed="left").agg(agg)
    df_30 = df_30[df_30["volume"] > 0].dropna(subset=["open", "close"])

    path_30 = CACHE_DIR / f"{clean_symbol}_30min.parquet"
    df_30.to_parquet(path_30, compression="snappy")
    print(f"  {clean_symbol} 30-min: {len(df_30):,} bars → {path_30.name} "
          f"({path_30.stat().st_size/1024:.0f} KB)")

    # Daily bars for HMM calibration
    agg_daily = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    df_daily = df.resample("1D", label="left", closed="left").agg(agg_daily)
    df_daily = df_daily[df_daily["volume"] > 0].dropna(subset=["open", "close"])
    df_daily["range"] = df_daily["high"] - df_daily["low"]

    path_daily = CACHE_DIR / f"{clean_symbol}_daily.parquet"
    df_daily.to_parquet(path_daily, compression="snappy")
    print(f"  {clean_symbol} daily:  {len(df_daily):,} bars → {path_daily.name}")

    # Stats
    print(f"  Date range:      {df_30.index[0].date()} to {df_30.index[-1].date()}")
    print(f"  Price range:     {df_30['low'].min():.1f} to {df_30['high'].max():.1f}")
    avg_range = df_daily["range"].mean()
    print(f"  Avg daily range: {avg_range:.1f} pts  "
          f"(${avg_range * 5:.0f}/contract at $5/pt)")
    print()

    return df_30, df_daily


def also_build_daily_for_existing(symbol):
    """Build daily bars for MNQ/MYM from existing 1-min cache."""
    path_1min = CACHE_DIR / f"{symbol}_1min.parquet"
    if not path_1min.exists():
        print(f"  {symbol} 1-min not found — skipping daily build")
        return

    path_daily = CACHE_DIR / f"{symbol}_daily.parquet"
    if path_daily.exists():
        print(f"  {symbol} daily already exists — skipping")
        return

    print(f"  Building {symbol} daily bars from existing 1-min cache...")
    df = pd.read_parquet(path_1min)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")

    df = df.between_time(SESSION_START, SESSION_END)

    agg_daily = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    df_daily = df.resample("1D").agg(agg_daily)
    df_daily = df_daily[df_daily["volume"] > 0].dropna(subset=["open", "close"])
    df_daily["range"] = df_daily["high"] - df_daily["low"]
    df_daily.to_parquet(path_daily, compression="snappy")

    avg_range = df_daily["range"].mean()
    print(f"  {symbol} daily: {len(df_daily)} bars  "
          f"avg range={avg_range:.1f}pts  "
          f"${avg_range * (2 if symbol == 'MNQ' else 0.5):.0f}/contract")


def main():
    print()
    print("Session 4 Data Fetch — MES + Daily Bars for HMM")
    print("=" * 55)
    print(f"  End date:   {END_DATE}")
    print(f"  1-year start: {START_1Y}")
    print(f"  2-year start: {START_2Y}")
    print()

    client = db.Historical(API_KEY)

    # Step 1: cost check
    costs = check_costs(client)

    # Step 2: build daily bars for existing MNQ/MYM from cache
    print("Building daily bars for existing instruments...")
    for sym in ["MNQ", "MYM", "MGC"]:
        also_build_daily_for_existing(sym)
    print()

    # Step 3: fetch MES (1 year)
    print("=" * 55)
    print("  FETCHING MES")
    print("=" * 55)
    print()

    import warnings
    warnings.filterwarnings("ignore")

    try:
        fetch_symbol(client, "MES.c.0", "MES", START_1Y)
    except Exception as e:
        print(f"  ERROR fetching MES: {e}")
        return

    # Step 4: summary
    print("=" * 55)
    print("  CACHE SUMMARY")
    print("=" * 55)
    instruments = {
        "MGC": 10.0, "MES": 5.0, "MNQ": 2.0, "MYM": 0.5
    }
    for sym, pv in instruments.items():
        path_30    = CACHE_DIR / f"{sym}_30min.parquet"
        path_daily = CACHE_DIR / f"{sym}_daily.parquet"
        if path_30.exists():
            df = pd.read_parquet(path_30)
            df.index = pd.to_datetime(df.index)
            n_days   = len(set(df.index.date))
            daily_df = pd.read_parquet(path_daily) if path_daily.exists() else None
            avg_rng  = daily_df["range"].mean() if daily_df is not None else 0
            print(f"  {sym:<5}  30min={len(df):>5} bars  "
                  f"days={n_days:>3}  "
                  f"avg_range={avg_rng:>6.1f}pt  "
                  f"${avg_rng*pv:>6.0f}/contract")
        else:
            print(f"  {sym:<5}  not in cache")

    print()
    print("Next steps:")
    print("  python signals/hoeffding.py     — Session 4 Phase 2")
    print("  python signals/regime_em.py     — Session 4 Phase 3")
    print("  python signals/wasserstein_hmm.py — Session 4 Phase 4")
    print()


if __name__ == "__main__":
    main()
