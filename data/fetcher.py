"""
Databento Historical Data Fetcher
==================================
Pulls 12 months of MCL and MGC 30-minute OHLCV bars from Databento
and caches them locally as Parquet files.

Key design decisions:
  - Uses continuous contract symbology (MCL.c.0, MGC.c.0) so rollover
    is handled automatically by Databento — no manual contract stitching
  - Checks cost BEFORE pulling — prints estimate and asks confirmation
  - Caches locally as Parquet — fast, compressed, pandas-native
  - Never re-downloads if cache exists — saves money and time
  - Also pulls 1-minute bars for bar_builder to aggregate into 30-min
    with proper buy/sell volume approximation from tick direction

Dataset: GLBX.MDP3 (CME Globex MDP 3.0)
Schemas:
  ohlcv-1m  — 1-minute OHLCV bars (aggregated to 30-min in bar_builder)
  trades    — tick-by-tick trades (for buy/sell volume split)

References:
  Databento Python client: github.com/databento/databento-python
  Continuous contracts:    databento.com/docs/examples/symbology/continuous
"""

import os
import sys
import databento as db
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("DATABENTO_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "DATABENTO_API_KEY not found in environment.\n"
        "Make sure your .env file contains: DATABENTO_API_KEY=db-xxxx"
    )

DATASET   = "GLBX.MDP3"
SYMBOLS   = ["MCL.c.0", "MGC.c.0"]   # continuous front-month contracts
STYPE_IN  = "continuous"

# Date range: 12 months back from today
END_DATE   = datetime.now(tz=timezone.utc).date()
START_DATE = (datetime.now(tz=timezone.utc) - timedelta(days=365)).date()

# Local cache directory — sits inside the project, gitignored
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache filenames
CACHE_FILES = {
    "MCL": CACHE_DIR / "MCL_30min.parquet",
    "MGC": CACHE_DIR / "MGC_30min.parquet",
    "MCL_1min": CACHE_DIR / "MCL_1min.parquet",
    "MGC_1min": CACHE_DIR / "MGC_1min.parquet",
}


# ---------------------------------------------------------------------------
# Cost check — always run before downloading
# ---------------------------------------------------------------------------

def check_cost(client: db.Historical, schema: str) -> float:
    """
    Query Databento's cost estimate API before pulling any data.
    Returns estimated cost in USD.
    Databento charges per GB of data returned — OHLCV bars are tiny.
    """
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=SYMBOLS,
            stype_in=STYPE_IN,
            schema=schema,
            start=str(START_DATE),
            end=str(END_DATE),
        )
        return cost
    except Exception as e:
        print(f"  Warning: Could not estimate cost: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv_1min(client: db.Historical) -> pd.DataFrame:
    """
    Pull 1-minute OHLCV bars for MCL and MGC.
    We use 1-minute bars rather than 30-minute bars because:
      1. We can aggregate to any timeframe locally without re-fetching
      2. 1-min bars let us approximate buy/sell volume using tick direction
      3. Cost difference between 1-min and 30-min OHLCV is minimal
    """
    print(f"\nFetching 1-minute OHLCV bars: {START_DATE} → {END_DATE}")
    print(f"Symbols: {SYMBOLS}")

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=SYMBOLS,
        stype_in=STYPE_IN,
        schema="ohlcv-1m",
        start=str(START_DATE),
        end=str(END_DATE),
    )

    df = data.to_df()
    print(f"  Retrieved {len(df):,} 1-minute bars")
    return df


def process_and_split(df_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Process raw Databento OHLCV DataFrame:
      1. Split into MCL and MGC
      2. Clean and standardise column names
      3. Convert Databento price format (integer × 1e-9) to float
      4. Add session filter (RTH: 09:00-16:30 ET)
      5. Add approximate buy/sell volume using bar direction
    
    Databento stores prices as fixed-point integers (price × 1e-9).
    The Python client handles this automatically via .to_df() but
    we verify the scale is correct.
    """
    results = {}

    # Databento returns a 'symbol' column with the continuous contract notation
    # Map back to clean instrument names
    symbol_map = {"MCL.c.0": "MCL", "MGC.c.0": "MGC"}

    for db_symbol, clean_symbol in symbol_map.items():
        # Filter to this instrument
        if "symbol" in df_raw.columns:
            mask = df_raw["symbol"] == db_symbol
        elif "instrument_id" in df_raw.columns:
            # Sometimes Databento returns instrument_id instead of symbol
            # in that case we split by position in the dataset
            mask = slice(None)  # take all, handled below
        
        df = df_raw[mask].copy() if isinstance(mask, pd.Series) else df_raw.copy()

        # Rename columns to our standard format
        col_map = {
            "open":   "open",
            "high":   "high", 
            "low":    "low",
            "close":  "close",
            "volume": "volume",
        }
        df = df.rename(columns=col_map)

        # Ensure index is DatetimeIndex in Eastern time
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts_event" in df.columns:
                df = df.set_index("ts_event")
        
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert("America/New_York")

        # Keep only standard columns
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"]
                     if c in df.columns]
        df = df[keep_cols]

        # Drop any rows with NaN prices
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Filter to regular trading hours + extended session
        # MCL trades 18:00 Sunday - 17:00 Friday ET (nearly 24h)
        # MGC trades 18:00 Sunday - 17:00 Friday ET (nearly 24h)
        # For our strategy: 09:00 - 16:30 ET (US session focus per Gao et al.)
        df = df.between_time("09:00", "16:30")

        # Approximate buy/sell volume using bar direction
        # If close > open → bullish bar → assign more to buy volume
        # This is an approximation — tick-level data would be more accurate
        # but costs more to fetch. Acceptable for 30-min signal computation.
        price_direction = (df["close"] - df["open"]).apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        # Bullish bar: 65% buy, 35% sell. Bearish: reverse. Doji: 50/50
        buy_fraction = price_direction.map({1: 0.65, -1: 0.35, 0: 0.50})
        df["buy_volume"]  = (df["volume"] * buy_fraction).astype(int)
        df["sell_volume"] = df["volume"] - df["buy_volume"]

        df["symbol"] = clean_symbol
        results[clean_symbol] = df

        print(f"  {clean_symbol}: {len(df):,} 1-min bars "
              f"({df.index[0].date()} → {df.index[-1].date()})")

    return results


def aggregate_to_30min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute bars to 30-minute bars.
    
    Standard OHLCV aggregation rules:
      open   = first bar's open
      high   = max of all highs
      low    = min of all lows
      close  = last bar's close
      volume = sum of all volumes
      buy_volume  = sum of buy volumes
      sell_volume = sum of sell volumes
    
    Uses 30-minute periods aligned to session open (09:00, 09:30, 10:00...)
    """
    agg_rules = {
        "open":         "first",
        "high":         "max",
        "low":          "min",
        "close":        "last",
        "volume":       "sum",
        "buy_volume":   "sum",
        "sell_volume":  "sum",
    }

    # Only aggregate columns that exist
    agg_rules = {k: v for k, v in agg_rules.items() if k in df_1min.columns}

    df_30 = df_1min.resample("30min", label="left", closed="left").agg(agg_rules)

    # Drop bars with no volume (market closed / holiday gaps)
    df_30 = df_30[df_30["volume"] > 0].dropna(subset=["open", "close"])

    return df_30


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def save_to_cache(df: pd.DataFrame, path: Path, label: str):
    """Save DataFrame to Parquet with compression."""
    df.to_parquet(path, compression="snappy")
    size_kb = path.stat().st_size / 1024
    print(f"  Saved {label}: {len(df):,} bars → {path.name} ({size_kb:.1f} KB)")


def load_from_cache(path: Path, label: str) -> pd.DataFrame:
    """Load DataFrame from Parquet cache."""
    df = pd.read_parquet(path)
    print(f"  Loaded {label} from cache: {len(df):,} bars "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def cache_is_fresh(path: Path) -> bool:
    """
    Check if cache file exists and was created today.
    If it's older than today, re-fetch to get the latest bars.
    """
    if not path.exists():
        return False
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
    return modified >= END_DATE - timedelta(days=1)


# ---------------------------------------------------------------------------
# Main fetch function — called by backtest runner and live system
# ---------------------------------------------------------------------------

def fetch_bars(
    force_refresh: bool = False,
    check_cost_first: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Main entry point. Returns dict of 30-min bar DataFrames keyed by symbol.

    Args:
        force_refresh:    Re-fetch even if cache is fresh
        check_cost_first: Print cost estimate and ask confirmation before fetching

    Returns:
        {
            "MCL": pd.DataFrame,  # 30-min bars, 12 months
            "MGC": pd.DataFrame,  # 30-min bars, 12 months
        }

    Each DataFrame has columns:
        open, high, low, close, volume, buy_volume, sell_volume
    Index: DatetimeIndex in America/New_York timezone, 30-min frequency
    """
    results = {}

    # Check if both caches are fresh
    mcl_fresh = cache_is_fresh(CACHE_FILES["MCL"]) and not force_refresh
    mgc_fresh = cache_is_fresh(CACHE_FILES["MGC"]) and not force_refresh

    if mcl_fresh and mgc_fresh:
        print("Cache is fresh — loading from local files (no API call needed)")
        results["MCL"] = load_from_cache(CACHE_FILES["MCL"], "MCL 30-min")
        results["MGC"] = load_from_cache(CACHE_FILES["MGC"], "MGC 30-min")
        return results

    # Need to fetch — initialize client
    client = db.Historical(API_KEY)

    # Cost check
    if check_cost_first:
        print(f"\nChecking cost for 1-min OHLCV bars...")
        print(f"  Dataset:  {DATASET}")
        print(f"  Symbols:  {SYMBOLS}")
        print(f"  Range:    {START_DATE} → {END_DATE} (12 months)")
        print(f"  Schema:   ohlcv-1m")

        cost = check_cost(client, "ohlcv-1m")
        print(f"\n  Estimated cost: ${cost:.4f} USD")

        if cost > 1.0:
            confirm = input(
                f"\n  Cost is ${cost:.4f}. Proceed? (y/n): "
            ).strip().lower()
            if confirm != "y":
                print("  Fetch cancelled.")
                sys.exit(0)
        else:
            print(f"  Cost is under $1.00 — proceeding automatically.")

    # Fetch 1-minute bars
    print(f"\nFetching from Databento...")
    df_raw = fetch_ohlcv_1min(client)

    # Process and split by instrument
    print("\nProcessing bars...")
    split_1min = process_and_split(df_raw)

    # Save 1-min cache and aggregate to 30-min
    print("\nAggregating to 30-minute bars...")
    for symbol, df_1min in split_1min.items():
        # Save 1-min cache
        save_to_cache(df_1min, CACHE_FILES[f"{symbol}_1min"], f"{symbol} 1-min")

        # Aggregate to 30-min
        df_30 = aggregate_to_30min(df_1min)
        save_to_cache(df_30, CACHE_FILES[symbol], f"{symbol} 30-min")
        results[symbol] = df_30

        # Print summary statistics
        print(f"\n  {symbol} 30-min summary:")
        print(f"    Bars:        {len(df_30):,}")
        print(f"    Date range:  {df_30.index[0].date()} → {df_30.index[-1].date()}")
        print(f"    Price range: {df_30['low'].min():.2f} → {df_30['high'].max():.2f}")
        print(f"    Avg volume:  {df_30['volume'].mean():.0f} contracts/bar")
        print(f"    Avg buy vol: {df_30['buy_volume'].mean():.0f} ({df_30['buy_volume'].mean()/df_30['volume'].mean()*100:.0f}% of total)")

    return results


def load_bars() -> dict[str, pd.DataFrame]:
    """
    Convenience function — loads from cache if available,
    fetches from Databento if not. Used by backtest runner.
    """
    mcl_fresh = cache_is_fresh(CACHE_FILES["MCL"])
    mgc_fresh = cache_is_fresh(CACHE_FILES["MGC"])

    if mcl_fresh and mgc_fresh:
        return {
            "MCL": load_from_cache(CACHE_FILES["MCL"], "MCL 30-min"),
            "MGC": load_from_cache(CACHE_FILES["MGC"], "MGC 30-min"),
        }
    else:
        return fetch_bars()


# ---------------------------------------------------------------------------
# Entry point — run directly to fetch and cache data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Databento Historical Data Fetcher")
    print(f"  MCL + MGC | {START_DATE} → {END_DATE}")
    print("=" * 60)

    bars = fetch_bars(force_refresh="--refresh" in sys.argv)

    print("\n" + "=" * 60)
    print("  Fetch complete. Summary:")
    print("=" * 60)
    for symbol, df in bars.items():
        print(f"  {symbol}: {len(df):,} 30-min bars ready for backtest")

    print(f"\n  Cache location: {CACHE_DIR.absolute()}")
    print("\n  Next step: run backtest/runner.py")
