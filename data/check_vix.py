"""
data/check_vix.py
==================
Check what VIX-related data is available on Databento.
VIX is the Cont & da Fonseca Factor 1 proxy for NQ momentum regime.
"""

import databento as db
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()
client = db.Historical(os.getenv("DATABENTO_API_KEY"))

start = (datetime.now(tz=timezone.utc) - timedelta(days=7)).date()
end   = datetime.now(tz=timezone.utc).date()

print("Checking VIX availability on Databento...")
print()

# Check CBOE dataset
try:
    datasets = client.metadata.list_datasets()
    print("Available datasets:")
    for d in datasets:
        print(f"  {d}")
except Exception as e:
    print(f"Error listing datasets: {e}")

print()

# Try CBOE dataset for VIX
vix_candidates = [
    ("CBOE.OPTIONS", "VIX"),
    ("DBEQ.BASIC", "VIX"),
    ("OPRA.PILLAR", "SPX"),
]

for dataset, symbol in vix_candidates:
    try:
        cost = client.metadata.get_cost(
            dataset=dataset,
            symbols=[symbol],
            stype_in="raw_symbol",
            schema="ohlcv-1d",
            start=str(start),
            end=str(end),
        )
        print(f"  {dataset} / {symbol}: available, cost=${cost:.4f}")
    except Exception as e:
        print(f"  {dataset} / {symbol}: not available ({str(e)[:60]})")

print()

# Also check if VIX futures are on CME (VX contracts)
try:
    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=["VXM.c.0"],  # Micro VIX futures
        stype_in="continuous",
        schema="ohlcv-1m",
        start=str(start),
        end=str(end),
    )
    print(f"  GLBX.MDP3 / VXM.c.0 (Micro VIX futures): available, cost=${cost:.4f}")
except Exception as e:
    print(f"  GLBX.MDP3 / VXM.c.0: {str(e)[:80]}")

try:
    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=["VX.c.0"],  # VIX futures
        stype_in="continuous",
        schema="ohlcv-1m",
        start=str(start),
        end=str(end),
    )
    print(f"  GLBX.MDP3 / VX.c.0 (VIX futures): available, cost=${cost:.4f}")
except Exception as e:
    print(f"  GLBX.MDP3 / VX.c.0: {str(e)[:80]}")

print()
print("Note: If VIX futures not available, we approximate Factor 1 using")
print("rolling realized volatility of NQ returns (available from MNQ bars).")
