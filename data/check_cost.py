"""
data/check_cost.py
==================
Check Databento cost for MNQ + MYM before fetching.
"""

import databento as db
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()
client = db.Historical(os.getenv("DATABENTO_API_KEY"))

start = (datetime.now(tz=timezone.utc) - timedelta(days=365)).date()
end   = datetime.now(tz=timezone.utc).date()

symbols = ["MNQ.c.0", "MYM.c.0"]

print(f"Cost check: {symbols}")
print(f"Range: {start} to {end}")
print(f"Schema: ohlcv-1m")
print()

cost = client.metadata.get_cost(
    dataset="GLBX.MDP3",
    symbols=symbols,
    stype_in="continuous",
    schema="ohlcv-1m",
    start=str(start),
    end=str(end),
)
print(f"Estimated cost: ${cost:.4f} USD")
