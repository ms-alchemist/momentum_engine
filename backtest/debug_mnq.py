"""
backtest/debug_mnq.py
Traces signal computation on the first few trading days to find why no trades fire.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time

df = pd.read_parquet("data/cache/MNQ_30min.parquet")
df.index = pd.to_datetime(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("America/New_York")
df = df.sort_index()

print(f"Loaded {len(df)} bars")
print(f"Columns: {df.columns.tolist()}")
print()

from signals.mnq_engine import MNQSignalEngine, SIGNAL_THRESHOLD, DIRECTION_THRESHOLD

engine = MNQSignalEngine()

ENTRY_START_ET   = time(9, 30)
ENTRY_END_ET     = time(14, 0)

trading_days = sorted(set(df.index.date))
signal_count = 0
tradeable_count = 0
skip_reasons = {}

print("Scanning first 20 trading days for signals...\n")

for day in trading_days[:20]:
    day_bars = df[df.index.date == day]
    prior    = df[df.index.date < day].tail(32)

    day_signals = []

    for i, (ts, bar) in enumerate(day_bars.iterrows()):
        bar_time = ts.time()
        if not (ENTRY_START_ET <= bar_time < ENTRY_END_ET):
            continue

        bars_for_signal = pd.concat([prior, day_bars.iloc[:i+1]])

        if len(bars_for_signal) < 20:
            skip_reasons['too_few_bars'] = skip_reasons.get('too_few_bars', 0) + 1
            continue

        signal = engine.compute(bars_for_signal)
        signal_count += 1

        if signal.is_tradeable:
            tradeable_count += 1
            day_signals.append(signal)

        # Print first signal of each day for inspection
        if i == 0 or (len(day_signals) == 0 and bar_time == time(13, 30)):
            print(f"  {day} {bar_time}  score={signal.combined_score:+.4f}  "
                  f"dir={signal.direction}  tradeable={signal.is_tradeable}  "
                  f"regime={signal.regime}")
            for name, layer in signal.layers.items():
                print(f"    {name:<14} score={layer.score:+.4f}  "
                      f"value={layer.value:.5f}  active={layer.active}  {layer.note}")
            print()
            break

print(f"\nTotal signals computed: {signal_count}")
print(f"Tradeable signals:      {tradeable_count}")
print(f"Skip reasons:           {skip_reasons}")
print(f"SIGNAL_THRESHOLD:       {SIGNAL_THRESHOLD}")
print(f"DIRECTION_THRESHOLD:    {DIRECTION_THRESHOLD}")
