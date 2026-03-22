"""
backtest/regime_filter_test.py
===============================
Tests whether filtering out low-range months improves strategy performance.
The hypothesis: first-bar directional signal fails in choppy/low-range months.
"""

import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("backtest/results")
trade_files = sorted(results_dir.glob("v4_trades_1ct_*.csv"))
daily_files = sorted(results_dir.glob("v4_daily_1ct_*.csv"))

if not trade_files:
    print("No v4 results found. Run backtest/runner_v4.py first.")
    exit()

trades = pd.read_csv(trade_files[-1])
trades["date"] = pd.to_datetime(trades["date"])

# Load MGC bars for daily range data
bars = pd.read_parquet("data/cache/MGC_30min.parquet")
bars.index = pd.to_datetime(bars.index)

# Build daily range series
daily_range = (
    bars.resample("1D")
        .agg({"high": "max", "low": "min"})
        .dropna()
)
daily_range["range"] = daily_range["high"] - daily_range["low"]

# Monthly average range
monthly_avg_range = daily_range["range"].resample("ME").mean()
monthly_avg_range.index = monthly_avg_range.index.to_period("M")

# Tag each trade with the PRIOR month's avg range
# (we can only know prior month's range at trade time)
trades["month"] = trades["date"].dt.to_period("M")
trades["prior_month"] = trades["month"] - 1
trades["prior_month_range"] = trades["prior_month"].map(
    monthly_avg_range.to_dict()
)

print("=" * 60)
print("  MONTHLY RANGE vs WIN RATE ANALYSIS")
print("=" * 60)
print()

# Show each month's prior range and win rate
monthly_stats = trades.groupby("month").apply(
    lambda x: pd.Series({
        "trades":            len(x),
        "win_rate":          x["win"].mean(),
        "pnl":               x["net_pnl"].sum(),
        "prior_month_range": x["prior_month_range"].iloc[0]
            if not x["prior_month_range"].isna().all() else 0,
    })
).reset_index()

print(f"{'Month':<12} {'PriorRange':>11} {'Trades':>7} {'WinRate':>9} {'P&L':>10}")
print("-" * 55)
for _, row in monthly_stats.iterrows():
    flag = " <-- LOW RANGE" if row["prior_month_range"] < 25 else ""
    print(f"  {str(row['month']):<10}  {row['prior_month_range']:>8.1f}pt  "
          f"{row['trades']:>5.0f}   {row['win_rate']:>7.1%}  "
          f"${row['pnl']:>8.0f}{flag}")

# Test different range thresholds
print()
print("=" * 60)
print("  FILTER TEST — skip months where prior range < threshold")
print("=" * 60)
print()
print(f"{'Threshold':>12} {'Trades':>8} {'Filtered':>9} {'WinRate':>9} {'P&L':>10} {'Sharpe':>8}")
print("-" * 60)

for threshold in [0, 20, 25, 28, 30, 32, 35]:
    if threshold == 0:
        filtered = trades
        label = "No filter"
    else:
        filtered = trades[
            trades["prior_month_range"].isna() |
            (trades["prior_month_range"] >= threshold)
        ]
        label = f">= {threshold}pt"

    if len(filtered) == 0:
        continue

    n          = len(filtered)
    removed    = len(trades) - n
    win_rate   = filtered["win"].mean()
    total_pnl  = filtered["net_pnl"].sum()

    # Simple daily Sharpe approximation
    daily_pnl  = filtered.groupby("date")["net_pnl"].sum()
    sharpe     = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
                  if daily_pnl.std() > 0 else 0)

    print(f"  {label:>10}   {n:>6}   {removed:>7}   "
          f"{win_rate:>7.1%}  ${total_pnl:>8.0f}  {sharpe:>7.2f}")

# Best threshold analysis
print()
print("=" * 60)
print("  CORRELATION: prior month range vs win rate")
print("=" * 60)
valid = monthly_stats[monthly_stats["prior_month_range"] > 0]
corr = valid["prior_month_range"].corr(valid["win_rate"])
print(f"\n  Correlation coefficient: {corr:.3f}")
if corr > 0.3:
    print("  POSITIVE correlation — higher range months have better win rates")
    print("  Filtering low-range months should improve performance")
elif corr < -0.3:
    print("  NEGATIVE correlation — lower range months have better win rates")
else:
    print("  Weak correlation — range filter may not help much")

print()
