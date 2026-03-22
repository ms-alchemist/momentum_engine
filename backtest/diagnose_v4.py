"""
backtest/diagnose_v4.py
=======================
Diagnostic analysis of v4 backtest results.
Run after runner_v4.py has produced results.
"""

import pandas as pd
from pathlib import Path

results_dir = Path("backtest/results")
trade_files = sorted(results_dir.glob("v4_trades_1ct_*.csv"))

if not trade_files:
    print("No v4 results found. Run backtest/runner_v4.py first.")
    exit()

df = pd.read_csv(trade_files[-1])
print(f"Loaded {len(df)} trades from {trade_files[-1].name}")
print()

# Direction-adjusted actual move in points
direction = df["direction"].map({"LONG": 1, "SHORT": -1})
df["actual_move_pts"] = direction * (df["exit_price"] - df["entry_price"])

# --- Exit reason breakdown ---
stops   = df[df["exit_reason"] == "stop"]
targets = df[df["exit_reason"] == "target"]
closes  = df[df["exit_reason"].isin(["session_close", "eod"])]

print("=" * 55)
print("  EXIT REASON ANALYSIS")
print("=" * 55)

print(f"\nSTOP trades ({len(stops)}):")
print(f"  Avg move at stop:  {stops['actual_move_pts'].mean():.1f} pts")
print(f"  Min / Max:         {stops['actual_move_pts'].min():.1f} / {stops['actual_move_pts'].max():.1f} pts")
print(f"  Avg P&L:           ${stops['net_pnl'].mean():.0f}")

print(f"\nTARGET trades ({len(targets)}):")
print(f"  Avg move at target: {targets['actual_move_pts'].mean():.1f} pts")
print(f"  Avg P&L:            ${targets['net_pnl'].mean():.0f}")

print(f"\nSESSION CLOSE / EOD trades ({len(closes)}):")
print(f"  Avg move:   {closes['actual_move_pts'].mean():.1f} pts")
print(f"  Wins:       {(closes['actual_move_pts'] > 0).sum()}")
print(f"  Losses:     {(closes['actual_move_pts'] <= 0).sum()}")
print(f"  Avg P&L:    ${closes['net_pnl'].mean():.0f}")

# --- Monthly breakdown ---
print()
print("=" * 55)
print("  MONTHLY BREAKDOWN")
print("=" * 55)
df["date"] = pd.to_datetime(df["date"])
monthly = df.groupby(df["date"].dt.to_period("M")).apply(
    lambda x: pd.Series({
        "trades":   len(x),
        "wins":     x["win"].sum(),
        "win_rate": x["win"].mean(),
        "pnl":      x["net_pnl"].sum(),
    })
).reset_index()

print(f"\n{'Month':<10} {'Trades':>7} {'Wins':>6} {'WinRate':>9} {'P&L':>10}")
print("-" * 46)
for _, row in monthly.iterrows():
    print(f"  {str(row['date']):<10} {row['trades']:>5.0f}   {row['wins']:>4.0f}   "
          f"{row['win_rate']:>7.1%}   ${row['pnl']:>7.0f}")

# --- Distribution of actual moves ---
print()
print("=" * 55)
print("  MOVE DISTRIBUTION — all trades")
print("=" * 55)
buckets = [-50, -30, -20, -16, -10, -5, 0, 5, 10, 16, 19, 25, 30, 50]
print(f"\n{'Range (pts)':<20} {'Count':>6} {'%':>6}")
print("-" * 35)
for i in range(len(buckets) - 1):
    lo, hi = buckets[i], buckets[i+1]
    count = ((df["actual_move_pts"] >= lo) & (df["actual_move_pts"] < hi)).sum()
    pct   = count / len(df) * 100
    bar   = "#" * int(pct)
    print(f"  {lo:>4} to {hi:>4} pts     {count:>4}   {pct:>5.1f}%  {bar}")

print()
print("=" * 55)
print("  SIGNAL SCORE vs OUTCOME")
print("=" * 55)
df["score_bucket"] = pd.cut(df["signal_score"].abs(),
                             bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                             labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
score_analysis = df.groupby("score_bucket", observed=True).apply(
    lambda x: pd.Series({
        "trades":   len(x),
        "win_rate": x["win"].mean(),
        "avg_pnl":  x["net_pnl"].mean(),
    })
)
print(f"\n{'Score bucket':<14} {'Trades':>7} {'WinRate':>9} {'AvgP&L':>9}")
print("-" * 42)
for bucket, row in score_analysis.iterrows():
    print(f"  {str(bucket):<14} {row['trades']:>5.0f}   {row['win_rate']:>7.1%}   ${row['avg_pnl']:>6.0f}")

print()
