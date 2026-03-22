"""
backtest/analyze_mnq.py
Detailed analysis of MNQ trade exits to find what needs fixing.
"""
import pandas as pd
import numpy as np
from pathlib import Path

results_dir = Path("backtest/results")
files = sorted(results_dir.glob("mnq_full_1ct_*.csv"))

if not files:
    print("No results found. Run runner_mnq_full.py first.")
    exit()

df = pd.read_csv(files[-1])
print(f"Loaded {len(df)} trades from {files[-1].name}")
print()

# Direction of each exit
direction = df["direction"].map({"LONG": 1, "SHORT": -1})
df["move_pts"] = direction * (df["exit_price"] - df["entry_price"])

print("=" * 55)
print("  EXIT BREAKDOWN")
print("=" * 55)
for reason in ["target", "stop", "session_close", "eod"]:
    sub = df[df["exit_reason"] == reason]
    if len(sub) == 0:
        continue
    wins = sub[sub["win"]]
    print(f"\n{reason.upper()} ({len(sub)} trades):")
    print(f"  Win rate:  {len(wins)/len(sub):.1%}")
    print(f"  Avg P&L:   ${sub['net_pnl'].mean():.0f}")
    print(f"  Total P&L: ${sub['net_pnl'].sum():.0f}")
    print(f"  Avg move:  {sub['move_pts'].mean():.1f} pts")
    print(f"  Move range: {sub['move_pts'].min():.1f} to {sub['move_pts'].max():.1f} pts")

print()
print("=" * 55)
print("  STOP/TARGET ANALYSIS")
print("=" * 55)
print(f"\nAvg stop distance:   {df['stop_pts'].mean():.1f} pts")
print(f"Avg target distance: {df['target_pts'].mean():.1f} pts")
print(f"Avg b/a ratio:       {(df['target_pts']/df['stop_pts']).mean():.2f}")

print()
print("=" * 55)
print("  SIGNAL SCORE vs OUTCOME")
print("=" * 55)
df["score_abs"] = df["signal_score"].abs()
bins = [0, 0.2, 0.3, 0.4, 0.5, 1.0]
labels = ["0-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5+"]
df["score_bin"] = pd.cut(df["score_abs"], bins=bins, labels=labels)
score_analysis = df.groupby("score_bin", observed=True).apply(
    lambda x: pd.Series({
        "trades":   len(x),
        "win_rate": x["win"].mean(),
        "avg_pnl":  x["net_pnl"].mean(),
        "total":    x["net_pnl"].sum(),
    })
)
print(f"\n{'Score':>10} {'Trades':>8} {'WinRate':>9} {'AvgPnL':>9} {'Total':>10}")
print("-" * 50)
for bucket, row in score_analysis.iterrows():
    print(f"  {str(bucket):<10} {row['trades']:>6.0f}   "
          f"{row['win_rate']:>7.1%}  ${row['avg_pnl']:>7.0f}  ${row['total']:>8.0f}")

print()
print("=" * 55)
print("  DIRECTION ANALYSIS")
print("=" * 55)
for d in ["LONG", "SHORT"]:
    sub = df[df["direction"] == d]
    wins = sub[sub["win"]]
    print(f"\n{d} ({len(sub)} trades):")
    print(f"  Win rate: {len(wins)/len(sub):.1%}  Avg PnL: ${sub['net_pnl'].mean():.0f}")

print()
print("=" * 55)
print("  KEY INSIGHT: What if we only took high-score signals?")
print("=" * 55)
for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    sub = df[df["score_abs"] >= threshold]
    if len(sub) == 0:
        continue
    wins = sub[sub["win"]]
    wr   = len(wins)/len(sub)
    pnl  = sub["net_pnl"].sum()
    losses = sub[~sub["win"]]
    pf = wins["net_pnl"].sum() / abs(losses["net_pnl"].sum()) if len(losses) and losses["net_pnl"].sum() != 0 else np.inf
    print(f"  Score >= {threshold:.2f}: {len(sub):>3} trades  WR={wr:.1%}  "
          f"PF={pf:.2f}  P&L=${pnl:,.0f}")
