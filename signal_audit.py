import pandas as pd
import numpy as np

df = pd.read_parquet('data/cache/MGC_30min.parquet')
df.index = pd.to_datetime(df.index, utc=True)

df['date'] = df.index.date
df['ret']  = df['close'].pct_change()
df['hour'] = df.index.hour

# First bar return (UTC 13:00 = 9am ET)
first_bars = df[df['hour'] == 13][['date', 'ret']].rename(columns={'ret': 'first_ret'})

# Session return
daily = df.groupby('date').agg(
    session_open=('open', 'first'),
    session_close=('close', 'last'),
    n_bars=('close', 'count')
)
daily['session_ret'] = daily['session_close'] - daily['session_open']
daily['session_dir'] = np.sign(daily['session_ret'])

merged = daily.join(first_bars.set_index('date'))
merged['first_dir'] = np.sign(merged['first_ret'])
merged = merged.dropna()

same_dir = (merged['first_dir'] == merged['session_dir']).mean()

print("=" * 50)
print("SIGNAL AUDIT — MGC 30-min")
print("=" * 50)
print()
print("TEST 1: First bar predicts session direction")
print(f"  Accuracy: {same_dir*100:.1f}%")
print(f"  Baseline: 50.0%  (need >55% for edge)")
print()

correct   = merged['first_dir'] == merged['session_dir']
wins      = merged[correct]['session_ret'].abs().mean()
losses    = merged[~correct]['session_ret'].abs().mean()
print("  When correct:")
print(f"    Avg session move: {wins:.1f} pts")
print("  When wrong:")
print(f"    Avg session move: {losses:.1f} pts")
print()

print("TEST 2: Return autocorrelation at 30-min bars")
rets = df['ret'].dropna()
for lag, label in [(1,'30 min'), (2,'1 hour'), (8,'4 hours'), (16,'8 hours')]:
    ac = rets.autocorr(lag=lag)
    interp = 'momentum' if ac > 0.02 else ('mean-revert' if ac < -0.02 else 'random')
    print(f"  Lag {lag:>2} ({label:<8}): {ac:>7.4f}  [{interp}]")
print()

print("TEST 3: Monthly first-bar prediction accuracy")
merged.index = pd.to_datetime(merged.index)
monthly = merged.groupby(merged.index.to_period('M')).apply(
    lambda x: pd.Series({
        'days': len(x),
        'accuracy': (x['first_dir'] == x['session_dir']).mean() * 100,
        'avg_range': x['session_ret'].abs().mean()
    })
)
print(monthly.round(1).to_string())