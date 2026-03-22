import pandas as pd
import numpy as np

df = pd.read_parquet('data/cache/MGC_30min.parquet')
df.index = pd.to_datetime(df.index, utc=True)
df['date'] = df.index.date
df['hour'] = df.index.hour

df['ret'] = df['close'].pct_change()
first_bars = df[df['hour'] == 13][['date','ret','close','open']].copy()
first_bars.columns = ['date','first_ret','first_close','first_open']
first_bars['ret'] = first_bars['first_close'].pct_change()

daily = df.groupby('date').agg(
    session_open=('open','first'),
    session_close=('close','last'),
    session_high=('high','max'),
    session_low=('low','min'),
    n_bars=('close','count')
)
daily['session_ret'] = daily['session_close'] - daily['session_open']
daily['session_dir'] = np.sign(daily['session_ret'])
daily['daily_range'] = daily['session_high'] - daily['session_low']

merged = daily.join(first_bars.set_index('date')).dropna()
merged['first_dir'] = np.sign(merged['first_ret'])
merged['correct']   = merged['first_dir'] == merged['session_dir']

wins   = merged[merged['correct']]['session_ret'].abs()
losses = merged[~merged['correct']]['session_ret'].abs()
wr     = merged['correct'].mean()
ev     = wr * wins.mean() - (1 - wr) * losses.mean()

print("FULL SESSION HOLD — FIRST BAR SIGNAL")
print("=" * 45)
print(f"Win rate:          {wr*100:.1f}%")
print(f"Avg win (pts):     {wins.mean():.1f}")
print(f"Avg loss (pts):    {losses.mean():.1f}")
print(f"Payoff ratio:      {wins.mean()/losses.mean():.2f}")
print(f"Expected value:    {ev:.1f} pts per trade")
print(f"EV per contract:   ${ev*10:.0f} per trade")
print()
print("STOP DISTANCE ANALYSIS")
print("(What % of winning days would survive each stop?)")
print()
for stop in [5, 8, 10, 15, 20, 25]:
    # On winning long days: how far did price dip below entry?
    long_wins  = merged[(merged['correct']) & (merged['first_dir'] == 1)]
    short_wins = merged[(merged['correct']) & (merged['first_dir'] == -1)]
    # Max adverse excursion approximation: use daily low vs open
    long_mae   = (long_wins['session_open'] - long_wins['session_low']).clip(lower=0)
    short_mae  = (short_wins['session_high'] - short_wins['session_open']).clip(lower=0)
    all_mae    = pd.concat([long_mae, short_mae])
    survival   = (all_mae <= stop).mean() * 100
    print(f"  {stop:>2}pt stop: {survival:.0f}% of winning trades survive")

print()
print("VOLATILITY FILTER IMPACT")
print("(Skip days where 5-day realized vol > 2x its 20-day avg)")
merged.index = pd.to_datetime(merged.index)
merged['range_5d']  = merged['daily_range'].rolling(5).mean()
merged['range_20d'] = merged['daily_range'].rolling(20).mean()
merged['vol_ok']    = merged['range_5d'] <= (merged['range_20d'] * 2.5)

filtered = merged[merged['vol_ok']]
if len(filtered) > 0:
    wr_f   = filtered['correct'].mean()
    wins_f = filtered[filtered['correct']]['session_ret'].abs().mean()
    loss_f = filtered[~filtered['correct']]['session_ret'].abs().mean()
    ev_f   = wr_f * wins_f - (1 - wr_f) * loss_f
    print(f"  Days traded after filter: {len(filtered)} of {len(merged)}")
    print(f"  Win rate after filter:    {wr_f*100:.1f}%")
    print(f"  EV after filter:          {ev_f:.1f} pts  (${ev_f*10:.0f}/contract)")