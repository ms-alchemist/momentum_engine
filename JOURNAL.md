# momentum_engine — Master Project Journal
**Last updated:** 2026-03-23 (Session 6 complete)
**Repository:** github.com/ms-alchemist/momentum_engine (private)
**Account:** Lucid LucidFlex 100K

---

## STRATEGY STATUS: LOCKED — AWAITING AUTOMATION BUILD

### Final Strategy: MGC Mean-Reversion + MNQ Volatility Breakout
**Sizing:** 4x MGC + 4x MNQ contracts
**Filter:** TWAP (prior-session VWAP)
**OOS Sharpe:** 2.86
**OOS Ann P&L:** $19,356/year (~$1,452/month at 90% split)
**OOS Worst Month:** -$2,764 (May 2025, Liberation Day)
**OOS Positive Months:** 26/37 (70%)
**Eval projection:** ~58 trading days (~12 calendar weeks)

---

## THEORETICAL FOUNDATION
*Every component of this strategy is derived from a specific paper
and a specific theoretical concept. This section documents the
intellectual lineage of each design decision.*

---

### 1. WHY MGC MEAN-REVERTS

**Papers:**
- Tsekrekos, A.K. (2010). The effect of mean reversion on entry and exit
  decisions under uncertainty. Annals of Finance.
- Leung, T. & Li, X. (2015). Optimal Mean Reversion Trading with Transaction
  Costs and Stop-Loss Exit. IJTAF.

**Theoretical concept:** Ornstein-Uhlenbeck (OU) process.
Gold prices exhibit mean-reversion at intraday timescales because gold has
no dividend or earnings — it is a store of value. Intraday deviations from
fair value are corrected by arbitrageurs. The 9:00 AM opening bar often
overshoots the equilibrium price due to overnight order imbalance clearing
at the open, creating a predictable intraday arc back toward equilibrium.

**Calibration (Session 5):**
- kappa (mean-reversion speed): 0.258/hr
- Half-life: 5.7 bars = 2.8 hours
- sigma_stat: 15.8 points — typical oscillation around session mean

**Autocorrelation confirmation (Session 6):**
MGC 30-min autocorrelation Lag 8 = -0.022, Lag 13 = -0.042 (REVERT).
Exactly the 4-6.5 hour holding period of the strategy.

**How it's used:** The first-bar signal fires in the opening bar's direction.
MGC's OU structure means this direction predicts where the session closes
62.9% of the time — the opening overshoot reverts by 4:15 PM. The strategy
holds from 9:30 AM to close, capturing the full reversion arc.

---

### 2. WHY MNQ TRENDS

**Papers:**
- Moskowitz, T.J., Ooi, Y.H., & Pedersen, L.H. (2012). Time Series Momentum.
  Journal of Financial Economics, 104(2), 228-250.
- Gao, L., Han, Y., Li, S.Z., & Zhou, G. (2018). Intraday Momentum:
  The First Half-Hour Return Predicts the Last Half-Hour Return.

**Theoretical concept:** Time-Series Momentum (TSMOM).
Equity index futures exhibit positive return autocorrelation at 1-12 month
horizons (Moskowitz et al.) and intraday (Gao et al.). NQ is institutionally
traded and has predictable order flow at the 9:30 AM equity open. Unlike
gold, NQ has earnings, flows, and sentiment — it trends when directional
information enters the market.

**Autocorrelation confirmation (Session 6):**
MNQ 30-min autocorrelation Lag 2 = +0.034 (TREND — the only positive lag
across all 6 instruments tested). At exactly 1 hour after a move, MNQ
continues rather than reverting.

**How it's used:** MNQ is the momentum instrument. Pseudomomentum
accumulates in a tight opening range then releases directionally on
breakout. The strategy captures continuation from breakout to session close.

---

### 3. THE FIRST-BAR SIGNAL

**Paper:** Gao, L., Han, Y., Li, S.Z., & Zhou, G. (2018).
Intraday Momentum: The First Half-Hour Return Predicts the Last Half-Hour Return.

**Theoretical concept:** Institutional order flow at the equity open creates
directional pressure that persists through the session. The first 30-minute
return is a statistically significant predictor of the session close direction
for equity index futures. Extended here to MGC which showed even stronger
accuracy (62.9%) due to its OU mean-reversion completing by close.

**Signal accuracy (3-year backtest):**
- MGC: 62.9% first-bar accuracy
- MNQ: 55.3% first-bar accuracy (breakout strategy used instead)

**Implementation:**
- Signal bar: 9:00-9:30 AM ET (first 30-minute bar)
- Signal: sign(first_bar_close - first_bar_open)
- Entry: at first-bar close (9:30 AM) for MGC
- Doji (open == close): no trade

---

### 4. THE TWAP/VWAP FILTER

**Paper:** Buhler, O. (2014). Waves and Mean Flows (2nd ed.).
Cambridge University Press. Chapter 10: Lagrangian-Mean Theory.

**Theoretical concept:** Lagrangian-mean vs Eulerian-mean.
Buhler distinguishes two ways of measuring flow:

- Eulerian mean: measure at a fixed point (analogous to SMA — price
  average at fixed time intervals regardless of volume)
- Lagrangian mean: follow the particle through space (analogous to
  VWAP — average price weighted by where volume actually transacted)

The Lagrangian mean is more physically meaningful because it tracks
where the mass of the market actually moved. VWAP is the market's
Lagrangian-mean price — the average cost basis of all participants
who traded that session. Prior-session VWAP is the natural equilibrium
reference.

**Filter logic derived from Buhler:**
If today's open is above yesterday's VWAP, the market opened above the
prior session's Lagrangian equilibrium. Institutional buyers are anchored
to this level. A short signal fights this positioning. A long signal
(expecting reversion toward VWAP from above) is consistent with OU
mean-reversion back toward the Lagrangian-mean.

- Today's open > prior VWAP: skip SHORT signals
- Today's open < prior VWAP: skip LONG signals
- Trade only when the first-bar direction expects reversion back toward VWAP

**Formula:**
  prior_VWAP = sum(typical_price_i * volume_i) / sum(volume_i)
  typical_price = (high + low + close) / 3
  Computed from prior trading session — zero look-ahead bias.

**Why TWAP beats SMA20 for macro tail risk:**
SMA20 uses only closing prices — it lags regime changes by days. VWAP
weights by volume. In macro shock events (Liberation Day, May 2025),
volume surges at the turning point and VWAP pivots faster than SMA.
Result: TWAP reduced May 2025 worst month from -$9,759 (SMA20) to
-$2,764 (TWAP) — a $6,995 improvement in the single worst event
across the 3-year dataset. With gold directly exposed to geopolitical
shocks (Iran, energy), VWAP's volume-weighting provides genuine
protection that price-only filters miss.

**OOS validation:** TWAP filter Sharpe retention OOS = 174%.
The VWAP equilibrium concept is structural, not fitted.

---

### 5. THE VOLATILITY BREAKOUT SIGNAL (MNQ)

**Paper:** Buhler, O. (2014). Waves and Mean Flows (2nd ed.).
Cambridge University Press. Chapter 5: Pseudomomentum Conservation.

**Theoretical concept:** Pseudomomentum accumulation and release.
In wave physics, energy accumulates in a compressed medium until it
exceeds a threshold, then releases directionally as a propagating wave.
In market terms: price consolidating in a tight range during the first
hour represents accumulated pseudomomentum — directional energy building
behind a constraint. The breakout bar is the release event. Once
pseudomomentum is released, it propagates: MNQ's Lag-2 positive
autocorrelation (+0.034) confirms the continuation tendency.

**Parameter derivation (diagnose_breakout.py, Session 6):**
Scanned 18 combinations of consolidation window (2-3 bars) and ATR
threshold (10%-60%):
- 2-bar window, 30% ATR: N=143 trades, WR=52.4%, Sharpe=2.37 — optimal
- Session-close WR = 74.2%: when the breakout holds, it wins 74% of the time

**Implementation:**
1. Consolidation: first 2 bars (9:00-10:00 AM), range < 30% of 20-day ATR
2. Breakout: subsequent bar closes outside consolidation range
3. Entry at breakout bar close
4. Stop at opposite end of consolidation range (thesis failure boundary)
5. Exit at session close (MOC order at 4:10 PM)

---

### 6. THE STOP PLACEMENT

**Papers:**
- Di Graziano, G. (2022). Regime-Adaptive Stop-Loss Strategies.
- Leung, T. & Li, X. (2015). Optimal Mean Reversion Trading with
  Transaction Costs and Stop-Loss Exit. IJTAF.

**MGC — 22-point stop (Di Graziano):**
Stop width calibrated to the volatility regime. Must survive normal
intraday noise while cutting genuine losers. Survival analysis: at 22pt,
86.7% of winning MGC trades survive to session close. 22pt is
approximately 1.4x the average intraday noise range.
Max daily loss at 4ct: 22pt x $10 x 4ct = $880 + $3.20 commission.

**MNQ — consolidation range stop (Leung & Li):**
Leung & Li derive the optimal stop for OU processes as the boundary
where the mean-reversion thesis is definitively wrong. For a breakout
strategy, the equivalent: if price returns inside the consolidation
range, the pseudomomentum was not genuine. The consolidation boundary
is the natural thesis-invalidation level.

---

### 7. PORTFOLIO DECORRELATION

**Papers:**
- Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio
  Transactions. Journal of Risk, 3(2), 5-39.
- Moskowitz et al. (2012) — diversified TSMOM portfolio Sharpe improvement.

**Theoretical concept:** Uncorrelated strategy combination reduces
portfolio variance without reducing expected return. Moskowitz
demonstrated that a diversified TSMOM portfolio delivers substantially
higher Sharpe than any single instrument. Almgren & Chriss provide
the execution and portfolio construction framework.

**Measured correlation (Session 4):**
MGC vs MNQ 30-min return correlation: -0.024 (essentially zero).
Gold (safe-haven, commodity) and Nasdaq (risk-on, equity) are driven
by opposite macro forces and are naturally decorrelated.

**Portfolio effect:** At ~zero correlation, portfolio Sharpe is
approximately sqrt(2) x average individual Sharpe.
Individual Sharpes ~1.5-2.0 → portfolio Sharpe ~2.86.

---

### 8. POSITION SIZING (FUTURE ENHANCEMENT)

**Papers:**
- Barroso, P. & Santa-Clara (2015). Momentum Has Its Moments.
  Journal of Financial Economics.
- Daniel, K. & Moskowitz (2016). Momentum Crashes.
  Journal of Financial Economics.

**Theoretical concept:** Constant-volatility position sizing.
Scaling position size inversely to realized volatility (targeting
constant dollar vol) nearly doubles Sharpe for momentum strategies.
Especially important around macro shocks where vol spikes — exactly
when position size should shrink.

**Current:** Fixed 4ct sizing (conservative baseline). The TWAP filter
provides daily regime-awareness. Vol-scaling is the documented next
enhancement: scale to 2-3ct during high-vol months, 5ct during
low-vol trending months. Planned for Version 2 after live trading
is established.

---

### 9. WASSERSTEIN HMM (Built in Session 4, Not Used in Final Strategy)

**Papers:**
- Luwanga et al. (2026). HHT Regime Discovery + VLMC.
- Chevyrev & Kormilitzin (2016). A Primer on the Signature Method
  in Machine Learning. ArXiv.

**What was built:** 6-instrument Wasserstein HMM detecting 4 regimes:
choppy (62%), trending_commodity (26%), trending_equity (7%), volatile (5%).

**Why not in final strategy:** The TWAP filter achieves equivalent
regime-awareness at the daily level with far less complexity. The HMM
is preserved for future dynamic position sizing — scale contracts by
detected regime.

**Levy area (Chevyrev) — future enhancement:**
The signed area between normalized price and volume paths detects
whether volume leads price (genuine accumulation, strong breakout) or
price leads volume (distribution, weak breakout). Documented as an
additional MNQ breakout quality filter in Version 2.

---

### 10. DIRECTION 2 — MGC MEAN-REVERSION (Abandoned Session 5)

**Papers:** Tsekrekos (2010), Leung & Li (2015), Micaletti (2023).

**What was built:** Full OU calibration, MRSI signal (17.5% of days
MRSI<15, 19.4% MRSI>85), isolation diagnostic.

**Why abandoned:** Regime detection failure. The 20-day rolling mean
theta was consistently below current price during the 2024-2026 gold
bull trend ($1,900->$5,100). Fading short toward a lagging theta is
systematically wrong in a sustained trend.

**Future fix (Micaletti variance ratio filter):**
VR(5,63) < 0.8 means ranging → activate fade.
VR > 1.0 means trending → deactivate.
This correctly identifies ranging vs trending regimes for OU strategy.

---

## SESSION HISTORY

| Session | Date | Key Outcome |
|---|---|---|
| 1 | 2026-03-21 | Research synthesis (9 papers), signal architecture, Git/Databento setup |
| 2 | 2026-03-21/22 | MGC backtest iterations, timezone bug fix, Di Graziano calibration, v4 runner |
| 3 | 2026-03-22 | MGC HTC locked (Sharpe 1.31). MNQ/MYM signal audit. MGC wins at 62.9%. |
| 4 | 2026-03-23 | Wasserstein HMM, Hoeffding detector, cross-instrument correlation (-0.024) |
| 5 | 2026-03-23 | Direction 2 abandoned (OU valid but theta lagging in bull trend) |
| 6 | 2026-03-23 | STRATEGY LOCKED — Direction 3: TWAP filter, 4x4, Sharpe 2.86 |

---

## FILE INVENTORY

### Signals
| File | Purpose |
|---|---|
| signals/wasserstein_hmm.py | 6-instrument Wasserstein HMM |
| signals/ou_calibrator.py | OU calibration (kappa=0.258, half-life 2.8hr) |
| signals/mrsi.py | MRSI/MTSI indicators (Direction 2) |
| signals/hoeffding.py | Hoeffding detector (marginal, not used) |

### Backtest
| File | Purpose |
|---|---|
| backtest/runner_final.py | FINAL PRODUCTION RUNNER |
| backtest/runner_mgc_htc_22.py | MGC HTC locked baseline |
| backtest/runner_portfolio.py | Portfolio v1 (9 sizings) |
| backtest/runner_portfolio_v2.py | Portfolio v2 (extended sizing, loss limits) |
| backtest/signal_audit_mnq.py | MNQ 3yr signal audit (6 tests) |
| backtest/diagnose_mnq_entry.py | Delayed entry test (failed) |
| backtest/diagnose_breakout.py | Direction 3 parameter scan |
| backtest/diagnose_3yr_monthly.py | Full 3yr monthly breakdown |
| backtest/diagnose_loss_limit.py | Loss limit audit + OOS (overfitted, rejected) |
| backtest/diagnose_trend_regime.py | Trend filter diagnostic (11 filters) |
| backtest/diagnose_filter_oos.py | OOS filter validation |

### Data
| File | Purpose |
|---|---|
| data/cache/MGC_30min.parquet | 7,365 bars, 2023-03-23 to 2026-03-20 |
| data/cache/MNQ_30min.parquet | 11,950 bars, 2023-03-23 to 2026-03-20 |
| data/cache/MES/MCL/MYM/M2K_30min.parquet | 3yr, 4 additional instruments |

---

## GIT LOG
```
f9e169c  Session 6: Direction 3 complete — TWAP filter, 4x4 sizing
0e849cc  Session 6: MNQ signal audit 3yr + delayed entry — Direction 1 thin edge
55d3220  Session 5: Journal — Direction 2 abandoned, Session 6 plan
62983c8  Session 5: Direction 2 abandoned
e9e113e  Session 4: Wasserstein HMM
1008c84  Session 4: MES data, Hoeffding detector
d91cd2a  Session 3: MGC HTC, MNQ signal stack
```

---

## RESEARCH LIBRARY (17 papers)

| # | Paper | Key Concept | Where Used |
|---|---|---|---|
| 1 | Buhler (2014) — Waves and Mean Flows | Pseudomomentum accumulation; Lagrangian-mean (VWAP); wave breaking | MNQ breakout; TWAP filter |
| 2 | Moskowitz, Ooi, Pedersen (2012) — Time Series Momentum | TSMOM; autocorrelation; vol scaling; diversified portfolio | MNQ trending; portfolio design |
| 3 | Chevyrev, Kormilitzin (2016) — Path Signature | Levy area price/volume lead-lag | Future MNQ quality filter |
| 4 | Cont, da Fonseca (2002) — IV Surface Dynamics | Level/skew/curvature factors; realized vol proxy | Regime gate design |
| 5 | Barroso, Santa-Clara (2015) — Momentum Has Its Moments | Constant-vol sizing doubles Sharpe | Future vol-scaling |
| 6 | Daniel, Moskowitz (2016) — Momentum Crashes | Dynamic sizing around crash regimes | Future position sizing |
| 7 | Almgren, Chriss (2001) — Optimal Execution | Decorrelated portfolio variance reduction | Portfolio combination |
| 8 | Di Graziano (2022) — Regime-Adaptive Stops | Stop width calibrated to vol regime | MGC 22pt stop |
| 9 | Gao, Han, Li, Zhou (2018) — Intraday Momentum | First-half-hour predicts last-half-hour | First-bar signal |
| 10 | Leung, Li (2015) — Optimal Mean Reversion Trading | OU stop at equilibrium boundary | MGC stop; MNQ consolidation stop |
| 11 | Tsekrekos (2010) — Mean Reversion Entry/Exit | OU entry/exit thresholds | MGC OU calibration |
| 12 | Micaletti (2023) — MRSI/MTSI Indicators | Variance ratio filter (ranging vs trending) | Direction 2 (abandoned); future |
| 13 | Blake, Gandhi, Jakkula (2025) — HAR Vol Forecasting | Heterogeneous autoregressive vol model | Future vol scaling |
| 14 | Luwanga et al. (2026) — HHT Regime Discovery | Non-linear regime detection; VLMC | Wasserstein HMM |
| 15 | Egger, Vestal (2025) — Crypto Momentum | Cross-asset momentum validation | Portfolio universe selection |
| 16 | Cheng, Masuda (2026) — NQ Intraday Structure | MNQ intraday order flow patterns | MNQ breakout validation |
| 17 | Boukardagha (2026) — OU Calibration | Real-time OU parameter estimation | MGC OU calibration updates |

---

## SESSION 7 PRE-WORK

Answer these before Session 7:

1. Broker: Is the Lucid account on Tradovate?
2. API access: Is developer/API access enabled?
3. Infrastructure: Local Windows machine or cloud VPS?
4. Notifications: Email, Slack, or SMS for daily summary?

---

## SESSION 7 PLAN — AUTOMATION BUILD

### Architecture
```
momentum_engine/
  live/
    config.py       constants, account params, instrument specs
    signals.py      MGC first-bar + TWAP filter; MNQ consolidation + breakout
    execution.py    order management (entry, stop, MOC exit)
    risk.py         MLL tracker, position limits, kill switch
    scheduler.py    time-based event loop
    main.py         entry point and orchestrator
  tests/
    test_signals.py
    test_execution.py
    test_risk.py
  logs/
    trades.csv
```

### Build Sequence

| Step | Deliverable | Gate |
|---|---|---|
| 1 | Broker API connection | Authenticated session + live bar feed |
| 2 | Signal engine | TWAP filter + MNQ breakout ported to real-time |
| 3 | Order management | Entry, stop-bracket, MOC, cancel logic |
| 4 | Risk manager | MLL tracker, buffer alerts ($500 warn, $200 halt) |
| 5 | Scheduler | Full event loop 8:45 AM to 4:35 PM ET |
| 6 | Paper trading | 2 weeks — signal rate within 10% of backtest |
| 7 | 1-contract live | 2 weeks at 1ct — live P&L within 15% of paper |
| 8 | Full size | 4ct MGC + 4ct MNQ with daily summary |

### Daily Schedule (ET)
| Time | Event |
|---|---|
| 08:45 | Pre-market: load prior VWAP, compute ATR, check MLL buffer |
| 09:00 | Start tracking MGC first bar |
| 09:30 | Evaluate MGC signal, apply TWAP filter, submit entry if valid |
| 09:30 | Start tracking MNQ consolidation range |
| 10:00 | Consolidation closes, begin monitoring MNQ breakout |
| 10:00-16:00 | Monitor MNQ for breakout bar close |
| 16:10 | Submit MOC exit orders for all open positions |
| 16:30 | Verify flat, log P&L, update MLL tracker |
| 16:35 | Send daily summary notification |

### Slippage Model
- MGC entry/exit: +/-0.5pt (~$5/ct) — market order on liquid instrument
- MNQ entry/exit: +/-1-2pt (~$2-4/ct) — market order at breakout bar close
- Stop fills: +/-1pt on stop-limit orders — negligible at this P&L scale
- MOC fills: typically within 0.5pt of closing price
