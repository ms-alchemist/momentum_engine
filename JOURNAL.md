# Momentum Engine — Development Journal

## Project
LucidFlex 100K prop firm evaluation — intraday momentum/mean-reversion on micro futures.
Repo: github.com/ms-alchemist/momentum_engine (private)

---

## Sessions 1-3 Summary
- MGC hold-to-close strategy built and finalized
- Signal: first 30-min bar direction (9:00-9:30 AM ET)
- Stop: 22pt catastrophic (Di Graziano calibration)
- Exit: 4:15 PM session close
- Vol filter: skip if 5-day range > 2.5× 20-day avg
- Prior-month filter: skip if prior month avg range > 50pt
- Result: Sharpe 1.85, $8,965/year (1-year backtest, 2 contracts)
- MLL mechanics verified — floor never breached

---

## Session 4 — Regime Detection & Cross-Instrument Analysis

### Data
- Extended all instruments to 3 years (2023-03-22 to 2026-03-20)
- Full universe: MGC, MCL, MNQ, MES, MYM, M2K
- Fetch script: data/fetch_all_3y.py

### Hoeffding Regime Detector
- File: signals/hoeffding.py
- Result: MARGINAL VALUE — best window (+$92, +0.05 Sharpe)
- Decision: not worth standalone complexity

### Cross-Instrument Correlation
- File: backtest/regime_correlation.py
- MGC vs MNQ: -0.024 (essentially independent)
- Key finding: clear decorrelation — when MGC profitable, MNQ often flat/negative

### Wasserstein HMM (6-instrument, 3-year)
- File: signals/wasserstein_hmm.py
- 4 distinct regimes identified: choppy (62%), trending_commodity (26%),
  trending_equity (7%), volatile (5%)
- HMM overlay vs baseline: approximately NEUTRAL (+0.02 Sharpe, -$176 P&L)
- MNQ in trending_equity regime: 12 trades, 58.3% WR, PF 2.14 — GENUINE EDGE

### 3-Year Baseline Results
- Pure MGC 2ct: Sharpe 1.31, $2,072/year average
- Edge is real but thin — not strong enough to pass eval quickly
- Decision: build stronger strategy rather than scale contracts

### New Papers (Session 4)
- Blake, Gandhi, Jakkula (2025) — Regime-switching HAR vol forecasting
- Luwanga et al. (2026) — HHT regime discovery + VLMC dynamics
- Tsekrekos (2010) — OU entry/exit thresholds
- Micaletti (2023) — MRSI/MTSI mean-reversion indicators
- Leung & Li (2015) — Optimal mean-reversion trading with stop-loss

---

## Session 5 — Direction 2: MGC Mean-Reversion (ABANDONED)

### Hypothesis
MGC shows negative lag-1 autocorrelation at 30-min resolution.
Fading extreme first-bar moves should exploit intraday mean-reversion.

### OU Calibration (signals/ou_calibrator.py)
- κ = 0.258/hr, CoV = 23.7% STABLE — mean-reversion is real
- Half-life = 5.7 bars = 2.8 hours — valid intraday timescale
- σ_stat = 15.8 pts — typical oscillation around session mean
- Optimal take-profit: ~9pts linked to 15pt stop (Leung-Li)

### MRSI Signal (signals/mrsi.py)
- 17.5% of days MRSI < 15 (close near HIGH)
- 19.4% of days MRSI > 85 (close near LOW)
- Signal distribution looked promising before isolation test

### Isolation Diagnostic (backtest/diagnose_mr.py) — DECISIVE
| MRSI Band        |  N | WR    | Target% | cstop% |
|------------------|----|-------|---------|--------|
| MRSI 0-15        | 72 | 11.1% | 61.1%   | 16.7%  |
| MRSI 85-100      | 78 | 29.5% | 35.9%   | 20.5%  |

High target% but catastrophically low WR — apparent paradox.

### Root Cause: Regime Detection Failure
**The mean-reversion mechanism is valid. The regime identification failed.**

The Wasserstein HMM labeled 2024-2026 as "choppy" when MGC was in a
sustained bull trend ($1,900 → $5,100, +168% over 3 years). θ computed
as a 20-day rolling session mean was consistently BELOW current price.
Fading short toward a lagging θ requires a 30-50pt move to reach target
but the catastrophic stop fires at 15pts — systematically losing.

The OU process IS real (κ stable) but it mean-reverts around a DRIFTING
mean in a trending market. Direction 2 requires a regime filter that
correctly identifies trending vs ranging environments BEFORE activating
the fade signal.

### Decision: ABANDON Direction 2 (current form)
Future enhancement: reactivate when HMM correctly identifies ranging regime.
The full research framework (Tsekrekos + Micaletti + Leung-Li + OU calibrator)
is preserved and valid — just needs proper regime gating.

### Files Built (preserved)
- signals/ou_calibrator.py
- signals/mrsi.py
- backtest/diagnose_mr.py
- backtest/runner_mgc_mr.py

---

## Session 6 Plan — Direction 1: MNQ Multi-Timeframe Momentum

### Rationale
- MNQ: $669/contract daily range vs $246 MGC (2.7x higher dollar vol)
- MGC and MNQ essentially independent (-0.024 correlation)
- MNQ in trending_equity regime: WR 58.3%, PF 2.14 — verified edge
- 6-layer signal stack already built (signals/mnq_engine.py)
- 3-year data in cache (11,950 30-min bars)

### Research Foundation
- Gao et al. (2018) — Intraday momentum (first-bar signal)
- Moskowitz et al. (2012) — Time series momentum (EWMA signal)
- Barroso & Santa-Clara (2015) — Vol scaling
- Daniel & Moskowitz (2016) — Momentum crashes + regime gate
- Wasserstein HMM — already built, use trending_equity filter

### Key Questions for Session 6
1. Does first-bar signal work on MNQ with same strength as MGC?
2. Optimal stop distance for MNQ? (Di Graziano calibration)
3. Does trending_equity HMM filter genuinely improve MNQ results?
4. Can MNQ standalone hit Sharpe 2.0 on 3-year backtest?
5. Is MGC + MNQ regime-switched portfolio stronger than either alone?

### Target
Sharpe ≥ 2.0 on 3-year backtest. MLL never breached.
Eval profit target ($6,000) within 6-9 months.

---

## Strategy Inventory

| Strategy              | Status      | 3yr Sharpe | Notes                        |
|-----------------------|-------------|------------|------------------------------|
| MGC HTC follow (2ct)  | BASELINE    | 1.31       | Thin edge, viable but slow   |
| MGC + HMM rotation    | TESTED      | 1.74       | Marginal improvement         |
| MGC mean-reversion    | ABANDONED   | 0.29       | Regime detection failure     |
| MNQ momentum          | IN PROGRESS | TBD        | Session 6 target             |

---

## Lucid LucidFlex 100K Rules
- Profit target: $6,000
- MLL: floor = min(peak - $3,000, $100,000)
- Consistency rule: 50% during eval only (soft gate)
- Once funded: no consistency rule, 90/10 split
- Hard close: 4:45 PM ET
- MGC ($0.80 RT) approved, max 60 micro contracts

## Git Log
```
d91cd2a  Session 3: MGC HTC strategy, MNQ signal stack
e9e113e  Session 4: Wasserstein HMM 6-instrument, 3yr history
[latest] Session 5: Direction 2 abandoned — regime detection failure
```
