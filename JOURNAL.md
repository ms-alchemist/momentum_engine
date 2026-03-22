\# Strategy Development Journal

\## MCL/MGC Intraday Momentum Strategy — LucidFlex 100K



\---



\## 2026-03-21 — Session 1: Signal Engine Build



\### What was built

A complete algorithmic momentum signal engine for Micro Crude Oil (MCL)

and Micro Gold (MGC) futures, targeting the Lucid Trading LucidFlex 100K

prop firm evaluation account.



\*\*Modules created:\*\*

\- `config/settings.py` — Single source of truth for all constants.

&#x20; Instrument specs (point values, tick sizes, commissions), account

&#x20; constraints (MLL, profit target, max contracts), signal parameters,

&#x20; risk parameters, and Almgren-Chriss execution config. Nothing is

&#x20; hardcoded in logic modules — everything imports from here.



\- `signals/engine.py` — The four-paper signal stack unified into one

&#x20; pipeline. Each layer is independently auditable via the layers dict

&#x20; on SignalOutput. All signal outputs are in the range \[-1, +1] with

&#x20; an expected move magnitude in points for the threshold engine.



\- `execution/threshold.py` — Almgren-Chriss (2001) cost decomposition.

&#x20; Computes the minimum signal strength required to justify a trade after

&#x20; accounting for commission drag, temporary market impact, and permanent

&#x20; market impact. Includes a scaling table showing how costs rise with

&#x20; contract count — the natural ceiling for position sizing.



\- `risk/manager.py` — Translates signal output into a sized, validated

&#x20; trade decision. Enforces all Lucid LucidFlex 100K constraints: EOD

&#x20; max loss limit ($3,000), daily trade circuit breaker, consistency rule

&#x20; (best day ≤ 50% of cumulative eval P\&L), MLL buffer check, and

&#x20; volatility-targeted position sizing (Barroso \& Santa-Clara 2015).



\- `tests/test\_engine.py` — 33 tests across four classes validating every

&#x20; component with synthetic data.



\### Research foundation (9 papers)

1\. Bühler — Waves and Mean Flows (wave physics / pseudomomentum)

2\. Moskowitz, Ooi \& Pedersen (2012) — Time Series Momentum (TSMOM)

3\. Chevyrev \& Kormilitzin (2016) — Path Signature Method (feature encoding)

4\. Cont \& da Fonseca (2002) — Dynamics of Implied Volatility Surfaces

5\. Daniel \& Moskowitz (2016) — Momentum Crashes (regime gate / crash filter)

6\. Liu, Tsyvinski \& Wu (2022) — Common Risk Factors in Cryptocurrency

7\. Gao, Han, Li \& Zhou (2018) — Market Intraday Momentum

8\. Almgren \& Chriss (2001) — Optimal Execution of Portfolio Transactions

9\. Barroso \& Santa-Clara (2015) — Momentum Has Its Moments (vol scaling)



\### What the 33 tests cover



\*\*TestInstrumentSpecs (6 tests)\*\*

Validates core contract math for MCL and MGC — point values ($10/pt

each), tick arithmetic (tick\_size × point\_value = tick\_value), Lucid

commission rates ($0.50 MCL / $0.80 MGC round-trip), and account

limit constants ($3,000 MLL, $6,000 profit target, 60 micro max).



\*\*TestThresholdEngine (10 tests)\*\*

Validates the Almgren-Chriss cost decomposition. Confirms commission

conversion from dollars to points, that threshold rises with contract

count (market impact scaling), safety multiplier application (1.5×),

strong signals clear the gate, weak signals are blocked, zero signal

produces zero contracts, breakeven win rate is mathematically correct

at \~33.5% for 2:1 R:R, cost stays below 10% of stop at 1 contract,

and scaling table returns the correct number of rows.



\*\*TestSignalEngine (7 tests)\*\*

Validates the four-layer signal stack. Confirms insufficient data

returns FLAT (safety check), output structure is correct with enough

data, strong uptrend produces positive score, strong downtrend produces

negative score, panic regime blocks trading, all five layers appear in

output, and MCL/MGC engines operate independently.



\*\*TestRiskManager (7 tests)\*\*

Validates all Lucid constraint enforcement. Confirms HOLD on flat

signal, HOLD when daily trade limit reached, HOLD when position already

open, contracts never exceed account maximums, dollar risk within

per-trade limits, consistency rule caps size near the 50% daily limit,

and MLL buffer prevents oversizing when account is near the floor.



\*\*TestIntegration (3 tests)\*\*

Full pipeline end-to-end on synthetic data. MCL uptrend: 0.920 signal

score, LONG 1 contract, 8.75pt expected move, threshold cleared at

0.092pts. MGC downtrend: -0.286 signal score, SHORT 1 contract,

threshold cleared. Scaling table printed for manual review — MGC cost

crosses 20% of stop at \~10-11 contracts (natural scaling ceiling).



\### Key insight from scaling table

MCL is cheaper to scale than MGC. At 5 contracts MCL execution cost

is 3.0% of stop. MGC is 9.5% at the same size. MGC's natural ceiling

without a stronger signal is \~10 contracts. This is the Almgren-Chriss

efficient frontier applied to micro futures.



\### One bug fixed

`np.trapz` was renamed `np.trapezoid` in NumPy 2.0. The path signature

layer uses numerical integration for iterated integrals (Lévy area

computation). Fixed with:

`\_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')`

Now compatible with both old and new NumPy versions.



\### Infrastructure set up

\- Git repository initialized at C:\\Users\\skins\\projects\\momentum\_engine

\- Private GitHub repository at github.com/ms-alchemist/momentum\_engine

\- .gitignore protecting credentials, cache files, and log files

\- Windows PowerShell workflow established (one command per Enter)



\### Next session

1\. Install Databento Python library (`pip install databento`)

2\. Store Databento API key in .env file (never committed to Git)

3\. Build `data/fetcher.py` — pull 12 months of MCL + MGC 30-min

&#x20;  historical bars from Databento using continuous contract symbology

&#x20;  (MCL.c.0, MGC.c.0) on dataset GLBX.MDP3

4\. Build `data/bar\_builder.py` — cache bars locally so we don't pay

&#x20;  for the same data twice

5\. Build `backtest/runner.py` — feed historical bars through the signal

&#x20;  engine bar by bar, simulate fills and commissions realistically

6\. Review backtest results: win rate vs breakeven threshold, avg R,

&#x20;  max drawdown vs MLL limits, trades per day



\### Decision log

\- Instruments: MCL + MGC (micro crude oil + micro gold)

\- Timeframe: Intraday only (swing layer deferred to LucidLive phase)

\- Account: LucidFlex 100K ($157.50 eval fee, $3,000 MLL, $6,000 target)

\- Hard close: 4:30 PM ET (15 min before Lucid's 4:45 PM auto-liquidation)

\- Starting size: 2-5 micros per instrument

\- Data source: Databento GLBX.MDP3 (CME Globex MDP 3.0)

\- Prop firm path: Eval → Funded (6 payouts) → LucidLive → Swing layer



\---
---

## 2026-03-21 — Session 2: Data Layer Complete

### What was built
`data/fetcher.py` — Databento historical data fetcher with local cache.

### Data pulled
- MCL (Micro Crude Oil): 4,016 30-min bars, 2025-03-21 → 2026-03-20
- MGC (Micro Gold): 2,511 30-min bars, 2025-03-21 → 2026-03-20
- Source: Databento GLBX.MDP3, continuous contract (MCL.c.0, MGC.c.0)
- Cost: $0.00 (included in Standard plan)
- Cached locally as Parquet — not committed to Git

### Key observations from the data
- MCL price range: $54.98 → $104.56 (highly volatile year for crude)
- MGC price range: $2,960 → $5,542 (gold all-time highs — strong trend)
- Buy/sell volume split: ~49-50% (approximation method working correctly)
- 3 degraded quality days flagged (rollovers + Thanksgiving) — noted for
  backtest filter
- MCL has more bars than MGC (4,016 vs 2,511) due to extended session
  liquidity differences within 09:00-16:30 ET window

### Data quality notes
- BentoWarning on Sep 17, Sep 24, Nov 28 — degraded quality days
- Will add degraded day filter in backtest runner
- 1-min bars also cached locally for future reference

### Next session
Build backtest/runner.py — feed 12 months of bars through the signal
engine bar by bar, simulate fills and commissions, output equity curve,
win rate, avg R, max drawdown, and trades per day.


---

## 2026-03-21 — Session 3: Di Graziano Calibration + v4 Backtest Runner

### Research added
Di Graziano (2014) — Optimal Trading Stops and Algorithmic Trading.
Full paper reviewed. Key findings applied:
- Optimal stops derived by maximising expected discounted utility of P&L
- P&L modelled as Markov modulated diffusion (signal-active state → noise state)
- b/a ratio (target/stop) always > 1; grows with signal strength and signal life
- Fast-decaying signals → tighter stops; slow-decaying signals → wider stops
- Table 2 directly calibrated to MGC: normal regime stop 16pt, target 19pt
- Independent validation: Di Graziano's 16pt stop converged with our empirical
  90% survival threshold from session_test.py — two methods, same answer

Di Graziano calibration results (digraziano_calibrate.py):
- sigma (daily vol): 1.31% of price
- mu1 (signal drift): 0.25% of price
- Signal decay q (monthly): 0.066 (avg life 15.2 months — persistent signal)
- Signal decay q (daily): 0.0031 (avg life 318 trading days)
- Signal-to-noise ratio: 0.1879
- Implied b/a ratio: 1.20:1
- Di Graziano optimal stop: 15.7 pts → rounded to 16 pts
- Di Graziano optimal target: 18.8 pts → rounded to 19 pts
- Conservative EV: $44/contract/trade (vs earlier $65-78 estimate)

### Signal audit findings (from signal_audit.py — previous session)
- MCL dropped: avg daily range 1.64 pts, stop/target structurally unreachable
- MGC kept: avg daily range 41.79 pts, 83% of days see >10pt range
- First-bar accuracy: 58.3% (baseline 50%, target >55%) ✓
- Autocorrelation: MGC mean-reverts at 30-min resolution
  Lag 1 (30 min): -0.0417, Lag 2 (1 hr): -0.0296 → TSMOM layers add noise
- Primary edge is first-bar directional prediction, not bar-level TSMOM
- Monthly accuracy: Aug 71.4%, Apr 64.7%, Sep 65.1% (strong)
  Mar 2026 45.0% (extreme vol, 63.7pt avg range — filtered out)

### Files created this session
- `digraziano_calibrate.py` — standalone calibration script
- `signal_audit.py` — autocorrelation + monthly accuracy analysis
- `session_test.py` — stop distance survival analysis
- `backtest/runner_v4.py` — full v4 backtest with multi-contract comparison

### v3 → v4 changes (complete list)

**Instrument scope**
  v3: MCL + MGC simultaneously
  v4: MGC only — MCL daily range too small for stops/targets

**Signal source**
  v3: Full four-layer stack (intraday momentum, TSMOM, path signature,
      vol regime proxy) evaluated on every bar throughout session
  v4: Single signal — first 30-min bar direction (9:00-9:30 AM ET only)
  Reason: MGC autocorrelation is mean-reverting at 30-min resolution.
  TSMOM layers added noise. First-bar accuracy: 58.3% vs stack's 44-47%.

**Stop and target levels**
  v3: Fixed 10pt stop, 20pt target (both instruments)
  v4: Di Graziano regime-adaptive (MGC only):
      Low-vol  (<20pt daily range): stop 9pt,  target 12pt
      Normal   (20-50pt range):     stop 16pt, target 19pt  ← primary
      High-vol (>50pt range):       stop 22pt, target 26pt

**Volatility filter**
  v3: Session time filter (UTC 13:00-15:30, 19:00 entries only)
  v4: Vol ratio filter — skip if 5-day avg range > 2.5× 20-day avg range
  Effect: removes ~9% of extreme-vol days where win rate drops to 45%

**MLL mechanics**
  v3: Fixed $3,000 floor
  v4: Exact Lucid trailing mechanics:
      - Trails peak EOD balance at $3,000 below
      - Locks permanently at $100,000 once balance reaches that level
      - Buffer = current_balance - max($97,000, peak_EOD - $3,000)

**Multi-contract comparison**
  v3: Single contract size
  v4: 1, 2, 3, 4 MGC contracts run simultaneously on identical trade sequence
  Outputs per size: P&L, win rate, Sharpe, max drawdown, min MLL buffer,
                    worst losing streak, consistency flags, days to $6k target

**New reporting sections**
  - Min MLL buffer seen during backtest
  - Worst consecutive losing streak (days)
  - Lucid consistency flag count (best day > 50% of cumulative P&L)
  - Path to target: did each size hit $6,000 and in how many trading days
  - Regime breakdown: P&L by low-vol / normal / high-vol regime
  - Recommendation block: largest safe contract size + estimated eval timeline

### Expected value (Di Graziano conservative estimate)
  At 58.3% win rate, 19pt target, 16pt stop, $0.80 commission:
  EV = 0.583 × $190 - 0.417 × $160 - $0.80 = $43.88/trade/contract
  At ~1 trade/day, 20 trading days/month: ~$878/month at 1 contract

### Next step
Run: python backtest/runner_v4.py
Review multi-contract comparison output, confirm MLL safety at 2-3 contracts,
determine optimal contract size for fastest eval pass without MLL risk.
Commit results to JOURNAL.md after running.

