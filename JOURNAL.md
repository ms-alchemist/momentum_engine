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

