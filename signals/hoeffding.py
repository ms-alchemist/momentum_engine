"""
signals/hoeffding.py
====================
Hoeffding Regime Change Detector — Phase 2

Based on: Egger & Vestal (2025) arXiv:2512.08851
"A New Application of Hoeffding's Inequality Can Give Traders
Early Warning of Financial Regime Change"

Core idea:
  Our strategy has an expected win rate μ (e.g. 0.621 for MGC).
  After each trade, we compute the maximum probability that the
  observed deviation from μ is due to chance alone.

  P[X̄ - μ ≥ t] ≤ exp(-2t²N)    [Hoeffding bound, binary case]

  Where:
    μ   = expected win rate (from signal audit)
    X̄   = observed win rate over last N trades
    t   = μ - X̄  (deviation below expectation)
    N   = number of recent trades in window
    H   = exp(-2t²N) = Hoeffding probability

  Interpretation:
    H > 0.50  → regime likely intact, trade normally
    H < 0.50  → regime may have shifted, monitor closely
    H < 0.25  → significant evidence of regime change, reduce to 1 contract
    H < 0.10  → regime change likely, halt instrument, check rotation

Usage:
    from signals.hoeffding import HoeffdingDetector
    detector = HoeffdingDetector(mu=0.621, window=20)
    detector.update(win=True)
    state = detector.state   # 'normal', 'caution', 'reduce', 'halt'
    prob  = detector.h_prob  # current Hoeffding probability

    # Or run standalone backtest analysis:
    python signals/hoeffding.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Expected win rates from signal audit (session 3)
MU = {
    "MGC": 0.461,   # 46.1% realized win rate (backtest, 22pt stop HTC)
    "MNQ": 0.481,   # 48.1% realized win rate (backtest)
    "MES": 0.481,   # assume same as MNQ
    "MYM": 0.452,   # 45.2% realized win rate
}

# Hoeffding probability thresholds
THRESHOLD_CAUTION = 0.50   # monitor — more likely than not regime shifted
THRESHOLD_REDUCE  = 0.25   # reduce size — significant evidence of regime change
THRESHOLD_HALT    = 0.10   # halt instrument — regime change likely

# Rolling window for win rate calculation
DEFAULT_WINDOW = 20   # last 20 trades (~1 month of daily trades)
MIN_TRADES     = 5    # minimum trades before detector activates


# ---------------------------------------------------------------------------
# Core Hoeffding computation
# ---------------------------------------------------------------------------

def hoeffding_prob(observed_win_rate: float,
                   expected_win_rate: float,
                   n_trades: int) -> float:
    """
    Compute Hoeffding probability that observed deviation is due to chance.

    Formula: H = exp(-2 * t² * N)
    Where t = max(0, μ - X̄)  [one-sided: only flag when below expectation]

    Returns probability in [0, 1]:
      High (→1): deviation consistent with chance, regime likely intact
      Low  (→0): deviation unlikely to be chance, regime likely changed
    """
    if n_trades < MIN_TRADES:
        return 1.0  # not enough data — assume regime intact

    t = max(0.0, expected_win_rate - observed_win_rate)
    if t == 0:
        return 1.0  # performing at or above expectation

    h = np.exp(-2.0 * t**2 * n_trades)
    return float(np.clip(h, 0.0, 1.0))


def regime_state(h_prob: float) -> str:
    """Map Hoeffding probability to regime state."""
    if h_prob >= THRESHOLD_CAUTION:
        return "normal"
    elif h_prob >= THRESHOLD_REDUCE:
        return "caution"
    elif h_prob >= THRESHOLD_HALT:
        return "reduce"
    else:
        return "halt"


def contract_scalar(state: str) -> float:
    """Position size scalar based on regime state."""
    return {"normal": 1.0, "caution": 0.75, "reduce": 0.5, "halt": 0.0}[state]


# ---------------------------------------------------------------------------
# Detector class — used in live trading and backtesting
# ---------------------------------------------------------------------------

@dataclass
class HoeffdingState:
    n_trades:     int
    n_wins:       int
    win_rate:     float
    h_prob:       float
    state:        str
    size_scalar:  float
    t_deviation:  float


class HoeffdingDetector:
    """
    Rolling Hoeffding regime change detector.

    Maintains a rolling window of trade outcomes and computes
    the Hoeffding probability after each trade.

    Args:
        mu:     Expected win rate (from historical signal audit)
        window: Rolling window size (default 20 trades)
        symbol: Instrument name for logging
    """

    def __init__(self, mu: float, window: int = DEFAULT_WINDOW,
                 symbol: str = ""):
        self.mu      = mu
        self.window  = window
        self.symbol  = symbol
        self._outcomes: list[int] = []  # 1=win, 0=loss

    def update(self, win: bool) -> HoeffdingState:
        """Record a trade outcome and return current state."""
        self._outcomes.append(1 if win else 0)
        # Keep only rolling window
        if len(self._outcomes) > self.window:
            self._outcomes = self._outcomes[-self.window:]
        return self.current_state

    @property
    def current_state(self) -> HoeffdingState:
        n = len(self._outcomes)
        if n == 0:
            return HoeffdingState(0, 0, 0.0, 1.0, "normal", 1.0, 0.0)

        n_wins   = sum(self._outcomes)
        win_rate = n_wins / n
        h_prob   = hoeffding_prob(win_rate, self.mu, n)
        state    = regime_state(h_prob)
        t_dev    = max(0.0, self.mu - win_rate)

        return HoeffdingState(
            n_trades    = n,
            n_wins      = n_wins,
            win_rate    = win_rate,
            h_prob      = h_prob,
            state       = state,
            size_scalar = contract_scalar(state),
            t_deviation = t_dev,
        )

    def reset(self):
        """Reset detector (e.g. after switching instruments)."""
        self._outcomes = []


# ---------------------------------------------------------------------------
# Backtest analysis — apply to existing MGC HTC results
# ---------------------------------------------------------------------------

def backtest_hoeffding(trades_df: pd.DataFrame,
                       symbol: str,
                       window: int = DEFAULT_WINDOW,
                       n_contracts_base: int = 2) -> pd.DataFrame:
    """
    Replay Hoeffding detector on historical trades.
    Shows when it would have flagged regime changes and
    how much P&L would have been preserved.
    """
    mu       = MU.get(symbol, 0.55)
    detector = HoeffdingDetector(mu=mu, window=window, symbol=symbol)

    results = []
    for _, row in trades_df.iterrows():
        # State BEFORE this trade (what we knew going in)
        pre_state = detector.current_state

        # Record the trade outcome
        post_state = detector.update(bool(row["win"]))

        # Actual P&L — scaled by position size from Hoeffding
        actual_pnl    = row["net_pnl"]
        contracts_used = max(1, round(n_contracts_base * pre_state.size_scalar))
        # Rescale P&L to reflect Hoeffding-adjusted size
        # (original was at n_contracts_base, we're using contracts_used)
        hoeffding_pnl = actual_pnl * (contracts_used / n_contracts_base)

        results.append({
            "date":            row.get("date", row.get("entry_time", "unknown")),
            "win":             row["win"],
            "actual_pnl":      actual_pnl,
            "hoeffding_pnl":   hoeffding_pnl,
            "pre_h_prob":      pre_state.h_prob,
            "pre_state":       pre_state.state,
            "pre_win_rate":    pre_state.win_rate,
            "post_h_prob":     post_state.h_prob,
            "post_state":      post_state.state,
            "contracts_used":  contracts_used,
            "size_scalar":     pre_state.size_scalar,
        })

    return pd.DataFrame(results)


def print_backtest_analysis(results_df: pd.DataFrame, symbol: str, window: int = DEFAULT_WINDOW):
    """Print analysis of Hoeffding detector performance."""
    if results_df.empty:
        print("No results to analyze.")
        return

    print()
    print("=" * 65)
    print(f"  HOEFFDING BACKTEST ANALYSIS — {symbol}")
    print(f"  μ = {MU.get(symbol, 0.461):.1%}  window = {window} trades")
    print("=" * 65)

    # State distribution
    state_counts = results_df["pre_state"].value_counts()
    total        = len(results_df)
    print(f"\n  State distribution ({total} trades):")
    for state in ["normal", "caution", "reduce", "halt"]:
        n    = state_counts.get(state, 0)
        pct  = n / total * 100
        print(f"    {state:<10} {n:>4} trades  ({pct:>5.1f}%)")

    # P&L comparison
    actual_total   = results_df["actual_pnl"].sum()
    hoeffding_total = results_df["hoeffding_pnl"].sum()
    improvement    = hoeffding_total - actual_total

    print(f"\n  P&L comparison (2-contract baseline):")
    print(f"    Without Hoeffding:  ${actual_total:>8,.0f}")
    print(f"    With Hoeffding:     ${hoeffding_total:>8,.0f}")
    print(f"    Improvement:        ${improvement:>8,.0f}  "
          f"({improvement/abs(actual_total)*100:+.1f}%)" if actual_total != 0
          else f"    Improvement:        ${improvement:>8,.0f}")

    # Monthly breakdown
    results_df2 = results_df.copy()
    results_df2["date"] = pd.to_datetime(results_df2["date"])
    monthly = results_df2.groupby(
        results_df2["date"].dt.to_period("M")
    ).agg(
        trades         = ("win", "count"),
        win_rate       = ("win", "mean"),
        actual_pnl     = ("actual_pnl", "sum"),
        hoeffding_pnl  = ("hoeffding_pnl", "sum"),
        avg_h_prob     = ("pre_h_prob", "mean"),
        pct_normal     = ("pre_state", lambda x: (x == "normal").mean()),
    )

    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':<10} {'Trades':>6} {'WR':>6} {'Actual':>9} "
          f"{'Hoeff':>9} {'AvgH':>6} {'%Normal':>8}")
    print("  " + "-" * 62)

    for period, row in monthly.iterrows():
        delta = row["hoeffding_pnl"] - row["actual_pnl"]
        flag  = " ↑" if delta > 100 else (" ↓" if delta < -100 else "  ")
        print(f"  {str(period):<10} {row['trades']:>6.0f}  "
              f"{row['win_rate']:>5.1%}  "
              f"${row['actual_pnl']:>7,.0f}  "
              f"${row['hoeffding_pnl']:>7,.0f}  "
              f"{row['avg_h_prob']:>5.2f}  "
              f"{row['pct_normal']:>7.1%}{flag}")

    # Key regime change events
    halts   = results_df[results_df["pre_state"] == "halt"]
    reduces = results_df[results_df["pre_state"] == "reduce"]

    if len(halts) > 0:
        print(f"\n  HALT signals ({len(halts)} trades skipped/halved):")
        for _, row in halts.iterrows():
            print(f"    {row['date']}  H={row['pre_h_prob']:.3f}  "
                  f"WR={row['pre_win_rate']:.1%}  "
                  f"actual_pnl=${row['actual_pnl']:+.0f}")

    # Sharpe comparison
    daily_actual    = results_df.groupby("date")["actual_pnl"].sum()
    daily_hoeffding = results_df.groupby("date")["hoeffding_pnl"].sum()

    sharpe_actual = ((daily_actual.mean() / daily_actual.std() * np.sqrt(252))
                     if daily_actual.std() > 0 else 0)
    sharpe_hoeff  = ((daily_hoeffding.mean() / daily_hoeffding.std() * np.sqrt(252))
                     if daily_hoeffding.std() > 0 else 0)

    print(f"\n  Sharpe comparison:")
    print(f"    Without Hoeffding:  {sharpe_actual:.2f}")
    print(f"    With Hoeffding:     {sharpe_hoeff:.2f}")
    print(f"    Improvement:        {sharpe_hoeff - sharpe_actual:+.2f}")
    print()


# ---------------------------------------------------------------------------
# Main — run standalone analysis on MGC HTC backtest results
# ---------------------------------------------------------------------------

def main():
    results_dir = Path("backtest/results")

    print()
    print("Hoeffding Regime Detector — Backtest Analysis")
    print("=" * 55)
    print(f"Based on: Egger & Vestal (2025) arXiv:2512.08851")
    print()

    # Test on MGC HTC results (2-contract, 22pt stop)
    files = sorted(results_dir.glob("mgc_htc22_2ct_*.csv"))
    if not files:
        files = sorted(results_dir.glob("mgc_htc_2ct_*.csv"))
    if not files:
        files = sorted(results_dir.glob("mgc_htc*2ct*.csv"))

    if not files:
        print("No MGC HTC 2-contract results found.")
        print("Run backtest/runner_mgc_htc_22.py first.")
        return

    df = pd.read_csv(files[-1])
    print(f"Loaded {len(df)} trades from {files[-1].name}")

    # Ensure net_pnl column exists
    if "net_pnl" not in df.columns and "pnl" in df.columns:
        df["net_pnl"] = df["pnl"]

    # Run analysis with different window sizes
    for window in [10, 15, 20, 30]:
        results = backtest_hoeffding(df, "MGC", window=window, n_contracts_base=2)
        print_backtest_analysis(results, f"MGC (window={window})", window=window)

    # Also test on MNQ if available
    mnq_files = sorted(results_dir.glob("mnq_full_1ct_*.csv"))
    if mnq_files:
        df_mnq = pd.read_csv(mnq_files[-1])
        if "net_pnl" not in df_mnq.columns and "pnl" in df_mnq.columns:
            df_mnq["net_pnl"] = df_mnq["pnl"]
        results_mnq = backtest_hoeffding(
            df_mnq, "MNQ", window=DEFAULT_WINDOW, n_contracts_base=1)
        print_backtest_analysis(results_mnq, "MNQ (window=20)", window=DEFAULT_WINDOW)

    print("Hoeffding detector ready for integration.")
    print("Import: from signals.hoeffding import HoeffdingDetector")
    print()


if __name__ == "__main__":
    main()
