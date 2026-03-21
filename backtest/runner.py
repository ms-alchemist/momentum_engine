"""
Backtest Runner
===============
Feeds 12 months of MCL/MGC 30-min bars through the signal engine
bar by bar, simulates realistic fills, and produces a full
performance report.

Design principles:
  - Zero lookahead bias: signal at bar N only uses data from bars 0..N
  - Realistic fills: entry at next bar's open after signal fires
  - Commission drag applied on every trade (Almgren-Chriss threshold enforced)
  - Lucid account constraints simulated exactly as live system would enforce
  - Results saved to backtest/results/ as CSV + summary report

Performance metrics reported:
  - Total return, annualised return
  - Sharpe ratio (annualised, risk-free = 0)
  - Max drawdown (dollar and percent)
  - Win rate, avg winner, avg loser, profit factor
  - Avg R-multiple (how many R per winning trade)
  - Trades per day, avg hold time in bars
  - Breakeven win rate vs actual win rate (edge quantification)
  - Per-instrument breakdown (MCL vs MGC)
  - Monthly P&L table (consistency check)

References:
  Moskowitz et al. (2012) — TSMOM signal
  Barroso & Santa-Clara (2015) — vol scaling
  Daniel & Moskowitz (2016) — panic regime gate
  Almgren & Chriss (2001) — execution cost threshold
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional

from momentum_engine.config.settings import INSTRUMENTS, ACCOUNT, RISK_CONFIG
from momentum_engine.signals.engine import MomentumSignalEngine, SignalDirection
from momentum_engine.risk.manager import RiskManager, AccountState
from momentum_engine.execution.threshold import SignalThresholdEngine

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Complete record of a single simulated trade."""
    symbol:          str
    entry_time:      pd.Timestamp
    exit_time:       pd.Timestamp
    direction:       str           # 'long' or 'short'
    contracts:       int
    entry_price:     float
    exit_price:      float
    exit_reason:     str           # 'target', 'stop', 'eod_close', 'signal_exit'
    stop_price:      float
    target_price:    float
    gross_pnl:       float         # before commission
    commission:      float         # round-trip commission
    net_pnl:         float         # gross - commission
    r_multiple:      float         # net_pnl / (stop_pts * point_value * contracts)
    hold_bars:       int
    signal_score:    float
    regime:          str

    @property
    def won(self) -> bool:
        return self.net_pnl > 0

    @property
    def dollar_risk(self) -> float:
        spec = INSTRUMENTS[self.symbol]
        stop_pts = abs(self.entry_price - self.stop_price)
        return self.contracts * stop_pts * spec.point_value


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    Bar-by-bar backtest simulation.

    For each bar:
      1. Check if open position needs to be managed (stop/target/EOD)
      2. Compute signal on bars up to and including current bar
      3. If no position and signal fires + threshold clears → queue entry
      4. Entry executes at NEXT bar's open (zero lookahead)
      5. Update account state

    EOD close: any open position is closed at 16:25 bar close
    (simulating our 16:30 strategy close before Lucid's 16:45 auto-liq)
    """

    def __init__(self, starting_balance: float = 100_000.0):
        self.starting_balance  = starting_balance
        self.signal_engines    = {
            "MCL": MomentumSignalEngine("MCL"),
            "MGC": MomentumSignalEngine("MGC"),
        }
        self.risk_manager      = SignalThresholdEngine()
        self.rm                = RiskManager()
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[dict]  = []

    def run(
        self,
        bars: dict[str, pd.DataFrame],
        verbose: bool = True,
    ) -> "BacktestResults":
        """
        Run the full backtest across all instruments simultaneously.

        Args:
            bars:    dict of {symbol: DataFrame} from data/fetcher.py
            verbose: print progress every 500 bars

        Returns:
            BacktestResults with full performance analysis
        """
        if verbose:
            print("\nStarting backtest...")
            for sym, df in bars.items():
                print(f"  {sym}: {len(df):,} bars "
                      f"({df.index[0].date()} → {df.index[-1].date()})")

        # Align all instruments to the same timeline
        # Use MCL as the master index (more bars, more liquid session)
        master_index = bars["MCL"].index
        balance      = self.starting_balance
        peak_balance = self.starting_balance

        # Track open positions per instrument
        open_positions: dict[str, Optional[dict]] = {
            "MCL": None, "MGC": None
        }

        # Daily tracking for Lucid constraints
        current_date      = None
        daily_pnl         = 0.0
        daily_trades      = 0
        cumulative_pnl    = 0.0
        best_day_pnl      = 0.0

        # Pending entries (queued for next bar open)
        pending_entries: dict[str, Optional[dict]] = {
            "MCL": None, "MGC": None
        }

        total_bars = len(master_index)

        for bar_idx, timestamp in enumerate(master_index):
            # Progress reporting
            if verbose and bar_idx % 500 == 0 and bar_idx > 0:
                pct = bar_idx / total_bars * 100
                print(f"  [{pct:.0f}%] Bar {bar_idx:,}/{total_bars:,} | "
                      f"Balance: ${balance:,.0f} | "
                      f"Trades: {len(self.trades)}")

            # Day boundary reset
            bar_date = timestamp.date()
            if bar_date != current_date:
                if current_date is not None:
                    # Record EOD equity
                    self.equity_curve.append({
                        "date":    current_date,
                        "balance": balance,
                        "daily_pnl": daily_pnl,
                        "drawdown": balance - peak_balance,
                    })
                    best_day_pnl = max(best_day_pnl, daily_pnl)
                    peak_balance = max(peak_balance, balance)

                current_date = bar_date
                daily_pnl    = 0.0
                daily_trades = 0

            # Daily loss circuit breaker
            mll_floor        = peak_balance - ACCOUNT.max_loss_limit
            daily_loss_limit = ACCOUNT.max_loss_limit * RISK_CONFIG.max_daily_loss_pct
            circuit_broken   = (
                daily_pnl < -daily_loss_limit or
                balance < mll_floor
            )

            # EOD close check — close all positions before 16:30
            is_eod = (timestamp.hour == 16 and timestamp.minute >= 25)

            # --- Process each instrument ---
            for symbol in ["MCL", "MGC"]:
                spec = INSTRUMENTS[symbol]

                # Get bars up to current bar for this instrument
                sym_bars = bars.get(symbol)
                if sym_bars is None:
                    continue

                # Find current bar index in this instrument's timeline
                if timestamp not in sym_bars.index:
                    continue

                sym_bar_idx  = sym_bars.index.get_loc(timestamp)
                bars_to_here = sym_bars.iloc[: sym_bar_idx + 1]
                current_bar  = sym_bars.iloc[sym_bar_idx]
                current_price = float(current_bar["close"])

                # --- Execute pending entry from previous bar ---
                if pending_entries[symbol] is not None and not circuit_broken:
                    entry = pending_entries[symbol]
                    # Execute at this bar's open
                    exec_price = float(current_bar["open"])
                    entry["entry_price"]  = exec_price
                    entry["entry_time"]   = timestamp

                    if entry["direction"] == "long":
                        entry["stop_price"]   = exec_price - entry["stop_pts"]
                        entry["target_price"] = exec_price + entry["target_pts"]
                    else:
                        entry["stop_price"]   = exec_price + entry["stop_pts"]
                        entry["target_price"] = exec_price - entry["target_pts"]

                    open_positions[symbol] = entry
                    pending_entries[symbol] = None

                # --- Manage open position ---
                pos = open_positions[symbol]
                if pos is not None:
                    exit_reason = None
                    exit_price  = current_price

                    if is_eod:
                        exit_reason = "eod_close"
                        exit_price  = current_price

                    elif pos["direction"] == "long":
                        if current_bar["low"] <= pos["stop_price"]:
                            exit_reason = "stop"
                            exit_price  = pos["stop_price"]
                        elif current_bar["high"] >= pos["target_price"]:
                            exit_reason = "target"
                            exit_price  = pos["target_price"]

                    elif pos["direction"] == "short":
                        if current_bar["high"] >= pos["stop_price"]:
                            exit_reason = "stop"
                            exit_price  = pos["stop_price"]
                        elif current_bar["low"] <= pos["target_price"]:
                            exit_reason = "target"
                            exit_price  = pos["target_price"]

                    if exit_reason:
                        # Calculate P&L
                        direction_mult = 1 if pos["direction"] == "long" else -1
                        price_diff     = (exit_price - pos["entry_price"]) * direction_mult
                        gross_pnl      = price_diff * spec.point_value * pos["contracts"]
                        commission     = spec.commission_rt * pos["contracts"]
                        net_pnl        = gross_pnl - commission

                        # R-multiple
                        risk_per_trade = pos["stop_pts"] * spec.point_value * pos["contracts"]
                        r_multiple     = net_pnl / risk_per_trade if risk_per_trade > 0 else 0

                        hold_bars = sym_bar_idx - pos.get("entry_bar_idx", sym_bar_idx)

                        trade = TradeRecord(
                            symbol       = symbol,
                            entry_time   = pos["entry_time"],
                            exit_time    = timestamp,
                            direction    = pos["direction"],
                            contracts    = pos["contracts"],
                            entry_price  = pos["entry_price"],
                            exit_price   = exit_price,
                            exit_reason  = exit_reason,
                            stop_price   = pos["stop_price"],
                            target_price = pos["target_price"],
                            gross_pnl    = gross_pnl,
                            commission   = commission,
                            net_pnl      = net_pnl,
                            r_multiple   = r_multiple,
                            hold_bars    = hold_bars,
                            signal_score = pos.get("signal_score", 0.0),
                            regime       = pos.get("regime", "normal"),
                        )

                        self.trades.append(trade)
                        balance      += net_pnl
                        daily_pnl    += net_pnl
                        cumulative_pnl += net_pnl
                        daily_trades += 1
                        open_positions[symbol] = None

                # --- Generate new signal (only if no position and no circuit break) ---
                if (open_positions[symbol] is None
                        and pending_entries[symbol] is None
                        and not circuit_broken
                        and not is_eod
                        and daily_trades < RISK_CONFIG.max_daily_trades
                        and len(bars_to_here) >= self.signal_engines[symbol].cfg.ewma_slow_span):

                    # Build account state for risk manager
                    account_state = AccountState(
                        balance            = balance,
                        peak_eod_balance   = peak_balance,
                        realized_pnl_today = daily_pnl,
                        unrealized_pnl     = 0.0,
                        trades_today       = daily_trades,
                        best_day_pnl       = best_day_pnl,
                        cumulative_eval_pnl = cumulative_pnl,
                        open_positions     = {k: (1 if v else 0)
                                              for k, v in open_positions.items()},
                        is_eval_phase      = True,
                    )

                    # Compute signal
                    signal = self.signal_engines[symbol].compute(bars_to_here)

                    # Risk manager decision
                    decision = self.rm.evaluate(signal, account_state, bars_to_here)

                    # Queue entry for next bar if trade approved
                    if decision.is_trade:
                        pending_entries[symbol] = {
                            "direction":     "long" if decision.action == "buy" else "short",
                            "contracts":     decision.contracts,
                            "stop_pts":      decision.stop_points,
                            "target_pts":    decision.target_points,
                            "entry_price":   None,
                            "entry_time":    None,
                            "stop_price":    None,
                            "target_price":  None,
                            "entry_bar_idx": sym_bar_idx + 1,
                            "signal_score":  signal.combined_score,
                            "regime":        signal.regime,
                        }

        # Final EOD record
        self.equity_curve.append({
            "date":      current_date,
            "balance":   balance,
            "daily_pnl": daily_pnl,
            "drawdown":  balance - peak_balance,
        })

        if verbose:
            print(f"  [100%] Backtest complete | "
                  f"Final balance: ${balance:,.0f} | "
                  f"Total trades: {len(self.trades)}")

        return BacktestResults(
            trades        = self.trades,
            equity_curve  = pd.DataFrame(self.equity_curve).set_index("date"),
            starting_balance = self.starting_balance,
            final_balance    = balance,
        )


# ---------------------------------------------------------------------------
# Results analysis
# ---------------------------------------------------------------------------

@dataclass
class BacktestResults:
    trades:           list[TradeRecord]
    equity_curve:     pd.DataFrame
    starting_balance: float
    final_balance:    float

    def summary(self) -> str:
        if not self.trades:
            return "No trades executed."

        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        winners   = trades_df[trades_df["net_pnl"] > 0]
        losers    = trades_df[trades_df["net_pnl"] <= 0]
        n         = len(trades_df)

        # Core metrics
        total_pnl      = trades_df["net_pnl"].sum()
        win_rate       = len(winners) / n * 100
        avg_winner     = winners["net_pnl"].mean() if len(winners) > 0 else 0
        avg_loser      = losers["net_pnl"].mean()  if len(losers)  > 0 else 0
        profit_factor  = (winners["net_pnl"].sum() / abs(losers["net_pnl"].sum())
                          if len(losers) > 0 and losers["net_pnl"].sum() != 0 else float('inf'))
        avg_r          = trades_df["r_multiple"].mean()
        total_comm     = trades_df["commission"].sum()

        # Drawdown
        eq = self.equity_curve
        max_dd         = eq["drawdown"].min()
        max_dd_pct     = max_dd / self.starting_balance * 100

        # Sharpe (annualised, assuming ~252 trading days)
        daily_rets     = eq["daily_pnl"] / self.starting_balance
        sharpe         = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)
                          if daily_rets.std() > 0 else 0)

        # Annualised return
        n_days         = len(eq)
        annual_ret     = (self.final_balance / self.starting_balance) ** (252 / max(n_days, 1)) - 1

        # Trades per day
        trades_per_day = n / max(n_days, 1)

        # Avg hold time
        avg_hold       = trades_df["hold_bars"].mean()

        # Breakeven win rate at avg R:R
        avg_win_pts    = avg_winner / (INSTRUMENTS["MCL"].point_value *
                         trades_df["contracts"].mean()) if avg_winner > 0 else 0
        avg_loss_pts   = abs(avg_loser) / (INSTRUMENTS["MCL"].point_value *
                         trades_df["contracts"].mean()) if avg_loser != 0 else 1
        breakeven_wr   = avg_loss_pts / (avg_win_pts + avg_loss_pts) * 100

        # Exit reason breakdown
        exit_counts    = trades_df["exit_reason"].value_counts()

        # Per-instrument breakdown
        mcl_trades     = trades_df[trades_df["symbol"] == "MCL"]
        mgc_trades     = trades_df[trades_df["symbol"] == "MGC"]

        # Monthly P&L
        eq_copy        = eq.copy()
        eq_copy.index  = pd.to_datetime(eq_copy.index)
        monthly_pnl    = eq_copy["daily_pnl"].resample("ME").sum()

        lines = [
            "",
            "=" * 60,
            "  BACKTEST RESULTS — MCL/MGC Momentum Strategy",
            "=" * 60,
            "",
            "  ACCOUNT",
            f"    Starting balance:  ${self.starting_balance:>10,.0f}",
            f"    Final balance:     ${self.final_balance:>10,.0f}",
            f"    Total P&L:         ${total_pnl:>+10,.0f}",
            f"    Total commission:  ${total_comm:>10,.0f}",
            f"    Annualised return: {annual_ret*100:>+9.1f}%",
            "",
            "  RISK",
            f"    Max drawdown:      ${max_dd:>10,.0f}  ({max_dd_pct:.1f}%)",
            f"    Sharpe ratio:      {sharpe:>10.2f}",
            f"    MLL limit:         ${-ACCOUNT.max_loss_limit:>10,.0f}",
            f"    Drawdown vs MLL:   {'SAFE' if abs(max_dd) < ACCOUNT.max_loss_limit else 'BREACH'}",
            "",
            "  TRADE STATISTICS",
            f"    Total trades:      {n:>10,}",
            f"    Trades per day:    {trades_per_day:>10.2f}",
            f"    Win rate:          {win_rate:>9.1f}%",
            f"    Breakeven WR:      {breakeven_wr:>9.1f}%",
            f"    Edge:              {win_rate - breakeven_wr:>+9.1f}%  ({'POSITIVE' if win_rate > breakeven_wr else 'NEGATIVE'})",
            f"    Avg winner:        ${avg_winner:>+10,.0f}",
            f"    Avg loser:         ${avg_loser:>+10,.0f}",
            f"    Profit factor:     {profit_factor:>10.2f}",
            f"    Avg R-multiple:    {avg_r:>10.2f}",
            f"    Avg hold (bars):   {avg_hold:>10.1f}",
            "",
            "  EXIT REASONS",
        ]
        for reason, count in exit_counts.items():
            pct = count / n * 100
            lines.append(f"    {reason:<20} {count:>5,}  ({pct:.0f}%)")

        lines += [
            "",
            "  PER-INSTRUMENT",
            f"    MCL trades:        {len(mcl_trades):>10,}  "
            f"P&L: ${mcl_trades['net_pnl'].sum():>+,.0f}  "
            f"WR: {len(mcl_trades[mcl_trades['net_pnl']>0])/max(len(mcl_trades),1)*100:.0f}%",
            f"    MGC trades:        {len(mgc_trades):>10,}  "
            f"P&L: ${mgc_trades['net_pnl'].sum():>+,.0f}  "
            f"WR: {len(mgc_trades[mgc_trades['net_pnl']>0])/max(len(mgc_trades),1)*100:.0f}%",
            "",
            "  MONTHLY P&L",
        ]
        for month, pnl in monthly_pnl.items():
            bar    = "+" * int(abs(pnl) / 100) if pnl >= 0 else "-" * int(abs(pnl) / 100)
            bar    = bar[:30]
            sign   = "+" if pnl >= 0 else ""
            lines.append(f"    {str(month)[:7]}   {sign}${pnl:>7,.0f}  {bar}")

        lines += ["", "=" * 60, ""]
        return "\n".join(lines)

    def save(self, label: str = ""):
        """Save trades and equity curve to CSV."""
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag   = f"_{label}" if label else ""
        trades_path = RESULTS_DIR / f"trades{tag}_{ts}.csv"
        equity_path = RESULTS_DIR / f"equity{tag}_{ts}.csv"

        pd.DataFrame([t.__dict__ for t in self.trades]).to_csv(trades_path)
        self.equity_curve.to_csv(equity_path)

        print(f"  Saved trades:       {trades_path.name}")
        print(f"  Saved equity curve: {equity_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load cached bars (fetcher.py must have been run first)
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    mcl_path  = cache_dir / "MCL_30min.parquet"
    mgc_path  = cache_dir / "MGC_30min.parquet"

    if not mcl_path.exists() or not mgc_path.exists():
        print("Cache not found. Run data/fetcher.py first.")
        sys.exit(1)

    print("Loading cached bars...")
    bars = {
        "MCL": pd.read_parquet(mcl_path),
        "MGC": pd.read_parquet(mgc_path),
    }
    print(f"  MCL: {len(bars['MCL']):,} bars")
    print(f"  MGC: {len(bars['MGC']):,} bars")

    # Run backtest
    runner  = BacktestRunner(starting_balance=ACCOUNT.account_size)
    results = runner.run(bars, verbose=True)

    # Print and save results
    print(results.summary())
    results.save(label="MCL_MGC_12mo")

    print("Done. Review results in backtest/results/")
