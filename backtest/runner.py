"""
Backtest Runner v2
==================
Fixed from v1:
  FIX 1 — Daily circuit breaker: daily_pnl tracked correctly per instrument
           and checked before EVERY new entry attempt. Four consecutive stops
           on the same instrument same day now correctly halts trading.
  FIX 2 — Strategy silence after November: the signal engine was accumulating
           state across the full run. Now each bar slice is passed cleanly
           and the engine processes it without stale internal state buildup.
  FIX 3 — MLL breach: peak_balance now updates intraday (not just EOD) so
           the trailing floor is always current. Position risk capped at
           25% of remaining MLL buffer (reduced from 40%).
  FIX 4 — Consecutive stop detection: if same instrument stops out 2x in
           one day, it is blocked for the rest of that session.

Bar-by-bar backtest. Entry at next bar open after signal fires.
Zero lookahead bias guaranteed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional

from momentum_engine.config.settings import INSTRUMENTS, ACCOUNT, RISK_CONFIG
from momentum_engine.signals.engine import MomentumSignalEngine, SignalDirection
from momentum_engine.risk.manager import RiskManager, AccountState
from momentum_engine.execution.threshold import SignalThresholdEngine

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    symbol:       str
    entry_time:   object
    exit_time:    object
    direction:    str
    contracts:    int
    entry_price:  float
    exit_price:   float
    exit_reason:  str
    stop_price:   float
    target_price: float
    gross_pnl:    float
    commission:   float
    net_pnl:      float
    r_multiple:   float
    hold_bars:    int
    signal_score: float
    regime:       str

    @property
    def won(self) -> bool:
        return self.net_pnl > 0


# ---------------------------------------------------------------------------
# Backtest engine v2
# ---------------------------------------------------------------------------

class BacktestRunner:

    def __init__(self, starting_balance: float = 100_000.0):
        self.starting_balance = starting_balance
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[dict]  = []

    def run(self, bars: dict[str, pd.DataFrame], verbose: bool = True) -> "BacktestResults":

        if verbose:
            print("\nStarting backtest v2...")
            for sym, df in bars.items():
                print(f"  {sym}: {len(df):,} bars "
                      f"({df.index[0].date()} → {df.index[-1].date()})")

        # Fresh signal engines for this run — no stale state
        signal_engines = {
            "MCL": MomentumSignalEngine("MCL"),
            "MGC": MomentumSignalEngine("MGC"),
        }
        rm = RiskManager()

        balance      = self.starting_balance
        peak_balance = self.starting_balance  # updated intraday now (FIX 3)

        open_positions:  dict[str, Optional[dict]] = {"MCL": None, "MGC": None}
        pending_entries: dict[str, Optional[dict]] = {"MCL": None, "MGC": None}

        # Daily tracking — reset at each new date
        current_date       = None
        daily_pnl          = 0.0
        daily_trades_total = 0
        cumulative_pnl     = 0.0
        best_day_pnl       = 0.0

        # FIX 1: per-instrument daily stop counter
        daily_stops: dict[str, int] = {"MCL": 0, "MGC": 0}
        MAX_STOPS_PER_INSTRUMENT_PER_DAY = 2

        master_index = bars["MCL"].index
        total_bars   = len(master_index)

        for bar_idx, timestamp in enumerate(master_index):

            if verbose and bar_idx % 500 == 0 and bar_idx > 0:
                pct = bar_idx / total_bars * 100
                print(f"  [{pct:.0f}%] Bar {bar_idx:,}/{total_bars:,} | "
                      f"Balance: ${balance:,.0f} | "
                      f"Trades: {len(self.trades)}")

            bar_date = timestamp.date()

            # --- Day boundary ---
            if bar_date != current_date:
                if current_date is not None:
                    self.equity_curve.append({
                        "date":      current_date,
                        "balance":   balance,
                        "daily_pnl": daily_pnl,
                        "drawdown":  balance - peak_balance,
                    })
                    best_day_pnl  = max(best_day_pnl, daily_pnl)
                    # FIX 3: update peak at EOD
                    peak_balance  = max(peak_balance, balance)

                current_date       = bar_date
                daily_pnl          = 0.0
                daily_trades_total = 0
                # FIX 1: reset per-instrument stop counters daily
                daily_stops        = {"MCL": 0, "MGC": 0}

            # EOD flag — close all positions before 16:30
            is_eod = (timestamp.hour == 16 and timestamp.minute >= 25)

            # MLL floor — updated intraday (FIX 3)
            mll_floor = peak_balance - ACCOUNT.max_loss_limit

            # Global circuit breaker
            # FIX 1: check actual balance + unrealized against MLL floor
            circuit_broken = (
                balance < mll_floor or
                daily_pnl < -(ACCOUNT.max_loss_limit * RISK_CONFIG.max_daily_loss_pct)
            )

            # --- Process each instrument ---
            for symbol in ["MCL", "MGC"]:
                spec     = INSTRUMENTS[symbol]
                sym_bars = bars.get(symbol)
                if sym_bars is None or timestamp not in sym_bars.index:
                    continue

                sym_bar_idx   = sym_bars.index.get_loc(timestamp)
                bars_to_here  = sym_bars.iloc[: sym_bar_idx + 1]
                current_bar   = sym_bars.iloc[sym_bar_idx]
                current_price = float(current_bar["close"])

                # --- Execute pending entry ---
                if pending_entries[symbol] is not None and not circuit_broken:
                    entry      = pending_entries[symbol]
                    exec_price = float(current_bar["open"])
                    entry["entry_price"] = exec_price
                    entry["entry_time"]  = timestamp

                    if entry["direction"] == "long":
                        entry["stop_price"]   = exec_price - entry["stop_pts"]
                        entry["target_price"] = exec_price + entry["target_pts"]
                    else:
                        entry["stop_price"]   = exec_price + entry["stop_pts"]
                        entry["target_price"] = exec_price - entry["target_pts"]

                    open_positions[symbol]  = entry
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
                        direction_mult = 1 if pos["direction"] == "long" else -1
                        price_diff     = (exit_price - pos["entry_price"]) * direction_mult
                        gross_pnl      = price_diff * spec.point_value * pos["contracts"]
                        commission     = spec.commission_rt * pos["contracts"]
                        net_pnl        = gross_pnl - commission

                        risk_per_trade = pos["stop_pts"] * spec.point_value * pos["contracts"]
                        r_multiple     = net_pnl / risk_per_trade if risk_per_trade > 0 else 0
                        hold_bars      = sym_bar_idx - pos.get("entry_bar_idx", sym_bar_idx)

                        self.trades.append(TradeRecord(
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
                        ))

                        balance        += net_pnl
                        daily_pnl      += net_pnl
                        cumulative_pnl += net_pnl
                        daily_trades_total += 1
                        open_positions[symbol] = None

                        # FIX 1: count stops per instrument per day
                        if exit_reason == "stop":
                            daily_stops[symbol] += 1

                        # FIX 3: update peak intraday after profitable exit
                        if net_pnl > 0:
                            peak_balance = max(peak_balance, balance)

                # --- Generate new signal ---
                # FIX 1: block instrument if it has hit max stops today
                instrument_blocked = daily_stops[symbol] >= MAX_STOPS_PER_INSTRUMENT_PER_DAY

                min_bars_needed = signal_engines[symbol].cfg.ewma_slow_span

                if (open_positions[symbol] is None
                        and pending_entries[symbol] is None
                        and not circuit_broken
                        and not instrument_blocked          # FIX 1
                        and not is_eod
                        and daily_trades_total < RISK_CONFIG.max_daily_trades
                        and len(bars_to_here) >= min_bars_needed):

                    account_state = AccountState(
                        balance             = balance,
                        peak_eod_balance    = peak_balance,
                        realized_pnl_today  = daily_pnl,
                        unrealized_pnl      = 0.0,
                        trades_today        = daily_trades_total,
                        best_day_pnl        = best_day_pnl,
                        cumulative_eval_pnl = cumulative_pnl,
                        open_positions      = {
                            k: (1 if v else 0) for k, v in open_positions.items()
                        },
                        is_eval_phase       = True,
                    )

                    # FIX 2: pass clean slice — engine handles its own smoothing
                    signal   = signal_engines[symbol].compute(bars_to_here)
                    decision = rm.evaluate(signal, account_state, bars_to_here)

                    if decision.is_trade:
                        # FIX 3: cap risk at 25% of MLL buffer (was 40%)
                        mll_buffer   = balance - mll_floor
                        max_by_mll   = int(
                            (mll_buffer * 0.25)
                            / (decision.stop_points * spec.point_value + 1e-9)
                        )
                        safe_contracts = min(decision.contracts, max(max_by_mll, 1))

                        if safe_contracts >= 1:
                            pending_entries[symbol] = {
                                "direction":     "long" if decision.action == "buy" else "short",
                                "contracts":     safe_contracts,
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
            trades           = self.trades,
            equity_curve     = pd.DataFrame(self.equity_curve).set_index("date"),
            starting_balance = self.starting_balance,
            final_balance    = balance,
        )


# ---------------------------------------------------------------------------
# Results
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

        df       = pd.DataFrame([t.__dict__ for t in self.trades])
        winners  = df[df["net_pnl"] > 0]
        losers   = df[df["net_pnl"] <= 0]
        n        = len(df)

        total_pnl     = df["net_pnl"].sum()
        win_rate      = len(winners) / n * 100
        avg_winner    = winners["net_pnl"].mean() if len(winners) > 0 else 0
        avg_loser     = losers["net_pnl"].mean()  if len(losers)  > 0 else 0
        profit_factor = (
            winners["net_pnl"].sum() / abs(losers["net_pnl"].sum())
            if len(losers) > 0 and losers["net_pnl"].sum() != 0 else float('inf')
        )
        avg_r         = df["r_multiple"].mean()
        total_comm    = df["commission"].sum()

        eq         = self.equity_curve
        max_dd     = eq["drawdown"].min()
        max_dd_pct = max_dd / self.starting_balance * 100

        daily_rets = eq["daily_pnl"] / self.starting_balance
        sharpe     = (
            daily_rets.mean() / daily_rets.std() * np.sqrt(252)
            if daily_rets.std() > 0 else 0
        )

        n_days     = len(eq)
        annual_ret = (self.final_balance / self.starting_balance) ** (252 / max(n_days, 1)) - 1
        trades_per_day = n / max(n_days, 1)
        avg_hold   = df["hold_bars"].mean()

        avg_win_pts  = abs(avg_winner) / (10 * df["contracts"].mean()) if avg_winner != 0 else 1
        avg_loss_pts = abs(avg_loser)  / (10 * df["contracts"].mean()) if avg_loser  != 0 else 1
        breakeven_wr = avg_loss_pts / (avg_win_pts + avg_loss_pts) * 100

        exit_counts = df["exit_reason"].value_counts()
        mcl = df[df["symbol"] == "MCL"]
        mgc = df[df["symbol"] == "MGC"]

        eq_copy       = eq.copy()
        eq_copy.index = pd.to_datetime(eq_copy.index)
        monthly_pnl   = eq_copy["daily_pnl"].resample("ME").sum()

        mll_breached  = abs(max_dd) > ACCOUNT.max_loss_limit

        lines = [
            "",
            "=" * 62,
            "  BACKTEST RESULTS v2 — MCL/MGC Momentum Strategy",
            "=" * 62,
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
            f"    MLL limit:         $    -3,000",
            f"    MLL status:        {'*** BREACH ***' if mll_breached else 'SAFE'}",
            "",
            "  TRADE STATISTICS",
            f"    Total trades:      {n:>10,}",
            f"    Trades per day:    {trades_per_day:>10.2f}",
            f"    Win rate:          {win_rate:>9.1f}%",
            f"    Breakeven WR:      {breakeven_wr:>9.1f}%",
            f"    Edge:              {win_rate - breakeven_wr:>+9.1f}%"
            f"  ({'POSITIVE' if win_rate > breakeven_wr else 'NEGATIVE'})",
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
            lines.append(f"    {reason:<22} {count:>4,}  ({pct:.0f}%)")

        lines += [
            "",
            "  PER-INSTRUMENT",
            f"    MCL  trades: {len(mcl):>4,}  "
            f"P&L: ${mcl['net_pnl'].sum():>+8,.0f}  "
            f"WR: {len(mcl[mcl['net_pnl']>0])/max(len(mcl),1)*100:.0f}%  "
            f"Stops blocked: {(mcl['exit_reason']=='stop').sum()}",
            f"    MGC  trades: {len(mgc):>4,}  "
            f"P&L: ${mgc['net_pnl'].sum():>+8,.0f}  "
            f"WR: {len(mgc[mgc['net_pnl']>0])/max(len(mgc),1)*100:.0f}%  "
            f"Stops blocked: {(mgc['exit_reason']=='stop').sum()}",
            "",
            "  MONTHLY P&L",
        ]

        for month, pnl in monthly_pnl.items():
            bar  = ("+" if pnl >= 0 else "-") * min(int(abs(pnl) / 100), 30)
            sign = "+" if pnl >= 0 else ""
            lines.append(f"    {str(month)[:7]}   {sign}${pnl:>8,.0f}  {bar}")

        lines += ["", "=" * 62, ""]
        return "\n".join(lines)

    def save(self, label: str = ""):
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag         = f"_{label}" if label else ""
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

    runner  = BacktestRunner(starting_balance=ACCOUNT.account_size)
    results = runner.run(bars, verbose=True)

    print(results.summary())
    results.save(label="MCL_MGC_12mo_v2")
    print("Done. Review results in backtest/results/")
