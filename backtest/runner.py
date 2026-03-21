"""
Backtest Runner v3 — MGC Only, Adaptive Stops
===============================================
Changes from v2:
  - MGC only (MCL dropped — daily range too small for 30-min momentum)
  - Adaptive stop/target based on recent 5-day ATR (Average True Range)
    Stop  = 0.5 × 5-day ATR
    Target = 1.0 × 5-day ATR (1:1 minimum, scales to market conditions)
    This means in June (low vol, 13pt range) stops are ~6pts
    and in January (high vol, 103pt range) stops are ~50pts
  - Minimum 8 bars per session before allowing entries (sparse data filter)
  - Session filter: UTC 13:00-15:30 and 19:00-20:00 only (Fix A)
  - Daily signal reset carried in engine (Fix B)
  - Threshold 0.25 in engine (Fix C)
  - Per-instrument stop counter: 2 stops per day max (Fix 1)
  - MLL buffer capped at 25% per trade (Fix 3)
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

# MGC-specific constants
MGC_SPEC          = INSTRUMENTS["MGC"]
ATR_LOOKBACK_DAYS = 5     # days of history for ATR calculation
ATR_STOP_MULT     = 0.50  # stop = 0.5 × ATR
ATR_TARGET_MULT   = 1.00  # target = 1.0 × ATR (1:1 R:R minimum)
MIN_BARS_PER_DAY  = 8     # skip signal generation on sparse data days
MAX_STOPS_PER_DAY = 2     # per-instrument daily stop limit


# ---------------------------------------------------------------------------
# ATR calculator
# ---------------------------------------------------------------------------

def compute_atr(bars: pd.DataFrame, lookback_days: int = 5) -> float:
    """
    Compute the Average True Range over the last N calendar days.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    Returns ATR in points. Falls back to recent high-low range if insufficient data.
    """
    if len(bars) < 2:
        return 10.0  # safe default

    # Get recent N days of data
    cutoff = bars.index[-1] - pd.Timedelta(days=lookback_days)
    recent = bars[bars.index >= cutoff]

    if len(recent) < 4:
        recent = bars.tail(20)  # fallback to last 20 bars

    highs  = recent["high"].values
    lows   = recent["low"].values
    closes = recent["close"].values

    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]

    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - prev_closes),
            np.abs(lows - prev_closes)
        )
    )

    atr = float(np.mean(tr))
    # Clamp ATR to reasonable bounds for MGC
    # Min 8pts (low vol months like June), Max 80pts (extreme months)
    return float(np.clip(atr, 8.0, 80.0))


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
    stop_pts:     float
    target_pts:   float
    atr_at_entry: float
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
# Backtest engine v3
# ---------------------------------------------------------------------------

class BacktestRunner:

    def __init__(self, starting_balance: float = 100_000.0):
        self.starting_balance = starting_balance
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[dict]  = []

    def run(self, bars: pd.DataFrame, verbose: bool = True) -> "BacktestResults":
        """
        MGC-only backtest with adaptive ATR-based stops and targets.

        Args:
            bars: MGC 30-min bar DataFrame from cache
            verbose: print progress

        Returns:
            BacktestResults
        """
        if verbose:
            print(f"\nStarting backtest v3 — MGC only, adaptive ATR stops")
            print(f"  MGC: {len(bars):,} bars "
                  f"({bars.index[0].date()} → {bars.index[-1].date()})")
            print(f"  ATR lookback: {ATR_LOOKBACK_DAYS} days")
            print(f"  Stop mult: {ATR_STOP_MULT}× ATR | Target mult: {ATR_TARGET_MULT}× ATR")

        signal_engine = MomentumSignalEngine("MGC")
        rm            = RiskManager()

        balance      = self.starting_balance
        peak_balance = self.starting_balance

        open_position:  Optional[dict] = None
        pending_entry:  Optional[dict] = None

        current_date        = None
        daily_pnl           = 0.0
        daily_trades        = 0
        daily_stops         = 0
        cumulative_pnl      = 0.0
        best_day_pnl        = 0.0
        bars_today          = 0

        total_bars = len(bars)

        for bar_idx, timestamp in enumerate(bars.index):

            if verbose and bar_idx % 300 == 0 and bar_idx > 0:
                pct = bar_idx / total_bars * 100
                print(f"  [{pct:.0f}%] Bar {bar_idx:,}/{total_bars:,} | "
                      f"Balance: ${balance:,.0f} | "
                      f"Trades: {len(self.trades)}")

            bar_date    = timestamp.date()
            current_bar = bars.iloc[bar_idx]

            # --- Day boundary ---
            if bar_date != current_date:
                if current_date is not None:
                    self.equity_curve.append({
                        "date":      current_date,
                        "balance":   balance,
                        "daily_pnl": daily_pnl,
                        "drawdown":  balance - peak_balance,
                        "bars":      bars_today,
                    })
                    best_day_pnl = max(best_day_pnl, daily_pnl)
                    peak_balance = max(peak_balance, balance)

                current_date = bar_date
                daily_pnl    = 0.0
                daily_trades = 0
                daily_stops  = 0
                bars_today   = 0

            bars_today += 1

            # EOD close flag
            is_eod = (timestamp.hour == 16 and timestamp.minute >= 25)

            # MLL floor
            mll_floor = peak_balance - ACCOUNT.max_loss_limit

            # Circuit breakers
            circuit_broken = (
                balance < mll_floor or
                daily_pnl < -(ACCOUNT.max_loss_limit * RISK_CONFIG.max_daily_loss_pct)
            )

            current_price = float(current_bar["close"])

            # --- Execute pending entry ---
            if pending_entry is not None and not circuit_broken:
                exec_price = float(current_bar["open"])
                pending_entry["entry_price"] = exec_price
                pending_entry["entry_time"]  = timestamp

                if pending_entry["direction"] == "long":
                    pending_entry["stop_price"]   = exec_price - pending_entry["stop_pts"]
                    pending_entry["target_price"] = exec_price + pending_entry["target_pts"]
                else:
                    pending_entry["stop_price"]   = exec_price + pending_entry["stop_pts"]
                    pending_entry["target_price"] = exec_price - pending_entry["target_pts"]

                open_position = pending_entry
                pending_entry = None

            # --- Manage open position ---
            if open_position is not None:
                exit_reason = None
                exit_price  = current_price

                if is_eod:
                    exit_reason = "eod_close"
                    exit_price  = current_price

                elif open_position["direction"] == "long":
                    if current_bar["low"] <= open_position["stop_price"]:
                        exit_reason = "stop"
                        exit_price  = open_position["stop_price"]
                    elif current_bar["high"] >= open_position["target_price"]:
                        exit_reason = "target"
                        exit_price  = open_position["target_price"]

                elif open_position["direction"] == "short":
                    if current_bar["high"] >= open_position["stop_price"]:
                        exit_reason = "stop"
                        exit_price  = open_position["stop_price"]
                    elif current_bar["low"] <= open_position["target_price"]:
                        exit_reason = "target"
                        exit_price  = open_position["target_price"]

                if exit_reason:
                    direction_mult = 1 if open_position["direction"] == "long" else -1
                    price_diff     = (exit_price - open_position["entry_price"]) * direction_mult
                    gross_pnl      = price_diff * MGC_SPEC.point_value * open_position["contracts"]
                    commission     = MGC_SPEC.commission_rt * open_position["contracts"]
                    net_pnl        = gross_pnl - commission

                    stop_pts       = open_position["stop_pts"]
                    risk_per_trade = stop_pts * MGC_SPEC.point_value * open_position["contracts"]
                    r_multiple     = net_pnl / risk_per_trade if risk_per_trade > 0 else 0
                    hold_bars      = bar_idx - open_position.get("entry_bar_idx", bar_idx)

                    self.trades.append(TradeRecord(
                        symbol       = "MGC",
                        entry_time   = open_position["entry_time"],
                        exit_time    = timestamp,
                        direction    = open_position["direction"],
                        contracts    = open_position["contracts"],
                        entry_price  = open_position["entry_price"],
                        exit_price   = exit_price,
                        exit_reason  = exit_reason,
                        stop_price   = open_position["stop_price"],
                        target_price = open_position["target_price"],
                        stop_pts     = stop_pts,
                        target_pts   = open_position["target_pts"],
                        atr_at_entry = open_position["atr"],
                        gross_pnl    = gross_pnl,
                        commission   = commission,
                        net_pnl      = net_pnl,
                        r_multiple   = r_multiple,
                        hold_bars    = hold_bars,
                        signal_score = open_position.get("signal_score", 0.0),
                        regime       = open_position.get("regime", "normal"),
                    ))

                    balance        += net_pnl
                    daily_pnl      += net_pnl
                    cumulative_pnl += net_pnl
                    daily_trades   += 1
                    open_position   = None

                    if exit_reason == "stop":
                        daily_stops += 1

                    if net_pnl > 0:
                        peak_balance = max(peak_balance, balance)

            # --- Generate new signal ---
            # Session filter: morning (UTC 13-15) and late session (UTC 19)
            in_session = (13 <= timestamp.hour <= 15) or (timestamp.hour == 19)

            # Sparse data filter: need at least MIN_BARS_PER_DAY bars today
            enough_bars_today = bars_today >= MIN_BARS_PER_DAY

            min_bars_needed = signal_engine.cfg.ewma_slow_span
            bars_to_here    = bars.iloc[: bar_idx + 1]

            if (open_position is None
                    and pending_entry is None
                    and not circuit_broken
                    and not is_eod
                    and in_session
                    and enough_bars_today
                    and daily_stops < MAX_STOPS_PER_DAY
                    and daily_trades < RISK_CONFIG.max_daily_trades
                    and len(bars_to_here) >= min_bars_needed):

                # Compute ATR-based stop and target
                atr        = compute_atr(bars_to_here, ATR_LOOKBACK_DAYS)
                stop_pts   = round(atr * ATR_STOP_MULT, 1)
                target_pts = round(atr * ATR_TARGET_MULT, 1)

                # Ensure minimum viable stop (larger than commission drag)
                stop_pts   = max(stop_pts, 5.0)
                target_pts = max(target_pts, 10.0)

                # Compute signal
                signal = signal_engine.compute(bars_to_here)

                if not signal.is_tradeable:
                    continue

                # Account state for risk manager
                account_state = AccountState(
                    balance             = balance,
                    peak_eod_balance    = peak_balance,
                    realized_pnl_today  = daily_pnl,
                    unrealized_pnl      = 0.0,
                    trades_today        = daily_trades,
                    best_day_pnl        = best_day_pnl,
                    cumulative_eval_pnl = cumulative_pnl,
                    open_positions      = {"MGC": 0, "MCL": 0},
                    is_eval_phase       = True,
                )

                decision = rm.evaluate(signal, account_state, bars_to_here)

                if decision.is_trade:
                    # MLL buffer cap at 25%
                    mll_buffer     = balance - mll_floor
                    max_by_mll     = int(
                        (mll_buffer * 0.25)
                        / (stop_pts * MGC_SPEC.point_value + 1e-9)
                    )
                    safe_contracts = min(decision.contracts, max(max_by_mll, 1))

                    if safe_contracts >= 1:
                        pending_entry = {
                            "direction":     "long" if decision.action == "buy" else "short",
                            "contracts":     safe_contracts,
                            "stop_pts":      stop_pts,
                            "target_pts":    target_pts,
                            "atr":           atr,
                            "entry_price":   None,
                            "entry_time":    None,
                            "stop_price":    None,
                            "target_price":  None,
                            "entry_bar_idx": bar_idx + 1,
                            "signal_score":  signal.combined_score,
                            "regime":        signal.regime,
                        }

        # Final EOD record
        self.equity_curve.append({
            "date":      current_date,
            "balance":   balance,
            "daily_pnl": daily_pnl,
            "drawdown":  balance - peak_balance,
            "bars":      bars_today,
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

        df      = pd.DataFrame([t.__dict__ for t in self.trades])
        winners = df[df["net_pnl"] > 0]
        losers  = df[df["net_pnl"] <= 0]
        n       = len(df)

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
        avg_atr       = df["atr_at_entry"].mean()
        avg_stop      = df["stop_pts"].mean()
        avg_target    = df["target_pts"].mean()

        eq         = self.equity_curve
        max_dd     = eq["drawdown"].min()
        max_dd_pct = max_dd / self.starting_balance * 100

        daily_rets = eq["daily_pnl"] / self.starting_balance
        sharpe     = (
            daily_rets.mean() / daily_rets.std() * np.sqrt(252)
            if daily_rets.std() > 0 else 0
        )

        n_days         = len(eq)
        annual_ret     = (self.final_balance / self.starting_balance) ** (252 / max(n_days, 1)) - 1
        trades_per_day = n / max(n_days, 1)
        avg_hold       = df["hold_bars"].mean()

        avg_win_pts  = abs(avg_winner) / (MGC_SPEC.point_value * df["contracts"].mean()) if avg_winner != 0 else 1
        avg_loss_pts = abs(avg_loser)  / (MGC_SPEC.point_value * df["contracts"].mean()) if avg_loser  != 0 else 1
        breakeven_wr = avg_loss_pts / (avg_win_pts + avg_loss_pts) * 100

        exit_counts  = df["exit_reason"].value_counts()

        eq_copy        = eq.copy()
        eq_copy.index  = pd.to_datetime(eq_copy.index)
        monthly_pnl    = eq_copy["daily_pnl"].resample("ME").sum()

        mll_breached = abs(max_dd) > ACCOUNT.max_loss_limit

        lines = [
            "",
            "=" * 62,
            "  BACKTEST RESULTS v3 — MGC Only, Adaptive ATR Stops",
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
            "  ATR CALIBRATION",
            f"    Avg ATR at entry:  {avg_atr:>10.1f} pts",
            f"    Avg stop:          {avg_stop:>10.1f} pts  (${avg_stop*MGC_SPEC.point_value:.0f})",
            f"    Avg target:        {avg_target:>10.1f} pts  (${avg_target*MGC_SPEC.point_value:.0f})",
            "",
            "  EXIT REASONS",
        ]

        for reason, count in exit_counts.items():
            pct = count / n * 100
            lines.append(f"    {reason:<22} {count:>4,}  ({pct:.0f}%)")

        lines += ["", "  MONTHLY P&L"]

        for month, pnl in monthly_pnl.items():
            bar  = ("+" if pnl >= 0 else "-") * min(int(abs(pnl) / 200), 25)
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
    mgc_path  = cache_dir / "MGC_30min.parquet"

    if not mgc_path.exists():
        print("MGC cache not found. Run data/fetcher.py first.")
        sys.exit(1)

    print("Loading MGC bars...")
    bars = pd.read_parquet(mgc_path)
    print(f"  MGC: {len(bars):,} bars")

    runner  = BacktestRunner(starting_balance=ACCOUNT.account_size)
    results = runner.run(bars, verbose=True)

    print(results.summary())
    results.save(label="MGC_only_v3")
    print("Done. Review results in backtest/results/")
