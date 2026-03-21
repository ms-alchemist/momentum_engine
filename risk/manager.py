"""
Risk Manager
============
Translates signal output into a sized, validated trade decision.

Responsibilities:
  1. Volatility-targeted position sizing (Barroso & Santa-Clara 2015)
  2. Almgren-Chriss threshold gate (signal must clear execution cost)
  3. Lucid account constraint enforcement (MLL, max contracts, close time)
  4. Daily loss circuit breaker
  5. Consistency rule tracker (eval phase: best day ≤ 50% of cumulative P&L)

All dollar amounts are in USD. All size outputs are in contracts (integer).

The position size formula follows Barroso & Santa-Clara:
    target_vol = RISK_CONFIG.target_daily_vol_pct × account_balance
    contracts  = target_vol / (realized_vol_per_contract × point_value)
    contracts  = min(contracts, max_contracts_for_signal(threshold))
    contracts  = min(contracts, account_max, daily_remaining)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from momentum_engine.config.settings import (
    INSTRUMENTS, ACCOUNT, RISK_CONFIG, EXECUTION_CONFIG
)
from momentum_engine.signals.engine import SignalOutput, SignalDirection
from momentum_engine.execution.threshold import SignalThresholdEngine, ThresholdResult


@dataclass
class TradeDecision:
    """Output of the risk manager for a single signal evaluation."""
    symbol: str
    action: str                     # 'buy', 'sell', 'hold'
    contracts: int                  # 0 if hold
    stop_points: float
    target_points: float
    stop_price: Optional[float]
    target_price: Optional[float]
    entry_price: Optional[float]    # current bar close (indicative)
    dollar_risk: float              # contracts × stop_points × point_value
    threshold: Optional[ThresholdResult]
    sizing_reason: str              # audit trail — why this size

    @property
    def is_trade(self) -> bool:
        return self.action in ('buy', 'sell') and self.contracts > 0

    def summary(self) -> str:
        if not self.is_trade:
            return f"HOLD — {self.sizing_reason}"
        direction = "LONG" if self.action == 'buy' else "SHORT"
        return (
            f"{direction} {self.contracts} {self.symbol} @ ~{self.entry_price:.2f} | "
            f"Stop: {self.stop_price:.2f} ({self.stop_points:.1f}pts) | "
            f"Target: {self.target_price:.2f} ({self.target_points:.1f}pts) | "
            f"Risk: ${self.dollar_risk:.0f} | "
            f"Reason: {self.sizing_reason}"
        )


@dataclass
class AccountState:
    """
    Real-time account state — updated after each trade and each bar close.
    Tracks all Lucid-specific constraints.
    """
    balance: float                   # current balance (sim)
    peak_eod_balance: float          # highest EOD balance (drives MLL)
    realized_pnl_today: float        # closed P&L today
    unrealized_pnl: float            # open position P&L
    trades_today: int                # number of completed trades today
    best_day_pnl: float              # best single day in eval cycle
    cumulative_eval_pnl: float       # total P&L since eval started
    open_positions: dict             # symbol → contracts (+ = long, - = short)
    is_eval_phase: bool = True

    @property
    def current_mll(self) -> float:
        """Max loss limit floor = peak EOD balance - ACCOUNT.max_loss_limit"""
        return self.peak_eod_balance - ACCOUNT.max_loss_limit

    @property
    def eod_buffer_remaining(self) -> float:
        """How much EOD loss we can still absorb before breaching MLL."""
        current_eod = self.balance + self.unrealized_pnl
        return current_eod - self.current_mll

    @property
    def consistency_headroom(self) -> float:
        """
        In eval: how much more P&L today before we exceed 50% of cumulative.
        Returns float('inf') if in funded phase (no consistency rule).
        """
        if not self.is_eval_phase:
            return float('inf')
        max_today = ACCOUNT.consistency_rule_pct * self.cumulative_eval_pnl
        return max(0.0, max_today - self.realized_pnl_today)

    @property
    def daily_loss_consumed(self) -> float:
        """P&L loss today as fraction of daily loss limit."""
        daily_loss = min(0.0, self.realized_pnl_today)
        limit = ACCOUNT.max_loss_limit * RISK_CONFIG.max_daily_loss_pct
        return abs(daily_loss) / (limit + 1e-9)


class RiskManager:
    """
    Sizes positions and gates trades based on signal, account state,
    and all Lucid LucidFlex 100K constraints.

    Usage:
        rm = RiskManager()
        decision = rm.evaluate(signal_output, account_state, bars)
    """

    def __init__(self):
        self.threshold_engine = SignalThresholdEngine()
        self.cfg = RISK_CONFIG
        self.account = ACCOUNT

    def evaluate(
        self,
        signal: SignalOutput,
        state: AccountState,
        bars: pd.DataFrame,
    ) -> TradeDecision:
        """
        Main entry: given a signal and account state, return a trade decision.

        Pipeline:
          1. Hard gates (panic, no signal, market hours, daily limits)
          2. Compute stop/target from config
          3. Vol-targeted sizing (Barroso & Santa-Clara)
          4. Almgren-Chriss threshold check
          5. Lucid constraint clipping
          6. Final decision
        """
        symbol = signal.symbol
        spec = INSTRUMENTS[symbol]
        current_price = float(bars["close"].iloc[-1])

        stop_pts   = self.cfg.default_stop_points[symbol]
        target_pts = self.cfg.default_target_points[symbol]

        # ---- Gate 1: Signal must be tradeable ----
        if not signal.is_tradeable:
            return self._hold(symbol, current_price, stop_pts, target_pts,
                              f"Signal not tradeable: {signal.regime}")

        # ---- Gate 2: Daily trade limit ----
        if state.trades_today >= self.cfg.max_daily_trades:
            return self._hold(symbol, current_price, stop_pts, target_pts,
                              f"Daily trade limit reached ({self.cfg.max_daily_trades})")

        # ---- Gate 3: Daily loss circuit breaker ----
        if state.daily_loss_consumed >= 1.0:
            return self._hold(symbol, current_price, stop_pts, target_pts,
                              "Daily loss limit reached — flat for rest of session")

        # ---- Gate 4: Already in a position in this instrument ----
        existing = state.open_positions.get(symbol, 0)
        if existing != 0:
            return self._hold(symbol, current_price, stop_pts, target_pts,
                              f"Already holding {existing} {symbol} contracts")

        # ---- Gate 5: MLL buffer check — never risk breaching MLL in one trade ----
        if state.eod_buffer_remaining < 0:
            return self._hold(symbol, current_price, stop_pts, target_pts,
                              "EOD balance already below MLL floor")

        # ---- Step 1: Volatility-targeted size (Barroso & Santa-Clara) ----
        vol_sized_contracts = self._vol_target_size(symbol, bars, state)

        # ---- Step 2: Cap by Almgren-Chriss threshold ----
        # Find max contracts where signal still clears execution cost
        threshold_max = self.threshold_engine.max_contracts_for_signal(
            symbol=symbol,
            signal_strength_points=signal.expected_move_pts,
            stop_points=stop_pts,
            target_points=target_pts,
            max_contracts=self.cfg.max_contracts_per_instrument,
        )

        if threshold_max == 0:
            # Compute the threshold for the audit trail
            thresh_result = self.threshold_engine.compute(
                symbol, 1, stop_pts, target_pts
            )
            return self._hold(
                symbol, current_price, stop_pts, target_pts,
                f"Signal ({signal.expected_move_pts:.2f}pts) below threshold "
                f"({thresh_result.threshold_points:.2f}pts) even at 1 contract",
                threshold=thresh_result,
            )

        # ---- Step 3: Apply all constraints ----
        contracts = min(
            vol_sized_contracts,
            threshold_max,
            self.cfg.max_contracts_per_instrument,
            self.account.max_contracts_micro,
        )

        # ---- Step 4: Consistency rule clip (eval only) ----
        if state.is_eval_phase:
            max_profit_allowed = state.consistency_headroom
            max_by_consistency = int(max_profit_allowed / (target_pts * spec.point_value + 1e-9))
            contracts = min(contracts, max(max_by_consistency, 1))

        # ---- Step 5: MLL dollar risk check ----
        dollar_risk = contracts * stop_pts * spec.point_value
        # Never risk more than 40% of remaining MLL buffer in one trade
        max_by_mll = int((state.eod_buffer_remaining * 0.40) / (stop_pts * spec.point_value + 1e-9))
        contracts = min(contracts, max(max_by_mll, 1))

        # ---- Step 6: Minimum size check ----
        if contracts < 1:
            return self._hold(symbol, current_price, stop_pts, target_pts,
                              "Position size rounded to 0 after constraints")

        # ---- Build decision ----
        is_long = signal.direction == SignalDirection.LONG
        action = 'buy' if is_long else 'sell'

        if is_long:
            stop_price   = round(current_price - stop_pts, 2)
            target_price = round(current_price + target_pts, 2)
        else:
            stop_price   = round(current_price + stop_pts, 2)
            target_price = round(current_price - target_pts, 2)

        dollar_risk_final = contracts * stop_pts * spec.point_value
        thresh_result = self.threshold_engine.compute(symbol, contracts, stop_pts, target_pts)

        reason = (
            f"Vol-sized={vol_sized_contracts}, "
            f"threshold_max={threshold_max}, "
            f"final={contracts} | "
            f"signal={signal.combined_score:.3f} | "
            f"regime={signal.regime}"
        )

        return TradeDecision(
            symbol=symbol,
            action=action,
            contracts=contracts,
            stop_points=stop_pts,
            target_points=target_pts,
            stop_price=stop_price,
            target_price=target_price,
            entry_price=current_price,
            dollar_risk=dollar_risk_final,
            threshold=thresh_result,
            sizing_reason=reason,
        )

    # ------------------------------------------------------------------
    # Volatility-targeted sizing (Barroso & Santa-Clara 2015)
    # ------------------------------------------------------------------

    def _vol_target_size(
        self, symbol: str, bars: pd.DataFrame, state: AccountState
    ) -> int:
        """
        Target a fixed daily volatility in dollar terms, then back out contracts.

        target_dollar_vol = target_daily_vol_pct × account_balance
        contracts = target_dollar_vol / (realized_vol_per_bar × price × point_value)

        We use a short lookback for the vol estimate (responsive to regime changes).
        """
        spec = INSTRUMENTS[symbol]
        closes = bars["close"]
        returns = closes.pct_change().dropna()

        if len(returns) < self.cfg.vol_lookback_bars:
            return 1  # minimum size when insufficient history

        realized_vol = returns.tail(self.cfg.vol_lookback_bars).std()
        if realized_vol == 0 or np.isnan(realized_vol):
            return 1

        current_price = float(closes.iloc[-1])
        target_dollar_vol = self.cfg.target_daily_vol_pct * state.balance

        # Dollar vol per contract per bar
        dollar_vol_per_contract = realized_vol * current_price * spec.point_value

        if dollar_vol_per_contract == 0:
            return 1

        contracts = int(target_dollar_vol / dollar_vol_per_contract)
        return max(1, contracts)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _hold(
        self,
        symbol: str,
        price: float,
        stop_pts: float,
        target_pts: float,
        reason: str,
        threshold: Optional[ThresholdResult] = None,
    ) -> TradeDecision:
        return TradeDecision(
            symbol=symbol,
            action='hold',
            contracts=0,
            stop_points=stop_pts,
            target_points=target_pts,
            stop_price=None,
            target_price=None,
            entry_price=price,
            dollar_risk=0.0,
            threshold=threshold,
            sizing_reason=reason,
        )
