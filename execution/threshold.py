"""
Signal Threshold Engine
=======================
Implements the Almgren-Chriss (2001) breakeven cost framework.

Every signal must exceed a minimum threshold before we enter. That threshold
is the sum of:
  1. Commission drag (round-trip, both legs)
  2. Temporary market impact (moves price against us on entry + exit)
  3. Permanent market impact (price shift that doesn't revert)
  4. Safety multiplier (require signal >> cost, not just signal > cost)

This prevents us from trading when our edge doesn't clear execution costs —
the graveyard of theoretical strategies at scale.

As contracts increase, the threshold rises. This is the key mechanism that
tells the system when to stop scaling — when incremental signal strength
no longer exceeds incremental execution cost.

References:
    Almgren & Chriss (2001) — Optimal Execution of Portfolio Transactions
    Moskowitz, Ooi & Pedersen (2012) — Time Series Momentum (vol scaling)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from momentum_engine.config.settings import (
    INSTRUMENTS, ACCOUNT, EXECUTION_CONFIG, InstrumentSpec
)


@dataclass
class ThresholdResult:
    """All cost components and the final signal threshold in points."""
    symbol: str
    contracts: int
    commission_points: float      # round-trip commission converted to points
    temp_impact_points: float     # temporary price impact in points
    perm_impact_points: float     # permanent price impact in points
    total_cost_points: float      # sum of all costs
    threshold_points: float       # total_cost × safety_multiplier
    breakeven_win_rate: float     # win rate needed at this R:R to be profitable
    cost_as_pct_of_stop: float    # cost as % of default stop — key intuition metric

    def summary(self) -> str:
        return (
            f"{self.symbol} | {self.contracts} contracts\n"
            f"  Commission:      {self.commission_points:.3f} pts  (${self.commission_points * INSTRUMENTS[self.symbol].point_value:.2f})\n"
            f"  Temp impact:     {self.temp_impact_points:.3f} pts  (${self.temp_impact_points * INSTRUMENTS[self.symbol].point_value:.2f})\n"
            f"  Perm impact:     {self.perm_impact_points:.3f} pts  (${self.perm_impact_points * INSTRUMENTS[self.symbol].point_value:.2f})\n"
            f"  Total cost:      {self.total_cost_points:.3f} pts  (${self.total_cost_points * INSTRUMENTS[self.symbol].point_value:.2f})\n"
            f"  Threshold:       {self.threshold_points:.3f} pts  (${self.threshold_points * INSTRUMENTS[self.symbol].point_value:.2f})\n"
            f"  Cost/stop:       {self.cost_as_pct_of_stop:.1f}%\n"
            f"  Breakeven WR:    {self.breakeven_win_rate:.1f}%"
        )


class SignalThresholdEngine:
    """
    Computes the minimum signal strength required to justify a trade,
    accounting for all execution costs using Almgren-Chriss decomposition.

    The core insight: as we scale contracts, market impact grows (roughly
    linearly for micro contracts at these sizes), so the threshold rises.
    The signal engine should only generate a trade if the signal magnitude
    exceeds this rising threshold.

    Threshold in points = (commission_pts + temp_impact_pts + perm_impact_pts)
                          × safety_multiplier

    We express everything in points (not dollars) so the threshold can be
    directly compared against signal outputs, which are also in points.
    """

    def __init__(self):
        self.exec_cfg = EXECUTION_CONFIG
        self.account = ACCOUNT

    def compute(
        self,
        symbol: str,
        contracts: int,
        stop_points: float,
        target_points: float,
    ) -> ThresholdResult:
        """
        Compute the full cost breakdown and signal threshold.

        Args:
            symbol:    'MCL' or 'MGC'
            contracts: number of micro contracts sized for this trade
            stop_points:   stop loss distance in points
            target_points: profit target distance in points

        Returns:
            ThresholdResult with all cost components and the threshold
        """
        spec = INSTRUMENTS[symbol]

        # --- 1. Commission drag ---
        # Commission is per contract, round-trip. Convert to points by dividing
        # by point_value. This is the irreducible floor of execution cost.
        commission_pts = (spec.commission_rt * contracts) / spec.point_value

        # --- 2. Temporary market impact ---
        # Temporary impact: the bid-ask crossing + short-term order book pressure.
        # For micro contracts at 1-5 size, this is roughly 0.5 ticks per contract.
        # It reverts after the trade, so it affects entry and exit independently.
        # Total temporary cost = 2 legs × ticks × tick_size × contracts.
        temp_ticks = self.exec_cfg.temp_impact_ticks_per_contract.get(symbol, 0.5)
        temp_impact_pts = 2 * temp_ticks * spec.tick_size * contracts

        # --- 3. Permanent market impact ---
        # Permanent impact: the portion of our order that moves the price
        # permanently against us (informed flow repricing). Smaller for micros.
        perm_ticks = self.exec_cfg.perm_impact_ticks_per_contract.get(symbol, 0.1)
        perm_impact_pts = perm_ticks * spec.tick_size * contracts

        # --- 4. Total cost and threshold ---
        total_cost_pts = commission_pts + temp_impact_pts + perm_impact_pts
        threshold_pts = total_cost_pts * self.exec_cfg.safety_multiplier

        # --- 5. Breakeven win rate at this R:R ---
        # From basic probability: E[P&L] = WR × (target - cost) - (1-WR) × (stop + cost)
        # Setting E[P&L] = 0 and solving for WR:
        # WR = (stop + cost) / (target + stop)
        # This tells us the minimum win rate needed just to break even.
        rr = target_points / stop_points
        breakeven_wr = (stop_points + total_cost_pts) / (target_points + stop_points)
        breakeven_wr_pct = breakeven_wr * 100

        # --- 6. Cost as % of stop — the intuition metric ---
        # If cost is 30% of the stop, we need to be right more often just to
        # recover execution costs. This number should stay below 20% ideally.
        cost_pct_of_stop = (total_cost_pts / stop_points) * 100

        return ThresholdResult(
            symbol=symbol,
            contracts=contracts,
            commission_points=commission_pts,
            temp_impact_points=temp_impact_pts,
            perm_impact_points=perm_impact_pts,
            total_cost_points=total_cost_pts,
            threshold_points=threshold_pts,
            breakeven_win_rate=breakeven_wr_pct,
            cost_as_pct_of_stop=cost_pct_of_stop,
        )

    def signal_clears_threshold(
        self,
        symbol: str,
        signal_strength_points: float,
        contracts: int,
        stop_points: float,
        target_points: float,
    ) -> tuple[bool, ThresholdResult]:
        """
        Gate function: returns True only if the signal is strong enough
        to justify execution given all costs.

        Args:
            signal_strength_points: the raw signal magnitude in points
                                    (e.g., expected move from momentum model)
            Others: see compute()

        Returns:
            (clears: bool, result: ThresholdResult)
        """
        result = self.compute(symbol, contracts, stop_points, target_points)
        clears = signal_strength_points >= result.threshold_points
        return clears, result

    def scaling_table(
        self,
        symbol: str,
        stop_points: float,
        target_points: float,
        max_contracts: int = 20,
    ) -> list[ThresholdResult]:
        """
        Generate the full threshold table as contracts scale from 1 to max.
        This is the Almgren-Chriss efficient frontier applied to our strategy:
        shows at what contract size execution cost exceeds marginal signal value.

        Use this table to determine the maximum sensible position size given
        the current signal strength.
        """
        return [
            self.compute(symbol, n, stop_points, target_points)
            for n in range(1, max_contracts + 1)
        ]

    def max_contracts_for_signal(
        self,
        symbol: str,
        signal_strength_points: float,
        stop_points: float,
        target_points: float,
        max_contracts: int = 60,
    ) -> int:
        """
        Given a signal of known strength, return the maximum number of contracts
        where the signal still clears the threshold. This is the position size
        the risk engine should use.

        As signal strength is fixed but threshold rises with contracts,
        there is a crossover point. We return the last contract count
        where signal >= threshold.
        """
        best = 0
        for n in range(1, max_contracts + 1):
            result = self.compute(symbol, n, stop_points, target_points)
            if signal_strength_points >= result.threshold_points:
                best = n
            else:
                break  # threshold exceeded, stop scaling
        return best
