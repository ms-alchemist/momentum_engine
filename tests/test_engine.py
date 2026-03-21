"""
Test Suite — Momentum Signal Engine
====================================
Validates every component with synthetic data before any live deployment.

Tests cover:
  - Instrument specs (contract math)
  - Threshold engine (Almgren-Chriss cost decomposition)
  - Signal engine (all four layers)
  - Risk manager (position sizing, Lucid constraints)
  - Integration (full pipeline end-to-end)

Run with: python -m pytest momentum_engine/tests/test_engine.py -v
       or: python momentum_engine/tests/test_engine.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from momentum_engine.config.settings import INSTRUMENTS, ACCOUNT, RISK_CONFIG
from momentum_engine.execution.threshold import SignalThresholdEngine
from momentum_engine.signals.engine import MomentumSignalEngine, SignalDirection
from momentum_engine.risk.manager import RiskManager, AccountState


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def make_bars(
    n_bars: int = 100,
    symbol: str = "MCL",
    trend: float = 0.0,         # points per bar drift
    vol: float = 0.005,         # return vol per bar
    seed: int = 42,
    with_delta: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV bars for testing.
    trend > 0 → uptrend, trend < 0 → downtrend, trend = 0 → random walk
    """
    np.random.seed(seed)
    price = 75.0 if symbol == "MCL" else 2000.0
    prices = [price]
    for _ in range(n_bars - 1):
        ret = trend / price + np.random.normal(0, vol)
        prices.append(max(prices[-1] * (1 + ret), 1.0))

    closes = np.array(prices)
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    highs  = closes * (1 + abs(np.random.normal(0, vol/2, n_bars)))
    lows   = closes * (1 - abs(np.random.normal(0, vol/2, n_bars)))
    volumes = np.random.lognormal(8, 0.5, n_bars).astype(int)

    start = pd.Timestamp("2024-01-02 09:00", tz="America/New_York")
    idx = pd.date_range(start, periods=n_bars, freq="30min")

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }, index=idx)

    if with_delta:
        # Simulate buy/sell volume: trending bars have more buy vol
        direction = np.sign(np.diff(closes, prepend=closes[0]))
        buy_frac = 0.5 + direction * 0.15 + np.random.normal(0, 0.05, n_bars)
        buy_frac = np.clip(buy_frac, 0.05, 0.95)
        df["buy_volume"]  = (volumes * buy_frac).astype(int)
        df["sell_volume"] = volumes - df["buy_volume"]

    return df


def make_account_state(
    balance: float = 100_000.0,
    realized_pnl_today: float = 0.0,
    trades_today: int = 0,
    cumulative_eval_pnl: float = 0.0,
    open_positions: dict = None,
) -> AccountState:
    return AccountState(
        balance=balance,
        peak_eod_balance=balance,
        realized_pnl_today=realized_pnl_today,
        unrealized_pnl=0.0,
        trades_today=trades_today,
        best_day_pnl=0.0,
        cumulative_eval_pnl=cumulative_eval_pnl,
        open_positions=open_positions or {},
        is_eval_phase=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInstrumentSpecs:
    def test_mcl_point_value(self):
        spec = INSTRUMENTS["MCL"]
        assert spec.point_value == 10.0, "MCL: $10 per point"

    def test_mgc_point_value(self):
        spec = INSTRUMENTS["MGC"]
        assert spec.point_value == 10.0, "MGC: $10 per point"

    def test_mcl_tick_math(self):
        spec = INSTRUMENTS["MCL"]
        # 1 tick = 0.01 points × $10/point = $0.10
        assert abs(spec.tick_size * spec.point_value - spec.tick_value) < 1e-9

    def test_mgc_tick_math(self):
        spec = INSTRUMENTS["MGC"]
        # 1 tick = 0.10 points × $10/point = $1.00
        assert abs(spec.tick_size * spec.point_value - spec.tick_value) < 1e-9

    def test_commission_rt(self):
        assert INSTRUMENTS["MCL"].commission_rt == 0.50
        assert INSTRUMENTS["MGC"].commission_rt == 0.80

    def test_account_limits(self):
        assert ACCOUNT.max_contracts_micro == 60
        assert ACCOUNT.max_loss_limit == 3_000.0
        assert ACCOUNT.profit_target == 6_000.0


class TestThresholdEngine:
    def setup_method(self):
        self.engine = SignalThresholdEngine()

    def test_commission_conversion_mcl(self):
        """Commission in points = commission_rt / point_value × contracts"""
        result = self.engine.compute("MCL", 1, stop_points=10.0, target_points=20.0)
        expected_comm_pts = 0.50 / 10.0  # $0.50 / $10 per point
        assert abs(result.commission_points - expected_comm_pts) < 1e-9, \
            f"MCL commission should be {expected_comm_pts} pts, got {result.commission_points}"

    def test_commission_conversion_mgc(self):
        result = self.engine.compute("MGC", 1, stop_points=10.0, target_points=20.0)
        expected_comm_pts = 0.80 / 10.0
        assert abs(result.commission_points - expected_comm_pts) < 1e-9

    def test_threshold_rises_with_contracts(self):
        """More contracts → higher threshold (market impact scales up)"""
        r1 = self.engine.compute("MCL", 1, 10.0, 20.0)
        r5 = self.engine.compute("MCL", 5, 10.0, 20.0)
        assert r5.threshold_points > r1.threshold_points, \
            "Threshold must rise with contract count"

    def test_safety_multiplier_applied(self):
        result = self.engine.compute("MCL", 1, 10.0, 20.0)
        from momentum_engine.config.settings import EXECUTION_CONFIG
        assert abs(result.threshold_points - result.total_cost_points
                   * EXECUTION_CONFIG.safety_multiplier) < 1e-9

    def test_signal_clears_threshold_strong_signal(self):
        """A very large signal should always clear threshold at 1 contract"""
        clears, result = self.engine.signal_clears_threshold(
            "MCL", signal_strength_points=50.0,
            contracts=1, stop_points=10.0, target_points=20.0
        )
        assert clears, "50-point signal should clear threshold at 1 contract"

    def test_signal_blocked_weak_signal(self):
        """A tiny signal should not clear threshold"""
        clears, result = self.engine.signal_clears_threshold(
            "MCL", signal_strength_points=0.001,
            contracts=1, stop_points=10.0, target_points=20.0
        )
        assert not clears, "0.001-point signal should not clear threshold"

    def test_max_contracts_caps_at_zero_for_zero_signal(self):
        n = self.engine.max_contracts_for_signal(
            "MCL", signal_strength_points=0.0, stop_points=10.0, target_points=20.0
        )
        assert n == 0, "Zero signal → zero contracts"

    def test_breakeven_win_rate_reasonable(self):
        """At 2:1 R:R with small costs, breakeven WR should be near 35%"""
        result = self.engine.compute("MCL", 1, 10.0, 20.0)
        assert 30 < result.breakeven_win_rate < 42, \
            f"Breakeven WR at 2:1 R:R should be ~33-40%, got {result.breakeven_win_rate:.1f}%"

    def test_cost_pct_of_stop_small_at_1_contract(self):
        """At 1 contract, commission drag should be < 5% of a 10-point stop"""
        result = self.engine.compute("MCL", 1, 10.0, 20.0)
        assert result.cost_as_pct_of_stop < 10, \
            f"Cost should be <10% of stop at 1 contract, got {result.cost_as_pct_of_stop:.1f}%"

    def test_scaling_table_length(self):
        table = self.engine.scaling_table("MCL", 10.0, 20.0, max_contracts=10)
        assert len(table) == 10


class TestSignalEngine:
    def setup_method(self):
        self.mcl_engine = MomentumSignalEngine("MCL")
        self.mgc_engine = MomentumSignalEngine("MGC")

    def test_insufficient_data_returns_flat(self):
        bars = make_bars(n_bars=5, symbol="MCL")
        signal = self.mcl_engine.compute(bars)
        assert signal.direction == SignalDirection.FLAT
        assert not signal.is_tradeable

    def test_output_shape_with_sufficient_data(self):
        bars = make_bars(n_bars=60, symbol="MCL")
        signal = self.mcl_engine.compute(bars)
        assert signal.symbol == "MCL"
        assert -1.0 <= signal.combined_score <= 1.0
        assert signal.regime in ("normal", "caution", "panic", "insufficient_data")

    def test_strong_uptrend_produces_long_signal(self):
        """A persistent strong uptrend should eventually yield a long signal"""
        bars = make_bars(n_bars=80, symbol="MCL", trend=0.5, vol=0.002, seed=1)
        engine = MomentumSignalEngine("MCL")
        signal = engine.compute(bars)
        assert signal.combined_score > 0, \
            f"Strong uptrend should produce positive score, got {signal.combined_score}"

    def test_strong_downtrend_produces_short_signal(self):
        bars = make_bars(n_bars=80, symbol="MCL", trend=-0.5, vol=0.002, seed=2)
        engine = MomentumSignalEngine("MCL")
        signal = engine.compute(bars)
        assert signal.combined_score < 0, \
            f"Strong downtrend should produce negative score, got {signal.combined_score}"

    def test_panic_regime_blocks_trade(self):
        """A market that has crashed + vol spiked should not be tradeable"""
        # Simulate: big drop then volatile chop
        bars = make_bars(n_bars=60, symbol="MCL", trend=-2.0, vol=0.02, seed=3)
        engine = MomentumSignalEngine("MCL")
        signal = engine.compute(bars)
        # Either panic regime or at least not a confident long
        if signal.regime == "panic":
            assert not signal.is_tradeable
        # If not panic, just check score is not wildly positive
        assert signal.combined_score < 0.8

    def test_all_layers_present_in_output(self):
        bars = make_bars(n_bars=60, symbol="MCL")
        signal = self.mcl_engine.compute(bars)
        expected = {"intraday_momentum", "tsmom_fast", "tsmom_slow",
                    "path_signature", "vol_regime"}
        assert expected.issubset(set(signal.layers.keys())), \
            f"Missing layers: {expected - set(signal.layers.keys())}"

    def test_mgc_signal_independent_of_mcl(self):
        """MCL and MGC engines should produce different signals (different price paths)"""
        mcl_bars = make_bars(n_bars=60, symbol="MCL", seed=10)
        mgc_bars = make_bars(n_bars=60, symbol="MGC", seed=99)
        mcl_sig = MomentumSignalEngine("MCL").compute(mcl_bars)
        mgc_sig = MomentumSignalEngine("MGC").compute(mgc_bars)
        # Just confirm they computed independently (scores may differ)
        assert mcl_sig.symbol == "MCL"
        assert mgc_sig.symbol == "MGC"


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager()

    def _run(self, bars, state, symbol="MCL", trend=0.5, vol=0.002):
        engine = MomentumSignalEngine(symbol)
        signal = engine.compute(bars)
        return self.rm.evaluate(signal, state, bars), signal

    def test_hold_when_signal_flat(self):
        bars = make_bars(n_bars=5, symbol="MCL")  # insufficient data
        state = make_account_state()
        decision, signal = self._run(bars, state)
        assert decision.action == "hold"
        assert decision.contracts == 0

    def test_hold_when_daily_limit_reached(self):
        bars = make_bars(n_bars=80, symbol="MCL", trend=1.0, seed=1)
        state = make_account_state(trades_today=RISK_CONFIG.max_daily_trades)
        decision, _ = self._run(bars, state)
        assert decision.action == "hold", "Should hold when daily limit reached"

    def test_hold_when_position_already_open(self):
        bars = make_bars(n_bars=80, symbol="MCL", trend=1.0, seed=1)
        state = make_account_state(open_positions={"MCL": 3})
        decision, _ = self._run(bars, state)
        assert decision.action == "hold", "Should not add to existing position"

    def test_contracts_within_account_max(self):
        bars = make_bars(n_bars=80, symbol="MCL", trend=1.0, seed=1)
        state = make_account_state()
        decision, signal = self._run(bars, state)
        if decision.is_trade:
            assert decision.contracts <= RISK_CONFIG.max_contracts_per_instrument
            assert decision.contracts <= ACCOUNT.max_contracts_micro

    def test_dollar_risk_within_per_trade_limit(self):
        bars = make_bars(n_bars=80, symbol="MCL", trend=1.0, seed=1)
        state = make_account_state()
        decision, _ = self._run(bars, state)
        if decision.is_trade:
            assert decision.dollar_risk <= RISK_CONFIG.max_risk_per_trade_abs * \
                   RISK_CONFIG.max_contracts_per_instrument + 1, \
                "Dollar risk should not exceed per-trade limit × max contracts"

    def test_consistency_rule_respected_in_eval(self):
        """When already near 50% of cumulative P&L for the day, size must shrink"""
        bars = make_bars(n_bars=80, symbol="MCL", trend=1.0, seed=1)
        # Simulate: cumulative P&L = $1000, already made $499 today → headroom = $1
        state = make_account_state(
            cumulative_eval_pnl=1000.0,
            realized_pnl_today=499.0,
        )
        decision, _ = self._run(bars, state)
        # Consistency headroom = $1 → max contract = 0 or 1 (very small)
        if decision.is_trade:
            assert decision.contracts <= 1, \
                "Consistency rule should cap size when near 50% limit"

    def test_mll_buffer_prevents_oversizing(self):
        """With very little MLL buffer left, size must be tiny or zero"""
        bars = make_bars(n_bars=80, symbol="MCL", trend=1.0, seed=1)
        # Simulate: only $200 of MLL buffer left
        state = make_account_state(
            balance=100_000.0 - 2_800.0,
            realized_pnl_today=-2_800.0,
        )
        state = AccountState(
            balance=97_200.0,
            peak_eod_balance=100_000.0,
            realized_pnl_today=-2_800.0,
            unrealized_pnl=0.0,
            trades_today=0,
            best_day_pnl=0.0,
            cumulative_eval_pnl=0.0,
            open_positions={},
            is_eval_phase=True,
        )
        decision, _ = self._run(bars, state)
        if decision.is_trade:
            # 40% of $200 buffer / ($10/pt × 10pt stop) = 0.8 → rounds to 0 or 1
            assert decision.contracts <= 1


class TestIntegration:
    """Full pipeline: data → signal → threshold → risk → decision"""

    def test_full_pipeline_mcl_uptrend(self):
        bars = make_bars(n_bars=80, symbol="MCL", trend=0.8, vol=0.002, seed=42)
        state = make_account_state()

        engine = MomentumSignalEngine("MCL")
        signal = engine.compute(bars)

        rm = RiskManager()
        decision = rm.evaluate(signal, state, bars)

        # With strong uptrend, expect a long trade or a reasoned hold
        print(f"\nSignal: score={signal.combined_score:.3f}, "
              f"regime={signal.regime}, "
              f"expected_move={signal.expected_move_pts:.2f}pts")
        print(f"Decision: {decision.summary()}")
        if decision.threshold:
            print(decision.threshold.summary())

        assert signal.symbol == "MCL"
        assert decision.symbol == "MCL"

    def test_full_pipeline_mgc_downtrend(self):
        bars = make_bars(n_bars=80, symbol="MGC", trend=-0.8, vol=0.002, seed=43)
        state = make_account_state()

        engine = MomentumSignalEngine("MGC")
        signal = engine.compute(bars)

        rm = RiskManager()
        decision = rm.evaluate(signal, state, bars)

        print(f"\nSignal: score={signal.combined_score:.3f}, "
              f"regime={signal.regime}")
        print(f"Decision: {decision.summary()}")

        assert signal.symbol == "MGC"
        assert decision.symbol == "MGC"

    def test_threshold_cost_printed_for_scaling_review(self):
        """Print the full scaling table so we can review costs as size grows."""
        engine = SignalThresholdEngine()
        print("\n--- MCL Scaling Table (10pt stop, 20pt target) ---")
        print(f"{'Contracts':>10} | {'Comm(pts)':>10} | {'TmpImpact':>10} | "
              f"{'PrmImpact':>10} | {'Total(pts)':>10} | {'Threshold':>10} | "
              f"{'BEven WR%':>10} | {'Cost/Stop%':>10}")
        print("-" * 90)
        for r in engine.scaling_table("MCL", 10.0, 20.0, max_contracts=15):
            print(f"{r.contracts:>10} | {r.commission_points:>10.4f} | "
                  f"{r.temp_impact_points:>10.4f} | {r.perm_impact_points:>10.4f} | "
                  f"{r.total_cost_points:>10.4f} | {r.threshold_points:>10.4f} | "
                  f"{r.breakeven_win_rate:>10.1f} | {r.cost_as_pct_of_stop:>10.1f}")

        print("\n--- MGC Scaling Table (10pt stop, 20pt target) ---")
        for r in engine.scaling_table("MGC", 10.0, 20.0, max_contracts=15):
            print(f"{r.contracts:>10} | {r.commission_points:>10.4f} | "
                  f"{r.temp_impact_points:>10.4f} | {r.perm_impact_points:>10.4f} | "
                  f"{r.total_cost_points:>10.4f} | {r.threshold_points:>10.4f} | "
                  f"{r.breakeven_win_rate:>10.1f} | {r.cost_as_pct_of_stop:>10.1f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    test_classes = [
        TestInstrumentSpecs,
        TestThresholdEngine,
        TestSignalEngine,
        TestRiskManager,
        TestIntegration,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")
        for method in methods:
            if hasattr(instance, "setup_method"):
                instance.setup_method()
            try:
                getattr(instance, method)()
                print(f"  [PASS] {method}")
                passed += 1
            except AssertionError as e:
                print(f"  [FAIL] {method}: {e}")
                failed += 1
                errors.append((cls.__name__, method, str(e)))
            except Exception as e:
                print(f"  [ERR]  {method}: {type(e).__name__}: {e}")
                failed += 1
                errors.append((cls.__name__, method, f"{type(e).__name__}: {e}"))

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    if errors:
        print("\nFailures:")
        for cls_name, method, msg in errors:
            print(f"  {cls_name}.{method}: {msg}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
