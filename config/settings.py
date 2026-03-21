"""
Central configuration for the MCL/MGC momentum strategy.
All instrument specs, account constraints, and signal parameters live here.
Nothing is hardcoded in logic modules — they all import from this file.
"""

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Instrument specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    full_symbol: str          # parent contract (CL, GC)
    point_value: float        # USD per full point per contract
    tick_size: float          # minimum price increment in points
    tick_value: float         # USD per tick per contract
    commission_rt: float      # round-trip commission per contract (Lucid)
    exchange: str
    session_open: str         # HH:MM ET
    session_close: str        # HH:MM ET (strategy closes by 4:30, before 4:45 auto-liq)


INSTRUMENTS: Dict[str, InstrumentSpec] = {
    "MCL": InstrumentSpec(
        symbol="MCL",
        full_symbol="CL",
        point_value=10.0,      # $10 per point (CL=$1000, MCL=1/10th)
        tick_size=0.01,        # $0.01/bbl
        tick_value=0.10,       # $0.10 per tick
        commission_rt=0.50,    # Lucid Trading round-trip
        exchange="NYMEX",
        session_open="09:00",
        session_close="16:30", # strategy hard close — before Lucid 16:45 auto-liq
    ),
    "MGC": InstrumentSpec(
        symbol="MGC",
        full_symbol="GC",
        point_value=10.0,      # $10 per point (GC=$100/oz × 100oz, MGC=1/10th × 10oz)
        tick_size=0.10,        # $0.10/oz
        tick_value=1.00,       # $1.00 per tick
        commission_rt=0.80,    # Lucid Trading round-trip
        exchange="COMEX",
        session_open="08:20",
        session_close="16:30",
    ),
}


# ---------------------------------------------------------------------------
# Lucid Trading 100K LucidFlex account constraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccountConfig:
    account_size: float          # nominal simulated balance
    max_loss_limit: float        # EOD max loss limit (trailing from peak EOD balance)
    profit_target: float         # eval profit target
    max_contracts_mini: int      # max mini contracts
    max_contracts_micro: int     # max micro contracts
    consistency_rule_pct: float  # eval only: best day <= X% of cumulative profit
    drawdown_type: str           # 'EOD' — only end-of-day balance matters
    hard_close_time: str         # HH:MM ET — Lucid auto-liquidates at 16:45
    strategy_close_time: str     # HH:MM ET — we close 15 min early for safety
    reset_fee: float


ACCOUNT = AccountConfig(
    account_size=100_000.0,
    max_loss_limit=3_000.0,
    profit_target=6_000.0,
    max_contracts_mini=6,
    max_contracts_micro=60,
    consistency_rule_pct=0.50,   # no single day > 50% of cumulative profit
    drawdown_type="EOD",
    hard_close_time="16:45",
    strategy_close_time="16:30",
    reset_fee=140.0,
)


# ---------------------------------------------------------------------------
# Signal engine parameters
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    # Intraday momentum (Gao et al. 2018)
    intraday_bar_minutes: int = 30        # bar resolution
    intraday_lookback_bars: int = 1       # first bar predicts last bar

    # TSMOM lookback windows (Moskowitz et al. 2012) — in bars
    ewma_fast_span: int = 8               # ~4 hours at 30-min bars
    ewma_slow_span: int = 40              # ~1 week at 30-min bars

    # Path signature (Chevyrev & Kormilitzin)
    signature_depth: int = 3             # truncation level L
    signature_channels: list = field(    # input streams for path construction
        default_factory=lambda: ["price", "volume", "cumulative_delta"]
    )
    signature_window_bars: int = 8        # rolling window for signature computation

    # Volatility surface proxy (Cont & da Fonseca) — for MCL/MGC without options
    vol_proxy_window: int = 20           # bars for realized vol (Factor 1 proxy)
    skew_proxy_window: int = 8           # bars for volume imbalance (Factor 2 proxy)

    # Regime gate (Daniel & Moskowitz 2016 crash filter)
    panic_lookback_bars: int = 40        # ~1 week
    panic_return_threshold: float = -0.04  # -4% triggers caution mode
    panic_vol_multiplier: float = 1.5    # vol must be > 1.5x 20-bar avg

    # Signal combination
    min_signal_strength: float = 0.0     # overridden by threshold engine
    signal_smoothing_bars: int = 2       # prevent flip-flopping


SIGNAL_CONFIG = SignalConfig()


# ---------------------------------------------------------------------------
# Risk & position sizing parameters
# Grounded in Barroso & Santa-Clara (2015) vol-targeting
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    # Volatility targeting (Barroso & Santa-Clara)
    target_daily_vol_pct: float = 0.006   # 0.6% of account = $6/day per $1000
    vol_lookback_bars: int = 12           # bars for realized vol estimate (6 hours)

    # Per-trade hard limits
    max_risk_per_trade_pct: float = 0.015 # max 1.5% of account per trade = $150
    max_risk_per_trade_abs: float = 150.0 # hard dollar cap per trade

    # Daily limits
    max_daily_loss_pct: float = 0.40      # stop trading if down 40% of MLL ($1200)
    max_daily_trades: int = 6             # circuit breaker on overtrading

    # Portfolio limits
    max_concurrent_positions: int = 2     # 1 MCL + 1 MGC max at once
    max_contracts_per_instrument: int = 5 # conservative start, scale up after funded

    # Stop loss
    default_stop_points: Dict[str, float] = field(
        default_factory=lambda: {"MCL": 10.0, "MGC": 10.0}
    )
    default_target_points: Dict[str, float] = field(
        default_factory=lambda: {"MCL": 20.0, "MGC": 20.0}
    )
    min_rr_ratio: float = 1.5            # minimum reward:risk to take a trade


RISK_CONFIG = RiskConfig()


# ---------------------------------------------------------------------------
# Almgren-Chriss execution / signal threshold config
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    # Market impact estimates (conservative for micro contracts)
    # MCL: ~0.5–1 tick temporary impact per contract at 5 contracts
    # MGC: ~0.5–1 tick temporary impact per contract at 5 contracts
    temp_impact_ticks_per_contract: Dict[str, float] = field(
        default_factory=lambda: {"MCL": 0.5, "MGC": 0.5}
    )
    perm_impact_ticks_per_contract: Dict[str, float] = field(
        default_factory=lambda: {"MCL": 0.1, "MGC": 0.1}
    )

    # Signal threshold: signal must exceed total execution cost to enter
    # threshold = commission_rt + temp_impact_cost + perm_impact_cost (in points)
    # Computed dynamically by threshold engine based on contracts sized
    safety_multiplier: float = 1.5       # require signal > 1.5x breakeven cost

    # TWAP/VWAP execution preference
    use_limit_orders: bool = True
    limit_order_aggression: float = 0.3  # 0=passive, 1=aggressive
    max_fill_wait_bars: int = 1          # cancel and re-evaluate after 1 bar


EXECUTION_CONFIG = ExecutionConfig()
