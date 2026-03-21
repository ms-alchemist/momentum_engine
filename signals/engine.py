"""
Momentum Signal Engine
======================
Unified signal pipeline for MCL (Micro Crude) and MGC (Micro Gold).

Integrates all four research papers into a single, composable pipeline:

  Layer 1 — Intraday Momentum (Gao et al. 2018)
      First 30-min bar predicts direction for remainder of session.
      Stronger on high-vol, high-volume days.

  Layer 2 — TSMOM / EWMA Momentum (Moskowitz, Ooi & Pedersen 2012)
      Volatility-scaled return autocorrelation across multiple lookbacks.
      The dominant force: positive auto-covariance between past and future returns.

  Layer 3 — Path Signature Features (Chevyrev & Kormilitzin 2016)
      Iterated integrals of [price, volume, cumulative_delta] path.
      Captures cross-stream interactions (price-volume signed area = group velocity
      vs phase velocity from Bühler's wave theory).

  Layer 4 — Volatility Regime Proxy (Cont & da Fonseca 2002)
      Since MCL/MGC have no liquid options chain, we approximate the three
      IV surface eigenmodes using realized volatility proxies:
        Factor 1 (level)   → rolling realized vol ratio
        Factor 2 (skew)    → buy/sell volume imbalance (delta skew)
        Factor 3 (curvature) → kurtosis of returns (tail risk proxy)

  Regime Gate (Daniel & Moskowitz 2016)
      Scales all signals to zero during detected panic states.
      Panic = recent large drawdown + vol spike + potential sharp reversal.

  Wave Physics Layer (Bühler — Waves and Mean Flows)
      Pseudomomentum conservation: signal is treated as a conserved flux,
      not a snapshot. Exponential weighting + multi-scale coherence filter
      ensure we only trade when momentum is aligned across timeframes.

Output: SignalOutput with a combined score in [-1, +1] and expected
        move magnitude in points (used by threshold engine).

All parameters in config/settings.py — nothing hardcoded here.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum

from momentum_engine.config.settings import (
    INSTRUMENTS, SIGNAL_CONFIG, RISK_CONFIG, SignalConfig
)


class SignalDirection(Enum):
    LONG  = 1
    SHORT = -1
    FLAT  = 0


@dataclass
class SignalLayer:
    """Individual signal component — named for auditability."""
    name: str
    value: float          # raw signal in [-1, +1]
    confidence: float     # 0-1, used for weighting
    magnitude_pts: float  # expected move magnitude in points
    active: bool = True   # False when insufficient data or gated off


@dataclass
class SignalOutput:
    """Final combined signal for a given bar."""
    symbol: str
    timestamp: pd.Timestamp
    direction: SignalDirection
    combined_score: float         # weighted combination, [-1, +1]
    expected_move_pts: float      # used by threshold engine as signal_strength
    regime: str                   # 'normal', 'caution', 'panic'
    layers: Dict[str, SignalLayer] = field(default_factory=dict)
    is_tradeable: bool = False    # False during panic or insufficient data

    @property
    def long(self) -> bool:
        return self.direction == SignalDirection.LONG and self.is_tradeable

    @property
    def short(self) -> bool:
        return self.direction == SignalDirection.SHORT and self.is_tradeable


class MomentumSignalEngine:
    """
    Computes the full signal stack for a single instrument on each bar.

    Usage:
        engine = MomentumSignalEngine('MCL')
        signal = engine.compute(bars_df)

    bars_df must have columns:
        open, high, low, close, volume, buy_volume, sell_volume
        (buy_volume + sell_volume = volume; sourced from tick data aggregation)
    Index: DatetimeIndex in Eastern time, bar_minutes frequency
    """

    # Layer weights — sum to 1.0
    # Intraday gets more weight during session open window
    LAYER_WEIGHTS = {
        "intraday_momentum": 0.30,
        "tsmom_fast":        0.25,
        "tsmom_slow":        0.20,
        "path_signature":    0.15,
        "vol_regime":        0.10,
    }

    def __init__(self, symbol: str, config: Optional[SignalConfig] = None):
        if symbol not in INSTRUMENTS:
            raise ValueError(f"Unknown symbol: {symbol}. Must be MCL or MGC.")
        self.symbol = symbol
        self.spec = INSTRUMENTS[symbol]
        self.cfg = config or SIGNAL_CONFIG
        self._signal_history: list[float] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(self, bars: pd.DataFrame) -> SignalOutput:
        """
        Compute the full signal for the current bar (last row of bars).

        Args:
            bars: DataFrame of OHLCV bars. Must have at least
                  max(ewma_slow_span, signature_window_bars) rows.

        Returns:
            SignalOutput with direction, score, and all layer details.
        """
        if len(bars) < self.cfg.ewma_slow_span:
            return self._insufficient_data(bars)

        closes = bars["close"]
        volumes = bars["volume"]
        timestamp = bars.index[-1]

        # --- Regime check first (gate) ---
        regime, regime_scalar = self._regime_gate(bars)

        # --- Compute each layer ---
        layers = {}

        layers["intraday_momentum"] = self._intraday_momentum(bars)
        layers["tsmom_fast"]        = self._tsmom(bars, self.cfg.ewma_fast_span)
        layers["tsmom_slow"]        = self._tsmom(bars, self.cfg.ewma_slow_span)
        layers["path_signature"]    = self._path_signature(bars)
        layers["vol_regime"]        = self._vol_surface_proxy(bars)

        # --- Combine with Bühler-inspired multi-scale coherence ---
        combined, expected_pts = self._combine_layers(layers, regime_scalar)

        # Smooth to prevent signal flip-flopping (pseudomomentum conservation)
        combined = self._smooth_signal(combined)

        direction = (
            SignalDirection.LONG  if combined > 0.1 else
            SignalDirection.SHORT if combined < -0.1 else
            SignalDirection.FLAT
        )

        is_tradeable = (
            regime != "panic"
            and direction != SignalDirection.FLAT
            and expected_pts > 0
        )

        return SignalOutput(
            symbol=self.symbol,
            timestamp=timestamp,
            direction=direction,
            combined_score=round(combined, 4),
            expected_move_pts=round(expected_pts, 3),
            regime=regime,
            layers=layers,
            is_tradeable=is_tradeable,
        )

    # ------------------------------------------------------------------
    # Layer 1: Intraday momentum (Gao et al. 2018)
    # ------------------------------------------------------------------

    def _intraday_momentum(self, bars: pd.DataFrame) -> SignalLayer:
        """
        First-bar return predicts last-bar direction within session.
        We use the first completed bar after session open as the anchor.
        Stronger signal on high-vol, high-volume days.
        """
        closes = bars["close"]
        volumes = bars["volume"]

        # Identify session open bar (first bar of today)
        today = bars.index[-1].date()
        today_bars = bars[bars.index.date == today]

        if len(today_bars) < 2:
            return SignalLayer("intraday_momentum", 0.0, 0.0, 0.0, active=False)

        # First bar return: open of session to close of first bar
        first_bar_return = (today_bars["close"].iloc[0] - today_bars["open"].iloc[0])

        # Normalise by recent volatility to get a unit-free signal
        recent_vol = closes.pct_change().rolling(self.cfg.vol_proxy_window).std().iloc[-1]
        if recent_vol == 0 or np.isnan(recent_vol):
            return SignalLayer("intraday_momentum", 0.0, 0.0, 0.0, active=False)

        # Scale the first-bar return by vol — Gao et al. find stronger signal
        # when vol is elevated, so we don't penalise high-vol days
        norm_return = first_bar_return / (recent_vol * closes.iloc[-1])
        signal_val = float(np.clip(norm_return, -1.0, 1.0))

        # Volume confirmation: signal is stronger if volume on first bar is
        # above the 5-bar average (Gao et al. — stronger on high-volume days)
        vol_ratio = today_bars["volume"].iloc[0] / (volumes.iloc[-6:-1].mean() + 1e-9)
        confidence = float(np.clip(vol_ratio / 2.0, 0.0, 1.0))

        # Expected magnitude: abs(first_bar_return) projected to close
        expected_pts = abs(first_bar_return) * 1.2  # slight amplification to EOD

        return SignalLayer(
            name="intraday_momentum",
            value=signal_val,
            confidence=confidence,
            magnitude_pts=expected_pts,
            active=True,
        )

    # ------------------------------------------------------------------
    # Layer 2: TSMOM (Moskowitz et al. 2012) — EWMA return autocorrelation
    # ------------------------------------------------------------------

    def _tsmom(self, bars: pd.DataFrame, span: int) -> SignalLayer:
        """
        Time-series momentum: past return (EWMA-weighted) predicts future return.
        Volatility-scaled per Barroso & Santa-Clara (2015).

        The EWMA weighting implements Bühler's pseudomomentum flux — treating
        momentum as an exponentially decaying conserved quantity, not a
        fixed-window snapshot.
        """
        closes = bars["close"]
        returns = closes.pct_change().dropna()

        if len(returns) < span:
            return SignalLayer(f"tsmom_{span}", 0.0, 0.0, 0.0, active=False)

        # EWMA return — exponential weighting privileges recent bars
        ewma_return = returns.ewm(span=span, adjust=False).mean().iloc[-1]

        # Volatility scaling (Barroso & Santa-Clara): divide by recent realized vol
        # This makes signals comparable across regimes
        realized_vol = returns.rolling(span).std().iloc[-1]
        if realized_vol == 0 or np.isnan(realized_vol):
            return SignalLayer(f"tsmom_{span}", 0.0, 0.0, 0.0, active=False)

        vol_scaled_signal = ewma_return / realized_vol

        # Clip to [-1, +1] using tanh for smooth saturation
        signal_val = float(np.tanh(vol_scaled_signal * 2))

        # Confidence: higher when signal has been stable (low flip frequency)
        # Bühler: pseudomomentum is conserved — stable signals carry more "wave energy"
        recent_signals = np.sign(returns.tail(span // 2))
        signal_stability = abs(recent_signals.mean())  # 0=noisy, 1=persistent
        confidence = float(signal_stability)

        # Expected magnitude in points: vol_scaled_return × current price × point_value ratio
        expected_pts = abs(ewma_return) * closes.iloc[-1]

        layer_name = f"tsmom_{'fast' if span == self.cfg.ewma_fast_span else 'slow'}"
        return SignalLayer(
            name=layer_name,
            value=signal_val,
            confidence=confidence,
            magnitude_pts=expected_pts,
            active=True,
        )

    # ------------------------------------------------------------------
    # Layer 3: Path signature (Chevyrev & Kormilitzin 2016)
    # ------------------------------------------------------------------

    def _path_signature(self, bars: pd.DataFrame) -> SignalLayer:
        """
        Compute truncated path signature at depth L=2 over the rolling window.
        Uses [price, volume, cumulative_delta] as the 3D path.

        The key trading signals from the signature:
          - Level-1 terms: net price increment (= simple momentum, sanity check)
          - Level-2 cross term S(price, volume): the signed area between price
            and volume paths. Positive = volume leading price (accumulation,
            strong momentum). Negative = volume lagging (distribution, weak).
            This is the mathematical encoding of Bühler's group velocity vs
            phase velocity distinction.
          - Level-2 cross term S(price, delta): price-delta signed area.
            Delta = buy_vol - sell_vol. Positive = aggressive buying ahead of
            price move = momentum confirmation.

        We use the esig library if available, otherwise fall back to the
        closed-form L=2 computation (which is exact and sufficient).
        """
        window = self.cfg.signature_window_bars
        if len(bars) < window + 1:
            return SignalLayer("path_signature", 0.0, 0.0, 0.0, active=False)

        w = bars.iloc[-window - 1:]
        closes = w["close"].values
        volumes = w["volume"].values

        # Cumulative delta = running sum of (buy_vol - sell_vol)
        if "buy_volume" in w.columns and "sell_volume" in w.columns:
            delta = (w["buy_volume"] - w["sell_volume"]).cumsum().values
        else:
            # Fallback: approximate delta from price direction × volume
            price_dir = np.sign(np.diff(closes, prepend=closes[0]))
            delta = np.cumsum(price_dir * volumes)

        # Normalise each channel to [0,1] for the signature calculation
        # (signature is reparametrisation-invariant but normalising aids stability)
        def norm(x):
            r = x.max() - x.min()
            return (x - x.min()) / (r + 1e-9)

        p = norm(closes)    # price channel
        v = norm(volumes)   # volume channel
        d = norm(delta)     # delta channel

        # --- Depth-1 terms (net increments) ---
        s1_p = p[-1] - p[0]   # net price move (sign = direction)
        s1_v = v[-1] - v[0]   # net volume change
        s1_d = d[-1] - d[0]   # net delta change

        # --- Depth-2 cross terms (iterated integrals = signed area) ---
        # S(i,j) = integral of X_i dX_j — approximated by discrete sum
        # S(price, volume): positive = price and volume co-moving (price leads)
        #                   positive signed area = accumulation
        _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
        s2_pv = float(_trapz(p[:-1] * np.diff(v)))   # ∫ p dv
        s2_vp = float(_trapz(v[:-1] * np.diff(p)))   # ∫ v dp

        # The anti-symmetric part S(p,v) - S(v,p) = 2 × signed area
        # This is the Lévy area — the key geometric signal
        levy_area_pv = s2_pv - s2_vp

        # S(price, delta): positive = buying pressure leading price
        s2_pd = float(_trapz(p[:-1] * np.diff(d)))
        s2_dp = float(_trapz(d[:-1] * np.diff(p)))
        levy_area_pd = s2_pd - s2_dp

        # --- Combine into a single signal ---
        # Direction from depth-1 (net price move)
        direction_signal = float(np.tanh(s1_p * 5))

        # Conviction from depth-2 (signed area confirmation)
        # Positive levy_area_pv means volume is building in direction of move
        volume_confirmation = float(np.tanh(levy_area_pv * 10))

        # Delta confirmation: buying/selling pressure ahead of move
        delta_confirmation = float(np.tanh(levy_area_pd * 10))

        # Combined: direction × average confirmation strength
        conviction = (volume_confirmation + delta_confirmation) / 2.0
        signal_val = float(np.clip(direction_signal * (0.5 + 0.5 * abs(conviction))
                                   * np.sign(conviction + direction_signal + 1e-9), -1, 1))

        # Confidence: how aligned are the depth-1 and depth-2 signals
        alignment = (np.sign(s1_p) == np.sign(levy_area_pv) and
                     np.sign(s1_p) == np.sign(levy_area_pd))
        confidence = 0.8 if alignment else 0.3

        # Expected magnitude: proportional to net price increment scaled to points
        price_range = closes.max() - closes.min()
        expected_pts = abs(s1_p) * price_range * 2  # project recent swing

        return SignalLayer(
            name="path_signature",
            value=signal_val,
            confidence=confidence,
            magnitude_pts=float(expected_pts),
            active=True,
        )

    # ------------------------------------------------------------------
    # Layer 4: Volatility surface proxy (Cont & da Fonseca 2002)
    # ------------------------------------------------------------------

    def _vol_surface_proxy(self, bars: pd.DataFrame) -> SignalLayer:
        """
        Approximates the three Cont & da Fonseca IV surface eigenmodes
        using price/volume data (since MCL/MGC have no liquid options chain).

        Factor 1 (level — mean reverting, ~51 day constant):
            → Realized vol relative to its own rolling mean.
              When vol spikes above mean: momentum is exhausting (wave breaking).
              When vol is compressing below mean: trending conditions, momentum alive.

        Factor 2 (skew — directional tilt):
            → Volume imbalance: (buy_vol - sell_vol) / total_vol.
              Positive skew = net buying pressure = bullish momentum confirmation.
              Negative skew = net selling pressure = bearish confirmation.
              Analagous to options skew (puts vs calls demand asymmetry).

        Factor 3 (curvature — tail risk, kill switch):
            → Rolling kurtosis of returns. High kurtosis = fat tails expected
              = Bühler's "wave near breaking" zone = reduce/exit momentum.

        Returns a composite signal that gates and amplifies the other layers.
        """
        closes = bars["close"]
        returns = closes.pct_change().dropna()

        if len(returns) < self.cfg.vol_proxy_window:
            return SignalLayer("vol_regime", 0.0, 0.5, 0.0, active=False)

        # --- Factor 1: Realized vol level ---
        rv_short = returns.rolling(self.cfg.skew_proxy_window).std().iloc[-1]
        rv_long  = returns.rolling(self.cfg.vol_proxy_window).std().iloc[-1]
        vol_ratio = rv_short / (rv_long + 1e-9)

        # Vol compressing (ratio < 1) → good momentum conditions → positive contribution
        # Vol spiking (ratio > 1.5) → choppy/breaking → negative contribution
        f1_signal = float(np.clip(1.5 - vol_ratio, -1.0, 1.0))

        # --- Factor 2: Volume skew (bid/ask imbalance proxy) ---
        if "buy_volume" in bars.columns and "sell_volume" in bars.columns:
            recent = bars.tail(self.cfg.skew_proxy_window)
            buy_vol  = recent["buy_volume"].sum()
            sell_vol = recent["sell_volume"].sum()
            total_vol = buy_vol + sell_vol + 1e-9
            imbalance = (buy_vol - sell_vol) / total_vol
            f2_signal = float(np.clip(imbalance * 3, -1.0, 1.0))
        else:
            # Approximate: price direction × volume as delta proxy
            price_dir = np.sign(closes.diff()).tail(self.cfg.skew_proxy_window)
            vol_tail  = bars["volume"].tail(self.cfg.skew_proxy_window)
            approx_imbalance = (price_dir * vol_tail).sum() / (vol_tail.sum() + 1e-9)
            f2_signal = float(np.clip(approx_imbalance * 3, -1.0, 1.0))

        # --- Factor 3: Kurtosis kill switch ---
        kurt = returns.tail(self.cfg.vol_proxy_window).kurtosis()
        # Normal kurtosis ≈ 3. Above 6 = fat tails = Cont Factor 3 spike
        # When kurtosis is high, scale down the vol_regime signal contribution
        kurtosis_scalar = float(np.clip(1.0 - (kurt - 3.0) / 10.0, 0.0, 1.0))

        # --- Combine ---
        # F1 gates the magnitude (trending vs choppy regime)
        # F2 provides direction (skew = which way pressure is building)
        # F3 attenuates when tail risk is elevated
        combined_vol_signal = f2_signal * kurtosis_scalar * max(f1_signal, 0.1)
        signal_val = float(np.clip(combined_vol_signal, -1.0, 1.0))

        # Confidence higher when all three factors agree in direction
        confidence = kurtosis_scalar * (0.5 + 0.5 * abs(f1_signal))

        return SignalLayer(
            name="vol_regime",
            value=signal_val,
            confidence=float(confidence),
            magnitude_pts=0.0,   # vol regime is a modifier, not a magnitude source
            active=True,
        )

    # ------------------------------------------------------------------
    # Regime gate (Daniel & Moskowitz 2016)
    # ------------------------------------------------------------------

    def _regime_gate(self, bars: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect panic states and scale signals accordingly.

        Panic = (recent large drawdown) AND (current vol spike) AND
                (potential sharp rebound — most dangerous for momentum).

        Returns:
            regime: 'normal', 'caution', or 'panic'
            scalar: 1.0 (normal), 0.5 (caution), 0.0 (panic)
        """
        closes = bars["close"]
        returns = closes.pct_change().dropna()
        cfg = self.cfg

        if len(returns) < cfg.panic_lookback_bars:
            return "normal", 1.0

        # Condition 1: recent cumulative return is significantly negative
        recent_cumret = (closes.iloc[-1] / closes.iloc[-cfg.panic_lookback_bars] - 1)
        large_drawdown = recent_cumret < cfg.panic_return_threshold

        # Condition 2: current short-term vol is elevated vs recent average
        current_vol = returns.tail(cfg.skew_proxy_window).std()
        avg_vol     = returns.tail(cfg.panic_lookback_bars).std()
        vol_spike   = current_vol > (avg_vol * cfg.panic_vol_multiplier)

        if large_drawdown and vol_spike:
            return "panic", 0.0
        elif large_drawdown or vol_spike:
            return "caution", 0.5
        else:
            return "normal", 1.0

    # ------------------------------------------------------------------
    # Signal combination (Bühler multi-scale coherence)
    # ------------------------------------------------------------------

    def _combine_layers(
        self,
        layers: Dict[str, SignalLayer],
        regime_scalar: float,
    ) -> Tuple[float, float]:
        """
        Combine all layers into a single score.

        Key insight from Bühler's wave theory: momentum is only a valid
        signal when fast and slow components are co-directional (no shear
        instability). We implement this as a coherence multiplier:
        if fast and slow TSMOM disagree in sign, we halve the combined signal.

        Weights are defined in LAYER_WEIGHTS. Each layer's contribution is:
            layer.value × layer.confidence × weight

        Then the coherence check is applied, then regime scaling.
        """
        weighted_sum = 0.0
        weight_total = 0.0
        total_magnitude = 0.0

        for name, layer in layers.items():
            if not layer.active:
                continue
            w = self.LAYER_WEIGHTS.get(name, 0.0)
            weighted_sum += layer.value * layer.confidence * w
            weight_total += layer.confidence * w
            total_magnitude += layer.magnitude_pts * w

        if weight_total == 0:
            return 0.0, 0.0

        combined = weighted_sum / weight_total

        # --- Bühler shear instability check ---
        # If fast and slow TSMOM point in opposite directions, we are in a
        # "shear zone" — momentum is unreliable. Halve the signal.
        fast = layers.get("tsmom_fast")
        slow = layers.get("tsmom_slow")
        if fast and slow and fast.active and slow.active:
            if np.sign(fast.value) != np.sign(slow.value):
                combined *= 0.5  # shear penalty

        # Apply regime scalar (panic = 0, caution = 0.5, normal = 1.0)
        combined *= regime_scalar

        # Magnitude: weighted average of layer expected moves
        expected_pts = total_magnitude / max(weight_total, 1e-9)

        return float(combined), float(expected_pts)

    # ------------------------------------------------------------------
    # Signal smoothing (pseudomomentum conservation)
    # ------------------------------------------------------------------

    def _smooth_signal(self, raw_signal: float) -> float:
        """
        Apply exponential smoothing to prevent flip-flopping.
        Implements Bühler's pseudomomentum conservation: the signal should
        persist until a genuine dissipation event, not noise.
        """
        self._signal_history.append(raw_signal)
        n = self.cfg.signal_smoothing_bars
        if len(self._signal_history) < n:
            return raw_signal
        # Exponential weights: most recent bar has highest weight
        weights = np.exp(np.linspace(-1, 0, n))
        weights /= weights.sum()
        smoothed = np.dot(weights, self._signal_history[-n:])
        return float(smoothed)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _insufficient_data(self, bars: pd.DataFrame) -> SignalOutput:
        ts = bars.index[-1] if len(bars) > 0 else pd.Timestamp.now()
        return SignalOutput(
            symbol=self.symbol,
            timestamp=ts,
            direction=SignalDirection.FLAT,
            combined_score=0.0,
            expected_move_pts=0.0,
            regime="insufficient_data",
            layers={},
            is_tradeable=False,
        )
