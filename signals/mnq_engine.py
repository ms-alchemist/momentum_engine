"""
signals/mnq_engine.py
=====================
Full six-layer signal stack for MNQ (Micro Nasdaq) intraday momentum.

Research foundation:
  Layer 1 — TSMOM (Moskowitz, Ooi & Pedersen 2012)
      EWMA-weighted return autocorrelation. Momentum as a conserved flux,
      not a snapshot. Signal = vol-scaled EWMA of 30-min returns.

  Layer 2 — Multi-scale coherence (Bühler — Waves and Mean Flows)
      Shear instability filter. Only trade when intraday EWMA direction
      agrees with prior-day return. Divergence = Kelvin-Helmholtz zone.

  Layer 3 — Lévy area volume confirmation (Chevyrev & Kormilitzin 2016)
      Signed area between normalized price and volume paths (depth-2
      path signature). Positive = volume leading price = genuine
      accumulation. Negative = distribution. Must confirm signal direction.

  Layer 4 — Wave breaking / regime gate (Bühler + Daniel & Moskowitz 2016)
      Skip when short-term realized vol > 1.5× 20-bar average.
      This is Bühler's dissipation threshold — pseudomomentum breaks down
      when volatility spikes, exactly as Daniel & Moskowitz document for
      momentum crashes.

  Layer 5 — Cont Factor 1 proxy (Cont & da Fonseca 2002)
      IV surface level approximated by rv_short / rv_long ratio.
      Compressing vol (ratio < 1.0) = wave settling = momentum conditions.
      Spiking vol (ratio > 1.3) = wave breaking = reduce exposure.

  Layer 6 — Cont Factor 3 proxy / tail risk kill switch
      Rolling kurtosis of 30-min returns. Kurtosis > 6 = fat tails =
      market pricing in discontinuity = stand aside completely.

Output: MNQSignal with direction, strength (0-1), and all layer values
        for auditability.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# EWMA spans in bars (30-min bars)
EWMA_FAST_SPAN   = 4    # 2 hours
EWMA_SLOW_SPAN   = 16   # 8 hours (one full session)

# Coherence filter: prior day lookback
PRIOR_DAY_BARS   = 16   # ~one session of 30-min bars

# Lévy area window
LEVY_WINDOW      = 6    # 3 hours of 30-min bars

# Regime gate thresholds
VOL_SPIKE_MULT   = 1.5  # short vol > 1.5x long vol = wave breaking
VOL_SHORT_BARS   = 4    # 2 hours for short vol estimate
VOL_LONG_BARS    = 20   # 10 hours for long vol baseline

# Cont Factor 1 proxy
CONT_F1_COMPRESS = 1.0  # below this = good momentum conditions
CONT_F1_SPIKE    = 1.3  # above this = reduce signal weight

# Cont Factor 3 (kurtosis) kill switch
KURTOSIS_WINDOW  = 20
KURTOSIS_LIMIT   = 6.0  # above = fat tails = stand aside

# Signal thresholds
SIGNAL_THRESHOLD = 0.15  # minimum combined score to generate a signal
DIRECTION_THRESHOLD = 0.10  # minimum for directional classification


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    name:       str
    value:      float    # raw value in natural units
    score:      float    # normalized contribution [-1, +1]
    weight:     float    # weight in combination
    active:     bool     # False = insufficient data or gated off
    note:       str = ""


@dataclass
class MNQSignal:
    """Complete signal output for one bar."""
    timestamp:      pd.Timestamp
    direction:      int          # +1 long, -1 short, 0 flat
    strength:       float        # 0-1, overall signal confidence
    combined_score: float        # weighted combination [-1, +1]
    is_tradeable:   bool
    regime:         str          # 'normal', 'caution', 'blocked'
    layers:         dict = field(default_factory=dict)

    # Derived stop/target suggestion (ATR-based)
    atr_pts:        float = 0.0
    stop_pts:       float = 0.0
    target_pts:     float = 0.0

    @property
    def long(self):
        return self.direction == 1 and self.is_tradeable

    @property
    def short(self):
        return self.direction == -1 and self.is_tradeable


# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------

class MNQSignalEngine:
    """
    Computes the full six-layer signal stack for MNQ on each bar.

    Usage:
        engine = MNQSignalEngine()
        signal = engine.compute(bars_df)

    bars_df: 30-min OHLCV DataFrame with buy_volume, sell_volume columns.
             Index: DatetimeIndex (America/New_York timezone).
    """

    # Layer weights — sum to 1.0
    # TSMOM and coherence filter carry the most weight (primary edge)
    # Chevyrev and Cont layers are confirmation/gating
    WEIGHTS = {
        "tsmom":        0.30,
        "coherence":    0.25,
        "levy_area":    0.20,
        "regime_gate":  0.10,
        "cont_f1":      0.10,
        "cont_f3":      0.05,
    }

    def __init__(self):
        self._trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

    def compute(self, bars: pd.DataFrame) -> MNQSignal:
        """
        Compute full signal on the last bar of bars.

        Args:
            bars: DataFrame of 30-min bars up to and including current bar.
                  Must have: open, high, low, close, volume,
                             buy_volume, sell_volume

        Returns:
            MNQSignal
        """
        ts = bars.index[-1]

        if len(bars) < EWMA_SLOW_SPAN:
            return self._flat(ts, "insufficient_data")

        closes  = bars["close"]
        returns = closes.pct_change().dropna()

        # --- Regime check first (kill switch) ---
        regime, regime_scalar = self._regime_gate(returns, ts)
        if regime == "blocked":
            return self._flat(ts, "blocked")

        # --- Compute all layers ---
        layers = {}
        layers["tsmom"]       = self._tsmom(returns, closes)
        layers["coherence"]   = self._coherence_filter(bars, returns)
        layers["levy_area"]   = self._levy_area(bars)
        layers["regime_gate"] = self._regime_layer(returns)
        layers["cont_f1"]     = self._cont_f1(returns)
        layers["cont_f3"]     = self._cont_f3(returns)

        # --- Combine ---
        combined = self._combine(layers, regime_scalar)

        # --- Direction ---
        if combined > DIRECTION_THRESHOLD:
            direction = 1
        elif combined < -DIRECTION_THRESHOLD:
            direction = -1
        else:
            direction = 0

        strength = min(abs(combined) / 0.5, 1.0)  # normalize to 0-1

        is_tradeable = (
            regime != "blocked"
            and direction != 0
            and abs(combined) >= SIGNAL_THRESHOLD
        )

        # --- ATR-based stop/target ---
        atr = self._atr(bars)
        stop_pts   = round(atr * 1.5, 1)
        target_pts = round(atr * 2.5, 1)

        return MNQSignal(
            timestamp      = ts,
            direction      = direction,
            strength       = round(strength, 3),
            combined_score = round(combined, 4),
            is_tradeable   = is_tradeable,
            regime         = regime,
            layers         = layers,
            atr_pts        = round(atr, 1),
            stop_pts       = stop_pts,
            target_pts     = target_pts,
        )

    # ------------------------------------------------------------------
    # Layer 1: TSMOM — Moskowitz + Bühler pseudomomentum flux
    # ------------------------------------------------------------------

    def _tsmom(self, returns: pd.Series, closes: pd.Series) -> LayerResult:
        """
        EWMA-weighted return = Bühler pseudomomentum flux.
        Exponential decay treats momentum as a conserved quantity
        that decays until a breaking event dissipates it.
        """
        if len(returns) < EWMA_FAST_SPAN:
            return LayerResult("tsmom", 0, 0, self.WEIGHTS["tsmom"],
                               False, "insufficient data")

        # Fast EWMA (2-hour)
        ewma_fast = returns.ewm(span=EWMA_FAST_SPAN, adjust=False).mean().iloc[-1]

        # Slow EWMA (8-hour) for multi-scale coherence check
        if len(returns) >= EWMA_SLOW_SPAN:
            ewma_slow = returns.ewm(span=EWMA_SLOW_SPAN, adjust=False).mean().iloc[-1]
        else:
            ewma_slow = ewma_fast

        # Vol-scale (Barroso & Santa-Clara)
        rv = returns.tail(EWMA_FAST_SPAN).std()
        if rv == 0 or np.isnan(rv):
            return LayerResult("tsmom", 0, 0, self.WEIGHTS["tsmom"],
                               False, "zero vol")

        vol_scaled = ewma_fast / rv

        # Shear check: fast and slow same direction?
        shear_penalty = 1.0
        if np.sign(ewma_fast) != np.sign(ewma_slow):
            shear_penalty = 0.4  # Bühler shear instability penalty

        score = float(np.tanh(vol_scaled * 3)) * shear_penalty
        score = float(np.clip(score, -1, 1))

        return LayerResult(
            name   = "tsmom",
            value  = float(ewma_fast),
            score  = score,
            weight = self.WEIGHTS["tsmom"],
            active = True,
            note   = f"ewma={ewma_fast:.5f} rv={rv:.5f} shear={shear_penalty}"
        )

    # ------------------------------------------------------------------
    # Layer 2: Multi-scale coherence filter — Bühler shear instability
    # ------------------------------------------------------------------

    def _coherence_filter(self, bars: pd.DataFrame,
                          returns: pd.Series) -> LayerResult:
        """
        Only trade when intraday momentum agrees with prior-day trend.
        Disagreement = Kelvin-Helmholtz shear zone = unstable regime.
        """
        closes = bars["close"]

        # Prior day return
        today      = bars.index[-1].date()
        prior_bars = bars[bars.index.date < today]

        if len(prior_bars) < 2:
            # No prior day data — neutral, don't block
            return LayerResult("coherence", 0, 0.5,
                               self.WEIGHTS["coherence"], True,
                               "no prior day — neutral")

        prior_day_return = (prior_bars["close"].iloc[-1] /
                            prior_bars["close"].iloc[0] - 1)

        # Intraday EWMA direction
        intraday_ewma = returns.tail(EWMA_FAST_SPAN).mean()

        prior_sign   = np.sign(prior_day_return)
        intraday_sign = np.sign(intraday_ewma)

        if prior_sign == intraday_sign and prior_sign != 0:
            # Aligned — amplify signal
            alignment_score = 1.0
            note = "aligned"
        elif prior_sign == 0 or intraday_sign == 0:
            alignment_score = 0.5
            note = "neutral"
        else:
            # Divergent — shear zone, penalize heavily
            alignment_score = -0.3
            note = "SHEAR divergence"

        # Score reflects intraday direction weighted by coherence
        score = float(np.tanh(intraday_ewma * 200)) * alignment_score
        score = float(np.clip(score, -1, 1))

        return LayerResult(
            name   = "coherence",
            value  = float(prior_day_return),
            score  = score,
            weight = self.WEIGHTS["coherence"],
            active = True,
            note   = note
        )

    # ------------------------------------------------------------------
    # Layer 3: Lévy area — Chevyrev path signature (depth-2 cross term)
    # ------------------------------------------------------------------

    def _levy_area(self, bars: pd.DataFrame) -> LayerResult:
        """
        Signed area between normalized price and volume paths.
        This is the depth-2 cross-term of the path signature:
          S(price, volume) - S(volume, price) = 2 × Lévy area

        Positive: volume leading price = accumulation = momentum genuine
        Negative: price leading volume = distribution = momentum exhausting

        Connects to Bühler's group vs phase velocity distinction:
        volume = group velocity carrier, price = phase.
        """
        window = LEVY_WINDOW
        if len(bars) < window + 1:
            return LayerResult("levy_area", 0, 0,
                               self.WEIGHTS["levy_area"], False,
                               "insufficient data")

        w = bars.iloc[-window - 1:]

        closes  = w["close"].values.astype(float)
        volumes = w["volume"].values.astype(float)

        # Use buy_volume if available for cleaner signal
        if "buy_volume" in w.columns and w["buy_volume"].sum() > 0:
            vol_signal = w["buy_volume"].values.astype(float)
        else:
            vol_signal = volumes

        def norm(x):
            r = x.max() - x.min()
            return (x - x.min()) / (r + 1e-10)

        p = norm(closes)
        v = norm(vol_signal)

        # Depth-2 iterated integrals
        s_pv = float(self._trapz(p[:-1] * np.diff(v)))  # ∫ p dv
        s_vp = float(self._trapz(v[:-1] * np.diff(p)))  # ∫ v dp
        levy = s_pv - s_vp  # signed area = Lévy area

        # Depth-1: net price move (direction confirmation)
        net_price = p[-1] - p[0]
        direction = float(np.tanh(net_price * 10))

        # Conviction: Lévy area in same direction as price move
        if np.sign(levy) == np.sign(net_price) and abs(levy) > 0.01:
            conviction = min(abs(levy) * 20, 1.0)
        elif abs(levy) < 0.005:
            conviction = 0.3  # neutral
        else:
            conviction = -0.2  # contra-signal (distribution)

        score = float(np.clip(direction * (0.4 + 0.6 * conviction), -1, 1))

        return LayerResult(
            name   = "levy_area",
            value  = levy,
            score  = score,
            weight = self.WEIGHTS["levy_area"],
            active = True,
            note   = f"levy={levy:.4f} net_price={net_price:.4f}"
        )

    # ------------------------------------------------------------------
    # Layer 4: Regime gate — Bühler wave breaking + Daniel/Moskowitz
    # ------------------------------------------------------------------

    def _regime_gate(self, returns: pd.Series,
                     ts: pd.Timestamp) -> Tuple[str, float]:
        """
        Hard gate — returns regime string and scalar.
        blocked (0.0): don't trade at all
        caution (0.5): reduce signal weight
        normal  (1.0): full signal
        """
        if len(returns) < VOL_LONG_BARS:
            return "normal", 1.0

        rv_short = returns.tail(VOL_SHORT_BARS).std()
        rv_long  = returns.tail(VOL_LONG_BARS).std()

        if rv_long == 0:
            return "normal", 1.0

        vol_ratio = rv_short / rv_long

        # Kurtosis check (Cont Factor 3)
        if len(returns) >= KURTOSIS_WINDOW:
            kurt = returns.tail(KURTOSIS_WINDOW).kurtosis()
            if kurt > KURTOSIS_LIMIT:
                return "blocked", 0.0

        if vol_ratio > VOL_SPIKE_MULT:
            return "caution", 0.5
        elif vol_ratio > VOL_SPIKE_MULT * 0.8:
            return "caution", 0.7
        else:
            return "normal", 1.0

    def _regime_layer(self, returns: pd.Series) -> LayerResult:
        """Regime gate as a scored layer for the combination."""
        if len(returns) < VOL_LONG_BARS:
            return LayerResult("regime_gate", 1.0, 0.5,
                               self.WEIGHTS["regime_gate"], True,
                               "insufficient history")

        rv_short = returns.tail(VOL_SHORT_BARS).std()
        rv_long  = returns.tail(VOL_LONG_BARS).std()
        ratio    = rv_short / (rv_long + 1e-10)

        # Score: 1.0 when vol compressing, -1.0 when spiking
        score = float(np.clip(1.5 - ratio, -1, 1))

        return LayerResult(
            name   = "regime_gate",
            value  = ratio,
            score  = score,
            weight = self.WEIGHTS["regime_gate"],
            active = True,
            note   = f"vol_ratio={ratio:.3f}"
        )

    # ------------------------------------------------------------------
    # Layer 5: Cont Factor 1 proxy — realized vol compression
    # ------------------------------------------------------------------

    def _cont_f1(self, returns: pd.Series) -> LayerResult:
        """
        Approximates the Cont & da Fonseca IV surface level factor
        using rolling realized volatility ratio.

        When rv_short < rv_long (compression):
          IV level declining = wave settling after breaking
          = ideal momentum conditions (Bühler: post-breaking calm)

        When rv_short > rv_long (expansion):
          IV level rising = wave building or breaking
          = reduce momentum exposure
        """
        if len(returns) < VOL_LONG_BARS:
            return LayerResult("cont_f1", 1.0, 0.0,
                               self.WEIGHTS["cont_f1"], False,
                               "insufficient data")

        rv_short = returns.tail(VOL_SHORT_BARS).std()
        rv_long  = returns.tail(VOL_LONG_BARS).std()
        ratio    = rv_short / (rv_long + 1e-10)

        # When ratio < CONT_F1_COMPRESS: vol compressing = positive contribution
        # When ratio > CONT_F1_SPIKE: vol spiking = negative contribution
        if ratio < CONT_F1_COMPRESS:
            score = min((CONT_F1_COMPRESS - ratio) / CONT_F1_COMPRESS, 1.0)
        elif ratio > CONT_F1_SPIKE:
            score = -min((ratio - CONT_F1_SPIKE) / CONT_F1_SPIKE, 1.0)
        else:
            score = 0.0

        return LayerResult(
            name   = "cont_f1",
            value  = ratio,
            score  = float(score),
            weight = self.WEIGHTS["cont_f1"],
            active = True,
            note   = f"rv_ratio={ratio:.3f}"
        )

    # ------------------------------------------------------------------
    # Layer 6: Cont Factor 3 proxy — kurtosis kill switch
    # ------------------------------------------------------------------

    def _cont_f3(self, returns: pd.Series) -> LayerResult:
        """
        Approximates Cont & da Fonseca Factor 3 (smile curvature)
        using rolling kurtosis.

        High kurtosis = fat tails = market pricing in jump risk
        = Bühler wave near breaking = reduce/exit momentum
        """
        if len(returns) < KURTOSIS_WINDOW:
            return LayerResult("cont_f3", 3.0, 0.5,
                               self.WEIGHTS["cont_f3"], False,
                               "insufficient data")

        kurt = float(returns.tail(KURTOSIS_WINDOW).kurtosis())

        # Normal kurtosis ~3. Scale: 3=neutral, 6=warning, 9=danger
        if kurt > KURTOSIS_LIMIT:
            score = -1.0
        elif kurt > 4.5:
            score = -0.5
        elif kurt < 2.0:
            # Platykurtic = thin tails = clean trending = good
            score = 0.5
        else:
            score = 0.0

        return LayerResult(
            name   = "cont_f3",
            value  = kurt,
            score  = score,
            weight = self.WEIGHTS["cont_f3"],
            active = True,
            note   = f"kurtosis={kurt:.2f}"
        )

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def _combine(self, layers: dict, regime_scalar: float) -> float:
        """
        Weighted combination of all active layers.
        Regime scalar applied last (0=blocked, 0.5=caution, 1.0=normal).

        Key insight from Bühler: momentum is only reliable when
        fast and slow signals are co-directional (no shear).
        The coherence layer already penalizes shear — this combination
        respects that by weighting coherence highly.
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for name, layer in layers.items():
            if not layer.active:
                continue
            w = layer.weight
            weighted_sum += layer.score * w
            total_weight += w

        if total_weight == 0:
            return 0.0

        combined = weighted_sum / total_weight
        combined *= regime_scalar

        return float(np.clip(combined, -1, 1))

    # ------------------------------------------------------------------
    # ATR for stop/target sizing
    # ------------------------------------------------------------------

    def _atr(self, bars: pd.DataFrame, window: int = 4) -> float:
        """
        Average True Range over last N bars.
        Used for Di Graziano-consistent stop/target sizing.
        ATR stop = 1.5× ATR, ATR target = 2.5× ATR → b/a ratio = 1.67
        """
        if len(bars) < 2:
            return 20.0  # default for MNQ

        highs  = bars["high"].values
        lows   = bars["low"].values
        closes = bars["close"].values

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:]  - closes[:-1])
            )
        )

        n = min(window, len(tr))
        return float(np.mean(tr[-n:]))

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _flat(self, ts: pd.Timestamp, reason: str) -> MNQSignal:
        return MNQSignal(
            timestamp      = ts,
            direction      = 0,
            strength       = 0.0,
            combined_score = 0.0,
            is_tradeable   = False,
            regime         = reason,
            layers         = {},
            atr_pts        = 0.0,
            stop_pts       = 0.0,
            target_pts     = 0.0,
        )
