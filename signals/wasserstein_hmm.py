"""
signals/wasserstein_hmm.py
==========================
Wasserstein Hidden Markov Model for cross-asset regime detection.

Based on: Boukardagha (2026) arXiv:2603.04441
"Explainable Regime-Aware Investing"

Architecture:
  1. Feature construction: daily log return, 60-day vol, 20-day mean (t-1 causal)
  2. Predictive model-order selection: test K=2..5, pick best one-step PredLL
  3. Gaussian HMM: fit on rolling window, get filtered regime probabilities
  4. Wasserstein template tracking: map HMM components to stable regime identities
     using W2 distance between Gaussian distributions (closed-form)
  5. Template update: exponential smoothing to allow slow regime evolution

For our strategy:
  - Instruments: MGC, MNQ, MES (cross-asset daily returns)
  - Output: regime label per day (trending/choppy/volatile/crisis) + probabilities
  - Allocation rule: trade MGC when Gold in trending regime,
                     trade MNQ/MES when equity in trending regime,
                     reduce/halt when volatile/crisis regime detected

Usage:
    python signals/wasserstein_hmm.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMM_BACKEND = "hmmlearn"
except ImportError:
    HMM_BACKEND = None

from scipy.linalg import sqrtm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR  = Path("data/cache")

# Instruments and their point values
INSTRUMENTS = {
    "MGC": {"pv": 10.0,  "comm": 0.80},
    "MCL": {"pv": 100.0, "comm": 0.80},
    "MNQ": {"pv":  2.0,  "comm": 0.35},
    "MES": {"pv":  5.0,  "comm": 0.35},
    "MYM": {"pv":  0.5,  "comm": 0.35},
    "M2K": {"pv":  5.0,  "comm": 0.35},
}

# HMM parameters
K_MIN          = 2     # minimum regimes
K_MAX          = 5     # maximum regimes
ROLLING_WINDOW = 338   # ~16 months — half of 3-year history
VAL_WINDOW     = 42    # ~2 months for predictive log-likelihood validation
REFIT_FREQ     = 5     # refit model-order selection every 5 days (weekly)
COMPLEXITY_PEN = 0.5   # lambda_K — complexity penalty per regime

# Feature parameters (Boukardagha Section 4)
VOL_WINDOW     = 60    # 60-day rolling volatility
MEAN_WINDOW    = 20    # 20-day rolling mean return

# Template tracking
TEMPLATE_ALPHA = 0.05  # exponential smoothing for template updates (slow)
N_TEMPLATES    = 4     # number of persistent regime templates

# Regime labels mapped to our strategy
REGIME_LABELS = {
    0: "trending_commodity",   # gold trending, equity neutral/down
    1: "trending_equity",      # equity trending, gold neutral
    2: "volatile",             # high vol across assets
    3: "choppy",               # low directional persistence both
}

# Allocation scalars per regime per instrument
ALLOCATION = {
    # Trade instrument, 1.0 = full size, 0.5 = half size, 0.0 = skip
    "trending_commodity": {"MGC": 1.0, "MCL": 0.5, "MNQ": 0.0, "MES": 0.0, "MYM": 0.0, "M2K": 0.0},
    "trending_equity":    {"MGC": 0.0, "MCL": 0.0, "MNQ": 1.0, "MES": 0.5, "MYM": 0.0, "M2K": 0.5},
    "volatile":           {"MGC": 0.5, "MCL": 0.0, "MNQ": 0.0, "MES": 0.0, "MYM": 0.0, "M2K": 0.0},
    "choppy":             {"MGC": 0.5, "MCL": 0.0, "MNQ": 0.5, "MES": 0.0, "MYM": 0.0, "M2K": 0.0},
}


# ---------------------------------------------------------------------------
# Data loading and feature construction
# ---------------------------------------------------------------------------

def load_daily(symbol: str) -> Optional[pd.DataFrame]:
    path = CACHE_DIR / f"{symbol}_daily.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def build_features(daily_data: dict) -> pd.DataFrame:
    """
    Build feature matrix: [returns, vols, means] per instrument.
    All features use t-1 information (strict causality via shift).

    Feature vector per day: [r_MGC, r_MNQ, r_MES,
                              vol_MGC, vol_MNQ, vol_MES,
                              mean_MGC, mean_MNQ, mean_MES]
    """
    dfs = []
    for sym, df in daily_data.items():
        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol     = log_ret.rolling(VOL_WINDOW).std()
        mean    = log_ret.rolling(MEAN_WINDOW).mean()

        sym_df  = pd.DataFrame({
            f"ret_{sym}":  log_ret,
            f"vol_{sym}":  vol,
            f"mean_{sym}": mean,
        }, index=df.index)
        dfs.append(sym_df)

    features = pd.concat(dfs, axis=1).dropna()
    # Shift by 1 for strict causality — features at t use data up to t-1
    features = features.shift(1).dropna()
    return features


# ---------------------------------------------------------------------------
# Wasserstein distance between two Gaussians (closed-form)
# ---------------------------------------------------------------------------

def w2_gaussian(mu1: np.ndarray, sigma1: np.ndarray,
                mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Closed-form 2-Wasserstein distance between N(mu1,Sigma1) and N(mu2,Sigma2).

    W2^2 = ||mu1-mu2||^2 + Tr(Sigma1 + Sigma2 - 2*(Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2})

    Boukardagha (2026) Section 5.4, equation for W2^2.
    """
    mean_term = float(np.sum((mu1 - mu2) ** 2))

    try:
        sqrt_s2   = sqrtm(sigma2).real
        inner     = sqrt_s2 @ sigma1 @ sqrt_s2
        sqrt_inner = sqrtm(inner).real
        cov_term  = float(np.trace(sigma1 + sigma2 - 2 * sqrt_inner))
        cov_term  = max(cov_term, 0.0)   # numerical safety
    except Exception:
        # Fallback: use diagonal approximation
        cov_term = float(np.sum((np.sqrt(np.diag(sigma1)) -
                                 np.sqrt(np.diag(sigma2))) ** 2))

    return float(np.sqrt(mean_term + cov_term))


# ---------------------------------------------------------------------------
# Simple Gaussian HMM (EM) — fallback if hmmlearn not available
# ---------------------------------------------------------------------------

class SimpleGaussianHMM:
    """
    Minimal Gaussian HMM with Baum-Welch EM.
    Used when hmmlearn is not installed.
    """

    def __init__(self, n_components: int, n_iter: int = 50, tol: float = 1e-4):
        self.n_components = n_components
        self.n_iter       = n_iter
        self.tol          = tol
        self.means_       = None
        self.covars_      = None
        self.transmat_    = None
        self.startprob_   = None

    def fit(self, X: np.ndarray):
        n, d = X.shape
        K    = self.n_components

        # Initialise with k-means
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=3, random_state=42).fit(X)
        labels = km.labels_

        self.means_    = np.array([X[labels==k].mean(axis=0) for k in range(K)])
        self.covars_   = np.array([np.cov(X[labels==k].T) + 1e-4*np.eye(d)
                                   for k in range(K)])
        self.transmat_ = np.full((K, K), 1/K)
        self.startprob_= np.full(K, 1/K)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            # E-step: forward-backward
            alpha, c = self._forward(X)
            beta      = self._backward(X, c)
            gamma     = alpha * beta
            gamma    /= gamma.sum(axis=1, keepdims=True) + 1e-300

            xi = np.zeros((K, K))
            for t in range(n - 1):
                for i in range(K):
                    for j in range(K):
                        xi[i,j] += (alpha[t,i] * self.transmat_[i,j] *
                                    self._emission(X[t+1], j) * beta[t+1,j])
            xi_sum = xi.sum(axis=1, keepdims=True) + 1e-300
            xi    /= xi_sum

            # M-step
            self.startprob_ = gamma[0] + 1e-10
            self.startprob_ /= self.startprob_.sum()
            self.transmat_   = xi + 1e-10
            self.transmat_  /= self.transmat_.sum(axis=1, keepdims=True)

            for k in range(K):
                w = gamma[:, k] + 1e-300
                self.means_[k]  = (w[:, None] * X).sum(0) / w.sum()
                diff = X - self.means_[k]
                self.covars_[k] = ((w[:, None] * diff).T @ diff
                                   / w.sum() + 1e-4*np.eye(d))

            ll = np.log(c + 1e-300).sum()
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def _emission(self, x: np.ndarray, k: int) -> float:
        d     = len(x)
        diff  = x - self.means_[k]
        cov   = self.covars_[k]
        try:
            sign, logdet = np.linalg.slogdet(cov)
            inv_cov = np.linalg.inv(cov)
            exponent = -0.5 * diff @ inv_cov @ diff
            return float(np.exp(exponent - 0.5*(d*np.log(2*np.pi) + logdet)))
        except Exception:
            return 1e-300

    def _forward(self, X):
        n, _  = X.shape
        K     = self.n_components
        alpha = np.zeros((n, K))
        c     = np.zeros(n)

        alpha[0] = self.startprob_ * np.array([self._emission(X[0], k) for k in range(K)])
        c[0]     = alpha[0].sum() + 1e-300
        alpha[0] /= c[0]

        for t in range(1, n):
            em = np.array([self._emission(X[t], k) for k in range(K)])
            alpha[t] = (alpha[t-1] @ self.transmat_) * em
            c[t]     = alpha[t].sum() + 1e-300
            alpha[t] /= c[t]
        return alpha, c

    def _backward(self, X, c):
        n, _  = X.shape
        K     = self.n_components
        beta  = np.ones((n, K))

        for t in range(n-2, -1, -1):
            em = np.array([self._emission(X[t+1], k) for k in range(K)])
            beta[t] = (self.transmat_ * em[None, :] * beta[t+1][None, :]).sum(1)
            beta[t] /= c[t+1] + 1e-300
        return beta

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        alpha, c = self._forward(X)
        beta     = self._backward(X, c)
        gamma    = alpha * beta
        gamma   /= gamma.sum(axis=1, keepdims=True) + 1e-300
        return gamma

    def score(self, X: np.ndarray) -> float:
        _, c = self._forward(X)
        return float(np.log(c + 1e-300).sum())


def fit_hmm(X: np.ndarray, K: int) -> object:
    """Fit a K-state Gaussian HMM. Uses hmmlearn if available."""
    if HMM_BACKEND == "hmmlearn":
        model = hmmlearn_hmm.GaussianHMM(
            n_components=K, covariance_type="full",
            n_iter=100, tol=1e-4, random_state=42
        )
        model.fit(X)
        return model
    else:
        return SimpleGaussianHMM(K, n_iter=100).fit(X)


def hmm_score(model, X: np.ndarray) -> float:
    if HMM_BACKEND == "hmmlearn":
        return model.score(X)
    return model.score(X)


def hmm_means_covars(model, K: int):
    if HMM_BACKEND == "hmmlearn":
        return model.means_, model.covars_
    return model.means_, model.covars_


def hmm_predict_proba(model, X: np.ndarray) -> np.ndarray:
    if HMM_BACKEND == "hmmlearn":
        return model.predict_proba(X)
    return model.predict_proba(X)


# ---------------------------------------------------------------------------
# Predictive model-order selection (Boukardagha Section 5.2)
# ---------------------------------------------------------------------------

def select_k(X_train: np.ndarray, X_val: np.ndarray,
             k_min: int = K_MIN, k_max: int = K_MAX,
             penalty: float = COMPLEXITY_PEN) -> int:
    """
    Select number of regimes K by predictive log-likelihood on validation set.
    PredLL(K) - lambda_K * K  (penalize complexity)
    """
    best_score = -np.inf
    best_k     = k_min

    for K in range(k_min, k_max + 1):
        try:
            model = fit_hmm(X_train, K)
            score = hmm_score(model, X_val) - penalty * K
            if score > best_score:
                best_score = score
                best_k     = K
        except Exception:
            continue

    return best_k


# ---------------------------------------------------------------------------
# Wasserstein template tracker (Boukardagha Section 5.4)
# ---------------------------------------------------------------------------

class WassersteinTemplateTracker:
    """
    Maintains G persistent regime templates.
    Maps HMM components to templates via W2 distance.
    Updates templates via exponential smoothing.

    Solves the label-switching problem in rolling HMM estimation.
    """

    def __init__(self, n_templates: int = N_TEMPLATES,
                 alpha: float = TEMPLATE_ALPHA):
        self.G          = n_templates
        self.alpha      = alpha
        self.templates  = None   # list of (mu, Sigma) tuples
        self._initialized = False

    def initialize(self, means: np.ndarray, covars: np.ndarray):
        """Initialize templates from first HMM fit."""
        K = len(means)
        # Pad or truncate to G templates
        self.templates = []
        for g in range(self.G):
            k = g % K
            self.templates.append((means[k].copy(), covars[k].copy()))
        self._initialized = True

    def map_and_update(self, means: np.ndarray,
                       covars: np.ndarray) -> np.ndarray:
        """
        Map K HMM components to G templates via nearest W2 distance.
        Update templates via exponential smoothing.
        Returns assignment array: assignment[k] = template_index g
        """
        if not self._initialized:
            self.initialize(means, covars)
            return np.arange(min(len(means), self.G))

        K = len(means)
        assignment = np.zeros(K, dtype=int)

        # Assign each HMM component to nearest template
        for k in range(K):
            distances = [
                w2_gaussian(means[k], covars[k],
                            self.templates[g][0], self.templates[g][1])
                for g in range(self.G)
            ]
            assignment[k] = int(np.argmin(distances))

        # Update templates via exponential smoothing
        for k in range(K):
            g = assignment[k]
            mu_old, sig_old = self.templates[g]
            mu_new  = (1 - self.alpha) * mu_old + self.alpha * means[k]
            sig_new = (1 - self.alpha) * sig_old + self.alpha * covars[k]
            self.templates[g] = (mu_new, sig_new)

        return assignment

    def get_template_probs(self, component_probs: np.ndarray,
                           assignment: np.ndarray) -> np.ndarray:
        """
        Aggregate HMM component probabilities to template probabilities.
        p_template[g] = sum of p_component[k] for all k assigned to g
        """
        template_probs = np.zeros(self.G)
        for k, p in enumerate(component_probs):
            g = assignment[k]
            template_probs[g] += p
        return template_probs / (template_probs.sum() + 1e-300)


# ---------------------------------------------------------------------------
# Regime labeler — maps template index to economic label
# ---------------------------------------------------------------------------

def label_template(template_probs: np.ndarray,
                   templates: list,
                   feature_names: list) -> tuple:
    """
    Assign economic label to dominant template.
    Uses feature means to distinguish regimes:
      - High gold return + low equity return → trending_commodity
      - High equity return + low gold return → trending_equity
      - High volatility across assets        → volatile
      - Low mean returns across assets       → choppy
    """
    dominant = int(np.argmax(template_probs))
    dominant_prob = float(template_probs[dominant])

    if templates is None:
        return "choppy", dominant, dominant_prob

    mu = templates[dominant][0]

    # Extract feature indices
    ret_idx  = [i for i, n in enumerate(feature_names) if n.startswith("ret_")]
    vol_idx  = [i for i, n in enumerate(feature_names) if n.startswith("vol_")]
    mgc_ret  = mu[ret_idx[0]] if ret_idx else 0
    mnq_ret  = mu[ret_idx[1]] if len(ret_idx) > 1 else 0
    avg_vol  = np.mean([mu[i] for i in vol_idx]) if vol_idx else 0
    avg_ret  = np.mean([mu[i] for i in ret_idx]) if ret_idx else 0

    # Use rolling mean return (more stable than single-day return)
    mgc_mean_idx = [i for i, n in enumerate(feature_names) if n == "mean_MGC"]
    mnq_mean_idx = [i for i, n in enumerate(feature_names) if n == "mean_MNQ"]
    mcl_mean_idx = [i for i, n in enumerate(feature_names) if n == "mean_MCL"]
    mgc_trend = mu[mgc_mean_idx[0]] if mgc_mean_idx else mgc_ret
    mnq_trend = mu[mnq_mean_idx[0]] if mnq_mean_idx else mnq_ret
    mcl_trend = mu[mcl_mean_idx[0]] if mcl_mean_idx else 0.0
    # Commodity trending = gold OR crude trending positively
    commodity_trend = max(mgc_trend, mcl_trend)

    # Thresholds calibrated to actual feature distributions (p50 level)
    # ret p50=0.00127, vol p75=0.014, mean p75=0.00230
    vol_threshold  = 0.014    # p75 of vol — elevated volatility
    trend_pos      = 0.0008   # p60 of mean return — positive trend
    trend_neg      = -0.0002  # slight negative mean — not trending

    if avg_vol > vol_threshold:
        label = "volatile"
    elif commodity_trend > trend_pos and mnq_trend < trend_neg:
        label = "trending_commodity"
    elif mnq_trend > trend_pos and commodity_trend < trend_neg:
        label = "trending_equity"
    elif commodity_trend > trend_pos or mnq_trend > trend_pos:
        label = "trending_equity" if mnq_trend > commodity_trend else "trending_commodity"
    else:
        label = "choppy"

    return label, dominant, dominant_prob


# ---------------------------------------------------------------------------
# Main Wasserstein HMM engine
# ---------------------------------------------------------------------------

class WassersteinHMM:
    """
    Full Wasserstein HMM regime detection pipeline.

    Usage:
        whmm = WassersteinHMM()
        whmm.fit(features_df)
        regime_series = whmm.regimes_   # pd.Series of regime labels
        probs_df      = whmm.probs_     # pd.DataFrame of template probabilities
    """

    def __init__(self,
                 rolling_window: int = ROLLING_WINDOW,
                 val_window:     int = VAL_WINDOW,
                 refit_freq:     int = REFIT_FREQ,
                 n_templates:    int = N_TEMPLATES,
                 template_alpha: float = TEMPLATE_ALPHA):

        self.rolling_window  = rolling_window
        self.val_window      = val_window
        self.refit_freq      = refit_freq
        self.tracker         = WassersteinTemplateTracker(n_templates, template_alpha)
        self.regimes_        = None
        self.probs_          = None
        self._current_k      = K_MIN
        self._model          = None
        self._feature_names  = None

    def fit(self, features: pd.DataFrame) -> "WassersteinHMM":
        """
        Run full rolling Wasserstein HMM on feature matrix.
        Results stored in self.regimes_ and self.probs_
        """
        self._feature_names = list(features.columns)
        X_all   = features.values.astype(float)
        n       = len(X_all)
        dates   = features.index

        regime_labels  = []
        template_probs = []
        k_history      = []

        min_start = max(self.rolling_window, self.val_window + 20)

        print(f"  Running Wasserstein HMM on {n} daily bars...")
        print(f"  Window={self.rolling_window}d  Val={self.val_window}d  "
              f"RefitFreq={self.refit_freq}d  Templates={self.tracker.G}")

        for t in range(n):
            if t < min_start:
                regime_labels.append("insufficient_data")
                template_probs.append(np.ones(self.tracker.G) / self.tracker.G)
                k_history.append(self._current_k)
                continue

            # Window for training
            t_start = max(0, t - self.rolling_window)
            t_val   = max(t_start, t - self.val_window)
            X_train = X_all[t_start:t_val]
            X_val   = X_all[t_val:t]
            X_now   = X_all[t:t+1]

            # Refit model-order selection weekly
            if t % self.refit_freq == 0 and len(X_train) >= 30 and len(X_val) >= 5:
                try:
                    self._current_k = select_k(X_train, X_val)
                except Exception:
                    pass

            # Fit HMM on full training window
            X_fit = X_all[t_start:t]
            if len(X_fit) < self.rolling_window // 2:
                regime_labels.append("insufficient_data")
                template_probs.append(np.ones(self.tracker.G) / self.tracker.G)
                k_history.append(self._current_k)
                continue

            try:
                model = fit_hmm(X_fit, self._current_k)
                means, covars = hmm_means_covars(model, self._current_k)

                # Get regime probs for current bar
                probs_full = hmm_predict_proba(model, X_fit)
                probs_now  = probs_full[-1]   # latest filtered probs

                # Wasserstein template matching
                assignment = self.tracker.map_and_update(means, covars)

                # Aggregate to template probabilities
                tprobs = self.tracker.get_template_probs(probs_now, assignment)

                # Label dominant regime
                label, _, _ = label_template(
                    tprobs, self.tracker.templates, self._feature_names)

                regime_labels.append(label)
                template_probs.append(tprobs)
                k_history.append(self._current_k)

            except Exception as e:
                regime_labels.append("error")
                template_probs.append(np.ones(self.tracker.G) / self.tracker.G)
                k_history.append(self._current_k)

            # Progress
            if t % 50 == 0 and t > min_start:
                pct = (t - min_start) / (n - min_start) * 100
                print(f"    {dates[t].date()}  K={self._current_k}  "
                      f"regime={regime_labels[-1]}  [{pct:.0f}%]")

        self.regimes_ = pd.Series(regime_labels, index=dates, name="regime")
        self.probs_   = pd.DataFrame(
            template_probs, index=dates,
            columns=[f"template_{g}" for g in range(self.tracker.G)]
        )
        self._k_history = pd.Series(k_history, index=dates, name="K")

        return self

    def get_allocation(self, date) -> dict:
        """Get instrument allocation scalars for a given date."""
        if self.regimes_ is None or date not in self.regimes_.index:
            return {"MGC": 0.5, "MNQ": 0.25, "MES": 0.25}
        regime = self.regimes_[date]
        return ALLOCATION.get(regime, {"MGC": 0.5, "MNQ": 0.25, "MES": 0.25})


# ---------------------------------------------------------------------------
# Standalone analysis
# ---------------------------------------------------------------------------

def analyze_regimes(whmm: WassersteinHMM, daily_data: dict):
    """Print regime analysis and monthly breakdown."""
    regimes = whmm.regimes_
    active  = regimes[regimes != "insufficient_data"]

    print("\n" + "=" * 60)
    print("  WASSERSTEIN HMM — REGIME ANALYSIS")
    print("=" * 60)

    # Regime distribution
    counts = active.value_counts()
    print(f"\n  Regime distribution ({len(active)} trading days):")
    for label, count in counts.items():
        pct = count / len(active) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<25} {count:>4} days  ({pct:>5.1f}%)  {bar}")

    # K distribution
    k_hist = whmm._k_history[regimes != "insufficient_data"]
    k_counts = k_hist.value_counts().sort_index()
    print(f"\n  Adaptive K distribution:")
    for k, count in k_counts.items():
        print(f"    K={k}: {count} days ({count/len(k_hist)*100:.1f}%)")

    # Monthly regime breakdown
    print(f"\n  Monthly regime breakdown:")
    print(f"  {'Month':<10} {'Dominant':>22} {'Allocation: MGC/MNQ/MES'}")
    print("  " + "-" * 60)

    monthly_regime = active.resample("ME").agg(lambda x: x.value_counts().index[0])
    for period, regime in monthly_regime.items():
        alloc = ALLOCATION.get(regime, {})
        mgc   = alloc.get("MGC", 0)
        mnq   = alloc.get("MNQ", 0)
        mes   = alloc.get("MES", 0)
        print(f"  {str(period.date()):<12} {regime:<22} "
              f"{mgc:.0%} / {mnq:.0%} / {mes:.0%}")

    # Cross-reference with actual P&L from backtest results
    results_dir = Path("backtest/results")
    mgc_files   = sorted(results_dir.glob("mgc_htc22_2ct_*.csv"))
    if mgc_files:
        trades = pd.read_csv(mgc_files[-1])
        trades["date"] = pd.to_datetime(trades["date"])

        print(f"\n  Regime vs MGC P&L (2-contract baseline):")
        print(f"  {'Month':<10} {'Regime':>22} {'MGC_PnL':>10} {'Suggested':>15}")
        print("  " + "-" * 62)

        monthly_trades = trades.groupby(
            trades["date"].dt.to_period("M"))["net_pnl"].sum()

        for period, regime in monthly_regime.items():
            month_key = period.to_period("M")
            mgc_pnl   = monthly_trades.get(month_key, 0)
            alloc     = ALLOCATION.get(regime, {})
            mgc_alloc = alloc.get("MGC", 0)
            suggestion = "TRADE MGC" if mgc_alloc >= 1.0 else \
                         "REDUCE MGC" if mgc_alloc >= 0.5 else \
                         "SKIP MGC → MNQ/MES"
            flag = " ←" if mgc_pnl < -200 and mgc_alloc >= 1.0 else ""
            print(f"  {str(period.date()):<12} {regime:<22} "
                  f"${mgc_pnl:>8,.0f}  {suggestion}{flag}")

    print()


def main():
    print()
    print("Wasserstein HMM — Session 4 Phase 4")
    print("=" * 55)
    print(f"Backend: {HMM_BACKEND or 'SimpleGaussianHMM (hmmlearn not installed)'}")
    print()

    # Load daily data
    daily_data = {}
    for sym in ["MGC", "MCL", "MNQ", "MES", "MYM", "M2K"]:
        df = load_daily(sym)
        if df is not None:
            daily_data[sym] = df
            print(f"  {sym}: {len(df)} daily bars "
                  f"({df.index[0].date()} to {df.index[-1].date()})")
        else:
            print(f"  {sym}: not found")

    if len(daily_data) < 2:
        print("Need at least 2 instruments. Run data/fetch_mes.py first.")
        return

    # Build features
    print("\nBuilding features...")
    features = build_features(daily_data)
    print(f"  Feature matrix: {features.shape[0]} days × {features.shape[1]} features")
    print(f"  Features: {list(features.columns)}")

    # Fit Wasserstein HMM
    print()
    whmm = WassersteinHMM(
        rolling_window = min(ROLLING_WINDOW, len(features) // 2),
        val_window     = VAL_WINDOW,
        refit_freq     = REFIT_FREQ,
        n_templates    = N_TEMPLATES,
    )
    whmm.fit(features)

    # Save results
    output_dir = Path("backtest/results")
    output_dir.mkdir(exist_ok=True)
    whmm.regimes_.to_csv(output_dir / "wasserstein_regimes.csv")
    whmm.probs_.to_csv(output_dir / "wasserstein_probs.csv")
    print(f"\n  Saved regime series and probabilities to backtest/results/")

    # Analyze
    analyze_regimes(whmm, daily_data)

    print("Next step: python backtest/runner_portfolio.py")
    print()


if __name__ == "__main__":
    main()
