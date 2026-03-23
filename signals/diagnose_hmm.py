"""
signals/diagnose_hmm.py
Inspects fitted HMM template means to calibrate label_template thresholds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from signals.wasserstein_hmm import (
    load_daily, build_features, fit_hmm, hmm_means_covars,
    WassersteinTemplateTracker, ROLLING_WINDOW, N_TEMPLATES, TEMPLATE_ALPHA
)

CACHE_DIR = Path("data/cache")

def main():
    # Load data
    daily_data = {}
    for sym in ["MGC", "MNQ", "MES"]:
        df = load_daily(sym)
        if df is not None:
            daily_data[sym] = df

    features = build_features(daily_data)
    X = features.values.astype(float)
    feat_names = list(features.columns)

    print("Feature statistics (all 398 days):")
    print(f"{'Feature':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    for i, name in enumerate(feat_names):
        col = X[:, i]
        print(f"  {name:<13} {col.mean():>10.6f} {col.std():>10.6f} "
              f"{col.min():>10.6f} {col.max():>10.6f}")

    print()

    # Fit HMM on full data and inspect template means
    print("Fitting K=3 HMM on full dataset...")
    model = fit_hmm(X, 3)
    means, covars = hmm_means_covars(model, 3)

    print()
    print("HMM component means (K=3):")
    print(f"{'Feature':<15}" + "".join(f"  {'State'+str(k):>12}" for k in range(3)))
    print("-" * (15 + 3*14))
    for i, name in enumerate(feat_names):
        row = f"  {name:<13}"
        for k in range(3):
            row += f"  {means[k][i]:>12.6f}"
        print(row)

    print()
    print("Key regime indicators:")
    ret_idx  = [i for i,n in enumerate(feat_names) if n.startswith("ret_")]
    vol_idx  = [i for i,n in enumerate(feat_names) if n.startswith("vol_")]
    mean_idx = [i for i,n in enumerate(feat_names) if n.startswith("mean_")]

    for k in range(3):
        mu = means[k]
        mgc_ret  = mu[ret_idx[0]] if ret_idx else 0
        mnq_ret  = mu[ret_idx[1]] if len(ret_idx)>1 else 0
        mes_ret  = mu[ret_idx[2]] if len(ret_idx)>2 else 0
        mgc_vol  = mu[vol_idx[0]] if vol_idx else 0
        mnq_vol  = mu[vol_idx[1]] if len(vol_idx)>1 else 0
        avg_vol  = np.mean([mu[i] for i in vol_idx])
        avg_ret  = np.mean([mu[i] for i in ret_idx])
        avg_mean = np.mean([mu[i] for i in mean_idx])

        print(f"  State {k}: MGC_ret={mgc_ret:.5f}  MNQ_ret={mnq_ret:.5f}  "
              f"avg_vol={avg_vol:.5f}  avg_mean={avg_mean:.5f}")

    print()
    print("Suggested thresholds for label_template():")
    # Compute percentile-based thresholds
    all_rets = X[:, ret_idx].flatten()
    all_vols = X[:, vol_idx].flatten()
    all_means= X[:, mean_idx].flatten()
    print(f"  ret   p25={np.percentile(all_rets,25):.5f}  "
          f"p50={np.percentile(all_rets,50):.5f}  "
          f"p75={np.percentile(all_rets,75):.5f}")
    print(f"  vol   p25={np.percentile(all_vols,25):.5f}  "
          f"p50={np.percentile(all_vols,50):.5f}  "
          f"p75={np.percentile(all_vols,75):.5f}")
    print(f"  mean  p25={np.percentile(all_means,25):.5f}  "
          f"p50={np.percentile(all_means,50):.5f}  "
          f"p75={np.percentile(all_means,75):.5f}")

if __name__ == "__main__":
    main()
