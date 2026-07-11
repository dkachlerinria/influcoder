"""Spearman rank agreement between a predicted score matrix and ground truth."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: [n_anchors, n_pool].

    per_anchor_*  -- Spearman of each anchor's candidate ranking, averaged
    aggregated    -- Spearman of the anchor-averaged score vectors
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    per = []
    for i in range(gt.shape[0]):
        r = spearmanr(pred[i], gt[i]).statistic
        per.append(0.0 if np.isnan(r) else float(r))
    agg = spearmanr(pred.mean(0), gt.mean(0)).statistic
    return {
        "per_anchor_mean": float(np.mean(per)),
        "per_anchor_std": float(np.std(per)),
        "aggregated": 0.0 if np.isnan(agg) else float(agg),
        "per_anchor": per,
    }
