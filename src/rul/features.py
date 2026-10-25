"""Hand-crafted window features for the tree-based baselines."""
from __future__ import annotations

import numpy as np


def window_features(x: np.ndarray) -> np.ndarray:
    """(N, W, F) -> (N, 4F): last value, window mean, window std, linear slope.

    Trees cannot read a time series directly, so we summarise each window.
    The slope captures the *rate* of degradation, which is the real signal.
    """
    n, w, f = x.shape
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    t = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    slope = (x * t[None, :, None]).sum(axis=1) / (t**2).sum()
    return np.concatenate([last, mean, std, slope], axis=1).astype(np.float32)


def feature_names(features) -> list:
    return [f"{f}_{s}" for s in ("last", "mean", "std", "slope") for f in features]
