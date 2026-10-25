"""Regression and alerting metrics for RUL prediction."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 score. Late predictions (pred > true) cost more than early ones.

    d = pred - true;  s = exp(-d/13) - 1 if d < 0 else exp(d/10) - 1.  Lower is better.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def alert_metrics(true_rul, pred_rul, horizon: int = 30, threshold: float | None = None) -> dict:
    """Treat 'fails within `horizon` cycles' as the positive class.

    An alert fires when predicted RUL <= `threshold` (default: the horizon itself).
    A missed failure is far costlier than a false alarm, so recall matters most.
    """
    threshold = horizon if threshold is None else threshold
    truth = np.asarray(true_rul) <= horizon
    alert = np.asarray(pred_rul) <= threshold
    tp = int(np.sum(truth & alert))
    fp = int(np.sum(~truth & alert))
    fn = int(np.sum(truth & ~alert))
    tn = int(np.sum(~truth & ~alert))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "horizon": horizon,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize(true_rul, pred_rul) -> dict:
    return {
        "rmse": rmse(true_rul, pred_rul),
        "mae": mae(true_rul, pred_rul),
        "r2": r2(true_rul, pred_rul),
        "nasa_score": nasa_score(true_rul, pred_rul),
    }
