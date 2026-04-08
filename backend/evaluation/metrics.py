"""
Evaluation metrics for probabilistic match outcome predictions.

All metrics operate on:
    y_true: integer array of class labels [0=home win, 1=draw, 2=away win]
    y_proba: float array of shape (n, 3): [p_home, p_draw, p_away]

Metrics:
    brier_score — mean squared error of probabilities (PRIMARY)
    log_loss_score — negative log-likelihood
    rps — Ranked Probability Score (best for ordered outcomes)
    accuracy — argmax accuracy
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Multi-class Brier score: mean squared error between predicted and true probabilities.

    For each sample, the true outcome is one-hot encoded, and we compute
    the MSE across all 3 classes. Lower is better. Perfect = 0, random ≈ 0.67.
    """
    n = len(y_true)
    one_hot = np.zeros((n, 3))
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def log_loss_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Cross-entropy log loss. Lower is better. Random baseline ≈ 1.099 (ln 3).
    """
    return float(log_loss(y_true, y_proba, labels=[0, 1, 2]))


def rps(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Ranked Probability Score (RPS) for 3-class ordered outcomes.

    RPS = mean over samples of: sum over K-1 thresholds of (F_k - O_k)^2
    where F_k = cumulative predicted probability, O_k = cumulative true probability.

    RPS accounts for the *distance* between predicted and actual outcome —
    predicting 70% home win when the away team wins is penalised more than
    predicting 50% draw when the away team wins, because draw is "closer" to away win.

    Lower RPS = better. Random baseline ≈ 0.37, expert models typically 0.19–0.23.
    """
    n = len(y_true)
    one_hot = np.zeros((n, 3))
    one_hot[np.arange(n), y_true] = 1.0

    cum_pred = np.cumsum(y_proba[:, :2], axis=1)  # Cumulative probs for first 2 thresholds
    cum_true = np.cumsum(one_hot[:, :2], axis=1)
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1)))


def accuracy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Argmax accuracy — fraction of matches where predicted outcome is correct."""
    return float((y_proba.argmax(axis=1) == y_true).mean())


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Return all metrics as a dict."""
    return {
        "brier_score": brier_score(y_true, y_proba),
        "log_loss": log_loss_score(y_true, y_proba),
        "rps": rps(y_true, y_proba),
        "accuracy": accuracy(y_true, y_proba),
        "n_samples": int(len(y_true)),
    }


def calibration_summary(y_true: np.ndarray, y_proba: np.ndarray,
                         n_bins: int = 10) -> list[dict]:
    """
    Per-class calibration stats: compare average predicted probability vs actual win rate
    within each probability bin.

    A well-calibrated model has points near the diagonal (predicted ≈ actual).
    Returns a list of dicts with: class_idx, bin_start, bin_end, avg_predicted, actual_rate, count.
    """
    results = []
    bins = np.linspace(0, 1, n_bins + 1)
    for cls_idx in range(3):
        probs = y_proba[:, cls_idx]
        actuals = (y_true == cls_idx).astype(float)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            results.append({
                "class_idx": cls_idx,
                "bin_start": float(lo),
                "bin_end": float(hi),
                "avg_predicted": float(probs[mask].mean()),
                "actual_rate": float(actuals[mask].mean()),
                "count": int(mask.sum()),
            })
    return results
