"""
Evaluation metrics for probabilistic match outcome predictions.

All metrics operate on:
    y_true: integer array of class labels [0=home win, 1=draw, 2=away win]
    y_proba: float array of shape (n, 3): [p_home, p_draw, p_away]

Convention (literature, since T0.2 on 2026-04-24):
    brier_score — MEAN across K=3 classes. Competitive models: 0.18–0.22.
                  Perfect=0, random≈0.22. Range is 0–1 (not 0–2).
    rps         — sum over K-1 thresholds, divided by (K-1)=2.
                  Competitive models: 0.19–0.23. Random≈0.37 (wait: random on
                  3-class uniform is ≈0.22 after division; 0.37 refers to the
                  un-normalised form — do not compare pre/post-T0.2 numbers).
    log_loss_score — unchanged. Random 3-class baseline ≈ ln(3) ≈ 1.099.
    accuracy    — unchanged.

PRE-T0.2 CONVENTION was sum-based: brier returned sum across classes (≈3×
literature) and rps returned raw sum over thresholds (≈2× literature).
Eval artifacts in `backend/data/output/eval_*.json` produced before
2026-04-24 use the old convention. See `backend/evaluation/METRICS_CHANGELOG.md`.

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
    Multi-class Brier score, literature convention (mean across K=3 classes).

    For each sample we compute the mean squared error across the 3 predicted
    probabilities vs one-hot truth, then average across samples. Lower = better.

    Range: [0, 1]. Perfect model = 0. Published competitive range: 0.18–0.22.
    """
    n = len(y_true)
    one_hot = np.zeros((n, 3))
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean((y_proba - one_hot) ** 2))


def log_loss_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Cross-entropy log loss. Lower is better. Random baseline ≈ 1.099 (ln 3).
    """
    return float(log_loss(y_true, y_proba, labels=[0, 1, 2]))


def rps(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Ranked Probability Score (RPS) for 3-class ordered outcomes, literature
    convention (normalised by K-1 so RPS is in [0, 1]).

    RPS = mean over samples of (1/(K-1)) · sum over K-1 thresholds of
          (F_k - O_k)^2, where F_k = cumulative predicted probability and
          O_k = cumulative true probability.

    RPS accounts for the *distance* between predicted and actual outcome —
    predicting 70% home win when the away team wins is penalised more than
    predicting 50% draw when the away team wins, because draw is "closer"
    to away win.

    Lower = better. Published competitive range: 0.19–0.23.
    """
    n = len(y_true)
    num_classes = y_proba.shape[1]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y_true] = 1.0

    cum_pred = np.cumsum(y_proba[:, :-1], axis=1)  # cumulative probs over K-1 thresholds
    cum_true = np.cumsum(one_hot[:, :-1], axis=1)
    per_sample = np.sum((cum_pred - cum_true) ** 2, axis=1)
    return float(np.mean(per_sample) / (num_classes - 1))


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
