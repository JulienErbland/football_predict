"""
Weighted average ensemble model.

Combines predictions from multiple trained models using configurable weights.
Weights are normalised to sum to 1 using only enabled models, so you can disable
a model in model_config.yaml without touching the other weights.

Also supports a stacking variant: a logistic meta-learner is trained on
out-of-fold (OOF) predictions from the base models. This can outperform
weighted averaging when models are diverse and have different error patterns,
but requires more data and careful cross-validation to avoid leakage.

Note: pickle is used for persistence because sklearn/XGBoost objects don't have
a standardised cross-library serialisation format. Load only from trusted local files.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from models.base import BaseModel


class EnsembleModel:
    """
    Weighted average ensemble over a dict of trained models.

    Usage:
        ensemble = EnsembleModel(models, weights)
        proba = ensemble.predict_proba(X)  # shape (n, 3)
    """

    def __init__(self, models: dict[str, BaseModel], weights: dict[str, float]):
        """
        Args:
            models: dict of model_name → trained model (with .predict_proba())
            weights: dict of model_name → weight (need not sum to 1; normalised internally)
        """
        if not models:
            raise ValueError("Ensemble requires at least one model.")
        self.models = models
        # Normalise weights to sum to 1 over the provided (enabled) models
        total = sum(weights.get(name, 0.0) for name in models)
        if total == 0:
            # Fallback: equal weights
            self.weights = {name: 1.0 / len(models) for name in models}
        else:
            self.weights = {name: weights.get(name, 0.0) / total for name in models}
        logger.info(
            "Ensemble weights: "
            + ", ".join(f"{n}={w:.3f}" for n, w in self.weights.items())
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return weighted average of all model probabilities. Shape: (n, 3)."""
        weighted_sum = np.zeros((len(X), 3))
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            weighted_sum += self.weights[name] * proba
        # Renormalise (should already sum to 1, but floating-point arithmetic)
        row_sums = weighted_sum.sum(axis=1, keepdims=True).clip(min=1e-9)
        return weighted_sum / row_sums

    def save(self, path: str | Path) -> None:
        # Trusted local file — pickle is necessary for composite model objects
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleModel":
        # Only load from trusted local paths, never from user input
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301


class StackingEnsemble:
    """
    Logistic meta-learner trained on out-of-fold predictions.

    Use this instead of EnsembleModel when you want the ensemble to learn
    how to weight models based on their error correlation on training data.

    Workflow:
        1. Split training data into K folds
        2. For each fold: train base models on K-1 folds, predict the held-out fold
        3. Stack OOF predictions as meta-features
        4. Train a logistic regressor on meta-features → true labels
        5. At inference: concatenate base model predictions → feed to meta-learner

    Warning: requires more data than weighted averaging to avoid overfitting.
    """

    def __init__(self, models: dict[str, BaseModel], meta_C: float = 1.0):
        self.models = models
        self._meta = LogisticRegression(C=meta_C, multi_class="multinomial",
                                         max_iter=500, random_state=42)

    def fit(self, X_oof_preds: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the meta-learner on OOF predictions.

        X_oof_preds: shape (n_samples, n_models * 3) — concatenated OOF probas
        y: true labels (0/1/2)
        """
        self._meta.fit(X_oof_preds, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run base models → concatenate probas → meta-learner prediction."""
        meta_input = np.concatenate(
            [model.predict_proba(X) for model in self.models.values()], axis=1
        )
        return self._meta.predict_proba(meta_input)
