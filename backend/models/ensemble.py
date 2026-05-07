"""
Weighted average ensemble model.

Combines predictions from multiple trained models using configurable weights.
Weights are normalised to sum to 1 using only enabled models, so you can disable
a model in model_config.yaml without touching the other weights.

Note: pickle is used for persistence because sklearn/XGBoost objects don't have
a standardised cross-library serialisation format. Load only from trusted local files.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from loguru import logger

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
