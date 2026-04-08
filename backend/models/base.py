"""
Abstract base class for all football prediction models.

Every model implements fit/predict_proba/calibrate/save/load.
predict_proba always returns shape (n_samples, 3): [p_home_win, p_draw, p_away_win].

Note on serialization: models are persisted with pickle because sklearn/XGBoost/LightGBM
objects don't have a standardised cross-library JSON format. These files are only ever
loaded from trusted local paths — never from external or user-supplied sources.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


class BaseModel(ABC):
    """Abstract base class for all match outcome prediction models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the model. y contains class labels 0/1/2."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix of shape (n_samples, 3): [p_home, p_draw, p_away]."""
        ...

    def calibrate(
        self, X: np.ndarray, y: np.ndarray, method: str = "isotonic"
    ) -> "CalibratedModel":
        """
        Calibrate the model using isotonic regression on a held-out set.

        Isotonic regression is chosen over Platt (sigmoid) scaling because it's
        non-parametric — it makes no assumption about the shape of the calibration
        curve, which matters for multi-class sports prediction where draw probability
        is often underestimated by the raw model.
        """
        return CalibratedModel(self, X, y, method=method)

    def save(self, path: str | Path) -> None:
        """Persist the model to disk using pickle (trusted local files only)."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load a model from a trusted local pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 — local trusted file only


class CalibratedModel:
    """
    Wraps a BaseModel with per-class isotonic calibration.

    Fits a separate IsotonicRegression for each of the 3 outcome classes,
    then renormalises so probabilities sum to 1.
    """

    def __init__(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                 method: str = "isotonic"):
        self.model = model
        self._calibrators: list[IsotonicRegression] = []
        raw_proba = model.predict_proba(X)
        for cls_idx in range(3):
            y_bin = (y == cls_idx).astype(float)
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw_proba[:, cls_idx], y_bin)
            self._calibrators.append(cal)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(X)
        calibrated = np.stack([
            self._calibrators[i].predict(raw[:, i]) for i in range(3)
        ], axis=1)
        # Renormalise so probabilities sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True).clip(min=1e-9)
        return calibrated / row_sums

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedModel":
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 — local trusted file only
