"""
XGBoost multi-class classifier for match outcome prediction.

Uses multi:softprob objective to output probabilities for all 3 classes simultaneously.
Early stopping on validation log-loss prevents overfitting.
Feature importances are exposed for model interpretation.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from models.base import BaseModel

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


class XGBoostModel(BaseModel):
    """XGBoost multi-class classifier: outputs (p_home, p_draw, p_away)."""

    def __init__(self, **kwargs):
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        defaults = {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "tree_method": "hist",  # Fast histogram method
            "random_state": 42,
        }
        defaults.update(kwargs)
        self._params = defaults
        self._model: xgb.XGBClassifier | None = None
        self.feature_importances_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train XGBoost with early stopping on a validation set.

        kwargs may include:
            X_val, y_val: validation data for early stopping
            feature_names: list of feature column names
        """
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        feature_names = kwargs.get("feature_names")
        early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)

        self._model = xgb.XGBClassifier(**self._params)
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            fit_kwargs["verbose"] = False

        self._model.fit(X, y, **fit_kwargs)
        self.feature_importances_ = self._model.feature_importances_
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        logger.info(
            f"XGBoost trained: {self._model.n_estimators} estimators, "
            f"best_iteration={getattr(self._model, 'best_iteration', 'N/A')}"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        return self._model.predict_proba(X)

    def get_feature_importance_df(self) -> "pd.DataFrame | None":
        """Return a sorted DataFrame of feature names and importances."""
        if self.feature_importances_ is None or self.feature_names_ is None:
            return None
        import pandas as pd
        return (
            pd.DataFrame({
                "feature": self.feature_names_,
                "importance": self.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
