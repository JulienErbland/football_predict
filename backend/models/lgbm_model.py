"""
LightGBM multi-class classifier for match outcome prediction.

Same interface as XGBoostModel. LightGBM tends to be faster and can sometimes
outperform XGBoost on tabular data with many categorical features (like team names
if one-hot encoded). We use it as a complementary model in the ensemble.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from models.base import BaseModel

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False


class LGBMModel(BaseModel):
    """LightGBM multi-class classifier: outputs (p_home, p_draw, p_away)."""

    def __init__(self, **kwargs):
        if not _LGB_AVAILABLE:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
        defaults = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "metric": "multi_logloss",
            "random_state": 42,
            "verbose": -1,
        }
        defaults.update(kwargs)
        self._params = defaults
        self._model: lgb.LGBMClassifier | None = None
        self.feature_importances_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train LightGBM with optional early stopping.

        kwargs may include:
            X_val, y_val: validation data for early stopping
            feature_names: list of column names
            early_stopping_rounds: int (default 50)
        """
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        feature_names = kwargs.get("feature_names")
        early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)
        sample_weight = kwargs.get("sample_weight")

        self._model = lgb.LGBMClassifier(**self._params)
        callbacks = []
        fit_kwargs: dict = {}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=-1))
            fit_kwargs["callbacks"] = callbacks
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        self._model.fit(X, y, **fit_kwargs)
        self.feature_importances_ = self._model.feature_importances_
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        logger.info(
            f"LightGBM trained: {self._model.n_estimators_} estimators"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        if self.feature_names_ is not None and not hasattr(X, "columns"):
            import pandas as pd
            X = pd.DataFrame(X, columns=self.feature_names_)
        return self._model.predict_proba(X)

    def get_feature_importance_df(self) -> "pd.DataFrame | None":
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
