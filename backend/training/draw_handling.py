"""
T2.2 draw-class handling primitives.

Module-level architecture per design doc §2.1:
    resample(X, y, sampling_strategy, k_neighbors) -> (X', y')
    class_sample_weights(y, weights_dict) -> np.ndarray
    find_draw_threshold(...)              # Task 5
    predict_with_threshold(...)           # Task 5
    recompute_discrete_metrics(...)       # Task 5

Class index convention: {0: home_win, 1: draw, 2: away_win}.
"""
from __future__ import annotations

import numpy as np

try:
    from imblearn.over_sampling import SMOTE
    _IMBLEARN_AVAILABLE = True
except ImportError:
    _IMBLEARN_AVAILABLE = False


_DRAW_CLASS = 1  # locked: cv.py, train.py, predict.py all use this convention


def resample(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE-resample (X, y) per the requested sampling_strategy.

    Args:
        X: feature array, shape (n, n_features).
        y: integer class labels in {0, 1, 2}.
        sampling_strategy: one of:
            "off"          - return inputs unchanged (identity).
            "auto"         - SMOTE 'auto' (full balance to majority class count).
            "partial_70"   - oversample draws (class 1) to 70% of home count.
        k_neighbors: SMOTE k_neighbors parameter.
        random_state: SMOTE random_state for reproducibility.

    Returns: (X_resampled, y_resampled). For "off", returns (X, y) unchanged.
    """
    if sampling_strategy == "off":
        return X, y

    if not _IMBLEARN_AVAILABLE:
        raise ImportError("imbalanced-learn is not installed. Run: pip install imbalanced-learn")

    if sampling_strategy == "auto":
        smote = SMOTE(
            sampling_strategy="auto",
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
    elif sampling_strategy == "partial_70":
        n_home = int((y == 0).sum())
        target_draws = int(0.7 * n_home)
        n_draw_current = int((y == _DRAW_CLASS).sum())
        if target_draws <= n_draw_current:
            return X, y
        smote = SMOTE(
            sampling_strategy={_DRAW_CLASS: target_draws},
            k_neighbors=k_neighbors,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown sampling_strategy={sampling_strategy!r}. "
            "Expected 'off', 'auto', or 'partial_70'."
        )

    return smote.fit_resample(X, y)


def class_sample_weights(
    y: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """Return per-sample weight array derived from class weights.

    Args:
        y: integer class labels in {0, 1, 2}.
        weights: dict mapping {"H", "D", "A"} → float.

    Returns: float64 array of shape (n,) with the weight for each sample's class.
    """
    label_to_weight = np.array([weights["H"], weights["D"], weights["A"]], dtype=np.float64)
    return label_to_weight[y]
