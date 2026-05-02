"""
Regression tests for sample_weight plumbing in XGBoostModel.fit / LGBMModel.fit.

T2.2 commit 2 prerequisite: Task 2's SMOTE+class_weight ablation harness passes
``sample_weight=...`` into ``model.fit(...)``. Before this commit the wrappers
silently dropped the kwarg, which would make the 6-cell ablation effectively a
3-cell ablation (uniform vs weighted cells would coincide).

The tests below assert that:
    1. fit(..., sample_weight=...) does not error.
    2. Different sample_weight vectors produce different predict_proba outputs
       (i.e. the kwarg is actually consumed by the underlying booster, not
       discarded into **kwargs limbo).
"""
from __future__ import annotations

import numpy as np
import pytest

from models.lgbm_model import LGBMModel
from models.xgboost_model import XGBoostModel


def _toy_dataset(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6)).astype(np.float32)
    # Make class 1 (draw) easier to overweight: separable along feature 0.
    y = np.where(X[:, 0] < -0.5, 1, np.where(X[:, 0] < 0.5, 0, 2)).astype(int)
    return X, y


@pytest.mark.parametrize("Model", [XGBoostModel, LGBMModel])
def test_fit_accepts_sample_weight_without_error(Model):
    X, y = _toy_dataset()
    sw = np.ones(len(y), dtype=np.float64)

    m = Model(n_estimators=20, learning_rate=0.1)
    m.fit(X, y, sample_weight=sw)

    proba = m.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("Model", [XGBoostModel, LGBMModel])
def test_sample_weight_wrong_length_raises(Model):
    """If sample_weight were silently dropped (not propagated to the underlying
    booster), the wrong length wouldn't matter. We assert that the booster
    receives it and validates length."""
    X, y = _toy_dataset()
    sw_too_short = np.ones(len(y) - 5, dtype=np.float64)

    m = Model(n_estimators=10, learning_rate=0.1)
    with pytest.raises((ValueError, RuntimeError, Exception)) as exc_info:
        m.fit(X, y, sample_weight=sw_too_short)
    # Just confirm an error was raised — exact message varies between
    # XGBoost / LightGBM and across versions.
    assert exc_info.value is not None


@pytest.mark.parametrize("Model", [XGBoostModel, LGBMModel])
def test_sample_weight_changes_predictions_on_imbalanced_data(Model):
    """With an imbalanced + noisy dataset and a held-out test set, heavy
    upweighting of a minority class must shift predicted probabilities for
    that class on the test set. Uniform vs weighted boosts should differ."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 4)).astype(np.float32)
    # Draws (class 1) are a 10% minority; everything else is balanced 45/45.
    y_clean = np.where(X[:, 0] > 0.6, 2, 0)
    minority_mask = rng.random(n) < 0.10
    y = np.where(minority_mask, 1, y_clean).astype(int)
    # Add 15% label noise so probabilities don't saturate.
    flip = rng.random(n) < 0.15
    y_noisy = np.where(flip, rng.integers(0, 3, n), y).astype(int)

    split = n // 2
    X_tr, y_tr = X[:split], y_noisy[:split]
    X_te = X[split:]

    sw_uniform = np.ones(len(y_tr), dtype=np.float64)
    sw_skewed = np.where(y_tr == 1, 30.0, 1.0).astype(np.float64)

    m_u = Model(n_estimators=80, learning_rate=0.1)
    m_u.fit(X_tr, y_tr, sample_weight=sw_uniform)
    mean_draw_uniform = m_u.predict_proba(X_te)[:, 1].mean()

    m_s = Model(n_estimators=80, learning_rate=0.1)
    m_s.fit(X_tr, y_tr, sample_weight=sw_skewed)
    mean_draw_skewed = m_s.predict_proba(X_te)[:, 1].mean()

    assert mean_draw_skewed > mean_draw_uniform + 0.05, (
        f"sample_weight effect not visible: mean_draw_uniform={mean_draw_uniform:.3f} "
        f"vs mean_draw_skewed={mean_draw_skewed:.3f} — expected >5pp increase under 30× upweighting."
    )
