"""
Tests for backend/training/draw_handling.py — T2.2's draw-class handling helpers.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_resample_off_returns_input_unchanged():
    from training.draw_handling import resample

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y = np.array([0] * 46 + [1] * 24 + [2] * 30)

    X_out, y_out = resample(X, y, sampling_strategy="off", k_neighbors=5)

    assert X_out is X, "off mode should return the input X unchanged (identity)"
    assert y_out is y, "off mode should return the input y unchanged (identity)"


def test_resample_auto_balances_to_majority():
    from training.draw_handling import resample

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y = np.array([0] * 46 + [1] * 24 + [2] * 30)

    X_out, y_out = resample(X, y, sampling_strategy="auto", k_neighbors=5)

    counts = np.bincount(y_out, minlength=3)
    assert counts[0] == counts[1] == counts[2], (
        f"auto mode should balance all classes to majority count; got {counts.tolist()}"
    )
    assert counts[0] == 46, "majority class count should be preserved"


def test_resample_partial_lifts_draws_to_target_fraction():
    """partial: 'draws → 70% of H count' means draws should equal int(0.7 * n_home)."""
    from training.draw_handling import resample

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y = np.array([0] * 46 + [1] * 24 + [2] * 30)

    X_out, y_out = resample(X, y, sampling_strategy="partial_70", k_neighbors=5)

    counts = np.bincount(y_out, minlength=3)
    expected_draws = int(0.7 * 46)  # 32
    assert counts[1] == expected_draws, (
        f"partial_70 should set draws to int(0.7 * n_home) = {expected_draws}; "
        f"got {counts[1]}"
    )
    assert counts[0] == 46, "home count should be unchanged in partial mode"
    assert counts[2] == 30, "away count should be unchanged in partial mode"


def test_class_sample_weights_returns_per_sample_array():
    from training.draw_handling import class_sample_weights

    y = np.array([0, 1, 2, 1, 0])
    weights = {"H": 1.0, "D": 2.5, "A": 1.2}

    sw = class_sample_weights(y, weights)

    expected = np.array([1.0, 2.5, 1.2, 2.5, 1.0])
    np.testing.assert_array_equal(sw, expected)


def test_class_sample_weights_preserves_dtype():
    from training.draw_handling import class_sample_weights

    y = np.array([0, 1, 2])
    weights = {"H": 1.0, "D": 2.5, "A": 1.2}

    sw = class_sample_weights(y, weights)

    assert sw.dtype == np.float64
    assert sw.shape == (3,)
