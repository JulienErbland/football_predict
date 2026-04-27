"""Regression test for the T0.2 metrics-convention alignment.

Guards against a reintroduction of the pre-T0.2 sum-based convention where
``brier_score`` returned the sum across K=3 classes (≈3× literature) and
``rps`` returned the raw un-normalised sum over K-1 thresholds (≈2× literature).

The tests pin down a small, hand-computable example for each metric so that
any accidental re-introduction of the sum convention fails loudly.

See ``backend/evaluation/METRICS_CHANGELOG.md`` for the migration context.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    accuracy,
    brier_score,
    evaluate_predictions,
    log_loss_score,
    rps,
)


# ── hand-computable fixture ──────────────────────────────────────────────────
#
# Two samples, perfect forecasts on sample 0, flat forecast on sample 1.
#
#   sample 0: true=H (class 0), pred=[1.0, 0.0, 0.0]
#     one_hot = [1, 0, 0]; diff² = [0, 0, 0]; sum = 0; mean-over-K = 0
#     cumsum_pred(K-1) = [1.0, 1.0]; cumsum_true(K-1) = [1.0, 1.0]
#     sum_sq_diff = 0
#
#   sample 1: true=D (class 1), pred=[1/3, 1/3, 1/3]
#     one_hot = [0, 1, 0]; diff² = [1/9, 4/9, 1/9]; sum = 6/9 = 2/3
#     mean-over-K = 2/9
#     cumsum_pred = [1/3, 2/3]; cumsum_true = [0, 1]
#     diff² = [1/9, 1/9]; sum = 2/9
#
# brier (mean over samples of mean-over-K): (0 + 2/9) / 2 = 1/9 ≈ 0.1111
# rps (mean over samples of sum / (K-1)):   (0 + 2/9) / 2 / 2 = 1/18 ≈ 0.0556
# ──────────────────────────────────────────────────────────────────────────────

Y_TRUE = np.array([0, 1], dtype=int)
Y_PROBA = np.array([[1.0, 0.0, 0.0],
                    [1 / 3, 1 / 3, 1 / 3]], dtype=float)


class TestBrierScoreConvention:
    """brier_score must be mean-across-K, not sum-across-K."""

    def test_hand_computed_value(self):
        # (0 + 2/9) / 2 = 1/9
        expected = 1.0 / 9.0
        assert brier_score(Y_TRUE, Y_PROBA) == pytest.approx(expected, abs=1e-12)

    def test_perfect_forecast_is_zero(self):
        y_true = np.array([0, 1, 2])
        y_proba = np.eye(3)
        assert brier_score(y_true, y_proba) == pytest.approx(0.0, abs=1e-12)

    def test_uniform_forecast_literature_range(self):
        # Uniform 1/3 predictions across N samples: mean-over-K diff² = 2/9
        # regardless of the true class. This sits inside the [0, 1] literature
        # range (not [0, 2]) — the T0.2 signature property.
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=1000)
        y_proba = np.full((1000, 3), 1 / 3)
        val = brier_score(y_true, y_proba)
        assert val == pytest.approx(2.0 / 9.0, abs=1e-12)
        assert 0.0 <= val <= 1.0  # must NOT be in [0, 2] any more

    def test_competitive_range_sanity(self):
        # A mildly confident-but-wrong forecast should land in the published
        # 0.18–0.22 "competitive" band — the value that motivates T0.2.
        y_true = np.array([0] * 100 + [1] * 60 + [2] * 80)
        # Realistic football-ish probability mix
        y_proba = np.tile([0.45, 0.27, 0.28], (240, 1))
        val = brier_score(y_true, y_proba)
        assert 0.15 <= val <= 0.25


class TestRpsConvention:
    """rps must be normalised by (K-1)."""

    def test_hand_computed_value(self):
        # (0 + 2/9) / 2 / 2 = 1/18
        expected = 1.0 / 18.0
        assert rps(Y_TRUE, Y_PROBA) == pytest.approx(expected, abs=1e-12)

    def test_perfect_forecast_is_zero(self):
        y_true = np.array([0, 1, 2])
        y_proba = np.eye(3)
        assert rps(y_true, y_proba) == pytest.approx(0.0, abs=1e-12)

    def test_literature_range(self):
        # Uniform 1/3 predictions: cumsum_pred = [1/3, 2/3].
        # For true=H: cumsum_true = [1, 1] → diff² = [(2/3)², (1/3)²] = 5/9
        # For true=D: cumsum_true = [0, 1] → diff² = [(1/3)², (1/3)²] = 2/9
        # For true=A: cumsum_true = [0, 0] → diff² = [(1/3)², (2/3)²] = 5/9
        # Expected per-sample average ≈ 4/9; normalise by K-1=2 → 2/9 ≈ 0.222
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=5000)
        y_proba = np.full((5000, 3), 1 / 3)
        val = rps(y_true, y_proba)
        # With the class mix drawn uniformly, expected value is (5+2+5)/27 / 2
        assert 0.20 <= val <= 0.24


class TestEvaluatePredictions:
    """evaluate_predictions must bundle the literature-convention metrics."""

    def test_returns_expected_keys(self):
        out = evaluate_predictions(Y_TRUE, Y_PROBA)
        assert set(out) == {"brier_score", "log_loss", "rps", "accuracy", "n_samples"}

    def test_values_match_individual_functions(self):
        out = evaluate_predictions(Y_TRUE, Y_PROBA)
        assert out["brier_score"] == pytest.approx(brier_score(Y_TRUE, Y_PROBA))
        assert out["rps"] == pytest.approx(rps(Y_TRUE, Y_PROBA))
        assert out["log_loss"] == pytest.approx(log_loss_score(Y_TRUE, Y_PROBA))
        assert out["accuracy"] == pytest.approx(accuracy(Y_TRUE, Y_PROBA))
        assert out["n_samples"] == 2
