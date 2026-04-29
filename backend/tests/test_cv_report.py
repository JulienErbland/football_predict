"""
Roundtrip and gate-assertion tests for CVReport (schema cv_report.v1).
"""

from __future__ import annotations

import pytest

from evaluation.cv_report import (
    CVReport, CVSection, FoldMetrics, FoldGuards, FoldResult,
    HoldoutSection, GatesSection, CalibrationSection,
)
from evaluation.exceptions import QualityGateFailure


def _zero_metrics() -> FoldMetrics:
    return FoldMetrics(
        brier=0.21, rps=0.20, log_loss=1.07, accuracy=0.50,
        draw_f1=0.10, home_recall=0.65, draw_recall=0.05, away_recall=0.55,
    )


def _make_report(passed: bool = True, failures: tuple[str, ...] = ()) -> CVReport:
    fm = _zero_metrics()
    fg = FoldGuards(
        n_train=4000, n_val=400, train_seasons=(2021, 2022),
        train_matchday_max=38, val_season=2023, val_matchday_range=(10, 18),
        leakage_check="passed",
    )
    fold = FoldResult(fold_id=0, metrics=fm, guards=fg)
    return CVReport(
        schema_version="cv_report.v1",
        feature_schema_version="2.0",
        timestamp="2026-04-29T16:00:00+00:00",
        cv=CVSection(folds=(fold,), mean_metrics=fm, std_metrics=fm),
        holdout=HoldoutSection(
            season=2024, n_test=1752, metrics=fm,
            snapshot_hash="sha256:" + "a" * 64,
        ),
        gates=GatesSection(
            max_rps=0.21, max_brier=0.22, min_draw_f1=0.25,
            cv_mean_rps=0.20, cv_mean_brier=0.21, cv_mean_draw_f1=0.10,
            holdout_rps=0.21, holdout_brier=0.22, holdout_draw_f1=0.12,
            passed=passed, failures=failures,
        ),
        calibration=CalibrationSection(method="isotonic", cv_folds=6),
        feature_importances={"elo_diff": 0.12, "form_diff_5": 0.08},
    )


def test_cvreport_json_roundtrip():
    original = _make_report()
    serialized = original.to_json()
    parsed = CVReport.from_json(serialized)
    assert parsed == original


def test_assert_gates_passes_when_passed_true():
    report = _make_report(passed=True)
    report.assert_gates()  # must not raise


def test_assert_gates_raises_when_passed_false():
    report = _make_report(passed=False, failures=("min_draw_f1: 0.10 < 0.25",))
    with pytest.raises(QualityGateFailure) as exc_info:
        report.assert_gates()
    assert exc_info.value.gates is report.gates


def test_quality_gate_failure_verbose_breakdown_includes_failures():
    report = _make_report(passed=False, failures=(
        "min_draw_f1: 0.10 < 0.25",
        "cv_mean_rps: 0.22 > 0.21",
    ))
    with pytest.raises(QualityGateFailure) as exc_info:
        report.assert_gates()
    breakdown = exc_info.value.verbose_breakdown()
    assert "min_draw_f1" in breakdown
    assert "cv_mean_rps" in breakdown
    assert "Thresholds" in breakdown
    assert "CV means" in breakdown
    assert "Holdout" in breakdown


def test_frozen_dataclass_immutability():
    report = _make_report()
    with pytest.raises((AttributeError, TypeError)):
        report.schema_version = "v2"  # type: ignore[misc]


def test_schema_version_is_v1():
    assert _make_report().schema_version == "cv_report.v1"
