"""
Tier 2 quality-gate safety net.

Reads `backend/data/output/eval_ensemble.json` (produced by commit 6's
training run) and asserts the trained model passed all gates. This test
is the *backup* for Tier 1 (`train.py` exit gate) — Tier 1 fails fast
during training; Tier 2 catches the case where someone somehow shipped
a bad artifact and CI runs without re-training.

Behaviors (per design doc §6 Tier 2 table):
  - file missing → skip (developer ergonomics; cloning + pytest works
    before training);
  - schema_version absent → skip (pre-T2.1 artifact predating
    cv_report.v1; will be auto-overwritten on first commit-6 run);
  - schema_version != "cv_report.v1" → hard-fail (someone shipped a
    forward-incompatible artifact);
  - feature_schema_version mismatch → hard-fail
    (FeatureSchemaMismatch);
  - gates.passed is not True → hard-fail.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from config.loader import settings
from features.build import FEATURE_SCHEMA_VERSION
from evaluation.cv_report import SCHEMA_VERSION as CV_REPORT_SCHEMA_VERSION
from evaluation.exceptions import FeatureSchemaMismatch


def _eval_path() -> Path:
    return Path(settings()["paths"]["output"]) / "eval_ensemble.json"


def _load_or_skip() -> dict:
    path = _eval_path()
    if not path.exists():
        pytest.skip(f"{path.name} missing — run training first.")
    with open(path) as f:
        report = json.load(f)
    if "schema_version" not in report:
        pytest.skip(
            f"{path.name} predates cv_report.v1 — Tier 2 activates after the "
            "first commit-6 training run."
        )
    return report


def test_eval_artifact_schema_version_matches():
    report = _load_or_skip()
    sv = report.get("schema_version")
    assert sv == CV_REPORT_SCHEMA_VERSION, (
        f"eval_ensemble.json schema_version={sv!r}, expected "
        f"{CV_REPORT_SCHEMA_VERSION!r}. Retrain to refresh."
    )


def test_eval_artifact_feature_schema_version_matches():
    report = _load_or_skip()
    fsv = report.get("feature_schema_version")
    if fsv != FEATURE_SCHEMA_VERSION:
        raise FeatureSchemaMismatch(
            f"eval_ensemble.json feature_schema_version={fsv!r}, expected "
            f"{FEATURE_SCHEMA_VERSION!r}. Retrain to refresh."
        )


def test_eval_artifact_gates_passed():
    report = _load_or_skip()
    gates = report.get("gates", {})
    passed = gates.get("passed")
    failures = gates.get("failures", [])
    assert passed is True, (
        f"Quality gates failed ({len(failures)} failure(s)): {failures}"
    )


def test_eval_artifact_age_warning():
    """Warn (not fail) if the artifact is older than 30 days."""
    path = _eval_path()
    if not path.exists():
        pytest.skip(f"{path.name} missing.")
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )
    if age > timedelta(days=30):
        warnings.warn(
            f"{path.name} is {age.days} days old — consider retraining.",
            stacklevel=2,
        )
