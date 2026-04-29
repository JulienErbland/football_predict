"""
Frozen-dataclass schema for the per-training CV report (`cv_report.v1`).

The report is the single artifact that captures everything a reviewer
needs to judge a training run: per-fold metrics + leakage guards, holdout
results pinned to a snapshot hash, the gate verdict, and the final
retrained model's feature importances. It serialises as JSON
(``to_json()``); ``CVReport.from_json()`` round-trips bit-for-bit.

Frozen dataclasses give us:
  - immutability (no in-place mutation after construction);
  - structural equality (used by the roundtrip test);
  - ``dataclasses.asdict()`` for serialization without a Pydantic dep.

Public surface (locked, YAGNI):
    CVReport.to_json() -> str
    CVReport.from_json(s: str) -> CVReport
    CVReport.assert_gates() -> None    # raises QualityGateFailure

No ``to_dict``, ``compare_to``, ``save``, or ``summary`` — those are
consumer concerns (T2.5 may add them when stacking lands).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

from evaluation.exceptions import QualityGateFailure


SCHEMA_VERSION = "cv_report.v1"


@dataclass(frozen=True)
class FoldMetrics:
    brier: float
    rps: float
    log_loss: float
    accuracy: float
    draw_f1: float
    home_recall: float
    draw_recall: float
    away_recall: float


@dataclass(frozen=True)
class FoldGuards:
    n_train: int
    n_val: int
    train_seasons: tuple[int, ...]
    train_matchday_max: int
    val_season: int
    val_matchday_range: tuple[int, int]
    leakage_check: str  # "passed" | "failed"


@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    metrics: FoldMetrics
    guards: FoldGuards


@dataclass(frozen=True)
class CVSection:
    folds: tuple[FoldResult, ...]
    mean_metrics: FoldMetrics
    std_metrics: FoldMetrics


@dataclass(frozen=True)
class HoldoutSection:
    season: int
    n_test: int
    metrics: FoldMetrics
    snapshot_hash: str  # "sha256:<hex>"


@dataclass(frozen=True)
class GatesSection:
    max_rps: float
    max_brier: float
    min_draw_f1: float
    cv_mean_rps: float
    cv_mean_brier: float
    cv_mean_draw_f1: float
    holdout_rps: float
    holdout_brier: float
    holdout_draw_f1: float
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationSection:
    method: str    # "isotonic" | "sigmoid"
    cv_folds: int


@dataclass(frozen=True)
class CVReport:
    schema_version: str
    feature_schema_version: str
    timestamp: str
    cv: CVSection
    holdout: HoldoutSection
    gates: GatesSection
    calibration: CalibrationSection
    feature_importances: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, s: str) -> "CVReport":
        return _from_dict(cls, json.loads(s))

    def assert_gates(self) -> None:
        if not self.gates.passed:
            raise QualityGateFailure(self.gates)


def _from_dict(cls: type, data: Any) -> Any:
    """Recursively rebuild frozen dataclasses from their asdict() form.

    Tuple-typed fields (``tuple[T, ...]``) are restored from JSON lists.
    Plain dicts and primitives pass through.
    """
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        raw = data[f.name]
        kwargs[f.name] = _coerce(f.type, raw)
    return cls(**kwargs)


def _coerce(annotation: Any, raw: Any) -> Any:
    """Coerce a JSON-loaded value back to the dataclass field's runtime type."""
    if raw is None:
        return None
    type_str = str(annotation)
    # Tuple of dataclass instances: tuple[FoldResult, ...]
    if type_str.startswith("tuple[") and "..." in type_str:
        inner = _resolve_inner_type(type_str)
        return tuple(_coerce_value(inner, v) for v in raw)
    # Tuple of fixed types: tuple[int, int]
    if type_str.startswith("tuple[") and "..." not in type_str:
        return tuple(raw)
    # Plain dataclass field
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _from_dict(annotation, raw)
    if isinstance(annotation, str):
        # forward ref string; look up by name in this module
        return _coerce_value(annotation, raw)
    return raw


def _resolve_inner_type(type_str: str) -> str:
    """Extract the inner type name from `tuple[X, ...]`."""
    inner = type_str[type_str.find("[") + 1 : type_str.rfind("]")]
    inner = inner.split(",")[0].strip()
    return inner


def _coerce_value(type_name: str, raw: Any) -> Any:
    """Look up a dataclass by name in this module and reconstruct, else passthrough."""
    candidates = {
        "FoldResult": FoldResult,
        "FoldMetrics": FoldMetrics,
        "FoldGuards": FoldGuards,
        "CVSection": CVSection,
        "HoldoutSection": HoldoutSection,
        "GatesSection": GatesSection,
        "CalibrationSection": CalibrationSection,
    }
    cls = candidates.get(type_name)
    if cls is None or not isinstance(raw, dict):
        return raw
    return _from_dict(cls, raw)
