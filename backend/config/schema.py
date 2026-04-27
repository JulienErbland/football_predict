"""Typed Pydantic accessors for the YAML configs.

The dict-based loaders in ``config.loader`` (``settings()``, ``feature_config()``,
``model_config()``) keep working unchanged — typed accessors are an additive
layer for code that wants validation and IDE autocomplete.

Usage::

    from config.schema import load_feature_config, load_model_config
    fc = load_feature_config()
    if fc.form.enabled:
        ...
    mc = load_model_config()
    weights = mc.enabled_model_weights()

Design notes:
  * Each model mirrors the corresponding YAML structure 1:1 — extra keys are
    permitted (``model_config = ConfigDict(extra="allow")``) so YAML can grow
    new fields without breaking the loader.
  * The Phase 2 quality-gate fields (``max_rps``, ``max_brier``,
    ``min_draw_f1``, etc.) are class defaults inside ``TrainingConfig``. They
    are NOT yet keys in ``model_config.yaml`` — when Phase 2 work adds them,
    Pydantic picks them up automatically.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ───────────────────────── feature_config.yaml ──────────────────────────────


class _FeatureGroup(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True


class FormGroup(_FeatureGroup):
    windows: list[int] = Field(default_factory=lambda: [3, 5, 10])


class XgGroup(_FeatureGroup):
    windows: list[int] = Field(default_factory=lambda: [5, 10])


class HeadToHeadGroup(_FeatureGroup):
    window_years: int = 5


class EloGroup(_FeatureGroup):
    k: float = 32
    initial_rating: float = 1500


class FeatureConfig(BaseModel):
    """Mirror of ``feature_config.yaml`` plus Phase 2 placeholders."""
    model_config = ConfigDict(extra="allow")

    form: FormGroup = Field(default_factory=FormGroup)
    xg: XgGroup = Field(default_factory=XgGroup)
    squad: _FeatureGroup = Field(default_factory=_FeatureGroup)
    tactics: _FeatureGroup = Field(default_factory=_FeatureGroup)
    context: _FeatureGroup = Field(default_factory=_FeatureGroup)
    head_to_head: HeadToHeadGroup = Field(default_factory=HeadToHeadGroup)
    elo: EloGroup = Field(default_factory=EloGroup)

    # Phase 2 placeholders — not yet in YAML; defaults take effect until added.
    enable_pi_ratings: bool = False
    pi_rating_lambda: float = 0.035
    pi_rating_gamma: float = 0.7
    enable_weather: bool = False


# ─────────────────────────── model_config.yaml ──────────────────────────────


class ModelSpec(BaseModel):
    """One entry under ``models:`` in model_config.yaml."""
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    weight: float = 0.0


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: Literal["isotonic", "sigmoid"] = "isotonic"
    cv_folds: int = 5


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: Literal["weighted_average", "stacking"] = "weighted_average"


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    test_seasons: list[int] = Field(default_factory=lambda: [2024])


class TrainingConfig(BaseModel):
    """Phase 2 training/quality-gate parameters.

    These fields are NOT yet present in ``model_config.yaml``. When Phase 2
    work adds them under a ``training:`` key, Pydantic picks them up via
    :func:`load_model_config`. Until then, the defaults below are the source
    of truth and are referenced by ``IMPLEMENTATION_PLAN_v2.md``.

    All metric thresholds use the literature convention adopted in T0.2:
    Brier is mean across K=3 classes (range [0, 1]); RPS is per-sample
    cumulative score divided by K-1 (range [0, 1]).
    """
    model_config = ConfigDict(extra="allow")

    # Class weights for draw-aware training (T2.2)
    class_weight_home: float = 1.0
    class_weight_draw: float = 2.5
    class_weight_away: float = 1.2

    # Calibrated draw threshold search range (T2.2)
    draw_threshold_min: float = 0.18
    draw_threshold_max: float = 0.32

    # Quality gates — literature convention, post-T0.2.
    # Phase 1 baseline lands at brier=0.2006 / rps=0.2069, so these are a
    # tight "do no harm" floor for Phase 2 changes.
    max_rps: float = 0.21       # competitive band 0.19–0.23
    max_brier: float = 0.22     # competitive band 0.18–0.22
    min_draw_f1: float = 0.25   # deployment gate

    # Sampling / CV (T2.1, T2.2)
    smote_k_neighbors: int = 5
    cv_seasons: int = 3


class ModelConfig(BaseModel):
    """Mirror of ``model_config.yaml``."""
    model_config = ConfigDict(extra="allow")

    models: dict[str, ModelSpec] = Field(default_factory=dict)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    def enabled_model_weights(self) -> dict[str, float]:
        """Renormalised weights for enabled models only.

        Mirrors the runtime-ensemble logic in ``models/ensemble.py``.
        """
        enabled = {n: m.weight for n, m in self.models.items() if m.enabled}
        total = sum(enabled.values())
        if total == 0:
            return {}
        return {n: w / total for n, w in enabled.items()}


# ───────────────────────────── accessors ────────────────────────────────────


def load_feature_config() -> FeatureConfig:
    """Typed view over ``feature_config.yaml``."""
    from .loader import feature_config  # noqa: PLC0415 — avoid import cycle
    return FeatureConfig.model_validate(feature_config())


def load_model_config() -> ModelConfig:
    """Typed view over ``model_config.yaml``."""
    from .loader import model_config  # noqa: PLC0415
    return ModelConfig.model_validate(model_config())
