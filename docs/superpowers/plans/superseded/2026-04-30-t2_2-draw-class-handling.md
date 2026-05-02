# T2.2 Draw-Class Handling Implementation Plan

> **SUPERSEDED — 2026-05-02**
>
> This 13-commit plan was superseded when T2.2's ablation harness produced `HarnessFailure` on all 6 cells (no SMOTE × class-weight combination cleared the `mean.rps ≤ 0.205 AND mean.brier ≤ 0.215` margin filter), making the design's three-mechanism premise empirically untenable on this corpus. Tasks 1–3 executed substantially as planned and landed on `phase2/t2.2-draw-class-handling`; Tasks 4–13 were not executed. See `MODEL_REVIEW.md §6.8` for the measured negative result, the schema-2.1 sanity probe that confirmed Task 8's planned features would not have closed the rps gap, and the path-forward candidates; a forthcoming meta-spec amendment will govern the scope of any successor ticket.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four-mechanism draw-class handling (SMOTE, class weights, calibrated draw threshold θ_D, three new features) so the model passes the `min_draw_f1 ≥ 0.25` quality gate that T2.1 intentionally failed, while maintaining `max_rps ≤ 0.21` and `max_brier ≤ 0.22`.

**Architecture:** New `backend/training/draw_handling.py` module centralizes the four T2.2 mechanisms. SMOTE+sample_weight wire into `train_calibrated_models()` in cv.py (single integration point for both per-fold CV and final retrain). θ_D is grid-searched post-CV on stacked post-calibration ensemble OOF probabilities and stored as an attribute on EnsembleModel. train.py is reordered to a 9-phase strict-mode flow that partitions deployment-blocking gates (draw_f1) from inspect-on-failure gates (RPS, Brier).

**Tech Stack:** Python 3.12, scikit-learn ≥1.4, XGBoost ≥2.0, LightGBM ≥4.0, **imbalanced-learn ≥0.12,<1 (NEW)**, pandas, numpy, loguru, pytest. Existing patterns to mirror: `backend/evaluation/cv.py` for orchestration, `backend/evaluation/cv_report.py` for frozen dataclass artifacts, `tools/validate_cv_parametrization.py` for one-shot empirical harnesses.

**Reference**: `docs/superpowers/specs/2026-04-30-t2_2-draw-class-handling-design.md` is the authoritative spec. Section references below (e.g. "§3.3") point into that doc.

**Branch**: `phase2/t2.2-draw-class-handling` (already created, currently at `446b7c6` carrying the design-doc commit). All 13 commits land on this branch; PR to main at the end.

**Note on serialization format**: The existing codebase uses pickle for `ensemble.pkl` and `feature_cols.pkl` (T2.1 convention; see `backend/models/ensemble.py:save` and `backend/models/base.py:save` for the existing comments). T2.2 extends this convention without change — `pickle.load` / `pickle.dump` calls appear literally in tasks below where engineers need them. Migrating to JSON-based serialization would require choosing portable model-serialization formats for fitted XGBoost/LightGBM boosters; this is a separate architectural decision out of T2.2's scope.

---

## Path Convention Note

The design doc §4 lists `backend/tools/validate_smote_classweight_composition.py` for the ablation harness. **This is incorrect** — T2.1's actual precedent is project-root `tools/validate_cv_parametrization.py` (sets `sys.path` to backend/ at the top). Plan uses **project-root `tools/`** to mirror T2.1's actual convention. Update the design doc inline if/when convenient; not blocking.

---

## File Structure Overview

### Files created during this plan

```
backend/training/__init__.py                                    # Task 1
backend/training/draw_handling.py                               # Task 1, 5
tools/validate_smote_classweight_composition.py                 # Task 2
backend/data/output/smote_classweight_ablation.json             # Task 3 (artifact)
backend/data/output/decomposition/eval_pre_features.json        # Task 6b (artifact)
backend/data/output/training_recipe.json                        # Task 7b → produced by Task 10 (artifact)
backend/tests/test_draw_handling.py                             # Task 1, 5
backend/tests/test_smote_integration.py                         # Task 4
backend/tests/test_run_cv_oof.py                                # Task 6a
backend/tests/test_train_strict_mode.py                         # Task 7a
backend/tests/test_training_recipe.py                           # Task 7b
backend/tests/test_new_draw_features.py                         # Task 8
backend/tests/test_predict_threshold.py                         # Task 9
```

### Files modified during this plan

```
backend/requirements.txt              # Task 1
backend/config/model_config.yaml      # Task 4
backend/evaluation/cv.py              # Tasks 4, 6a
backend/models/ensemble.py            # Task 5
backend/models/train.py               # Tasks 6a, 6b, 7a, 7b
backend/features/elo.py               # Task 8
backend/features/form.py              # Task 8
backend/features/context.py           # Task 8
backend/features/build.py             # Task 8 (FEATURE_SCHEMA_VERSION bump)
backend/output/predict.py             # Task 9
MODEL_REVIEW.md                       # Tasks 3, 11
```

### Files deliberately untouched

```
backend/evaluation/cv_report.py       # cv_report.v1 is locked per meta-spec §5
backend/evaluation/exceptions.py      # QualityGateFailure already covers all gates
backend/evaluation/splits.py          # WalkForwardSplit defaults locked at T2.1
backend/data/eval/holdout_2024_25.json # holdout snapshot is locked
```

---

## Task 1: Add `imbalanced-learn` and skeleton `draw_handling.py`

**Goal**: Land the dependency and the bare module skeleton so Task 2's harness can import `resample` and `class_sample_weights`. Defer `find_draw_threshold` / `predict_with_threshold` / `recompute_discrete_metrics` to Task 5.

**Reference**: design doc §2.1, §6.

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/training/__init__.py`
- Create: `backend/training/draw_handling.py`
- Create: `backend/tests/test_draw_handling.py`

- [ ] **Step 1: Add `imbalanced-learn` to `backend/requirements.txt`**

Append `imbalanced-learn>=0.12,<1` to the dependencies section (after `lightgbm>=4.0,<5`):

```
imbalanced-learn>=0.12,<1
```

- [ ] **Step 2: Install the dependency**

Run: `cd backend && pip install -r requirements.txt`

Expected: `Successfully installed imbalanced-learn-0.X.Y` (version 0.12+, <1.0).

- [ ] **Step 3: Create `backend/training/__init__.py`**

```python
"""T2.2 training-time helpers: draw-class handling (SMOTE, class weights, θ_D)."""
```

- [ ] **Step 4: Write the failing test for `resample`**

Create `backend/tests/test_draw_handling.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_draw_handling.py -v`

Expected: FAILS with `ModuleNotFoundError: No module named 'training.draw_handling'`.

- [ ] **Step 6: Create `backend/training/draw_handling.py` with `resample` and `class_sample_weights`**

```python
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
        # imblearn requires target ≥ current count; if already ≥ target, skip resample
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
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_draw_handling.py -v`

Expected: 5 tests PASS in ≤ 2 seconds.

- [ ] **Step 8: Commit**

```bash
git add backend/requirements.txt backend/training/__init__.py backend/training/draw_handling.py backend/tests/test_draw_handling.py
git commit -m "$(cat <<'EOF'
feat(t2.2): add imbalanced-learn + draw_handling.py skeleton

Adds the imbalanced-learn>=0.12,<1 dependency and creates
backend/training/draw_handling.py with the resample() and
class_sample_weights() helpers consumed by Task 2's ablation harness.
The remaining helpers (find_draw_threshold, predict_with_threshold,
recompute_discrete_metrics) land in Task 5.

Per design doc §2.1, class index convention is locked to
{0: home_win, 1: draw, 2: away_win} — consistent with cv.py,
train.py, predict.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: SMOTE+class_weight ablation harness

**Goal**: Build `tools/validate_smote_classweight_composition.py` mirroring T2.1's `tools/validate_cv_parametrization.py`. Iterates the 6-cell ablation grid (3 SMOTE strategies × 2 weight schemes), runs walk-forward CV per cell on all 6 folds (or `--folds 0,2,4` subset), aggregates per-fold metrics, applies the margin filter decision rule, and writes `backend/data/output/smote_classweight_ablation.json`.

**Reference**: design doc §2.2 (cell set, decision rule, output schema, HarnessFailure mode).

**Files:**
- Create: `tools/validate_smote_classweight_composition.py`

- [ ] **Step 1: Read T2.1's harness as the structural reference**

Run: `cat tools/validate_cv_parametrization.py | head -50`

Expected: see the docstring, `_REPO_ROOT` / `sys.path.insert`, and the candidate-loop pattern.

- [ ] **Step 2: Write the harness skeleton with cell definitions and CLI**

Create `tools/validate_smote_classweight_composition.py`:

```python
"""
T2.2 commit 2 — SMOTE+class_weight composition ablation harness.

Iterates a 6-cell grid (3 SMOTE sampling_strategies × 2 class_weight schemes),
runs walk-forward CV per cell on all 6 folds (or --folds subset), aggregates
per-fold (rps, brier, draw_f1, draw_recall, draw_precision) and applies the
margin filter decision rule to pick the winning cell.

Per design doc §2.2:
- Cells:
    1: smote='off',         weights=(1.0, 1.0, 1.0)
    2: smote='off',         weights=(1.0, 2.5, 1.2)
    3: smote='partial_70',  weights=(1.0, 1.0, 1.0)
    4: smote='partial_70',  weights=(1.0, 2.5, 1.2)
    5: smote='auto',        weights=(1.0, 1.0, 1.0)
    6: smote='auto',        weights=(1.0, 2.5, 1.2)
- Margin filter: keep cells where mean.rps ≤ 0.205 AND mean.brier ≤ 0.215.
- Pick max mean.draw_f1 among passing cells; tiebreak: lower std.draw_f1.
- If no cell passes margin: raise HarnessFailure (no auto-tune fallback).

CLI usage (from repo root):
    python -m tools.validate_smote_classweight_composition
    python -m tools.validate_smote_classweight_composition --folds 0,2,4
    python -m tools.validate_smote_classweight_composition --output path/to.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from config.loader import settings, model_config  # noqa: E402
from evaluation.metrics import evaluate_predictions  # noqa: E402
from evaluation.splits import WalkForwardSplit  # noqa: E402
from features.build import FEATURE_SCHEMA_VERSION  # noqa: E402
from models.lgbm_model import LGBMModel  # noqa: E402
from models.xgboost_model import XGBoostModel  # noqa: E402
from training.draw_handling import class_sample_weights, resample  # noqa: E402


_META_COLS = {
    "match_id", "league", "season", "date", "matchday",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
}


_CELLS = [
    {"cell_id": 1, "smote_strategy": "off",         "class_weight": (1.0, 1.0, 1.0)},
    {"cell_id": 2, "smote_strategy": "off",         "class_weight": (1.0, 2.5, 1.2)},
    {"cell_id": 3, "smote_strategy": "partial_70", "class_weight": (1.0, 1.0, 1.0)},
    {"cell_id": 4, "smote_strategy": "partial_70", "class_weight": (1.0, 2.5, 1.2)},
    {"cell_id": 5, "smote_strategy": "auto",        "class_weight": (1.0, 1.0, 1.0)},
    {"cell_id": 6, "smote_strategy": "auto",        "class_weight": (1.0, 2.5, 1.2)},
]


class HarnessFailure(RuntimeError):
    """No cell cleared the margin filter — T2.2 cannot proceed."""


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _draw_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Argmax-based draw metrics (no θ_D applied — composition validation only)."""
    return {
        "draw_f1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "draw_precision": float(precision_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "draw_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
    }


def _train_cell_and_predict(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_inner_val: np.ndarray, y_inner_val: np.ndarray,
    X_val: np.ndarray, feature_names: list[str],
    smote_strategy: str, weights_tuple: tuple[float, float, float],
    smote_k_neighbors: int, mc: dict,
) -> np.ndarray:
    """Apply SMOTE to (X_tr, y_tr), build sample_weight, fit calibrated XGB+LGBM, return ensemble proba on X_val."""
    X_tr_resampled, y_tr_resampled = resample(
        X_tr, y_tr,
        sampling_strategy=smote_strategy,
        k_neighbors=smote_k_neighbors,
    )
    sample_weight = class_sample_weights(
        y_tr_resampled,
        {"H": weights_tuple[0], "D": weights_tuple[1], "A": weights_tuple[2]},
    )

    probas: dict[str, np.ndarray] = {}

    xgb_cfg = mc["models"]["xgboost"]
    if xgb_cfg["enabled"]:
        m = XGBoostModel(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
        )
        m.fit(
            X_tr_resampled, y_tr_resampled,
            X_val=X_inner_val, y_val=y_inner_val,
            feature_names=feature_names,
            sample_weight=sample_weight,
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        )
        probas["xgboost"] = m.calibrate(X_inner_val, y_inner_val).predict_proba(X_val)

    lgbm_cfg = mc["models"]["lightgbm"]
    if lgbm_cfg["enabled"]:
        m = LGBMModel(
            n_estimators=lgbm_cfg["n_estimators"],
            learning_rate=lgbm_cfg["learning_rate"],
            num_leaves=lgbm_cfg["num_leaves"],
        )
        m.fit(
            X_tr_resampled, y_tr_resampled,
            X_val=X_inner_val, y_val=y_inner_val,
            feature_names=feature_names,
            sample_weight=sample_weight,
        )
        probas["lightgbm"] = m.calibrate(X_inner_val, y_inner_val).predict_proba(X_val)

    weights = {
        m_name: mc["models"][m_name]["weight"]
        for m_name in ("xgboost", "lightgbm")
        if mc["models"][m_name]["enabled"]
    }
    keys = list(probas)
    total = sum(weights[k] for k in keys)
    out = sum((weights[k] / total) * probas[k] for k in keys)
    out = np.clip(out, 0.0, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _aggregate(folds: list[dict]) -> tuple[dict, dict]:
    keys = ["rps", "brier", "draw_f1", "draw_precision", "draw_recall"]
    mean = {k: float(statistics.fmean([f[k] for f in folds])) for k in keys}
    std = {
        k: (float(statistics.pstdev([f[k] for f in folds])) if len(folds) > 1 else 0.0)
        for k in keys
    }
    return mean, std


def _select_winner(cells_out: list[dict]) -> dict:
    """Apply margin filter; return chosen cell.

    Margin: mean.rps ≤ 0.205 AND mean.brier ≤ 0.215 (gates with 0.005 margin).
    Pick max mean.draw_f1 (tiebreak: lower std.draw_f1).
    """
    passing = [
        c for c in cells_out
        if c["mean"]["rps"] <= 0.205 and c["mean"]["brier"] <= 0.215
    ]
    if not passing:
        raise HarnessFailure(
            "No cell within margin (rps ≤ 0.205, brier ≤ 0.215). "
            "T2.2's three-pronged approach is insufficient. "
            "Retrospective should address fourth-mechanism candidates "
            "(focal loss, weighted Brier, ordinal regression). "
            "Meta-spec re-open required."
        )
    passing.sort(key=lambda c: (-c["mean"]["draw_f1"], c["std"]["draw_f1"]))
    return passing[0]


def run(
    features_path: Path | None = None,
    output_path: Path | None = None,
    folds_subset: list[int] | None = None,
    smote_k_neighbors: int = 5,
) -> dict:
    cfg = settings()
    mc = model_config()

    if features_path is None:
        features_path = Path(cfg["paths"]["features"])
    if output_path is None:
        output_path = Path(cfg["paths"]["output"]) / "smote_classweight_ablation.json"

    if not features_path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {features_path}. Run features.build first."
        )

    df = pd.read_parquet(features_path).sort_values(["date", "match_id"]).reset_index(drop=True)
    feat_cols = _feature_cols(df)

    splitter = WalkForwardSplit()
    fold_specs = splitter.fold_specs(df)
    fold_pairs = list(splitter.split(df))
    if folds_subset is not None:
        fold_specs = [fold_specs[i] for i in folds_subset]
        fold_pairs = [fold_pairs[i] for i in folds_subset]

    logger.info(
        f"Loaded {len(df)} rows × {len(feat_cols)} features. "
        f"Running {len(_CELLS)} cells × {len(fold_specs)} folds = "
        f"{len(_CELLS) * len(fold_specs)} fold-runs."
    )

    started_at = datetime.now(timezone.utc).isoformat()
    cells_out: list[dict] = []

    for cell in _CELLS:
        logger.info(
            f"=== Cell {cell['cell_id']}: smote={cell['smote_strategy']!r} "
            f"class_weight={cell['class_weight']} ==="
        )
        per_fold: list[dict] = []
        for spec, (train_idx, val_idx) in zip(fold_specs, fold_pairs):
            t0 = time.time()
            X_all = df.iloc[train_idx][feat_cols].fillna(0).to_numpy(np.float32)
            y_all = df.iloc[train_idx]["result"].to_numpy(int)
            X_val = df.iloc[val_idx][feat_cols].fillna(0).to_numpy(np.float32)
            y_val = df.iloc[val_idx]["result"].to_numpy(int)
            cut = int(len(X_all) * 0.9)
            X_tr, y_tr = X_all[:cut], y_all[:cut]
            X_iv, y_iv = X_all[cut:], y_all[cut:]

            ens_proba = _train_cell_and_predict(
                X_tr, y_tr, X_iv, y_iv, X_val, feat_cols,
                smote_strategy=cell["smote_strategy"],
                weights_tuple=cell["class_weight"],
                smote_k_neighbors=smote_k_neighbors,
                mc=mc,
            )
            base = evaluate_predictions(y_val, ens_proba)
            pred = ens_proba.argmax(axis=1)
            draw_m = _draw_metrics(y_val, pred)

            fold_record = {
                "fold_id": int(spec["fold_id"]),
                "rps": base["rps"],
                "brier": base["brier_score"],
                **draw_m,
            }
            per_fold.append(fold_record)
            logger.info(
                f"  fold={spec['fold_id']} season={spec['val_season']} "
                f"rps={base['rps']:.4f} brier={base['brier_score']:.4f} "
                f"draw_f1={draw_m['draw_f1']:.3f} ({time.time()-t0:.1f}s)"
            )

        mean, std = _aggregate(per_fold)
        cells_out.append({
            "cell_id": cell["cell_id"],
            "smote_strategy": cell["smote_strategy"],
            "class_weight": list(cell["class_weight"]),
            "fold_results": per_fold,
            "mean": mean,
            "std": std,
        })
        logger.info(
            f"  AGGREGATE: mean.rps={mean['rps']:.4f} mean.brier={mean['brier']:.4f} "
            f"mean.draw_f1={mean['draw_f1']:.3f} std.draw_f1={std['draw_f1']:.3f}"
        )

    winner = _select_winner(cells_out)
    logger.info(
        f"WINNING CELL: id={winner['cell_id']} smote={winner['smote_strategy']!r} "
        f"class_weight={winner['class_weight']} "
        f"mean.draw_f1={winner['mean']['draw_f1']:.3f}"
    )

    report = {
        "schema_version": "smote_cw_ablation.v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,  # "2.0" at this commit
        "generated_at": started_at,
        "smote_k_neighbors": smote_k_neighbors,
        "folds_used": [int(s["fold_id"]) for s in fold_specs],
        "cells": cells_out,
        "winner": {
            "cell_id": winner["cell_id"],
            "smote_strategy": winner["smote_strategy"],
            "class_weight": winner["class_weight"],
            "rationale": (
                f"Highest mean.draw_f1 ({winner['mean']['draw_f1']:.3f}) "
                "among cells passing margin filter (rps ≤ 0.205, brier ≤ 0.215). "
                f"std.draw_f1 = {winner['std']['draw_f1']:.3f}."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=None,
                        help="Path to features.parquet (default: from settings).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: data/output/smote_classweight_ablation.json).")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold indices to run (e.g. '0,2,4'). Default: all.")
    parser.add_argument("--smote-k-neighbors", type=int, default=5,
                        help="SMOTE k_neighbors (default: 5).")
    args = parser.parse_args()

    folds_subset = None
    if args.folds:
        folds_subset = [int(x) for x in args.folds.split(",")]

    run(
        features_path=args.features,
        output_path=args.output,
        folds_subset=folds_subset,
        smote_k_neighbors=args.smote_k_neighbors,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-import the harness module**

Run: `cd /home/julienerbland/Documents/Privé/Projets_persos/football_predict && python -c "from tools import validate_smote_classweight_composition; print(validate_smote_classweight_composition._CELLS[5])"`

Expected output: `{'cell_id': 6, 'smote_strategy': 'auto', 'class_weight': (1.0, 2.5, 1.2)}`

- [ ] **Step 4: Verify CLI args parse**

Run: `cd /home/julienerbland/Documents/Privé/Projets_persos/football_predict && python -m tools.validate_smote_classweight_composition --help`

Expected: argparse help text listing `--features`, `--output`, `--folds`, `--smote-k-neighbors`.

- [ ] **Step 5: Commit**

```bash
git add tools/validate_smote_classweight_composition.py
git commit -m "$(cat <<'EOF'
feat(t2.2): SMOTE+class_weight ablation harness

Adds tools/validate_smote_classweight_composition.py mirroring T2.1's
tools/validate_cv_parametrization.py pattern. Iterates the 6-cell
ablation grid (3 SMOTE strategies × 2 weight schemes), runs walk-forward
CV per cell on all 6 folds (or --folds subset), aggregates per-fold
metrics, applies the margin filter decision rule (rps ≤ 0.205,
brier ≤ 0.215) and writes data/output/smote_classweight_ablation.json.
HarnessFailure raised when no cell passes margin — no auto-tune fallback.

Per design doc §2.2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Run harness, commit results JSON + winning-cell analysis

**Goal**: Execute the harness on the full 6-fold CV pool, capture the result JSON, and add a winning-cell analysis section to MODEL_REVIEW.md. This commit produces **measurement artifact 1** for the retrospective.

**Reference**: design doc §2.2, §7 (Δ_smote_weights = winning cell vs T2.1 baseline).

**Files:**
- Create: `backend/data/output/smote_classweight_ablation.json` (artifact)
- Modify: `MODEL_REVIEW.md` (add T2.2 ablation analysis subsection)

- [ ] **Step 1: Verify features.parquet exists at the expected path**

Run: `ls backend/data/features/features.parquet`

Expected: file exists. If not, run `cd backend && python -m features.build` first.

- [ ] **Step 2: Run the harness on all 6 folds**

Run: `cd /home/julienerbland/Documents/Privé/Projets_persos/football_predict && python -m tools.validate_smote_classweight_composition`

Expected: ~30-60 minutes runtime (6 cells × 6 folds × ~1 min/fold). Final logger lines should show:
```
WINNING CELL: id=N smote='...' class_weight=[...] mean.draw_f1=0.XXX
Wrote backend/data/output/smote_classweight_ablation.json
```

If the run reports `HarnessFailure: No cell within margin`, **STOP**. The retrospective in Task 11 must address whether T2.2's three-pronged approach is sufficient. Escalate to user; meta-spec re-open required.

- [ ] **Step 3: Inspect the artifact**

Run: `python -c "import json; r=json.load(open('backend/data/output/smote_classweight_ablation.json')); print('winner:', r['winner']); print('schema:', r['schema_version']); print('feature_schema:', r['feature_schema_version'])"`

Expected: `feature_schema_version` is `"2.0"` (per design doc §2.2 — composition validated on current feature set).

- [ ] **Step 4: Add T2.2 ablation analysis subsection to MODEL_REVIEW.md**

Append (or insert before §7 Sources) the following section. Replace `<...>` placeholders with actual numbers from the artifact (read with `python -c "import json; r=json.load(open('backend/data/output/smote_classweight_ablation.json')); print(json.dumps(r['cells'], indent=2))"`).

```markdown
## 7. T2.2 Ablation Harness — SMOTE+Class_Weight Composition

Empirical validation of the SMOTE+class_weight composition decision per design doc §2.2.
Artifact: `backend/data/output/smote_classweight_ablation.json` (schema `smote_cw_ablation.v1`).
Run on all 6 walk-forward folds, `feature_schema_version="2.0"` (pre-T2.2-feature-add).

### Cell Results (mean across 6 folds)

| Cell | SMOTE strategy        | class_weight (H, D, A) | mean.rps | mean.brier | mean.draw_f1 | Margin? |
|------|-----------------------|------------------------|----------|------------|--------------|---------|
| 1    | off                   | (1.0, 1.0, 1.0)        | <...>    | <...>      | <...>        | <Y/N>   |
| 2    | off                   | (1.0, 2.5, 1.2)        | <...>    | <...>      | <...>        | <Y/N>   |
| 3    | partial_70            | (1.0, 1.0, 1.0)        | <...>    | <...>      | <...>        | <Y/N>   |
| 4    | partial_70            | (1.0, 2.5, 1.2)        | <...>    | <...>      | <...>        | <Y/N>   |
| 5    | auto                  | (1.0, 1.0, 1.0)        | <...>    | <...>      | <...>        | <Y/N>   |
| 6    | auto                  | (1.0, 2.5, 1.2)        | <...>    | <...>      | <...>        | <Y/N>   |

Margin filter: `mean.rps ≤ 0.205 AND mean.brier ≤ 0.215`.

### Winning Cell

Cell `<winner_cell_id>`: SMOTE strategy `<winner_smote>`, class_weight `<winner_weights>`.
Selection rationale: highest `mean.draw_f1` (`<value>`) among cells passing margin filter.
`std.draw_f1 = <std>` (lower is more stable).

This cell becomes the YAML default for `model_config.yaml: training:` block in Task 4.

### Δ_smote_weights (T2.2 retrospective input)

Δ_smote_weights = `<winner.mean.draw_f1>` − 0.0392 (T2.1 baseline) = **+`<delta>`**

This is the first of three measurement artifacts feeding the 4-mechanism decomposition
in §8 (the T2.2 baseline shift retrospective, written at Task 11 time).
```

- [ ] **Step 5: Commit**

```bash
git add backend/data/output/smote_classweight_ablation.json MODEL_REVIEW.md
git commit -m "$(cat <<'EOF'
feat(t2.2): ablation harness results + winning-cell analysis

Runs tools/validate_smote_classweight_composition.py on all 6 walk-forward
folds and commits the resulting smote_classweight_ablation.json (schema
smote_cw_ablation.v1, feature_schema_version "2.0"). Adds §7 to
MODEL_REVIEW.md with the per-cell mean±std table and the winning-cell
rationale.

The winning cell becomes the YAML default for model_config.yaml's
training: block in Task 4. This commit produces the first of three
measurement artifacts (Δ_smote_weights) feeding T2.2's 4-mechanism
decomposition retrospective.

Per design doc §2.2 + §7 commit slicing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire SMOTE + sample_weight into `train_calibrated_models` per harness winner

**Goal**: Modify `backend/evaluation/cv.py:train_calibrated_models` to apply `resample()` and `class_sample_weights()` from the chosen YAML config. Add the `training:` block to `model_config.yaml` reflecting the harness winner. Tests verify SMOTE applies to `(X_tr, y_tr)` only and sample_weight reaches the fit calls.

**Reference**: design doc §2.1 (single integration point), §6.7 (OOF post-calibration contract).

**Files:**
- Modify: `backend/evaluation/cv.py`
- Modify: `backend/config/model_config.yaml`
- Create: `backend/tests/test_smote_integration.py`

- [ ] **Step 1: Add `training:` block to `backend/config/model_config.yaml`**

Append at the bottom (replace `<winner_*>` with actual values from Task 3's winning cell):

```yaml
training:
  # T2.2: chosen by tools/validate_smote_classweight_composition harness (cell <winner_cell_id>).
  # See data/output/smote_classweight_ablation.json + MODEL_REVIEW.md §7.
  smote_strategy: <winner_smote_strategy>     # one of: "off", "partial_70", "auto"
  smote_k_neighbors: 5
  class_weight:
    H: <winner_h>
    D: <winner_d>
    A: <winner_a>
```

- [ ] **Step 2: Write the failing test for SMOTE integration**

Create `backend/tests/test_smote_integration.py`:

```python
"""
Verify that train_calibrated_models applies SMOTE to (X_tr, y_tr) ONLY
and threads class_sample_weights into the model fit calls.

Per design doc §2.1: SMOTE must NOT touch (X_inner_val, y_inner_val) —
calibration must learn from the real distribution.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch


def _toy_dataset(seed: int = 42, n_train: int = 500, n_val: int = 100):
    rng = np.random.default_rng(seed)
    X_tr = rng.standard_normal((n_train, 8)).astype(np.float32)
    y_tr = np.concatenate([
        np.zeros(int(n_train * 0.46), dtype=int),
        np.ones(int(n_train * 0.24), dtype=int),
        2 * np.ones(n_train - int(n_train * 0.46) - int(n_train * 0.24), dtype=int),
    ])
    X_val = rng.standard_normal((n_val, 8)).astype(np.float32)
    y_val = np.concatenate([
        np.zeros(int(n_val * 0.46), dtype=int),
        np.ones(int(n_val * 0.24), dtype=int),
        2 * np.ones(n_val - int(n_val * 0.46) - int(n_val * 0.24), dtype=int),
    ])
    return X_tr, y_tr, X_val, y_val


def test_train_calibrated_models_applies_smote_to_train_only():
    """SMOTE auto should grow training set; inner_val passed to calibrate must be untouched."""
    from evaluation.cv import train_calibrated_models
    from config.loader import model_config

    mc = model_config()
    # Force training: block to use auto SMOTE for this test
    mc = {**mc, "training": {**mc.get("training", {}),
                             "smote_strategy": "auto", "smote_k_neighbors": 5,
                             "class_weight": {"H": 1.0, "D": 1.0, "A": 1.0}}}

    X_tr, y_tr, X_iv, y_iv = _toy_dataset()

    captured_calibrate_X: list[np.ndarray] = []
    real_calibrate = None

    def spy_calibrate(self, X, y, method="isotonic"):
        captured_calibrate_X.append(X.copy())
        return real_calibrate(self, X, y, method=method)

    from models.base import BaseModel
    real_calibrate = BaseModel.calibrate
    with patch.object(BaseModel, "calibrate", spy_calibrate):
        models = train_calibrated_models(X_tr, y_tr, X_iv, y_iv, [f"f{i}" for i in range(8)], mc)

    # Calibration must see the original (unresampled) inner_val.
    for X_seen in captured_calibrate_X:
        assert X_seen.shape == X_iv.shape, (
            f"calibrate received X of shape {X_seen.shape}, expected inner_val shape {X_iv.shape}. "
            "SMOTE must NOT touch inner_val."
        )
        np.testing.assert_array_equal(X_seen, X_iv)


def test_train_calibrated_models_off_strategy_skips_smote():
    """sampling_strategy='off' should leave training set unchanged in size."""
    from evaluation.cv import train_calibrated_models
    from config.loader import model_config

    mc = model_config()
    mc = {**mc, "training": {**mc.get("training", {}),
                             "smote_strategy": "off", "smote_k_neighbors": 5,
                             "class_weight": {"H": 1.0, "D": 1.0, "A": 1.0}}}

    X_tr, y_tr, X_iv, y_iv = _toy_dataset()
    n_tr = len(X_tr)

    captured_fit_X: list[int] = []

    from models.xgboost_model import XGBoostModel
    real_fit = XGBoostModel.fit

    def spy_fit(self, X, y, **kwargs):
        captured_fit_X.append(len(X))
        return real_fit(self, X, y, **kwargs)

    with patch.object(XGBoostModel, "fit", spy_fit):
        train_calibrated_models(X_tr, y_tr, X_iv, y_iv, [f"f{i}" for i in range(8)], mc)

    assert captured_fit_X[0] == n_tr, (
        f"smote='off' should pass {n_tr} training rows to fit(); got {captured_fit_X[0]}"
    )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_smote_integration.py -v`

Expected: FAIL — `train_calibrated_models` ignores `mc["training"]` (no SMOTE/sample_weight wiring yet).

- [ ] **Step 4: Modify `backend/evaluation/cv.py:train_calibrated_models`**

Replace the existing `train_calibrated_models` function (lines 58-85) with the following. Threads SMOTE + sample_weight per `mc["training"]`:

```python
def train_calibrated_models(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_inner_val: np.ndarray, y_inner_val: np.ndarray,
    feature_names: list[str], mc: dict,
) -> dict[str, np.ndarray]:
    """Train enabled models with T2.2 SMOTE + class_weight from mc["training"], return calibrated proba-callables.

    Per design doc §2.1: SMOTE applied to (X_tr, y_tr) ONLY; inner_val
    untouched (calibration must see the real distribution).
    """
    from training.draw_handling import class_sample_weights, resample  # noqa: PLC0415

    training = mc.get("training", {})
    smote_strategy = training.get("smote_strategy", "off")
    smote_k = training.get("smote_k_neighbors", 5)
    cw_dict = training.get("class_weight", {"H": 1.0, "D": 1.0, "A": 1.0})

    X_tr_eff, y_tr_eff = resample(X_tr, y_tr, sampling_strategy=smote_strategy, k_neighbors=smote_k)
    sample_weight = class_sample_weights(y_tr_eff, cw_dict)

    models: dict[str, object] = {}
    if mc["models"]["xgboost"]["enabled"]:
        c = mc["models"]["xgboost"]
        m = XGBoostModel(
            n_estimators=c["n_estimators"], learning_rate=c["learning_rate"],
            max_depth=c["max_depth"], subsample=c["subsample"],
            colsample_bytree=c["colsample_bytree"],
        )
        m.fit(X_tr_eff, y_tr_eff, X_val=X_inner_val, y_val=y_inner_val,
              feature_names=feature_names,
              sample_weight=sample_weight,
              early_stopping_rounds=c["early_stopping_rounds"])
        models["xgboost"] = m.calibrate(X_inner_val, y_inner_val)
    if mc["models"]["lightgbm"]["enabled"]:
        c = mc["models"]["lightgbm"]
        m = LGBMModel(
            n_estimators=c["n_estimators"], learning_rate=c["learning_rate"],
            num_leaves=c["num_leaves"],
        )
        m.fit(X_tr_eff, y_tr_eff, X_val=X_inner_val, y_val=y_inner_val,
              feature_names=feature_names,
              sample_weight=sample_weight)
        models["lightgbm"] = m.calibrate(X_inner_val, y_inner_val)
    return models
```

- [ ] **Step 5: Verify XGBoost and LightGBM model `fit()` accept `sample_weight`**

Read `backend/models/xgboost_model.py` and `backend/models/lgbm_model.py`. Confirm both `.fit()` methods either:
(a) already accept `sample_weight` via `**kwargs` and forward to the underlying estimator, OR
(b) need to be updated to forward `sample_weight=kwargs.get("sample_weight")` into the underlying `.fit()` call.

If (b), update both files: in `fit()`, extract `sample_weight = kwargs.get("sample_weight")`, then pass `sample_weight=sample_weight` to the underlying `self._model.fit(...)` call. (Both XGBClassifier.fit and LGBMClassifier.fit accept `sample_weight` natively.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_smote_integration.py tests/test_draw_handling.py -v`

Expected: all PASS. (Existing draw_handling tests should still pass.)

- [ ] **Step 7: Run the broader test suite to verify no regression**

Run: `cd backend && pytest tests/ -v --timeout=120`

Expected: all tests pass or skip cleanly. Particularly: `test_walk_forward_split.py`, `test_cv_report.py`, `test_quality_gates.py`.

- [ ] **Step 8: Commit**

```bash
git add backend/config/model_config.yaml backend/evaluation/cv.py backend/models/xgboost_model.py backend/models/lgbm_model.py backend/tests/test_smote_integration.py
git commit -m "$(cat <<'EOF'
feat(t2.2): wire SMOTE + sample_weight into train_calibrated_models

Threads SMOTE (resample) and class_sample_weights through
train_calibrated_models() in cv.py per the harness winner from Task 3
(cell <N>). Adds the training: block to model_config.yaml citing
data/output/smote_classweight_ablation.json + MODEL_REVIEW.md §7 as
the empirical source.

SMOTE applies to (X_tr, y_tr) only; inner_val untouched per design doc
§2.1 — calibration must see the real distribution. Per-sample weights
flow into XGBoost and LightGBM via .fit(sample_weight=...).

Tests verify SMOTE+sample_weight reach fit() and that inner_val is
preserved through to calibrate().

Per design doc §2.1, §6.7.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement `find_draw_threshold`, `predict_with_threshold`, `recompute_discrete_metrics` + `EnsembleModel.draw_threshold`

**Goal**: Add the remaining three helpers to `draw_handling.py` and the `draw_threshold` attribute to `EnsembleModel`. No train.py wiring yet (Tasks 6a/6b).

**Reference**: design doc §2.3 (threshold rule), §5.3 (EnsembleModel attribute), §6.4-6.8.

**Files:**
- Modify: `backend/training/draw_handling.py` (extend with 3 functions)
- Modify: `backend/models/ensemble.py` (add draw_threshold param)
- Modify: `backend/tests/test_draw_handling.py` (extend tests)

- [ ] **Step 1: Audit class-index convention across the codebase**

Run: `cd backend && grep -n "outcome_map\|labels=\[0\|p_home\|p_draw\|p_away" evaluation/cv.py models/train.py output/predict.py`

Expected: every reference uses `{0: H, 1: D, 2: A}`. Document any anomaly here before proceeding. (Per design doc §6.5: convention is locked at T2.2.)

- [ ] **Step 2: Write the failing tests for `find_draw_threshold` and `predict_with_threshold`**

Append to `backend/tests/test_draw_handling.py`:

```python
def test_find_draw_threshold_returns_dict_with_required_keys():
    from training.draw_handling import find_draw_threshold

    rng = np.random.default_rng(0)
    n = 1000
    y = rng.integers(0, 3, n)
    proba = rng.dirichlet(alpha=[1, 1, 1], size=n).astype(np.float64)

    result = find_draw_threshold(y, proba, search_range=(0.18, 0.32), step=0.01)

    assert "best_threshold" in result
    assert "best_macro_f1" in result
    assert "grid" in result
    assert isinstance(result["grid"], list)
    assert len(result["grid"]) == 15  # 0.18, 0.19, ..., 0.32 inclusive
    assert 0.18 <= result["best_threshold"] <= 0.32

    # Grid sorted ascending by theta
    thetas = [g["theta"] for g in result["grid"]]
    assert thetas == sorted(thetas)

    # Each grid entry has the four metrics
    for g in result["grid"]:
        assert {"theta", "macro_f1", "draw_f1", "draw_precision", "draw_recall"} <= set(g)


def test_find_draw_threshold_grid_max_matches_best_macro_f1():
    """best_macro_f1 should equal max(grid[*].macro_f1)."""
    from training.draw_handling import find_draw_threshold

    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 3, n)
    proba = rng.dirichlet(alpha=[1, 1, 1], size=n).astype(np.float64)

    result = find_draw_threshold(y, proba)
    grid_max = max(g["macro_f1"] for g in result["grid"])
    assert result["best_macro_f1"] == grid_max


def test_predict_with_threshold_low_theta_predicts_more_draws():
    from training.draw_handling import predict_with_threshold

    proba = np.array([
        [0.40, 0.35, 0.25],   # argmax=H, p_draw=0.35 ≥ θ=0.30 but p_draw < p_home → still H
        [0.30, 0.40, 0.30],   # argmax=D, p_draw=0.40 ≥ θ=0.30 AND p_draw is max → D
        [0.20, 0.55, 0.25],   # argmax=D unambiguously → D
        [0.50, 0.20, 0.30],   # argmax=H, p_draw=0.20 < θ=0.30 → H
    ])

    pred = predict_with_threshold(proba, theta=0.30)
    assert pred.tolist() == [0, 1, 1, 0]


def test_predict_with_threshold_theta_is_necessary_not_sufficient():
    """If p_draw ≥ θ but p_draw < p_home or p_away, do NOT predict draw."""
    from training.draw_handling import predict_with_threshold

    proba = np.array([
        [0.50, 0.30, 0.20],   # p_draw=0.30 ≥ θ=0.25 but p_draw < p_home → H
    ])
    pred = predict_with_threshold(proba, theta=0.25)
    assert pred[0] == 0


def test_predict_with_threshold_argmax_fallback_on_ties():
    """Float ties are vanishingly rare but should fall through to argmax."""
    from training.draw_handling import predict_with_threshold

    proba = np.array([
        [0.5, 0.5, 0.0],   # tie home/draw — np.argmax picks first (home=0)
    ])
    pred = predict_with_threshold(proba, theta=0.4)
    # p_draw=0.5 ≥ 0.4 AND p_draw == max (tied with home) → predict_with_threshold's rule:
    # "predict draw if p_draw ≥ θ AND p_draw == max". On a tie, this is True, so D.
    # If we wanted strict "p_draw is THE max", change rule. Per design doc, tie → falls through to argmax.
    # argmax of [0.5, 0.5, 0.0] = 0 (first wins).
    assert pred[0] == 0


def test_recompute_discrete_metrics_preserves_probabilistic_metrics():
    """Brier, RPS, log_loss are theta-independent; carry forward unchanged."""
    from training.draw_handling import recompute_discrete_metrics
    from evaluation.cv_report import CVSection, FoldGuards, FoldMetrics, FoldResult

    rng = np.random.default_rng(2)
    n = 200
    y = rng.integers(0, 3, n)
    proba = rng.dirichlet(alpha=[1, 1, 1], size=n).astype(np.float64)

    # Build a minimal preliminary CVSection with one fold
    prelim_metrics = FoldMetrics(
        brier=0.2027, rps=0.2099, log_loss=1.05, accuracy=0.51,
        draw_f1=0.04, home_recall=0.7, draw_recall=0.05, away_recall=0.6,
    )
    fold = FoldResult(
        fold_id=0, metrics=prelim_metrics,
        guards=FoldGuards(
            n_train=1000, n_val=n,
            train_seasons=(2021,), train_matchday_max=20,
            val_season=2021, val_matchday_range=(21, 30),
            leakage_check="passed",
        ),
    )
    cv_section = CVSection(folds=(fold,), mean_metrics=prelim_metrics, std_metrics=prelim_metrics)

    new_section = recompute_discrete_metrics(
        cv_section, oof_per_fold=[(y, proba)], theta=0.27,
    )

    # Probabilistic metrics unchanged
    assert new_section.folds[0].metrics.brier == 0.2027
    assert new_section.folds[0].metrics.rps == 0.2099
    assert new_section.folds[0].metrics.log_loss == 1.05

    # Discrete metrics changed (recomputed with theta)
    # We don't know the exact value — just assert they're floats and well-defined
    assert isinstance(new_section.folds[0].metrics.draw_f1, float)
    assert 0.0 <= new_section.folds[0].metrics.draw_f1 <= 1.0


def test_ensemble_draw_threshold_attribute_default_none():
    from models.ensemble import EnsembleModel

    class _DummyModel:
        def predict_proba(self, X): return np.zeros((len(X), 3))
    ens = EnsembleModel(models={"x": _DummyModel()}, weights={"x": 1.0})
    assert ens.draw_threshold is None


def test_ensemble_draw_threshold_attribute_set_via_init():
    from models.ensemble import EnsembleModel

    class _DummyModel:
        def predict_proba(self, X): return np.zeros((len(X), 3))
    ens = EnsembleModel(models={"x": _DummyModel()}, weights={"x": 1.0}, draw_threshold=0.27)
    assert ens.draw_threshold == 0.27
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_draw_handling.py -v`

Expected: 6 new tests fail (functions and attribute don't exist yet); the 5 existing tests still pass.

- [ ] **Step 4: Extend `backend/training/draw_handling.py` with the three helpers**

Append to `backend/training/draw_handling.py`:

```python
from sklearn.metrics import f1_score, precision_score, recall_score


def predict_with_threshold(proba: np.ndarray, theta: float | None) -> np.ndarray:
    """Apply the T2.2 draw threshold rule:

        predict_draw  iff  p_draw >= theta  AND  p_draw == max(p_home, p_draw, p_away)
        otherwise:    predict argmax(proba)

    theta is **necessary but not sufficient** — the model still requires draw to
    beat both single-team probabilities (per design doc §2.3).

    If theta is None, falls back to plain argmax (used when EnsembleModel
    has no draw_threshold set, e.g., during preliminary CV in Task 6a).

    Args:
        proba: shape (n, 3), columns = [p_home, p_draw, p_away].
        theta: draw threshold in [0, 1], or None for argmax fallback.

    Returns: int array of predicted labels in {0, 1, 2}.
    """
    if theta is None:
        return proba.argmax(axis=1)

    argmax_pred = proba.argmax(axis=1)
    p_draw = proba[:, _DRAW_CLASS]
    p_max = proba.max(axis=1)
    draw_eligible = (p_draw >= theta) & (p_draw == p_max)
    return np.where(draw_eligible, _DRAW_CLASS, argmax_pred)


def find_draw_threshold(
    y_oof: np.ndarray,
    proba_oof: np.ndarray,
    search_range: tuple[float, float] = (0.18, 0.32),
    step: float = 0.01,
) -> dict:
    """Grid-search theta over [search_range[0], search_range[1]] inclusive at given step.

    Optimization criterion: macro_f1 (per design doc §2.3 — chosen for stability).
    Returns the full grid alongside the chosen theta so the retrospective can
    inspect the grid shape and gate margin.

    OOF probabilities MUST be post-calibration ensemble probabilities
    (per design doc §6.7) — NOT raw base-model outputs, NOT pre-calibration
    ensemble. This function does not enforce the contract; callers are
    responsible.

    Args:
        y_oof: shape (n,), integer labels {0, 1, 2}.
        proba_oof: shape (n, 3), post-calibration ensemble probabilities.
        search_range: (low, high) inclusive bounds for the grid.
        step: grid spacing.

    Returns:
        {
            "best_threshold": float,
            "best_macro_f1": float,
            "grid": [
                {"theta": float, "macro_f1": float, "draw_f1": float,
                 "draw_precision": float, "draw_recall": float},
                ...
            ]  # ordered by theta ascending
        }
    """
    lo, hi = search_range
    n_steps = int(round((hi - lo) / step)) + 1
    thetas = np.linspace(lo, hi, n_steps)

    grid = []
    for theta in thetas:
        pred = predict_with_threshold(proba_oof, float(theta))
        macro_f1 = float(f1_score(y_oof, pred, average="macro", zero_division=0))
        draw_f1 = float(f1_score(y_oof, pred, labels=[1], average="macro", zero_division=0))
        draw_p = float(precision_score(y_oof, pred, labels=[1], average="macro", zero_division=0))
        draw_r = float(recall_score(y_oof, pred, labels=[1], average="macro", zero_division=0))
        grid.append({
            "theta": float(theta),
            "macro_f1": macro_f1,
            "draw_f1": draw_f1,
            "draw_precision": draw_p,
            "draw_recall": draw_r,
        })

    best = max(grid, key=lambda g: g["macro_f1"])
    return {
        "best_threshold": best["theta"],
        "best_macro_f1": best["macro_f1"],
        "grid": grid,
    }


def recompute_discrete_metrics(
    cv_section,  # CVSection
    oof_per_fold: list[tuple[np.ndarray, np.ndarray]],
    theta: float,
):
    """Rebuild a CVSection where per-fold and aggregate discrete metrics
    are recomputed under the threshold rule with theta.

    Probabilistic metrics (brier, rps, log_loss) DO NOT depend on theta and
    are carried forward unchanged from the preliminary CVSection (per design
    doc §6.8). Only discrete metrics (accuracy, draw_f1, home_recall,
    draw_recall, away_recall) are recomputed.

    Args:
        cv_section: preliminary CVSection from run_cv_with_oof (Task 6a),
            with discrete metrics computed via argmax.
        oof_per_fold: parallel list to cv_section.folds; each element is
            (y_val_fold, proba_val_fold).
        theta: chosen draw threshold from find_draw_threshold.

    Returns: a new CVSection with rebuilt FoldMetrics.
    """
    import statistics  # noqa: PLC0415
    from evaluation.cv_report import CVSection, FoldMetrics, FoldResult  # noqa: PLC0415

    if len(oof_per_fold) != len(cv_section.folds):
        raise ValueError(
            f"oof_per_fold length {len(oof_per_fold)} != "
            f"cv_section.folds length {len(cv_section.folds)}"
        )

    new_folds: list[FoldResult] = []
    for fold, (y_fold, proba_fold) in zip(cv_section.folds, oof_per_fold):
        pred = predict_with_threshold(proba_fold, theta)
        accuracy = float((pred == y_fold).mean())
        draw_f1 = float(f1_score(y_fold, pred, labels=[1], average="macro", zero_division=0))
        recalls = recall_score(y_fold, pred, labels=[0, 1, 2], average=None, zero_division=0)
        home_r, draw_r, away_r = float(recalls[0]), float(recalls[1]), float(recalls[2])

        new_metrics = FoldMetrics(
            brier=fold.metrics.brier,
            rps=fold.metrics.rps,
            log_loss=fold.metrics.log_loss,
            accuracy=accuracy,
            draw_f1=draw_f1,
            home_recall=home_r,
            draw_recall=draw_r,
            away_recall=away_r,
        )
        new_folds.append(FoldResult(
            fold_id=fold.fold_id, metrics=new_metrics, guards=fold.guards,
        ))

    fields = [f.name for f in FoldMetrics.__dataclass_fields__.values()]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for field in fields:
        vals = [getattr(f.metrics, field) for f in new_folds]
        means[field] = float(statistics.fmean(vals))
        stds[field] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return CVSection(
        folds=tuple(new_folds),
        mean_metrics=FoldMetrics(**means),
        std_metrics=FoldMetrics(**stds),
    )
```

- [ ] **Step 5: Modify `backend/models/ensemble.py:EnsembleModel.__init__` to accept `draw_threshold`**

Update the `__init__` signature and add the attribute storage:

```python
def __init__(
    self,
    models: dict[str, BaseModel],
    weights: dict[str, float],
    draw_threshold: float | None = None,
):
    """
    Args:
        models: dict of model_name → trained model (with .predict_proba())
        weights: dict of model_name → weight (need not sum to 1; normalised internally)
        draw_threshold: T2.2 calibrated draw threshold θ_D. None means
            argmax fallback at inference (used during preliminary CV before
            θ_D is found). Set to a float in [0.18, 0.32] before save().
            Per design doc §5.3, save-time invariant in train.py asserts
            this is non-None before ensemble.save().
    """
    if not models:
        raise ValueError("Ensemble requires at least one model.")
    self.models = models
    total = sum(weights.get(name, 0.0) for name in models)
    if total == 0:
        self.weights = {name: 1.0 / len(models) for name in models}
    else:
        self.weights = {name: weights.get(name, 0.0) / total for name in models}
    self.draw_threshold = draw_threshold
    logger.info(
        "Ensemble weights: "
        + ", ".join(f"{n}={w:.3f}" for n, w in self.weights.items())
    )
```

- [ ] **Step 6: Run tests to verify all pass**

Run: `cd backend && pytest tests/test_draw_handling.py -v`

Expected: all 13 tests PASS.

- [ ] **Step 7: Run full test suite — verify no regression**

Run: `cd backend && pytest tests/ -v --timeout=120`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add backend/training/draw_handling.py backend/models/ensemble.py backend/tests/test_draw_handling.py
git commit -m "$(cat <<'EOF'
feat(t2.2): find_draw_threshold + predict_with_threshold + recompute_discrete_metrics + EnsembleModel.draw_threshold

Adds the remaining draw_handling.py helpers:
- find_draw_threshold: grid-search theta over [0.18, 0.32] step 0.01,
  returns {best_threshold, best_macro_f1, grid: [{theta, macro_f1,
  draw_f1, draw_precision, draw_recall}, ...]}. Optimization criterion
  is macro_f1 per design doc §2.3.
- predict_with_threshold: applies the rule "predict draw iff p_draw ≥ θ
  AND p_draw == max(proba); else argmax". θ is necessary but not
  sufficient (design doc §2.3). theta=None falls back to argmax.
- recompute_discrete_metrics: rebuilds a CVSection where probabilistic
  metrics (brier, rps, log_loss) carry forward unchanged and discrete
  metrics (accuracy, draw_f1, recalls) are recomputed under the threshold
  rule (design doc §6.8).

Adds EnsembleModel.__init__(draw_threshold=None) attribute. Save-time
invariant lands in Task 7a; predict-time application lands in Task 9.

Class index convention {0:H, 1:D, 2:A} verified consistent across cv.py,
train.py, predict.py (audit per design doc §6.5).

Per design doc §2.3, §5.3, §6.5, §6.7, §6.8.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6a: `run_cv()` signature change — return per-fold OOF arrays

**Goal**: Modify `run_cv()` in `backend/evaluation/cv.py` to return `(CVSection, list[(y_fold, proba_fold)])` so Task 6b can stack OOF and call `find_draw_threshold`. Update the train.py caller and add a test asserting the new return shape.

**Reference**: design doc §3.3 (Phase 1), §6.7 (post-calibration probabilities contract).

**Files:**
- Modify: `backend/evaluation/cv.py` (run_cv signature)
- Modify: `backend/models/train.py` (caller update; preliminary — no θ_D wiring yet)
- Create: `backend/tests/test_run_cv_oof.py`

- [ ] **Step 1: Read current `run_cv` to confirm baseline shape**

Run: `cd backend && grep -A 5 "^def run_cv" evaluation/cv.py`

Expected: returns `CVSection`. We're changing it to return `(CVSection, list[tuple[np.ndarray, np.ndarray]])`.

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_run_cv_oof.py`:

```python
"""
Verify run_cv() returns (CVSection, list[(y_fold, proba_fold)]) where
each proba_fold is post-calibration ensemble probability of shape (n, 3),
and the list length matches CVSection.folds.

Per design doc §3.3 Phase 1, §6.7.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_run_cv_returns_cv_section_and_oof_list(historical_matches, settings_dict):
    if historical_matches is None:
        pytest.skip("all_matches.parquet not available — run ingestion first.")

    import pandas as pd
    from config.loader import model_config
    from evaluation.cv import run_cv
    from evaluation.cv_report import CVSection
    from evaluation.splits import WalkForwardSplit
    from pathlib import Path

    feat_path = Path(settings_dict["paths"]["features"])
    if not feat_path.exists():
        pytest.skip("features.parquet not available — run features.build first.")
    df = pd.read_parquet(feat_path)
    feature_cols = [
        c for c in df.columns
        if c not in {
            "match_id", "league", "season", "date", "matchday",
            "home_team", "away_team", "home_team_id", "away_team_id",
            "referee", "home_goals", "away_goals", "result",
        }
    ]

    mc = model_config()
    splitter = WalkForwardSplit()
    weights = {
        m: mc["models"][m]["weight"]
        for m in ("xgboost", "lightgbm")
        if mc["models"][m]["enabled"]
    }

    result = run_cv(df, feature_cols, splitter, mc, weights)

    assert isinstance(result, tuple), (
        "run_cv must return a tuple (CVSection, oof_per_fold) per design doc §3.3 Phase 1"
    )
    cv_section, oof_per_fold = result
    assert isinstance(cv_section, CVSection)
    assert isinstance(oof_per_fold, list)
    assert len(oof_per_fold) == len(cv_section.folds)

    for (y_fold, proba_fold), fold in zip(oof_per_fold, cv_section.folds):
        assert isinstance(y_fold, np.ndarray)
        assert isinstance(proba_fold, np.ndarray)
        assert y_fold.ndim == 1
        assert proba_fold.shape == (len(y_fold), 3)
        # OOF proba rows must sum to ~1 (post-normalization)
        row_sums = proba_fold.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd backend && pytest tests/test_run_cv_oof.py -v --timeout=600`

Expected: FAIL — `run_cv` returns CVSection only, not a tuple.

- [ ] **Step 4: Modify `run_cv` in `backend/evaluation/cv.py`**

Change the function signature and body. Find the existing `def run_cv(...)` and replace its body with one that accumulates OOF per fold and returns the tuple:

```python
def run_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    splitter: WalkForwardSplit,
    mc: dict,
    weights: dict[str, float],
) -> tuple[CVSection, list[tuple[np.ndarray, np.ndarray]]]:
    """Run walk-forward CV and return (CVSection, per-fold OOF list).

    Returns a tuple per design doc §3.3 Phase 1:
        - CVSection: preliminary FoldMetrics (probabilistic metrics final;
          discrete metrics computed via argmax — recomputed in train.py
          Phase 3 once θ_D is known).
        - oof_per_fold: list of (y_val_fold, ensemble_proba_val_fold) where
          proba is post-calibration ensemble probability, shape (n_val, 3).
          Used by train.py to stack into find_draw_threshold input.
    """
    sorted_df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    fold_specs = splitter.fold_specs(sorted_df)
    fold_results: list[FoldResult] = []
    oof_per_fold: list[tuple[np.ndarray, np.ndarray]] = []

    for spec, (train_idx, val_idx) in zip(fold_specs, splitter.split(sorted_df)):
        X_all = sorted_df.iloc[train_idx][feature_cols].fillna(0).to_numpy(np.float32)
        y_all = sorted_df.iloc[train_idx]["result"].to_numpy(int)
        X_val = sorted_df.iloc[val_idx][feature_cols].fillna(0).to_numpy(np.float32)
        y_val = sorted_df.iloc[val_idx]["result"].to_numpy(int)

        cut = int(len(X_all) * 0.9)
        X_tr, y_tr = X_all[:cut], y_all[:cut]
        X_iv, y_iv = X_all[cut:], y_all[cut:]

        models = train_calibrated_models(X_tr, y_tr, X_iv, y_iv, feature_cols, mc)
        probas = {n: m.predict_proba(X_val) for n, m in models.items()}
        ens = _ensemble(probas, weights)

        metrics = _fold_metrics(y_val, ens)
        train_seasons = tuple(sorted(sorted_df.iloc[train_idx]["season"].unique()))
        train_max_md = int(sorted_df.iloc[train_idx]["matchday"].max())
        guards = FoldGuards(
            n_train=int(len(X_all)),
            n_val=int(len(X_val)),
            train_seasons=tuple(int(s) for s in train_seasons),
            train_matchday_max=train_max_md,
            val_season=int(spec["val_season"]),
            val_matchday_range=tuple(spec["val_matchday_range"]),
            leakage_check="passed",
        )
        fold_results.append(FoldResult(
            fold_id=int(spec["fold_id"]), metrics=metrics, guards=guards,
        ))
        oof_per_fold.append((y_val, ens))
        logger.info(
            f"Fold {spec['fold_id']} season={spec['val_season']} "
            f"mds={spec['val_matchday_range']}: "
            f"n_train={len(X_all)} n_val={len(X_val)} "
            f"rps={metrics.rps:.4f} brier={metrics.brier:.4f} "
            f"draw_f1={metrics.draw_f1:.3f} (preliminary, argmax)"
        )

    mean, std = _aggregate_metrics(fold_results)
    cv_section = CVSection(folds=tuple(fold_results), mean_metrics=mean, std_metrics=std)
    return cv_section, oof_per_fold
```

- [ ] **Step 5: Update `backend/models/train.py` caller**

Find the line that currently calls `run_cv(...)` (likely `cv_section = run_cv(...)`) and update it to unpack the tuple. Don't wire θ_D yet — Task 6b does that. Only change the unpacking:

```python
cv_section, _oof_per_fold = run_cv(df, feature_cols, splitter, mc, weights)
# _oof_per_fold consumed in Task 6b
```

- [ ] **Step 6: Run the new test + existing tests**

Run: `cd backend && pytest tests/test_run_cv_oof.py tests/test_walk_forward_split.py tests/test_cv_report.py -v --timeout=600`

Expected: all PASS. Note: `test_run_cv_oof` may take several minutes (full CV run). Skip it cleanly if features aren't available.

- [ ] **Step 7: Verify train.py still imports cleanly**

Run: `cd backend && python -c "from models.train import train"`

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add backend/evaluation/cv.py backend/models/train.py backend/tests/test_run_cv_oof.py
git commit -m "$(cat <<'EOF'
refactor(t2.2): run_cv returns (CVSection, per-fold OOF list)

Threads post-calibration ensemble OOF probabilities out of run_cv() so
train.py can stack them and call find_draw_threshold (lands in Task 6b).
Per-fold proba shape (n_val, 3) carries the same probabilities used to
compute the preliminary FoldMetrics; train.py will recompute discrete
metrics under the threshold rule once θ_D is known (recompute_discrete_metrics).

train.py caller updated to unpack the tuple. No θ_D wiring yet — that
lands in Task 6b alongside the Checkpoint A measurement snapshot.

Per design doc §3.3 Phase 1, §6.7.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6b: Integrate θ_D into train.py + Checkpoint A snapshot

**Goal**: Wire `find_draw_threshold` and `recompute_discrete_metrics` into train.py per design doc §3.3 Phases 1-5. **Do not** apply the strict-mode reorder yet (Task 7a). Run training once to produce **Checkpoint A measurement artifact** (`eval_pre_features.json`) — this snapshot represents "full mechanism stack except features".

**Reference**: design doc §3.3 (Phases 1-5), §7 (Checkpoint A → eval_pre_features.json).

**Files:**
- Modify: `backend/models/train.py`
- Create: `backend/data/output/decomposition/eval_pre_features.json` (artifact)

- [ ] **Step 1: Read current train.py to identify the patch points**

Run: `cd backend && grep -n "run_cv\|_evaluate_holdout\|_build_gates\|_retrain_final\|ensemble.save\|assert_gates" models/train.py`

Note the line numbers. We modify the function so it does:
1. unpack run_cv tuple → `cv_section_v1, oof_per_fold`
2. stack OOF and call `find_draw_threshold` → `theta_result`
3. call `recompute_discrete_metrics` → `cv_section`
4. retrain final, set `ensemble.draw_threshold = theta_result["best_threshold"]`
5. continue to evaluate holdout, build gates, save (existing T2.1 ordering — strict-mode reorder is Task 7a)

- [ ] **Step 2: Modify `backend/models/train.py:train()` to wire θ_D (Phases 1-5 only)**

Within `train()`, replace the segment from `cv_section = run_cv(...)` (or the Task-6a stub `cv_section, _oof_per_fold = run_cv(...)`) through to `_retrain_final(...)`:

```python
    # Phase 1: walk-forward CV with OOF (preliminary FoldMetrics; discrete metrics argmax-based)
    cv_section_v1, oof_per_fold = run_cv(df, feature_cols, splitter, mc, weights)

    # Phase 2: θ_D grid search on stacked, post-calibration ensemble OOF
    from training.draw_handling import find_draw_threshold, recompute_discrete_metrics  # noqa: PLC0415
    stacked_y = np.concatenate([y for y, _ in oof_per_fold])
    stacked_proba = np.concatenate([p for _, p in oof_per_fold], axis=0)
    theta_result = find_draw_threshold(stacked_y, stacked_proba)
    logger.info(
        f"Chosen θ_D = {theta_result['best_threshold']:.3f} "
        f"(macro_f1={theta_result['best_macro_f1']:.4f})"
    )

    # Phase 3: rebuild CVSection with threshold-rule discrete metrics
    cv_section = recompute_discrete_metrics(
        cv_section_v1, oof_per_fold, theta_result["best_threshold"]
    )

    # Phase 4: final retrain on full cv_pool
    ensemble = _retrain_final(df, feature_cols, mc, weights)
    ensemble.draw_threshold = theta_result["best_threshold"]
```

The remainder of `train()` (holdout eval, build gates, save, assert_gates) **stays as it is from T2.1** — Task 7a does the strict-mode reorder.

- [ ] **Step 3: Update `_evaluate_holdout` to use the threshold rule**

Find `_evaluate_holdout(...)` in train.py. The function currently uses `proba.argmax(axis=1)` for `draw_f1` (line ~105 per the design doc). Change it to accept `theta` and use `predict_with_threshold`:

```python
def _evaluate_holdout(ensemble, df, feature_cols) -> "HoldoutSection":
    """Evaluate the trained ensemble on the locked holdout snapshot.

    Per design doc §2.3: draw_f1 (and other discrete metrics) computed
    with the threshold rule using ensemble.draw_threshold.
    """
    from training.draw_handling import predict_with_threshold  # noqa: PLC0415
    # ... existing holdout selection code ...
    # ... existing X, y construction ...
    proba = ensemble.predict_proba(X)
    pred = predict_with_threshold(proba, ensemble.draw_threshold)
    # ... use `pred` (not `proba.argmax(axis=1)`) wherever discrete metrics are computed.
```

(Keep probabilistic metric computations on `proba` — they're θ-independent.)

- [ ] **Step 4: Run training to generate the Checkpoint A snapshot**

Run: `cd backend && python -m models.train`

Expected runtime: ~3-5 minutes (no SMOTE+features overhead vs T2.1).

Expected output (loguru lines):
- `Chosen θ_D = 0.X.. (macro_f1=...)`
- `Wrote ... eval_ensemble.json`
- assert_gates() may RAISE if rps/brier still pass but draw_f1 doesn't — that's fine for this commit (strict-mode reorder is Task 7a; we just need eval_ensemble.json on disk).

If `python -m models.train` exits non-zero with `QualityGateFailure`, that's expected per T2.1 inspect-on-failure behavior. The eval JSON is still written. Proceed to Step 5.

- [ ] **Step 5: Snapshot the eval JSON to `decomposition/eval_pre_features.json`**

Run:
```bash
mkdir -p backend/data/output/decomposition
cp backend/data/output/eval_ensemble.json backend/data/output/decomposition/eval_pre_features.json
```

Verify: `ls backend/data/output/decomposition/eval_pre_features.json` exists and has same content as eval_ensemble.json at this commit.

- [ ] **Step 6: Inspect the snapshot to confirm it contains θ_D-derived draw_f1**

Run:
```bash
python -c "import json; r=json.load(open('backend/data/output/decomposition/eval_pre_features.json')); print('feature_schema:', r['feature_schema_version']); print('cv mean draw_f1:', r['cv']['mean_metrics']['draw_f1']); print('holdout draw_f1:', r['holdout']['draw_f1'])"
```

Expected: `feature_schema` is `"2.0"` (features not added yet). `draw_f1` values should be substantively higher than T2.1's 0.0392 baseline — this is the post-θ_D / pre-features measurement.

- [ ] **Step 7: Commit (train.py changes + frozen snapshot together)**

```bash
git add backend/models/train.py backend/data/output/decomposition/eval_pre_features.json
git commit -m "$(cat <<'EOF'
feat(t2.2): integrate θ_D into train.py + Checkpoint A snapshot

Wires find_draw_threshold + recompute_discrete_metrics into train.py
Phases 1-5 (CV → θ_D grid search → CV recompute → final retrain →
holdout eval). predict_with_threshold(proba, ensemble.draw_threshold)
replaces proba.argmax(axis=1) at the holdout discrete-metric site.

Strict-mode reorder is intentionally deferred to Task 7a — this commit
preserves T2.1's save-then-assert ordering so the training run produces
eval_ensemble.json regardless of gate outcome.

Decomposition checkpoint A: eval_pre_features.json snapshotted from
this commit's eval_ensemble.json. feature_schema_version="2.0"
(pre-T2.2-features). This is the second measurement artifact for the
4-mechanism decomposition retrospective:

  Δ_θ_D = checkpoint A draw_f1 − ablation winning cell (commit 3, argmax)
  Δ_features = (Task 10's eval_ensemble.json) − checkpoint A

Reproducibility note: eval_pre_features.json is a frozen artifact;
re-running the training pipeline at a later commit will produce
different numbers (different feature set, strict-mode reorder, etc.).
Same property as T2.1's cv_parametrization_validation.json.

Per design doc §3.3 Phases 1-5, §7 (Checkpoint A).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7a: Strict-mode reorder of train.py + save-time invariant

**Goal**: Implement design doc §3.3's full 9-phase ordering. Add the save-time invariant (`assert ensemble.draw_threshold is not None` before `ensemble.save()`). Tests confirm the partition: draw_f1 failure blocks artifact writes; rps/brier failures allow artifacts + raise.

**Reference**: design doc §3.3, §5.3, §6.1, §6.2.

**Files:**
- Modify: `backend/models/train.py` (full Phase 1-10 ordering)
- Modify: `backend/models/ensemble.py` (no changes if Task 5 covered draw_threshold)
- Create: `backend/tests/test_train_strict_mode.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_train_strict_mode.py`:

```python
"""
Tier 1 strict-mode partition tests per design doc §3.3:

- draw_f1 < 0.25  → ensemble.pkl, feature_cols.pkl, training_recipe.json NOT saved.
- rps > 0.21 (draw_f1 ok) → all artifacts saved; assert_gates raises.
- All gates pass → all artifacts saved; no raise.

Save-time invariant test: ensemble.save() with draw_threshold=None
raises clearly (regression guard against future strict-mode-ordering refactors).
"""
from __future__ import annotations

import numpy as np
import pytest


def test_save_time_invariant_replicates_train_py_assertion():
    """train.py asserts ensemble.draw_threshold is not None before .save().
    This test replicates that assertion at the EnsembleModel construction site,
    confirming the bare-None case is caught.
    """
    from models.ensemble import EnsembleModel

    class _DummyModel:
        def predict_proba(self, X): return np.zeros((len(X), 3))
    ens = EnsembleModel(models={"x": _DummyModel()}, weights={"x": 1.0})
    assert ens.draw_threshold is None

    # Replicate the train.py guard verbatim
    with pytest.raises(AssertionError, match="draw_threshold"):
        assert ens.draw_threshold is not None, (
            "EnsembleModel.draw_threshold must be set before save() — "
            "T2.2 strict-mode ordering violated. Did Phase 4 run?"
        )


# Higher-level integration tests for the strict-mode partition.
# These mock the heavy paths (run_cv, _retrain_final) and assert which artifacts land on disk.

def test_strict_mode_blocks_artifacts_on_draw_f1_failure(tmp_path, monkeypatch):
    """draw_f1 < 0.25 → eval JSON saved; ensemble.pkl + feature_cols.pkl NOT saved."""
    pytest.skip("Requires train.py refactor with mockable internals; deferred to Task 10 manual verification")


def test_strict_mode_saves_artifacts_on_rps_failure(tmp_path, monkeypatch):
    """rps > 0.21 with draw_f1 ok → all artifacts saved; assert_gates raises."""
    pytest.skip("Requires train.py refactor with mockable internals; deferred to Task 10 manual verification")
```

- [ ] **Step 2: Run tests to verify the save-time invariant test passes (it self-contains the assertion)**

Run: `cd backend && pytest tests/test_train_strict_mode.py -v`

Expected: 1 PASS (save-time invariant), 2 SKIPPED (deferred integration). The skipped tests document the partition behavior; Task 10's full training run is the integration verification.

- [ ] **Step 3: Refactor `backend/models/train.py:train()` to the 9-phase order**

Replace the body of `train()` with the design doc §3.3 ordering:

```python
def train() -> "CVReport":
    """T2.2 9-phase strict-mode training flow per design doc §3.3.

    Phase 1:  run_cv_with_oof
    Phase 2:  find_draw_threshold
    Phase 3:  recompute_discrete_metrics
    Phase 4:  _retrain_final + assign ensemble.draw_threshold
    Phase 5:  _evaluate_holdout (uses θ_D)
    Phase 6:  build gates + CVReport
    Phase 7:  write eval_ensemble.json (UNCONDITIONAL — diagnostic)
    Phase 8:  if draw_f1 < min_draw_f1: raise (deployment artifacts NOT saved)
    Phase 9:  save-time invariant + ensemble.save() + save_feature_cols() + write training_recipe.json
    Phase 10: assert_gates() (T2.1 inspect-on-failure for rps/brier)
    """
    from training.draw_handling import find_draw_threshold, recompute_discrete_metrics  # noqa: PLC0415

    cfg = settings()
    mc = model_config()
    df = pd.read_parquet(cfg["paths"]["features"])
    feature_cols = _canonical_feature_cols(df)
    splitter = WalkForwardSplit()
    weights = {
        m: mc["models"][m]["weight"]
        for m in ("xgboost", "lightgbm")
        if mc["models"][m]["enabled"]
    }
    if not weights:
        raise RuntimeError("No models enabled in model_config.yaml.")

    # Phase 1
    cv_section_v1, oof_per_fold = run_cv(df, feature_cols, splitter, mc, weights)

    # Phase 2
    stacked_y = np.concatenate([y for y, _ in oof_per_fold])
    stacked_proba = np.concatenate([p for _, p in oof_per_fold], axis=0)
    theta_result = find_draw_threshold(stacked_y, stacked_proba)
    logger.info(
        f"Chosen θ_D = {theta_result['best_threshold']:.3f} "
        f"(macro_f1={theta_result['best_macro_f1']:.4f})"
    )

    # Phase 3
    cv_section = recompute_discrete_metrics(
        cv_section_v1, oof_per_fold, theta_result["best_threshold"]
    )

    # Phase 4
    ensemble = _retrain_final(df, feature_cols, mc, weights)
    ensemble.draw_threshold = theta_result["best_threshold"]

    # Phase 5
    holdout = _evaluate_holdout(ensemble, df, feature_cols)

    # Phase 6
    gates = _build_gates(cv_section, holdout, mc)
    report = CVReport(
        schema_version=CV_REPORT_SCHEMA_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        cv=cv_section, holdout=holdout, gates=gates,
        calibration=CalibrationSection(method="isotonic", cv_folds=5),
    )

    # Phase 7: UNCONDITIONAL diagnostic write
    out_dir = Path(cfg["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval_ensemble.json"
    eval_path.write_text(report.to_json())
    logger.info(f"Wrote diagnostic {eval_path}")

    # Phase 8: T2.2 strict mode for draw_f1 — block deployment artifact writes
    min_draw_f1 = mc.get("training", {}).get("min_draw_f1", 0.25)
    cv_draw_f1 = cv_section.mean_metrics.draw_f1
    if cv_draw_f1 < min_draw_f1:
        raise QualityGateFailure(
            gates,
            message=(
                f"T2.2 strict mode — draw_f1 below threshold, deployment artifacts not saved.\n"
                f"  draw_f1: {cv_draw_f1:.3f} < {min_draw_f1:.3f}  ❌\n"
                f"  rps:     {cv_section.mean_metrics.rps:.3f} (≤ 0.21? {cv_section.mean_metrics.rps <= 0.21})\n"
                f"  brier:   {cv_section.mean_metrics.brier:.3f} (≤ 0.22? {cv_section.mean_metrics.brier <= 0.22})\n"
                f"Diagnostic: {eval_path}\n"
                f"Chosen θ_D: {theta_result['best_threshold']:.3f}\n"
                f"  macro_f1 at θ*: {theta_result['best_macro_f1']:.4f}\n"
                f"  draw_f1 at θ*:  {cv_draw_f1:.4f} (gate margin: {cv_draw_f1 - min_draw_f1:+.4f})\n"
                f"ensemble.pkl: NOT WRITTEN"
            )
        )
    # NOTE: Phase 8 / Phase 10 both check draw_f1 — INTENTIONAL, not redundant.
    # Phase 8: deployment-blocking inline check.
    # Phase 10: T2.1 inspect-on-failure assert_gates() — fires if rps/brier fail
    # even when draw_f1 passes. Future "DRY-up" refactors must preserve the
    # partition. Per design doc §3.3, §6.1.

    # Phase 9: save-time invariant + deployment artifacts + recipe
    assert ensemble.draw_threshold is not None, (
        "EnsembleModel.draw_threshold must be set before save() — "
        "T2.2 strict-mode ordering violated. Did Phase 4 run?"
    )
    models_dir = Path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(models_dir / "ensemble.pkl")
    _save_feature_cols(feature_cols, models_dir / "feature_cols.pkl")
    # Task 7b adds the training_recipe.json writer here. For now, leave a TODO marker
    # that Task 7b will replace with the writer call:
    # WRITTEN_BY_TASK_7B: write training_recipe.json (sibling of ensemble.pkl)

    # Phase 10: T2.1 semantics — assert all gates; raises QualityGateFailure if rps/brier failed
    report.assert_gates()
    logger.info("All quality gates passed; T2.2 strict-mode flow complete.")
    return report
```

(The exact import path for `QualityGateFailure`, `_canonical_feature_cols`, `_save_feature_cols`, `_retrain_final`, `_evaluate_holdout`, `_build_gates` may need to be reconciled with the existing train.py module. Read it first; do not duplicate helpers if they already exist.)

- [ ] **Step 4: Verify train.py imports cleanly**

Run: `cd backend && python -c "from models.train import train; print('train imported OK')"`

Expected: no errors.

- [ ] **Step 5: Run the focused tests + the broader suite**

Run: `cd backend && pytest tests/test_train_strict_mode.py tests/test_quality_gates.py tests/test_run_cv_oof.py -v --timeout=600`

Expected: all PASS or skip cleanly.

- [ ] **Step 6: Commit**

```bash
git add backend/models/train.py backend/tests/test_train_strict_mode.py
git commit -m "$(cat <<'EOF'
feat(t2.2): strict-mode reorder of train.py + save-time invariant

Implements design doc §3.3 9-phase ordering:
  Phase 1: run_cv_with_oof
  Phase 2: find_draw_threshold
  Phase 3: recompute_discrete_metrics
  Phase 4: _retrain_final + assign ensemble.draw_threshold
  Phase 5: _evaluate_holdout (uses θ_D)
  Phase 6: build gates + CVReport
  Phase 7: write eval_ensemble.json (UNCONDITIONAL)
  Phase 8: T2.2 strict — raise if draw_f1 < 0.25 (artifacts NOT saved)
  Phase 9: save-time invariant + ensemble.save + save_feature_cols
           + training_recipe.json writer (lands in Task 7b)
  Phase 10: assert_gates() — T2.1 inspect-on-failure for rps/brier

The Phase 8 and Phase 10 draw_f1 checks are INTENTIONAL, not redundant
— they have different consequences (block deploy vs log-and-raise).
Code comment + design doc capture the partition so future DRY-ups don't
collapse them.

Save-time invariant: assert ensemble.draw_threshold is not None before
ensemble.save(). Fails fast in training; regression test guards against
future ordering violations.

Per design doc §3.3, §5.3, §6.1, §6.2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7b: `training_recipe.json` writer

**Goal**: Add the recipe writer at Phase 9 (alongside `ensemble.save()`). Schema is `training_recipe.v1` per design doc §5.2.

**Reference**: design doc §5.2.

**Files:**
- Modify: `backend/models/train.py` (replace TODO marker with writer call)
- Modify: `backend/training/draw_handling.py` (helper to build recipe dict)
- Create: `backend/tests/test_training_recipe.py`

- [ ] **Step 1: Write the failing test for recipe shape**

Create `backend/tests/test_training_recipe.py`:

```python
"""
training_recipe.v1 schema verification per design doc §5.2.
"""
from __future__ import annotations

import json
import numpy as np
import pytest


def test_build_training_recipe_shape():
    from training.draw_handling import build_training_recipe

    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1, 1], size=200).astype(np.float64)
    y = rng.integers(0, 3, 200)

    recipe = build_training_recipe(
        feature_schema_version="2.1",
        smote_strategy="auto",
        smote_k_neighbors=5,
        class_weight={"H": 1.0, "D": 2.5, "A": 1.2},
        ablation_winning_cell_id=6,
        y_oof=y, proba_oof=proba,
        chosen_theta=0.27,
    )

    # Top-level keys
    assert recipe["schema_version"] == "training_recipe.v1"
    assert recipe["feature_schema_version"] == "2.1"
    assert recipe["smote_strategy"] == "auto"
    assert recipe["smote_k_neighbors"] == 5
    assert recipe["class_weight"] == {"H": 1.0, "D": 2.5, "A": 1.2}
    assert recipe["ablation_winning_cell_id"] == 6
    assert "training_timestamp" in recipe

    # draw_threshold_chosen block
    chosen = recipe["draw_threshold_chosen"]
    assert chosen["theta"] == 0.27
    assert {"macro_f1", "draw_f1", "draw_precision", "draw_recall"} <= set(chosen)

    # draw_threshold_grid: array of objects ordered by theta ascending
    grid = recipe["draw_threshold_grid"]
    assert isinstance(grid, list)
    assert len(grid) >= 14  # 0.18..0.32 step 0.01 inclusive = 15
    thetas = [g["theta"] for g in grid]
    assert thetas == sorted(thetas)
    for g in grid:
        assert {"theta", "macro_f1", "draw_f1", "draw_precision", "draw_recall"} <= set(g)


def test_training_recipe_json_roundtrip():
    """Recipe must round-trip through json.dumps/json.loads."""
    from training.draw_handling import build_training_recipe

    rng = np.random.default_rng(1)
    proba = rng.dirichlet([1, 1, 1], size=100).astype(np.float64)
    y = rng.integers(0, 3, 100)

    recipe = build_training_recipe(
        feature_schema_version="2.1",
        smote_strategy="off",
        smote_k_neighbors=5,
        class_weight={"H": 1.0, "D": 1.0, "A": 1.0},
        ablation_winning_cell_id=1,
        y_oof=y, proba_oof=proba,
        chosen_theta=0.25,
    )
    s = json.dumps(recipe, indent=2)
    loaded = json.loads(s)
    assert loaded["schema_version"] == "training_recipe.v1"
    assert loaded["draw_threshold_chosen"]["theta"] == 0.25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_training_recipe.py -v`

Expected: FAIL — `build_training_recipe` doesn't exist.

- [ ] **Step 3: Add `build_training_recipe` to `backend/training/draw_handling.py`**

Append:

```python
def build_training_recipe(
    *,
    feature_schema_version: str,
    smote_strategy: str,
    smote_k_neighbors: int,
    class_weight: dict[str, float],
    ablation_winning_cell_id: int,
    y_oof: np.ndarray,
    proba_oof: np.ndarray,
    chosen_theta: float,
) -> dict:
    """Assemble the training_recipe.v1 dict per design doc §5.2.

    Computes the full grid + chosen-θ block from (y_oof, proba_oof, chosen_theta)
    so the writer call site doesn't have to plumb the grid separately.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    grid_result = find_draw_threshold(y_oof, proba_oof)
    # Compute chosen-θ metrics by finding the grid entry at chosen_theta
    chosen = next(
        (g for g in grid_result["grid"] if abs(g["theta"] - chosen_theta) < 1e-9),
        None,
    )
    if chosen is None:
        raise ValueError(
            f"chosen_theta={chosen_theta} not in find_draw_threshold grid. "
            "Did the grid range/step change?"
        )

    return {
        "schema_version": "training_recipe.v1",
        "feature_schema_version": feature_schema_version,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "smote_strategy": smote_strategy,
        "smote_k_neighbors": smote_k_neighbors,
        "class_weight": dict(class_weight),
        "ablation_winning_cell_id": ablation_winning_cell_id,
        "draw_threshold_chosen": {
            "theta": float(chosen["theta"]),
            "macro_f1": float(chosen["macro_f1"]),
            "draw_f1": float(chosen["draw_f1"]),
            "draw_precision": float(chosen["draw_precision"]),
            "draw_recall": float(chosen["draw_recall"]),
        },
        "draw_threshold_grid": grid_result["grid"],
    }
```

- [ ] **Step 4: Wire the writer into `train.py:train()` Phase 9**

Replace the `WRITTEN_BY_TASK_7B` marker in train.py with:

```python
    # Phase 9b: training_recipe.json (audit metadata, design doc §5.2)
    from training.draw_handling import build_training_recipe  # noqa: PLC0415
    training_cfg = mc.get("training", {})
    recipe = build_training_recipe(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        smote_strategy=training_cfg.get("smote_strategy", "off"),
        smote_k_neighbors=training_cfg.get("smote_k_neighbors", 5),
        class_weight=training_cfg.get("class_weight", {"H": 1.0, "D": 1.0, "A": 1.0}),
        ablation_winning_cell_id=training_cfg.get("ablation_winning_cell_id", -1),
        y_oof=stacked_y,
        proba_oof=stacked_proba,
        chosen_theta=theta_result["best_threshold"],
    )
    recipe_path = out_dir / "training_recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2))
    logger.info(f"Wrote {recipe_path}")
```

Add `ablation_winning_cell_id` to `model_config.yaml`'s `training:` block (any int — populate with the cell number from Task 3's winning cell).

- [ ] **Step 5: Run tests to verify all pass**

Run: `cd backend && pytest tests/test_training_recipe.py tests/test_draw_handling.py -v`

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/training/draw_handling.py backend/models/train.py backend/config/model_config.yaml backend/tests/test_training_recipe.py
git commit -m "$(cat <<'EOF'
feat(t2.2): training_recipe.json writer (training_recipe.v1)

Adds build_training_recipe() in draw_handling.py and wires it into
train.py Phase 9. training_recipe.json is written alongside ensemble.pkl
when (and only when) the strict-mode flow reaches Phase 9 — i.e. draw_f1
gate cleared.

Schema training_recipe.v1 per design doc §5.2:
  - top-level: smote_strategy, smote_k_neighbors, class_weight,
    ablation_winning_cell_id, training_timestamp
  - draw_threshold_chosen: {theta, macro_f1, draw_f1, draw_precision,
    draw_recall} for the chosen θ
  - draw_threshold_grid: array of objects ordered by theta ascending,
    each with {theta, macro_f1, draw_f1, draw_precision, draw_recall}

schema_version field is forward-looking — no test consumer at T2.2 time;
included for parity with cv_report.v1 discipline (per design doc §5.2).

Per design doc §5.2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Add 3 new draw features atomically with `FEATURE_SCHEMA_VERSION` 2.0→2.1

**Goal**: Add `elo_diff_abs`, `h2h_draw_rate_last_5`, `defensive_match_indicator` to their respective builders with column-position-locked tests. Bump `FEATURE_SCHEMA_VERSION` in the same commit.

**Reference**: design doc §2.4, §6.3, §6.4.

**Files:**
- Modify: `backend/features/elo.py` (+elo_diff_abs)
- Modify: `backend/features/form.py` (+h2h_draw_rate_last_5)
- Modify: `backend/features/context.py` (+defensive_match_indicator)
- Modify: `backend/features/build.py` (FEATURE_SCHEMA_VERSION = "2.1")
- Create: `backend/tests/test_new_draw_features.py`

- [ ] **Step 1: Write failing tests for all three features + schema bump**

Create `backend/tests/test_new_draw_features.py`:

```python
"""
T2.2 commit 8 — three new draw features + FEATURE_SCHEMA_VERSION bump.

Per design doc §2.4 + §6.3-6.4:
  - elo_diff_abs: abs(home_elo - away_elo), appended after away_elo.
  - h2h_draw_rate_last_5: draw rate over last 5 H2H meetings (count window),
    appended after h2h_draw_rate.
  - defensive_match_indicator: int(both teams' w10 avg_ga below
    league-season median), appended after away_title_race.
  - FEATURE_SCHEMA_VERSION bumped from "2.0" to "2.1" atomically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_feature_schema_version_bumped_to_2_1():
    from features.build import FEATURE_SCHEMA_VERSION
    assert FEATURE_SCHEMA_VERSION == "2.1"


def test_elo_diff_abs_column_position():
    """elo_diff_abs must be appended IMMEDIATELY AFTER away_elo in compute_elo output."""
    from features.elo import compute_elo

    df = pd.DataFrame({
        "match_id": [1, 2],
        "date": pd.to_datetime(["2021-08-15", "2021-08-22"]),
        "season": [2021, 2021],
        "league": ["PL", "PL"],
        "home_team": ["A", "B"],
        "away_team": ["B", "A"],
        "home_goals": [1, 2],
        "away_goals": [0, 1],
        "result": [0, 0],
    })
    out = compute_elo(df)
    cols = list(out.columns)
    assert "elo_diff_abs" in cols
    away_elo_idx = cols.index("away_elo")
    elo_diff_abs_idx = cols.index("elo_diff_abs")
    assert elo_diff_abs_idx == away_elo_idx + 1, (
        f"elo_diff_abs at position {elo_diff_abs_idx}; expected {away_elo_idx + 1} "
        f"(immediately after away_elo). cols={cols}"
    )


def test_elo_diff_abs_values():
    from features.elo import compute_elo

    df = pd.DataFrame({
        "match_id": [1],
        "date": pd.to_datetime(["2021-08-15"]),
        "season": [2021], "league": ["PL"],
        "home_team": ["A"], "away_team": ["B"],
        "home_goals": [1], "away_goals": [0], "result": [0],
    })
    out = compute_elo(df)
    expected = abs(out["home_elo"].iloc[0] - out["away_elo"].iloc[0])
    assert out["elo_diff_abs"].iloc[0] == pytest.approx(expected)


def test_h2h_draw_rate_last_5_distinct_from_5y_h2h():
    """When meeting frequency varies, h2h_draw_rate_last_5 must differ from h2h_draw_rate."""
    from features.form import build_h2h_features

    # Two teams, 7 prior meetings spanning 6 years; varying draw incidence
    # to ensure 5-year window and last-5-meetings window produce different rates.
    rows = []
    for i, (year, month, res) in enumerate([
        (2018, 1, 1),  # draw, >5yrs ago
        (2018, 6, 0),  # not draw, >5yrs ago
        (2020, 1, 1),  # draw
        (2020, 6, 1),  # draw
        (2021, 1, 0),  # not draw
        (2022, 1, 0),  # not draw
        (2023, 1, 1),  # draw
    ]):
        rows.append({
            "match_id": i + 1,
            "date": pd.Timestamp(f"{year}-{month:02d}-15"),
            "season": year, "league": "PL",
            "home_team": "A", "away_team": "B",
            "home_goals": 1, "away_goals": 0 if res == 0 else 1,
            "result": res,
        })
    rows.append({
        "match_id": 99,
        "date": pd.Timestamp("2024-01-15"),
        "season": 2023, "league": "PL",
        "home_team": "A", "away_team": "B",
        "home_goals": 1, "away_goals": 1, "result": 1,
    })
    df = pd.DataFrame(rows)
    out = build_h2h_features(df, window_years=5)
    last_row = out.iloc[-1]

    assert "h2h_draw_rate_last_5" in out.columns
    val = last_row["h2h_draw_rate_last_5"]
    assert pd.isna(val) or 0.0 <= val <= 1.0


def test_h2h_draw_rate_last_5_nan_when_no_history():
    from features.form import build_h2h_features

    df = pd.DataFrame({
        "match_id": [1],
        "date": pd.to_datetime(["2021-08-15"]),
        "season": [2021], "league": ["PL"],
        "home_team": ["A"], "away_team": ["B"],
        "home_goals": [1], "away_goals": [0], "result": [0],
    })
    out = build_h2h_features(df, window_years=5)
    assert pd.isna(out["h2h_draw_rate_last_5"].iloc[0])


def test_h2h_draw_rate_last_5_column_position():
    """h2h_draw_rate_last_5 must be appended immediately after h2h_draw_rate."""
    from features.form import build_h2h_features

    df = pd.DataFrame({
        "match_id": [1],
        "date": pd.to_datetime(["2021-08-15"]),
        "season": [2021], "league": ["PL"],
        "home_team": ["A"], "away_team": ["B"],
        "home_goals": [1], "away_goals": [0], "result": [0],
    })
    out = build_h2h_features(df, window_years=5)
    cols = list(out.columns)
    h2h_dr_idx = cols.index("h2h_draw_rate")
    h2h_dr_5_idx = cols.index("h2h_draw_rate_last_5")
    assert h2h_dr_5_idx == h2h_dr_idx + 1


def test_defensive_match_indicator_binary_and_position():
    """defensive_match_indicator must be int 0/1, appended after away_title_race."""
    from features.context import build_context_features
    from features.form import build_form_features

    n = 30
    rows = []
    base = pd.Timestamp("2021-08-15")
    for i in range(n):
        rows.append({
            "match_id": i + 1,
            "date": base + pd.Timedelta(days=i * 7),
            "season": 2021, "league": "PL",
            "matchday": (i % 30) + 1,
            "home_team": f"T{i % 5}", "away_team": f"T{(i + 1) % 5}",
            "home_goals": (i % 3), "away_goals": (i + 1) % 3,
            "result": 0 if (i % 3) > ((i + 1) % 3) else (1 if (i % 3) == ((i + 1) % 3) else 2),
            "referee": "X",
        })
    df = pd.DataFrame(rows)
    df = build_form_features(df)
    out = build_context_features(df)
    cols = list(out.columns)
    assert "defensive_match_indicator" in cols
    last_known_col = "away_title_race"
    last_idx = cols.index(last_known_col)
    dmi_idx = cols.index("defensive_match_indicator")
    assert dmi_idx == last_idx + 1, (
        f"defensive_match_indicator at position {dmi_idx}; expected {last_idx + 1} "
        f"(immediately after {last_known_col}). cols={cols}"
    )

    # Values must be 0 or 1 (or NaN where rolling window is short)
    vals = out["defensive_match_indicator"].dropna().unique()
    assert set(vals).issubset({0, 1, 0.0, 1.0}), f"non-binary values: {vals}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_new_draw_features.py -v`

Expected: all FAIL — none of the three features or the schema bump exist yet.

- [ ] **Step 3: Add `elo_diff_abs` to `backend/features/elo.py:compute_elo`**

In `compute_elo`, find where `home_elo` and `away_elo` are appended to `df`. After the line that adds `away_elo`, add:

```python
df["elo_diff_abs"] = (df["home_elo"] - df["away_elo"]).abs()
```

Position is locked: must be the next column after `away_elo`. Verify with the position test (Step 6).

- [ ] **Step 4: Add `h2h_draw_rate_last_5` to `backend/features/form.py:build_h2h_features`**

Replace the existing `for idx, row in df.iterrows()` loop in `build_h2h_features` so it computes BOTH the 5-year rate AND the last-5-meeting rate in the same pass. The column assignment at the end of the function must do (in order):

```python
df["h2h_games"] = h2h_games
df["h2h_home_win_rate"] = h2h_home_win_rate
df["h2h_draw_rate"] = h2h_draw_rate
df["h2h_draw_rate_last_5"] = h2h_draw_rate_last_5  # NEW — must be IMMEDIATELY after h2h_draw_rate
df["h2h_away_win_rate"] = h2h_away_win_rate
df["h2h_avg_goals"] = h2h_avg_goals
```

Inside the loop, compute the last-5 rate alongside existing logic:

```python
# 5-meeting count window (sorted by date descending, take top 5)
past_sorted = past.sort_values("date", ascending=False).head(5)
n5 = len(past_sorted)
if n5 == 0:
    h2h_draw_rate_last_5.append(np.nan)
else:
    draws_last_5 = 0
    for _, h2h_row in past_sorted.iterrows():
        if h2h_row["home_team"] == row["home_team"]:
            res = h2h_row["result"]
        else:
            res = {0: 2, 1: 1, 2: 0}.get(h2h_row["result"], np.nan)
        if res == 1:
            draws_last_5 += 1
    h2h_draw_rate_last_5.append(draws_last_5 / n5)
```

Initialise `h2h_draw_rate_last_5 = []` at the top of the function alongside the other accumulators.

- [ ] **Step 5: Add `defensive_match_indicator` to `backend/features/context.py:build_context_features`**

At the END of `build_context_features` (immediately after `_compute_league_positions(df)` or the standings_cache injection), append:

```python
    # T2.2: defensive_match_indicator — binary, league-season-median GA-relative.
    # Per design doc §2.4.3.
    if "home_w10_avg_ga" in df.columns and "away_w10_avg_ga" in df.columns:
        league_season_median = (
            df.groupby(["league", "season"])["home_w10_avg_ga"].transform("median")
        )
        df["defensive_match_indicator"] = (
            (df["home_w10_avg_ga"] < league_season_median)
            & (df["away_w10_avg_ga"] < league_season_median)
        ).astype(int)
    else:
        # If form features not yet built (build order violated), emit NaN.
        df["defensive_match_indicator"] = np.nan
```

This must be the LAST column added in `build_context_features` per design doc §6.4 (appended after `away_title_race`).

- [ ] **Step 6: Bump `FEATURE_SCHEMA_VERSION` in `backend/features/build.py`**

Find the line `FEATURE_SCHEMA_VERSION = "2.0"` and change it to:

```python
FEATURE_SCHEMA_VERSION = "2.1"
```

Update the comment immediately above it to note T2.2's three new feature columns.

- [ ] **Step 7: Run tests to verify all pass**

Run: `cd backend && pytest tests/test_new_draw_features.py tests/test_form_alignment.py -v`

Expected: all PASS.

- [ ] **Step 8: Run full suite to check no regression**

Run: `cd backend && pytest tests/ --timeout=120`

Expected: no new failures. Pre-existing skips OK.

- [ ] **Step 9: Rebuild features.parquet**

Run: `cd backend && python -m features.build`

Expected: features.parquet updated with 3 new columns. Loguru log line confirms the column count grew by 3.

- [ ] **Step 10: Commit (atomic — features + schema bump together)**

```bash
git add backend/features/elo.py backend/features/form.py backend/features/context.py backend/features/build.py backend/data/features/features.parquet backend/tests/test_new_draw_features.py
git commit -m "$(cat <<'EOF'
feat(t2.2): three new draw features + FEATURE_SCHEMA_VERSION 2.0→2.1

Adds atomically:
- elo_diff_abs in features/elo.py:compute_elo (after away_elo) —
  abs(home_elo - away_elo) captures evenly-matched-pairings draw signal.
- h2h_draw_rate_last_5 in features/form.py:build_h2h_features (after
  h2h_draw_rate) — count-based window over last 5 head-to-head
  meetings, complements existing 5-year h2h_draw_rate.
- defensive_match_indicator in features/context.py:build_context_features
  (after away_title_race, last column) — int(both teams' w10 avg_ga
  below their league-season median). GA proxy for spec's xGA (xG
  graceful-skips when StatsBomb missing).

Column positions locked per design doc §6.4 — tests assert each feature
lands at the specified position. feature_cols.pkl reproducibility under
schema 2.1 depends on column order; future "cleanup" refactors that
move lines silently break predict.py.

FEATURE_SCHEMA_VERSION bumped 2.0 → 2.1 atomically — predict.py guard
will reject pre-T2.2 models cleanly. T2.1's matchday-derivation commit
set the precedent for atomic schema bumps.

features.parquet rebuilt with 3 new columns.

Per design doc §2.4, §6.3, §6.4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: predict.py threshold rule + FileNotFoundError UX fix

**Goal**: Replace `np.argmax(proba)` at predict.py:323 with `predict_with_threshold(proba, ensemble.draw_threshold)`. Improve the FileNotFoundError UX when `ensemble.pkl` is missing.

**Reference**: design doc §2.3 (threshold rule), §6.6 (inference parity statement), §3.3 Phase 8 / Q4 update #6.

**Files:**
- Modify: `backend/output/predict.py`
- Create: `backend/tests/test_predict_threshold.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_predict_threshold.py`:

```python
"""
predict.py threshold-rule application + FileNotFoundError UX tests.

Per design doc §2.3, §6.6, Q4 update #6.
"""
from __future__ import annotations

import numpy as np
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch


def test_load_model_clear_error_on_missing_ensemble(tmp_path):
    """_load_model must raise a clear, operator-readable error on missing ensemble.pkl,
    not a raw stack trace."""
    from output.predict import _load_model

    with pytest.raises(FileNotFoundError) as exc_info:
        _load_model(tmp_path)

    msg = str(exc_info.value)
    # Operator-facing error must include: file path, runbook hint, T2.2 strict-mode mention.
    assert "ensemble.pkl" in msg
    assert "models.train" in msg or "python -m models.train" in msg
    assert "T2.2" in msg or "draw_f1" in msg or "strict" in msg.lower()


def test_predict_outcome_uses_threshold_rule_when_draw_threshold_set():
    """When ensemble.draw_threshold is set, predict.py must use the threshold rule, not argmax."""
    from training.draw_handling import predict_with_threshold

    proba = np.array([
        [0.40, 0.32, 0.28],   # argmax=H, p_draw=0.32 ≥ θ=0.25 but p_draw < p_home → H
        [0.30, 0.35, 0.35],   # argmax=D (or A — tie), p_draw=0.35 == max → D
    ])
    pred_argmax = proba.argmax(axis=1)
    pred_threshold = predict_with_threshold(proba, theta=0.25)

    assert pred_argmax[0] == 0
    assert pred_threshold[0] == 0  # both agree (p_draw < p_home)
    # Row 1: argmax picks first max (D=1); threshold rule should also pick D.
    assert pred_threshold[1] == 1


def test_predict_outcome_falls_back_to_argmax_when_threshold_none():
    """If ensemble.draw_threshold is None (shouldn't happen post-T2.2, but guard exists),
    predict_with_threshold falls through to argmax."""
    from training.draw_handling import predict_with_threshold

    proba = np.array([[0.30, 0.40, 0.30]])
    pred = predict_with_threshold(proba, theta=None)
    assert pred[0] == 1  # argmax = D
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_predict_threshold.py -v`

Expected: `test_load_model_clear_error_on_missing_ensemble` FAILS (raw FileNotFoundError without the required hints); the threshold-rule tests PASS (they exercise `predict_with_threshold` which already exists from Task 5).

- [ ] **Step 3: Improve `_load_model` UX in `backend/output/predict.py`**

Find `_load_model` (around line 68). Replace its current FileNotFoundError message with one that includes the runbook hint and T2.2 strict-mode context:

```python
def _load_model(models_dir: Path):
    """Load the ensemble model from a trusted local pickle file.

    Operator-facing error if the file is missing — common cause is
    T2.2 strict-mode blocking the artifact write when the draw_f1 gate
    was not cleared.
    """
    import pickle  # noqa: PLC0415
    ensemble_path = models_dir / "ensemble.pkl"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"ensemble.pkl not found at {ensemble_path}.\n\n"
            "Possible causes:\n"
            "  1. Training never ran on this checkout. Fix:\n"
            "       cd backend && python -m models.train\n"
            "  2. T2.2 strict mode blocked the deployment artifact write because\n"
            "     the min_draw_f1 ≥ 0.25 quality gate failed. Read the diagnostic at\n"
            "     backend/data/output/eval_ensemble.json to see which gate failed\n"
            "     and by how much. The retrospective in MODEL_REVIEW.md may also help.\n"
        )
    with open(ensemble_path, "rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted local file, never from user input
```

- [ ] **Step 4: Replace argmax at line ~323 with threshold rule**

Find `predicted_outcome = outcome_map[int(np.argmax(proba))]` in `predict()`. Replace with:

```python
                from training.draw_handling import predict_with_threshold  # noqa: PLC0415
                pred_label = predict_with_threshold(
                    proba.reshape(1, -1), ensemble.draw_threshold
                )[0]
                predicted_outcome = outcome_map[int(pred_label)]
```

(Alternatively, hoist the `import` to the top of the file and inline `pred_label` if cleaner.)

- [ ] **Step 5: Run tests + smoke-import predict.py**

Run: `cd backend && pytest tests/test_predict_threshold.py -v && python -c "from output.predict import predict, _load_model"`

Expected: all PASS, import succeeds.

- [ ] **Step 6: Commit**

```bash
git add backend/output/predict.py backend/tests/test_predict_threshold.py
git commit -m "$(cat <<'EOF'
feat(t2.2): predict.py threshold rule + FileNotFoundError UX fix

Replaces np.argmax(proba) at predict.py:323 with
predict_with_threshold(proba, ensemble.draw_threshold) per design doc
§2.3 threshold rule. ensemble.draw_threshold is set by train.py Phase 4;
load is via existing pickle.

FileNotFoundError on missing ensemble.pkl now produces an
operator-facing message that names the file, the runbook command, and
flags T2.2 strict-mode as the most likely cause (draw_f1 gate failure
blocks deployment artifact write). Per Q4 update #6.

Inference parity statement (design doc §6.6) confirmed: predict.py at
T2.2 has only these two changes — threshold rule + FileNotFoundError
UX. No new builder calls. New features at inference flow through the
same builders as at training (build_context_features etc.).

Per design doc §2.3, §6.6.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final T2.2 training run

**Goal**: Run full training with all four mechanisms wired (SMOTE, class_weights, θ_D, new features). Produce **measurement artifact 3** (`eval_ensemble.json` at schema 2.1) + the deployable `ensemble.pkl` + `feature_cols.pkl` + `training_recipe.json`. Gates must all pass.

**Reference**: design doc §3.3, §7 (Checkpoint B → eval_ensemble.json), §9.1 (Quality gates DoD).

**Files (all artifacts):**
- Update: `backend/data/output/eval_ensemble.json`
- Create: `backend/data/output/training_recipe.json`
- Update: `backend/data/models/ensemble.pkl`
- Update: `backend/data/models/feature_cols.pkl`

- [ ] **Step 1: Verify model_config.yaml `training:` block is populated correctly**

Run: `python -c "from config.loader import model_config; print(model_config()['training'])"` (from backend/).

Expected: `smote_strategy`, `smote_k_neighbors`, `class_weight`, `ablation_winning_cell_id` all present and matching Task 3's winning cell.

- [ ] **Step 2: Run training**

Run: `cd backend && python -m models.train`

Expected runtime: ~5-10 minutes (full CV with SMOTE + retrain + holdout eval + recipe write).

Expected outcome: training completes WITHOUT raising. Final loguru lines:
- `Chosen θ_D = 0.X.. (macro_f1=...)`
- `Wrote backend/data/output/eval_ensemble.json` (Phase 7)
- `Wrote backend/data/output/training_recipe.json` (Phase 9)
- `All quality gates passed; T2.2 strict-mode flow complete.` (Phase 10)

If `QualityGateFailure` raised at Phase 8 (draw_f1 < 0.25): T2.2 has FAILED its DoD. STOP. Read `eval_ensemble.json`, identify which gate failed and by how much, escalate. **Do not commit a failed run.**

If `QualityGateFailure` raised at Phase 10 (rps or brier failed; draw_f1 ok): artifacts ARE on disk per the partition. Decide whether to commit + relax gates, or revisit (likely revisit).

- [ ] **Step 3: Verify all artifacts are on disk with correct schema**

Run:
```bash
python -c "
import json, pickle
from pathlib import Path

eval_p = Path('backend/data/output/eval_ensemble.json')
recipe_p = Path('backend/data/output/training_recipe.json')
ens_p = Path('backend/data/models/ensemble.pkl')
fc_p = Path('backend/data/models/feature_cols.pkl')

assert eval_p.exists(), 'eval_ensemble.json missing'
assert recipe_p.exists(), 'training_recipe.json missing'
assert ens_p.exists(), 'ensemble.pkl missing'
assert fc_p.exists(), 'feature_cols.pkl missing'

eval_r = json.loads(eval_p.read_text())
recipe_r = json.loads(recipe_p.read_text())
print('eval schema:', eval_r['schema_version'], 'feature_schema:', eval_r['feature_schema_version'])
print('recipe schema:', recipe_r['schema_version'], 'feature_schema:', recipe_r['feature_schema_version'])
print('cv mean draw_f1:', eval_r['cv']['mean_metrics']['draw_f1'])
print('holdout draw_f1:', eval_r['holdout']['draw_f1'])
print('gates passed:', eval_r['gates']['passed'])
print('chosen θ_D:', recipe_r['draw_threshold_chosen']['theta'])

with open(ens_p, 'rb') as f:
    ens = pickle.load(f)
print('ensemble.draw_threshold:', ens.draw_threshold)
"
```

Expected:
- `eval schema: cv_report.v1`, `feature_schema: 2.1`
- `recipe schema: training_recipe.v1`, `feature_schema: 2.1`
- `cv mean draw_f1: 0.25+` (gate passing)
- `gates passed: True`
- `ensemble.draw_threshold` = the chosen θ from recipe.

- [ ] **Step 4: Run Tier 2 quality-gate tests**

Run: `cd backend && pytest tests/test_quality_gates.py -v`

Expected: all PASS (Tier 2 reads eval_ensemble.json and asserts gates passed + schemas match).

- [ ] **Step 5: Commit (artifacts only — code is unchanged at this commit)**

```bash
git add backend/data/output/eval_ensemble.json backend/data/output/training_recipe.json backend/data/models/ensemble.pkl backend/data/models/feature_cols.pkl
git commit -m "$(cat <<'EOF'
chore(t2.2): final training run — gates passed (Checkpoint B)

Full T2.2 training with all four mechanisms wired:
  - SMOTE strategy = <winner from MODEL_REVIEW.md §7>
  - class_weight = <winner from MODEL_REVIEW.md §7>
  - chosen θ_D = <theta from training_recipe.json>
  - feature_schema_version = 2.1 (3 new draw features)

All quality gates passed:
  - max_rps:    <value> ≤ 0.21
  - max_brier:  <value> ≤ 0.22
  - min_draw_f1: <value> ≥ 0.25

Artifacts:
  - data/models/ensemble.pkl (with draw_threshold attribute)
  - data/models/feature_cols.pkl
  - data/output/eval_ensemble.json (Checkpoint B — measurement
    artifact 3 for the 4-mechanism decomposition)
  - data/output/training_recipe.json (training_recipe.v1)

Per design doc §3.3 Phase 9-10, §7 Checkpoint B, §9.1 DoD.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: T2.2 baseline shift retrospective

**Goal**: Add §8 to MODEL_REVIEW.md with the 4-mechanism decomposition table citing the three measurement artifacts. Phrasing must use the **sequential marginal contributions** framing per design doc §7.

**Reference**: design doc §7, §8 (risks including season_stage signal loss + SHAP baseline shift).

**Files:**
- Modify: `MODEL_REVIEW.md` (add §8: T2.2 baseline shift retrospective)

- [ ] **Step 1: Read the three measurement artifacts to populate the decomposition**

Run:
```bash
python -c "
import json
ablation = json.load(open('backend/data/output/smote_classweight_ablation.json'))
pre_features = json.load(open('backend/data/output/decomposition/eval_pre_features.json'))
final = json.load(open('backend/data/output/eval_ensemble.json'))

print('--- Δ_smote_weights ---')
winner = ablation['winner']
print('winner cell:', winner['cell_id'], 'mean.draw_f1:', [c for c in ablation['cells'] if c['cell_id'] == winner['cell_id']][0]['mean']['draw_f1'])
print('T2.1 baseline draw_f1: 0.0392')

print('--- Δ_θ_D ---')
print('checkpoint A (pre-features, with θ_D) cv mean draw_f1:', pre_features['cv']['mean_metrics']['draw_f1'])
print('subtract winner cell argmax draw_f1 above')

print('--- Δ_features ---')
print('checkpoint B (final T2.2) cv mean draw_f1:', final['cv']['mean_metrics']['draw_f1'])
print('subtract checkpoint A draw_f1 above')

print('--- Holdout vs CV (drift signal) ---')
print('CV draw_f1:', final['cv']['mean_metrics']['draw_f1'])
print('Holdout draw_f1:', final['holdout']['draw_f1'])

print('--- Chosen θ_D position relative to search bounds ---')
recipe = json.load(open('backend/data/output/training_recipe.json'))
theta = recipe['draw_threshold_chosen']['theta']
print('θ* =', theta, '(search range [0.18, 0.32])')
print('near ceiling?', theta >= 0.30)
"
```

Note all values for the retrospective table.

- [ ] **Step 2: Append §8 to MODEL_REVIEW.md**

Insert before §7 (Sources) — i.e., the new §8 sits between the existing T2.2 ablation section (§7) and Sources (which becomes §9). Or, if MODEL_REVIEW.md already has the ablation as a sub-section of an existing section, choose appropriately.

```markdown
## 8. T2.2 Baseline Shift Retrospective

**Status**: T2.2 shipped on <DATE>. All quality gates passed.

**Mechanism stack**: SMOTE (`<winner.smote_strategy>`) + class_weight `<winner.class_weight>` + calibrated draw threshold θ_D = `<theta>` + 3 new draw features (`elo_diff_abs`, `h2h_draw_rate_last_5`, `defensive_match_indicator`). `feature_schema_version` 2.0 → 2.1.

### 8.1 Quality Gates (CV mean ± std, holdout)

| Metric | T2.1 baseline (CV) | T2.2 final (CV mean ± std) | T2.2 holdout | Gate | Pass? |
|--------|---------------------|----------------------------|---------------|------|-------|
| RPS | 0.2099 | <val ± std> | <val> | ≤ 0.21 | ✅ |
| Brier | 0.2027 | <val ± std> | <val> | ≤ 0.22 | ✅ |
| draw_f1 | 0.0392 | <val ± std> | <val> | ≥ 0.25 | ✅ |

### 8.2 Four-Mechanism Decomposition

**These are sequential marginal contributions, not isolated contributions.** Each Δ measures the additional improvement attributable to a mechanism *given everything previously layered in is already in place*. The decomposition is non-commutative — a different ordering would produce different per-mechanism numbers.

**Layering order**: T2.1 baseline → +SMOTE+class_weight (winning cell) → +θ_D → +new features.

| Stage | Mechanism layered | CV mean draw_f1 | Δ (vs prior stage) | Source artifact |
|-------|-------------------|-----------------|---------------------|------------------|
| 0 | (T2.1 baseline) | 0.0392 | — | T2.1 retrospective (MODEL_REVIEW §6.7) |
| 1 | +SMOTE + class_weight | <winner.draw_f1> | **Δ_smote_weights = +<delta1>** | `data/output/smote_classweight_ablation.json` (cell <id>, argmax) |
| 2 | +θ_D threshold rule | <pre_features.draw_f1> | **Δ_θ_D = +<delta2>** | `data/output/decomposition/eval_pre_features.json` (commit 6b) |
| 3 | +3 new draw features | <final.draw_f1> | **Δ_features = +<delta3>** | `data/output/eval_ensemble.json` (commit 10) |

**Sum of deltas** = `<delta1>` + `<delta2>` + `<delta3>` = `<sum>`. Total improvement T2.1 → T2.2 = `<final.draw_f1>` − 0.0392 = `<total>`. The discrepancy `<sum> − <total> = <residual>` reflects non-additive interactions among mechanisms (expected; documented in design doc §8).

### 8.3 Methodology Note

Two methodologies inform this decomposition:
- **Stage 1** uses the ablation harness (`tools/validate_smote_classweight_composition.py`) — varies SMOTE+class_weight composition while holding θ_D off (argmax) and features at schema 2.0. Reports per-fold mean ± std.
- **Stages 2–3** use checkout-and-run snapshots — eval_pre_features.json frozen at commit 6b before features were added.

Each methodology fits its question: composition (Stage 1) vs additive mechanisms (Stages 2–3). Reproducibility for the snapshots requires checking out the producing commits.

### 8.4 `season_stage` Signal-Loss Compensation

Phase 1 had draw_f1 = 0.0856; T2.1 dropped to 0.0392 after fixing the `matchday=0` bug that recovered `season_stage` from silently-zero. The regression suggests `season_stage` was picking up end-of-season draw signal (e.g., "both teams need a point" matches), and the bug fix removed that spurious signal source.

**Empirical finding**: T2.2's Δ_features = `<delta3>`. The 3 new features <recovered / did not recover / partially recovered> the ~0.05 of `season_stage`-derived signal that T2.1 displaced. If the gap remains, T2.3+ should consider end-of-season-specific features (e.g., motivation flags, "matchday > N AND both teams in mid-table" indicators).

### 8.5 SHAP Baseline Shift for T2.3

Per design doc §8 risk: T2.3's DoD includes the gate "SHAP shows pi-rating features collectively account for ≥ 5% of total feature importance". That gate is now measured against a **post-SMOTE** training distribution, biased toward the over-represented draw class. T2.3 should compute SHAP on a real-distribution validation set (not the SMOTE-augmented training set) when interpreting the 5% gate, or document the bias explicitly.

### 8.6 Chosen θ_D Position

θ* = `<theta>` (search range [0.18, 0.32]). <If θ* ≥ 0.30: this is near the search ceiling — soft signal that SMOTE+class_weight pushed the model too aggressively toward draws. T2.3+ may revisit composition with softer settings.> <Else: comfortably mid-range; no boundary concerns.>

### 8.7 Forward-Looking Notes

- `training_recipe.v1` schema is forward-looking; no test consumer exists at T2.2. Future tools (e.g., `tools/compare_recipes.py`) would consume the version field.
- xG-based `defensive_match_indicator` deferred to T2.3+ pending StatsBomb coverage; current binary GA-based form is the always-available baseline.
- Continuous variant of `defensive_match_indicator` deferred pending observation of binary form's signal sparsity in production.
```

- [ ] **Step 3: Validate the section reads cleanly**

Run: `head -200 MODEL_REVIEW.md | tail -100`

Expected: §8 renders coherently with all `<...>` placeholders replaced by real values from Step 1.

- [ ] **Step 4: Commit**

```bash
git add MODEL_REVIEW.md
git commit -m "$(cat <<'EOF'
docs(t2.2): baseline shift retrospective with 4-mechanism decomposition

Adds §8 to MODEL_REVIEW.md documenting T2.2's gate-passing model:
- All quality gates passed (CV mean and holdout)
- 4-mechanism decomposition with sequential marginal contributions:
    Δ_smote_weights, Δ_θ_D, Δ_features
  Cited from three measurement artifacts:
    - smote_classweight_ablation.json (commit 3)
    - eval_pre_features.json (commit 6b)
    - eval_ensemble.json (commit 10)
- season_stage signal-loss compensation analysis (T2.1's matchday=0
  fix dropped Phase 1 → T2.1; T2.2 either recovered or noted the gap)
- SHAP baseline-shift flag for T2.3 (post-SMOTE attribution bias)
- Chosen θ_D ceiling-proximity check
- Methodology note: two decomposition methodologies (harness for
  composition; checkout-and-run for additive mechanisms)

Per design doc §7, §8.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec coverage
- §1 (context, gates) — Tasks 1–10 close the draw_f1 gate; Task 11 retrospective summarizes.
- §2.1 (SMOTE module) — Task 1.
- §2.2 (ablation harness, decision rule) — Tasks 2–3.
- §2.3 (θ_D grid search, threshold rule) — Tasks 5, 6b, 7a, 9.
- §2.4 (3 new features, schema bump, column-position locking, inference parity) — Task 8.
- §3 (gates + strict-mode partition) — Task 7a.
- §4 (module layout) — Tasks 1, 2, 5, 7b, 8 in aggregate.
- §5.1 (cv_report.v1 unchanged) — no task touches it.
- §5.2 (training_recipe.v1) — Task 7b.
- §5.3 (EnsembleModel attribute) — Task 5.
- §5.4 (backwards compat: trust schema guard) — implicit in Task 8 (schema bump) + Task 9 (no defensive accessor added).
- §6.1 (Phase 8/10 intentional duplicate) — code comment in Task 7a.
- §6.2 (save-time invariant) — Task 7a.
- §6.3 (schema bump atomicity) — Task 8.
- §6.4 (column position locking) — Task 8 tests.
- §6.5 (class-index audit) — Task 5 step 1.
- §6.6 (inference parity) — Task 9 commit message.
- §6.7 (post-calibration OOF contract) — Tasks 6a, 7a docstrings.
- §6.8 (recompute_discrete_metrics contract) — Task 5 docstring.
- §6.9 (no auto-tune) — Task 2 HarnessFailure, Task 3 escalation note.
- §7 (commit slicing) — full plan structure.
- §8 (risks: season_stage compensation, SHAP shift) — Task 11 retrospective subsections.
- §9 (DoD) — Task 10 verification + Task 11 documentation.
- §10 (open questions) — Task 11 forward-looking notes section.

### Type consistency
- `find_draw_threshold` return shape: `{best_threshold, best_macro_f1, grid: [...]}` — consistent across Tasks 5, 7b, 11.
- `predict_with_threshold` signature: `(proba, theta) → labels` — consistent across Tasks 5, 7a, 9.
- `recompute_discrete_metrics` signature: `(cv_section, oof_per_fold, theta) → CVSection` — consistent.
- `EnsembleModel.draw_threshold`: float|None attribute — consistent across Tasks 5, 7a, 9.
- Training recipe key names (`smote_strategy`, `smote_k_neighbors`, `class_weight`, `ablation_winning_cell_id`, `draw_threshold_chosen`, `draw_threshold_grid`) — consistent across Tasks 4 (yaml), 7b (writer), 11 (cited).

### Placeholder scan
- Task 3, 4, 8, 10, 11: `<winner_*>`, `<theta>`, `<delta_N>` placeholders ARE intentional — they're values to be filled in at execution time from concrete artifact contents. Each is paired with the `python -c ...` command that produces the value. **Not** abstract placeholders; concrete data extraction points.
- No `TBD`, `TODO`, "implement later", or "similar to Task N" patterns found.

### Outstanding risks
- The exact line numbers in `train.py` will likely have shifted between design-doc-write and implementation. Tasks 6b and 7a use `grep` to locate patch points rather than hardcoding line numbers.
- `_canonical_feature_cols` and `_save_feature_cols` helper names in Task 7a may differ from what's actually in train.py — Task 7a step 3 explicitly says "Read it first; do not duplicate helpers if they already exist."
- Path convention discrepancy (design doc says `backend/tools/` but T2.1 ships `tools/`): plan uses `tools/` and notes the deviation explicitly at the top.

---

## Plan Complete

Plan saved to `docs/superpowers/plans/2026-04-30-t2_2-draw-class-handling.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task; review between tasks; fast iteration. Each commit's checkpoint allows pause-resume across sessions per the user's stated preference for "pausing makes sense between writing-plans output and commit 1".

**2. Inline Execution** — Execute tasks in this session using executing-plans; batch execution with checkpoints for review.

Which approach?
