# T2.2 — Draw-Class Handling — Design

**Status**: Brainstorm complete (Q1–Q7), pending user review of this spec.
**Parent**: `docs/superpowers/specs/2026-04-27-phase-2-meta-spec.md`
**Predecessor**: `docs/superpowers/specs/2026-04-27-t2_1-walk-forward-cv-design.md` (T2.1 shipped 2026-04-29; established CV machinery and intentionally failed `min_draw_f1` gate).
**Date**: 2026-04-30
**Owner**: Julien

---

## 1. Context and Goal

T2.1 established Phase 2's quality-gate machinery (walk-forward CV, locked holdout snapshot, three-tier gate enforcement) and intentionally shipped a model that fails the `min_draw_f1 ≥ 0.25` gate. The Phase 2 baseline that lands at the start of T2.2:

| Metric | T2.1 CV mean | Threshold | Gate |
|--------|--------------|-----------|------|
| RPS | 0.2099 | ≤ 0.21 | ✅ |
| Brier | 0.2027 | ≤ 0.22 | ✅ |
| draw_f1 | 0.0392 | ≥ 0.25 | ❌ (intentional) |

**T2.2's job**: implement draw-class handling so that the model passes `min_draw_f1 ≥ 0.25` while maintaining `max_rps ≤ 0.21` and `max_brier ≤ 0.22`. The gap to close is large (0.04 → 0.25 → +0.21 absolute). No single mechanism is expected to suffice; the meta-spec prescribes a four-mechanism approach:

1. **SMOTE** inside CV folds (synthetic minority oversampling on the training portion only).
2. **Class weights** applied via `sample_weight` during model fits.
3. **Calibrated draw threshold θ_D** grid-searched over [0.18, 0.32] on post-calibration ensemble OOF predictions.
4. **Three new draw-aware features**: `elo_diff_abs`, `h2h_draw_rate_last_5` (5-meeting count window), `defensive_match_indicator` (binary, league-season-median GA-relative).

`feature_schema_version` bumps `"2.0"` → `"2.1"` atomically with the feature-add commit.

**Non-goals** (per meta-spec §6.6, restated): no new model architectures, no Dixon-Coles, no neural-network-based draw handling, no holdout snapshot modifications, no CVReport schema changes.

---

## 2. Composition Strategy

### 2.1 SMOTE integration surface

A new module `backend/training/draw_handling.py` houses the four T2.2-specific helpers:

```
resample(X, y, sampling_strategy, k_neighbors) -> (X', y')
class_sample_weights(y, weights_dict) -> np.ndarray
find_draw_threshold(y_oof, proba_oof, search_range, step) -> dict
predict_with_threshold(proba, theta) -> labels
recompute_discrete_metrics(cv_section, oof_per_fold, theta) -> CVSection
```

Single integration point: `train_calibrated_models()` in `backend/evaluation/cv.py`. This function is called by both `run_cv()` per-fold and `_retrain_final()` once; placing SMOTE+class_weight here covers both paths. SMOTE is applied to `(X_tr, y_tr)` ONLY — `X_inner_val` and `y_inner_val` (used for isotonic calibration) and the fold's val rows (the OOF predictions) remain at the real distribution.

**Rationale for module placement** (Q1 conclusion): meta-spec §4 names `backend/training/draw_handling.py` as a T2.2 deliverable. Centralising the four mechanisms (SMOTE, sample_weights, θ_D search, threshold rule) gives them a coherent home and lets `cv.py` stay a thin orchestrator.

`find_draw_threshold` returns more than the bare float — it returns `{best_threshold, best_macro_f1, grid: [...]}` so the choice is inspectable post-hoc. See §5.

### 2.2 SMOTE + class_weight composition: empirical ablation

The composition of SMOTE and class_weight is genuinely under-determined by the literature (paper §6.5.2 covers both mechanisms but not their composition). Stacked naively (SMOTE 'auto' full-balance + weights {1.0, 2.5, 1.2}) the effective draw loss share rises to ~53% — back-of-envelope, treated as a heuristic, not load-bearing. The ablation harness measures actual behavior.

**Tool**: `backend/tools/validate_smote_classweight_composition.py` (mirrors T2.1's `tools/validate_cv_parametrization.py`).

**Cell set** — 3 SMOTE strategies × 2 weight schemes = 6 cells:

| Cell | SMOTE `sampling_strategy` | class_weight (H, D, A) | Role |
|------|---------------------------|------------------------|------|
| 1 | off | (1.0, 1.0, 1.0) | Naive baseline (≈ T2.1 reference) |
| 2 | off | (1.0, 2.5, 1.2) | Class-weight-only (no resample) |
| 3 | partial: draws → 70% of H count | (1.0, 1.0, 1.0) | Soft-resample-only |
| 4 | partial: draws → 70% of H count | (1.0, 2.5, 1.2) | Soft resample + specced weights |
| 5 | 'auto' (full balance) | (1.0, 1.0, 1.0) | Full-resample-only |
| 6 | 'auto' (full balance) | (1.0, 2.5, 1.2) | Specced (full resample + specced weights) |

**Folds**: all 6 (full CV pool 2021–2023, n_splits=2, val_window=9). Documented `--folds 0,2,4` fallback if measured runtime makes 6-fold prohibitive; fold-0-only is **not** acceptable (per T2.1 precedent: aggregate fold behavior is what selected (2,9) over (3,6)).

**Decision rule** (margin filter per meta-spec §2 improvement classification):
1. Filter cells where `mean.rps ≤ 0.205` AND `mean.brier ≤ 0.215` (gates with 0.005 margin).
2. Among passing cells, pick max `mean.draw_f1`.
3. Tiebreak: lower `std.draw_f1` (stability over peak).
4. **Failure mode**: if no cell passes margin, harness exits with `HarnessFailure: no cell within margin`. T2.2 cannot proceed; the retrospective addresses whether a fourth mechanism (focal loss, weighted Brier, ordinal regression) is required — meta-spec re-open conversation. **No auto-tune knob-twist fallback.**

**Output schema**: `backend/data/output/smote_classweight_ablation.json` (note: `feature_schema_version="2.0"` — composition is validated on the current feature set; the new draw features are added after the harness runs, with T2.2's final training run as the empirical check that the chosen composition holds across the 2.0→2.1 bump).

```json
{
  "schema_version": "smote_cw_ablation.v1",
  "feature_schema_version": "2.0",
  "cells": [
    {
      "cell_id": 1,
      "smote_strategy": "off",
      "class_weight": [1.0, 1.0, 1.0],
      "fold_results": [
        {"fold_id": 0, "rps": ..., "brier": ..., "draw_f1": ..., "draw_recall": ..., "draw_precision": ...}
      ],
      "mean": {"rps": ..., "brier": ..., "draw_f1": ...},
      "std":  {"rps": ..., "brier": ..., "draw_f1": ...}
    }
  ]
}
```

The "approx effective draw loss share" column from the brainstorm is **not** part of the harness output — it's a back-of-envelope heuristic for design discussion, not a measured value the harness validates.

### 2.3 θ_D grid search

After CV completes (gates measurement), grid-search θ_D over [0.18, 0.32] with step 0.01 (14 candidate points) on the **stacked, post-calibration ensemble OOF probabilities**.

**Critical constraint**: the OOF arrays must be post-calibration ensemble probabilities — not raw base-model outputs, not pre-calibration ensemble. θ_D is applied post-calibration at inference, so it must be tuned on the same probability surface.

**Single θ across all folds**, not per-fold:
- 14 grid points × ~3000 stacked OOF samples = high-power grid scoring.
- Gate metric (`min_draw_f1`) is computed on CV mean draw_f1; tuning θ on the same data the gate measures is semantically aligned.
- Per-fold draw_f1 is still reported in CVReport (with the chosen θ applied), so per-fold stability remains visible without driving θ selection.
- The "fit-to-validation" framing does **not** apply: with one continuous parameter, 14 grid points, a smooth concave macro-F1 surface, and ~3000 samples, this is locating a single minimum on a stable curve — not multi-degree-of-freedom optimization that overfits.

**Decision criterion**: θ* = argmax(macro_f1) over grid.

**Threshold rule** (applied at inference and at metric computation):
```
predict_draw  ⟺  (proba[draw] ≥ θ)  AND  (proba[draw] == max(proba))
otherwise:     predict argmax(proba)
```
θ_D is a **necessary but not sufficient** condition — the model still requires draw to beat both single-team probabilities. Explicit so a future "improvement" doesn't drop the second clause.

**Three call sites** must use the rule:
- `recompute_discrete_metrics()` — CV per-fold + aggregate (after θ_D found).
- `_evaluate_holdout()` in train.py — locked holdout single-pass.
- `output/predict.py` line 323 — inference label assignment.

All three call the same `predict_with_threshold(proba, theta)` helper in `draw_handling.py`. No copy-paste of the rule.

**Tie behavior**: `proba[:,1] == proba.max(axis=1)` falls through to argmax on float ties (which would themselves be vanishingly rare). Class-index convention `{0:H, 1:D, 2:A}` is locked at T2.2; design includes a verification audit at implementation time (consistent across `cv.py:_per_class_recall labels=[0,1,2]`, `train.py outcome_map`, `predict.py outcome_map`).

### 2.4 New draw features

Three features, additive to the existing schema. `feature_schema_version` bumps `"2.0"` → `"2.1"` atomically with the feature-add commit.

#### 2.4.1 `elo_diff_abs`

Definition: `abs(home_elo - away_elo)` — captures "evenly-matched pairings draw more often than mismatched ones" (textbook draw predictor).

**Placement**: inside `features/elo.py:compute_elo`. **Column position locked**: appended after `away_elo` in `compute_elo`'s output column order. Test fixture asserts the position. Reason: `feature_cols.pkl` is locked at inference; column order matters for schema 2.1 reproducibility — a future "cleanup" that moves the line elsewhere silently breaks `predict.py`. Same locking discipline applies to the other two features below.

#### 2.4.2 `h2h_draw_rate_last_5`

Definition: draw rate over the **last 5 head-to-head meetings** between the two teams (count-based window), regardless of how far back in time those meetings occurred. NaN if zero prior meetings.

**Distinct from** existing `h2h_draw_rate` (5-year time window). Both coexist: long-term tendency (5-year) + recent specifics (5-meeting). Tree models can use both.

**Placement**: extend `features/form.py:build_h2h_features`. Same per-row scan as the existing 5-year H2H pass; add a second pass sorted by date descending, take top 5, compute draw rate. Column-position-locked.

**Test**: extend `tests/test_form_alignment.py` with a fixture pair confirming `h2h_draw_rate_last_5` differs from `h2h_draw_rate` when meeting frequency changes; cold-start cases (zero meetings) get NaN.

#### 2.4.3 `defensive_match_indicator`

Definition (per Q5b-iv pushback — sticking close to the original spec form):
```
home_below_median_ga = home_w10_avg_ga < league_season_median_w10_avg_ga
away_below_median_ga = away_w10_avg_ga < league_season_median_w10_avg_ga
defensive_match_indicator = int(home_below_median_ga AND away_below_median_ga)
```

Binary, league-season-relative (median over (league, season) pairs). Captures the spec's "both teams below median xGA over last 10" intent with the minimum data substitution: xGA → GA, since GA is always available where xG features graceful-skip when StatsBomb data is missing. Continuous variants (clean-sheet products, GA differentials) are **not** chosen at T2.2: the binary form is closer to the spec, and continuous variants can be revisited in T2.3+ if empirical results suggest the binary form is too sparse.

**Placement**: extend `features/context.py:build_context_features`. Depends on form features being already computed (build order: elo → form → h2h → xg → squad → tactical → context — already supports this). **Column-position-locked**: appended after `away_title_race` (the current last column emitted by `build_context_features`, last entry of `_STANDING_COLS`). Test asserts the position.

**Verified prerequisite** (during exploration): `home_w10_avg_ga` and `away_w10_avg_ga` already exist in form.py output (form features compute `avg_ga` for windows [3, 5, 10]). 5b-iv is a one-liner over existing columns plus the per-(league, season) median computation.

#### 2.4.4 feature_config.yaml toggling

New columns are gated by their **parent group toggles** (`elo.enabled`, `head_to_head.enabled`, `context.enabled`). No new `draw_features` toggle group: feature config should reflect feature semantics, not ticket lineage.

#### 2.4.5 Inference parity

Explicit statement (the kind that's obvious until it isn't): **new features at inference flow through the same builders as at training; predict.py changes are limited to (1) threshold rule application at line 323 and (2) FileNotFoundError UX fix. No new builder calls in predict.py.**

---

## 3. Quality Gates and Strict-Mode Partition

### 3.1 Gates (no change from T2.1 / meta-spec §3)

| Gate | Threshold | Source |
|------|-----------|--------|
| max_rps | ≤ 0.21 | TrainingConfig.max_rps |
| max_brier | ≤ 0.22 | TrainingConfig.max_brier |
| min_draw_f1 | ≥ 0.25 | TrainingConfig.min_draw_f1 |

### 3.2 Strict-mode partition (T2.2-specific)

T2.1's discipline: "save artifacts BEFORE asserting gates — operators need them to debug" (train.py line 262 comment).

T2.2 partitions strict mode by gate type:
- **draw_f1**: deployment-blocking. If `cv.mean.draw_f1 < 0.25`, train.py refuses to write `ensemble.pkl` and `feature_cols.pkl`. The diagnostic JSON (`eval_ensemble.json`) is always written so the operator can read why.
- **rps, brier**: T2.1 inspect-on-failure semantics retained. Artifacts saved, then `assert_gates()` raises.

The two checks of `draw_f1` (Phase 8 inline + Phase 10 `assert_gates`) are **intentional, not redundant** — different consequences (block deploy vs log-and-raise). Code comment + this design doc capture the reasoning so a future "DRY-up" doesn't silently break the partition.

### 3.3 train.py phase ordering (Q4 conclusion)

```
Phase 1:  run_cv_with_oof()   -> (cv_section_v1, per_fold_oof)
                                  (FoldMetrics: probabilistic metrics final;
                                   discrete metrics PRELIMINARY/argmax-derived)
Phase 2:  find_draw_threshold(stacked_oof_y, stacked_oof_proba) -> theta_result
Phase 3:  cv_section = recompute_discrete_metrics(cv_section_v1, per_fold_oof, theta_result.best_threshold)
Phase 4:  ensemble = _retrain_final()
          ensemble.draw_threshold = theta_result["best_threshold"]
Phase 5:  holdout = _evaluate_holdout(ensemble)        # uses θ_D
Phase 6:  gates = _build_gates(cv_section, holdout)
          report = CVReport(cv=cv_section, holdout=holdout, gates=gates, ...)
Phase 7:  write eval_ensemble.json                     # UNCONDITIONAL — diagnostic always saves
Phase 8:  if cv_section.mean_metrics.draw_f1 < min_draw_f1:
              raise QualityGateFailure(...)             # T2.2 strict — artifacts NOT saved
Phase 9:  assert ensemble.draw_threshold is not None    # save-time invariant (regression guard)
          ensemble.save()
          save_feature_cols()
          write training_recipe.json                    # alongside ensemble.pkl
Phase 10: report.assert_gates()                         # T2.1 semantics: raises if RPS/Brier fail
```

**Behavior matrix**:

| Failure mode | eval JSON | ensemble.pkl | feature_cols.pkl | training_recipe.json | Where it raises |
|--------------|-----------|--------------|------------------|----------------------|-----------------|
| draw_f1 < 0.25 | ✅ | ❌ | ❌ | ❌ | Phase 8 |
| RPS > 0.21 (draw_f1 ok) | ✅ | ✅ | ✅ | ✅ | Phase 10 |
| Brier > 0.22 (draw_f1 ok) | ✅ | ✅ | ✅ | ✅ | Phase 10 |
| Any combo with draw_f1 < 0.25 | ✅ | ❌ | ❌ | ❌ | Phase 8 |
| All gates pass | ✅ | ✅ | ✅ | ✅ | (no raise) |

### 3.4 Phase 8 error message

```
QualityGateFailure: T2.2 strict mode — draw_f1 below threshold, deployment artifacts not saved.

Gates:
  draw_f1: 0.214 < 0.250  ❌
  rps:     0.207 ≤ 0.210  ✅
  brier:   0.205 ≤ 0.220  ✅

Diagnostic: backend/data/output/eval_ensemble.json
Chosen θ_D: 0.27
  macro_f1 at θ*: 0.341
  draw_f1 at θ*:  0.214 (gate margin: -0.036)
ensemble.pkl: NOT WRITTEN
```

Including both `macro_f1_at_best` and `draw_f1_at_best` (per Q4 update #5) avoids the ambiguous reading "the threshold search failed" — it didn't; θ_D was chosen correctly per the optimization criterion, and the resulting draw_f1 simply doesn't clear the gate.

---

## 4. Module Layout

### New files

```
backend/training/draw_handling.py           # Q1: SMOTE + sample_weights + θ_D + threshold rule + recompute
backend/tools/validate_smote_classweight_composition.py   # Q2: ablation harness
backend/data/output/smote_classweight_ablation.json       # Q2: harness output
backend/data/output/training_recipe.json                  # Q6: T2.2 training recipe (training_recipe.v1)
backend/data/output/decomposition/eval_pre_features.json  # Q7: Checkpoint A snapshot (frozen, from commit 6b)
```

### Modified files

```
backend/requirements.txt                      # add imbalanced-learn>=0.12,<1
backend/config/model_config.yaml              # new training: block (smote_strategy, smote_k_neighbors, class_weight)
backend/config/feature_config.yaml            # no schema changes; existing toggles cover new columns
backend/features/build.py                     # FEATURE_SCHEMA_VERSION "2.0" -> "2.1" (atomic with commit 8)
backend/features/elo.py                       # +elo_diff_abs (column position locked)
backend/features/form.py                      # +h2h_draw_rate_last_5 (column position locked)
backend/features/context.py                   # +defensive_match_indicator (column position locked)
backend/evaluation/cv.py                      # run_cv() returns (CVSection, list[(y_fold, proba_fold)])
                                              # SMOTE + sample_weight wired into train_calibrated_models per harness winner
backend/models/ensemble.py                    # EnsembleModel.__init__ +draw_threshold: float|None=None
backend/models/train.py                       # 9-phase strict-mode reorder; save-time invariant; recipe writer
backend/output/predict.py                     # threshold rule at line 323; FileNotFoundError UX fix
```

### Unchanged (deliberately)

```
backend/evaluation/cv_report.py               # cv_report.v1 schema is locked per meta-spec §5
backend/evaluation/exceptions.py              # QualityGateFailure already covers all three gates
backend/evaluation/splits.py                  # WalkForwardSplit defaults locked at T2.1
backend/data/eval/holdout_2024_25.json        # holdout snapshot is locked
```

---

## 5. Artifact Schemas

### 5.1 `cv_report.v1` — UNCHANGED

Locked at T2.1. Per meta-spec §5: "Phase 2 tickets do NOT bump schema_version; only meta-changes do." Adding a top-level `TrainingSection` block (or extending `CalibrationSection`) qualifies as a shape change requiring meta-spec re-open. Therefore: T2.2 does not modify `cv_report.v1`.

T2.2-specific recipe metadata lives in a separate file (§5.2).

### 5.2 `training_recipe.v1` — NEW

**File**: `backend/data/output/training_recipe.json`. Written by train.py at Phase 9, alongside `ensemble.pkl`.

```json
{
  "schema_version": "training_recipe.v1",
  "feature_schema_version": "2.1",
  "training_timestamp": "2026-..-..T..:..:..Z",

  "smote_strategy": "auto",
  "smote_k_neighbors": 5,
  "class_weight": {"H": 1.0, "D": 2.5, "A": 1.2},
  "ablation_winning_cell_id": 6,

  "draw_threshold_chosen": {
    "theta": 0.27,
    "macro_f1": 0.341,
    "draw_f1": 0.262,
    "draw_precision": 0.330,
    "draw_recall": 0.218
  },
  "draw_threshold_grid": [
    {"theta": 0.18, "macro_f1": ..., "draw_f1": ..., "draw_precision": ..., "draw_recall": ...},
    {"theta": 0.19, "macro_f1": ..., "draw_f1": ..., "draw_precision": ..., "draw_recall": ...}
  ]
}
```

**Grid format** (Q6 update): array of objects ordered by `theta` ascending, each with `macro_f1`, `draw_f1`, `draw_precision`, `draw_recall`. Not a `θ → draw_f1` dict — full per-θ metrics let the retrospective cite "θ* maximizes macro_f1 at X, yielding draw_f1 Y (gate margin Z)" directly from one block.

**Schema_version forward-looking honesty**: `training_recipe.v1` has no test consumer at T2.2 time. The `schema_version` field is included for parity with the `cv_report.v1` discipline; not load-bearing yet. Future tooling that scans recipes (e.g., `tools/compare_recipes.py`) would consume the version field. This is the same forward-looking discipline T2.1 applied to its placeholder gate fields.

**Loose coupling**: `training_recipe.json` is read by humans (retrospective, debugging) and any future analysis tooling, NOT by `predict.py`. Predict.py reads only `ensemble.pkl` (which carries `draw_threshold`) and `feature_cols.pkl`.

### 5.3 EnsembleModel pickle — `draw_threshold` attribute

**Note on serialization format**: T2.2 extends T2.1's existing pickle-based ensemble artifact (`ensemble.pkl`). Pickle is appropriate here because (a) the artifact contains fitted XGBoost/LightGBM models that don't roundtrip cleanly through JSON, (b) the artifact is produced and consumed only within this project's trusted pipeline (predict.py loads only the locally-written file from `data/models/`), and (c) the codebase's existing serialization comments document this trust boundary (`models/ensemble.py:save` and `:load` carry `# Trusted local file` notes). T2.2 introduces no new pickle surfaces beyond extending the existing `EnsembleModel` class with one additional attribute. Migrating to JSON-based serialization would require choosing a portable model-serialization format (`xgboost.Booster.save_model`, `lightgbm.Booster.save_model`, plus a custom JSON wrapper for the ensemble metadata) — a separate architectural decision out of T2.2's scope.

θ_D is stored as a **direct attribute** on `EnsembleModel`, not in a `training_metadata` dict:

```python
class EnsembleModel:
    def __init__(self, models, weights, draw_threshold: float | None = None):
        ...
        self.draw_threshold = draw_threshold
```

**Inference state in pickle** (`models`, `weights`, `draw_threshold`); **audit metadata in JSON** (`training_recipe.json`). Two sources, two questions: "how do I serve this?" vs "how was it made?".

**Save-time invariant** (Q6 user addition):
```python
# In train.py, immediately before ensemble.save():
assert ensemble.draw_threshold is not None, (
    "EnsembleModel.draw_threshold must be set before save() — "
    "T2.2 strict-mode ordering violated. Did Phase 4 run?"
)
```

Save-time over load-time because (a) fails fast in training, (b) keeps load-time simple per Q6, (c) regression test on `save()` guards against future refactors that subtly violate strict-mode ordering. Test: assert that calling `save()` on an ensemble with `draw_threshold=None` raises a clear error.

### 5.4 Backwards compat — trust the schema guard

The existing `feature_schema_version` guard in `predict.py:_enforce_serving_guards` rejects mismatched eval JSON. T2.1 ensembles (schema "2.0") cannot serve under T2.2 code (schema "2.1") — guard rejects at JSON load.

T2.2 does **not** add a defensive accessor on `ensemble.draw_threshold`. T2.1 principle: don't add API surface without consumers. The schema guard is the canonical "this model is incompatible" signal; duplicating responsibility at the attribute access site doesn't strengthen it (someone bypassing the JSON guard would also bypass an attribute guard). A pre-T2.2 ensemble unpickled into T2.2 code wouldn't have the `draw_threshold` attribute set — `AttributeError` at access time is *correct* in that case, since the dev knowingly disabled the schema guard.

Silent argmax fallback when `draw_threshold` is missing is **explicitly rejected** (silent failure).

---

## 6. Implementation Discipline

### 6.1 Strict-mode partition is intentional

Phase 8 inline check on `draw_f1` and Phase 10 `assert_gates()` both check draw_f1. **This is not redundancy** — the two checks have different consequences (block deploy vs log-and-raise). Documented in train.py code comment AND in this design doc so a future "DRY-up" doesn't silently break the partition.

### 6.2 Save-time invariant on `draw_threshold`

See §5.3. Save-time enforcement.

### 6.3 Schema bump atomicity

`FEATURE_SCHEMA_VERSION` bumps `"2.0"` → `"2.1"` in the **same commit** that adds the 3 new feature builders (commit 8). Decoupling creates a window where a schema-2.0 model serves with schema-2.1 expectations. T2.1's matchday-derivation commit set the precedent (commit 1 of T2.1 added the function, wired it into ingestion, AND bumped the schema constant atomically).

### 6.4 Column position locking

Each of the 3 new feature columns lands at a **specific position** in its parent builder's output. Tests assert position. `feature_cols.pkl` reproducibility under schema 2.1 depends on column order; future "cleanup" refactors that move lines silently break `predict.py`.

- `elo_diff_abs`: appended after `away_elo` in `compute_elo` output.
- `h2h_draw_rate_last_5`: appended after `h2h_draw_rate` in `build_h2h_features` output.
- `defensive_match_indicator`: appended after `away_title_race` (the current last column emitted by `build_context_features`, last entry of `_STANDING_COLS`).

### 6.5 Class-index convention audit

`predict_with_threshold` hardcodes `proba[:,1]` for the draw class. The `{0:H, 1:D, 2:A}` convention is consistent across the codebase (verified during brainstorm exploration: `cv.py:_per_class_recall labels=[0,1,2]`, `train.py outcome_map`, `predict.py outcome_map`). Locked at T2.2; implementation includes a verification step before the helper is committed.

### 6.6 Inference parity statement

(Restated from §2.4.5.) New features at inference flow through the same builders as at training. `predict.py` changes at T2.2 are limited to:
1. Threshold rule application at line 323 (replace argmax with `predict_with_threshold(proba, ensemble.draw_threshold)`).
2. FileNotFoundError UX fix (clear operator-facing error on missing `ensemble.pkl`, not stack trace).

**No new builder calls in predict.py.**

### 6.7 OOF probability surface contract

The OOF arrays threaded from `run_cv()` into `find_draw_threshold` are **post-calibration ensemble** probabilities. Not raw base-model outputs. Not pre-calibration ensemble. θ_D is applied post-calibration at inference, so it MUST be tuned on the same probability surface. Easy to fudge in implementation — explicit in `recompute_discrete_metrics` and `find_draw_threshold` docstrings.

### 6.8 `recompute_discrete_metrics` contract

Probabilistic metrics (`brier`, `rps`, `log_loss`) **do not depend on θ_D** and are carried forward unchanged from the preliminary CVSection. Only discrete metrics (`accuracy`, `draw_f1`, `home_recall`, `draw_recall`, `away_recall`) are recomputed. Stated explicitly in the function docstring to prevent second-guessing during implementation.

### 6.9 Fail-fast harness, no auto-tune

The Q2 ablation harness's failure mode (no cell within margin) does **not** trigger automatic knob-twisting (e.g., trying weights `(1.0, 1.2, 1.0)` as a fallback). If no cell passes, that's a real signal that T2.2's three-pronged approach is insufficient — surfaced as `HarnessFailure`, addressed in retrospective, escalated to meta-spec re-open.

---

## 7. Commit Slicing

13 commits, three measurement artifacts, four mechanism deltas.

| # | Commit | Touches | Measurement |
|---|--------|---------|-------------|
| 1 | Add `imbalanced-learn>=0.12,<1` + skeleton `draw_handling.py` (`resample`, `class_sample_weights` consumed by commit 2's harness; `find_draw_threshold` and `predict_with_threshold` deferred to commit 5) | requirements.txt, draw_handling.py | — |
| 2 | SMOTE+class_weight ablation harness (`tools/validate_smote_classweight_composition.py`); 6 cells × 6 folds; margin-filter decision rule; `HarnessFailure` mode | tools/, draw_handling.py | — |
| 3 | Run harness, commit `smote_classweight_ablation.json` + winning-cell analysis section in MODEL_REVIEW.md | data/output/, MODEL_REVIEW.md | ✅ **Δ_smote_weights**: winning cell vs T2.1 baseline |
| 4 | Wire SMOTE + sample_weight into `train_calibrated_models` per harness winner (cell N from commit 3); update `model_config.yaml` `training:` block. Commit message explicitly traces the dependency | cv.py, model_config.yaml | — |
| 5 | Implement `find_draw_threshold`, `predict_with_threshold`, `recompute_discrete_metrics` + `EnsembleModel.draw_threshold` attribute; verify class-index convention | draw_handling.py, ensemble.py, tests | — |
| 6a | `run_cv()` signature change: returns `(CVSection, list[(y_fold, proba_fold)])` + caller updates + tests | cv.py, train.py callsite, tests | — |
| 6b | Integrate θ_D into train.py: stack OOF, call `find_draw_threshold`, `recompute_discrete_metrics`, assign `ensemble.draw_threshold`. NO strict-mode reorder yet | train.py | ✅ **Checkpoint A**: run training, snapshot to `data/output/decomposition/eval_pre_features.json` |
| 7a | Strict-mode reorder of train.py (Phases 1-10 from §3.3) + save-time invariant on `draw_threshold` + regression test | train.py, ensemble.py, tests | — |
| 7b | `training_recipe.json` writer (`training_recipe.v1`, grid array of objects, `draw_threshold_chosen` block) | train.py, draw_handling.py | — |
| 8 | Add 3 new draw features atomically with `FEATURE_SCHEMA_VERSION` 2.0→2.1 (column positions locked, tests assert position) | elo.py, form.py, context.py, build.py, tests | — |
| 9 | predict.py: threshold rule at line 323 + FileNotFoundError UX fix | predict.py | — |
| 10 | Final T2.2 training run: ensemble + eval JSON + recipe JSON | data/models/, data/output/ | ✅ **Checkpoint B**: `eval_ensemble.json` |
| 11 | T2.2 baseline shift retrospective in MODEL_REVIEW.md with 4-mechanism decomposition table citing the three artifacts | MODEL_REVIEW.md | — |

### Three measurement artifacts → four mechanism deltas

- **Δ_smote_weights** = winning ablation cell (commit 3) − T2.1 baseline (CV mean draw_f1 = 0.0392)
- **Δ_θ_D** = `eval_pre_features.json` (commit 6b, with θ_D rule applied) − winning ablation cell (commit 3, argmax)
- **Δ_features** = `eval_ensemble.json` (commit 10) − `eval_pre_features.json` (commit 6b)

**These are sequential marginal contributions, not isolated contributions.** Each Δ measures the additional improvement attributable to a mechanism *given everything previously layered in is already in place*: Δ_θ_D is "what θ_D adds on top of SMOTE+weights", not "what θ_D would add to the T2.1 baseline alone"; Δ_features is "what the 3 new features add on top of SMOTE+weights+θ_D", not their isolated effect. The decomposition is non-commutative — a different ordering would produce different per-mechanism numbers.

Retrospective phrasing must reflect this. Acceptable framings: "given SMOTE+weights wired, θ_D added X to draw_f1"; "with the full mechanism stack except features, draw_f1 was Y; adding features brought it to Z, contributing Δ_features = Z − Y." Unacceptable framings: "θ_D contributes X" (without the conditioning); "features account for Δ_features of the total improvement" (without the ordering caveat).

Sum of deltas should approximately equal the total improvement from T2.1 baseline to T2.2 final (modulo non-linear interactions and the ordering effect noted above). The retrospective discusses any non-additivity.

### Frozen artifact reproducibility

`eval_pre_features.json` is a **frozen artifact**: reproducibility requires checking out commit 6b. Same property as T2.1's `cv_parametrization_validation.json` — acceptable cost, documented honestly.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ablation harness reports `HarnessFailure` (no cell passes margin) | T2.2 cannot proceed | Surfaced as gate-shipping issue; meta-spec re-open conversation; retrospective documents which cells failed and by how much; no auto-tune knob-twist fallback |
| Chosen θ* lands at search ceiling (θ* ≥ 0.30) | Soft signal that SMOTE+class_weight pushed model too aggressively toward draws | Documented in retrospective; T2.3 may revisit composition with softer settings (e.g., draw weight 1.5×, partial SMOTE) |
| Holdout draw_f1 ≪ CV draw_f1 | Distribution shift between cv_pool (2021–2023) and holdout (2024–25) | Existing CVReport gate compares both; both gates must pass. Surfaces as `HoldoutSection.draw_f1` gate failure with clear breakdown |
| `defensive_match_indicator` too sparse (very few matches with both teams below median) | Feature contributes near-zero signal | Documented as known limitation; T2.3+ revisits with continuous variant if empirical results warrant |
| `feature_schema_version` 2.1 forces all production models to retrain | One-time operational disruption | Intentional; T2.1's matchday-derivation set the precedent. Predict-time guard rejects pre-T2.2 models with clear error |
| 4-mechanism decomposition shows non-additive interactions | Retrospective story is harder to tell | Acceptable; non-additivity is a real property of the system. Retrospective discusses honestly |
| θ_D grid search picks a θ where macro_f1 maxes but draw_f1 is below gate | T2.2 fails at Phase 8 despite "best" θ | Phase 8 error message includes both `macro_f1_at_best` and `draw_f1_at_best` so the failure mode is unambiguous; retrospective addresses whether macro_f1 is the wrong optimization criterion |
| Decomposition harness methodology diverges between checkpoint A (checkout-and-run) and ablation harness (config-driven) | Two methodologies in one retrospective | Documented honestly in retrospective preamble; each methodology fits its question (composition vs additive mechanisms) |
| **`season_stage` signal loss to compensate for** | Phase 1 had draw_f1 = 0.0856; T2.1 dropped to 0.0392 after fixing the matchday=0 bug that recovered `season_stage` from silently-zero to varying in [0, 1]. The regression suggests `season_stage` was picking up end-of-season draw signal (e.g., "both teams need a point" matches), and the bug fix removed it. T2.2's new features must close not only the original Phase 1 → 0.25 gap but also recover the ~0.05 signal lost in T2.1 | Δ_features measured at checkpoint B is the empirical answer to whether the 3 new features collectively recover the lost signal. Retrospective discusses if they don't, and whether end-of-season-specific features (e.g., motivation flags, "matchday > N AND both teams in mid-table" indicators) need consideration in T2.3+ |
| **SHAP baseline shift for T2.3** | T2.3's DoD includes the gate "SHAP shows pi-rating features collectively account for ≥ 5% of total feature importance". That gate is measured against a SHAP baseline T2.2 will have shifted: post-SMOTE training distribution biases SHAP attributions toward the over-represented draw class, so T2.3's pi-rating SHAP comparison is against a draw-inflated baseline rather than a real-distribution one | Not blocking T2.2. Flagged in T2.2 retrospective so T2.3 design accounts for it (T2.3 may need to compute SHAP on a real-distribution validation set rather than on the SMOTE-augmented training set, or document the bias explicitly when interpreting the 5% gate) |

---

## 9. Definition of Done

### 9.1 Quality gates

✅ Final T2.2 model passes all three gates on full holdout:
- `max_rps ≤ 0.21` (CV mean and holdout)
- `max_brier ≤ 0.22` (CV mean and holdout)
- `min_draw_f1 ≥ 0.25` (CV mean and holdout) — strict mode for deployment artifacts

### 9.2 Process artifacts

✅ Ablation harness ran on all 6 folds (or documented `--folds 0,2,4` fallback with rationale); winning cell selected by margin filter with no auto-tuning fallback applied.

✅ θ_D grid stored in `training_recipe.json` with full per-θ grid (array of objects with `macro_f1`, `draw_f1`, `draw_precision`, `draw_recall`) + `draw_threshold_chosen` block.

✅ `training_recipe.json` written alongside `ensemble.pkl` in strict-mode-compliant order (Phase 9 only, after Phase 8 gate clears).

✅ `eval_pre_features.json` snapshot committed at commit 6b; `eval_ensemble.json` from commit 10.

✅ MODEL_REVIEW.md retrospective with 4-mechanism decomposition table citing the three artifacts (`smote_classweight_ablation.json`, `eval_pre_features.json`, `eval_ensemble.json`).

### 9.3 Code

✅ `FEATURE_SCHEMA_VERSION` bumped to `"2.1"` atomically with commit 8.

✅ `predict.py:_enforce_serving_guards` rejects pre-T2.2 models with clear error (existing guard, no changes needed beyond schema version constant).

✅ `predict.py` FileNotFoundError UX produces clear operator-facing error on missing `ensemble.pkl`, not raw stack trace.

✅ Phase 8 / Phase 10 duplicate `draw_f1` check is documented as intentional in train.py code comment.

✅ Save-time invariant on `ensemble.draw_threshold` is enforced and tested.

### 9.4 Tests

✅ `draw_handling.py` module unit tests:
- `resample`: input shape preserved on output, class proportions match expected for each `sampling_strategy`, k_neighbors honored.
- `class_sample_weights`: returns correct per-sample weights for given (y, weights_dict).
- `find_draw_threshold`: returns dict with required keys, grid is sorted, `best_threshold` is in `search_range`, `best_macro_f1` matches grid max.
- `predict_with_threshold`: tie behavior correct (θ necessary AND p_draw == max), boundary cases handled.
- `recompute_discrete_metrics`: probabilistic metrics carried forward unchanged; discrete metrics recomputed.

✅ EnsembleModel save-time invariant: `save()` on `draw_threshold=None` raises clearly.

✅ Column position assertions for all 3 new features.

✅ Tier 2 quality gate tests still pass (extended for `draw_f1` computed with θ_D).

✅ Inference parity test: ≥5 rows covering edge cases (zero history, partial history, full history for the rolling-window inputs); float-equality on all columns including the 3 new ones (`elo_diff_abs`, `h2h_draw_rate_last_5`, `defensive_match_indicator`); test fails on any column-value mismatch.

### 9.5 Class-index convention audit

✅ Documented audit confirms `{0:H, 1:D, 2:A}` consistent across `cv.py`, `train.py`, `predict.py`. Audit committed alongside commit 5.

---

## 10. Open Questions / Follow-ups

T2.2 is scoped to the four specified mechanisms. The following are deferred to T2.3+ or noted as potential follow-ups:

- **xG-based `defensive_match_indicator`**: if StatsBomb coverage improves, revisit with xGA-based formulation. Current binary GA-based form is the always-available baseline.
- **Continuous variant of `defensive_match_indicator`**: if the binary form proves too sparse (very few matches satisfy both conditions), T2.3+ can revisit with continuous formulations (clean-sheet products, GA differentials).
- **Soft SMOTE+class_weight composition**: if θ* lands at search ceiling (≥ 0.30), this signals composition was too aggressive. T2.3+ may revisit with reduced draw weight (e.g., 1.5×) or `sampling_strategy='not majority'` instead of `'auto'`.
- **League-specific θ_D**: T2.2 ships a single global θ. If retrospective shows large per-league variance in optimal θ, T2.3+ could decompose θ by league/season.
- **`training_recipe.json` consumers**: T2.2 ships the schema with no test consumer. A future `tools/compare_recipes.py` or similar would consume the version field; until then, the schema is forward-looking.

---

*End of T2.2 design.*
