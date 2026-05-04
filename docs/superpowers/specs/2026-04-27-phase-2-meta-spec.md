# Phase 2 — Meta-Spec

**Status**: Approved design (amended 2026-05-04 — T2.2 closure; see §1.6).
**Author**: Brainstorm session, 2026-04-27. Amendment 2026-05-04.
**Scope**: Sequencing, shared interfaces, measurement protocol, and quality
gates that govern every Phase 2 ticket (T2.1 – T2.6). Per-ticket detailed specs
live in their own brainstorm cycles.
**Companion plan**: `IMPLEMENTATION_PLAN_v2.md` § Phase 2.
**Phase 1 baseline (reference)**: brier 0.2006 / rps 0.2069 / log_loss 1.0872 /
accuracy 0.5171 on the 2024 holdout (single-split, post-T0.1d).

---

## Section 1 — Scope & Sequencing

### In-scope tickets

| Ticket | Title | Notes |
|--------|-------|-------|
| T2.6 | Phase 2 non-goals doc | **Lands FIRST**, ahead of T2.1 — prevents scope creep proactively rather than retrospectively. Already shipped as commit `010a711`. |
| T2.1 | Walk-forward CV + locked holdout + quality gates | Foundation. Every later ticket measures itself against the surfaces this ticket builds. |
| T2.2 | Draw-class handling | Closed as ablation evidence (PR #2, 2026-05-02). Three-mechanism design superseded; see §1.6 and `MODEL_REVIEW.md §6.8`. |
| T2.3 | Pi-ratings alongside Elo | New feature builder. Bumps `feature_schema_version`. |
| T2.4 | CatBoost base model | Third tree model. Native categorical handling. |
| T2.5 | Stacking meta-learner | Rewrites the broken `StackingEnsemble` scaffold + adds `oof_generator`. |

### Sequencing

**Strict-sequential through T2.2 closure**: T2.6 ✅ → T2.1 ✅ → T2.2 ✅
(closed as ablation; see §1.6) → T2.3 → T2.4 → T2.5. T2.3/T2.4/T2.5
sequencing unchanged; the draw-handling question does not gate them.
Each remaining ticket branches off `main` after the previous has merged.

Solo-dev rationale: parallelism gains are theoretical when there is one
implementer; sequential keeps deltas attributable per ticket and lets the
leakage gate run end-to-end after every retraining.

### Per-ticket merge protocol

1. Branch off `main` after the previous ticket has merged.
2. All existing tests stay green throughout (Phase 1 closed at 19 tests).
3. Leakage gate (`tests/test_no_leakage.py`) passes after every retraining.
4. Quality-gates test (Tier 2 from §3) stays green at HEAD.
5. Atomic commit set per ticket — mirrors the 7-commit Phase 1 closure pattern.

### Quality gate failure protocol

- **Default**: gate failure BLOCKS merge until resolved.
- **Override allowed only if all three conditions met**:
  1. Failure is documented and understood (not mysterious).
  2. Follow-up ticket explicitly opened to address it.
  3. Justification recorded in `MODEL_REVIEW.md`.

### T2.2 closure status (amendment 2026-05-04)

T2.2 was attempted between 2026-04-30 and 2026-05-02 against the original
13-commit design (`docs/superpowers/plans/superseded/2026-04-30-t2_2-draw-class-handling.md`).
The first three tasks shipped (draw-handling primitives, ablation harness,
6×6 evidence run); the harness produced `HarnessFailure` (no SMOTE × class-weight
cell within margin filter on the full 6 walk-forward folds). A schema-2.1
sanity probe established that the planned new features moved RPS by 0.00002
(within float-rounding noise on the gate metric). Tasks 4–13 of the original
plan did not execute, per the locked decision rule applied to the schema-2.1
sanity probe (RPS > 0.207 → Path A). Ablation evidence and retrospective
merged 2026-05-02 (PR #2).

The `min_draw_f1 ≥ 0.25` gate is reclassified per §6 of this amendment to
diagnostic-only (also see §3). Any successor draw-handling work would live
in Phase 3+, contingent on Phase 2 closure surfacing new evidence and a
successor scope being justified by feasibility-probe-first methodology. Full
retrospective: `MODEL_REVIEW.md §6.8`.

### Phase 4 prep — deferred

- `T4.0a — Ingest 2018-2020`: 5+ aggregate-fold confidence; addresses COVID
  handling, xG sparseness, promotion/relegation continuity. Effort 2–3 days.
  Trigger: before public launch (Phase 4).
- `T4.0b — Rotate holdout to full 2024-25 season`: after the season ends;
  rotates `holdout_snapshot.json` and re-establishes the baseline.

These are recorded in `IMPLEMENTATION_PLAN_v2.md` Phase 4 as one-line stubs;
they are NOT designed in this meta-spec.

### Out of scope (cross-references T2.6 in `MODEL_REVIEW.md` §6.6)

Bivariate Poisson features, neural networks / LSTMs / Transformers, player
embeddings, Bayesian hierarchical models, and reactivating the disabled
Dixon-Coles Poisson. If a Phase 2 implementation step starts drifting toward
any of these, **stop and flag scope creep** rather than implementing.

---

## Section 2 — CV Strategy

`WalkForwardSplit` configuration:

```
CV pool: 2021-22, 2022-23, 2023-24 (held-out 2024-25 unchanged)
Per CV season: 3 walk-forward splits
  Fold structure: train=md[1..N], val=md[N+1..N+6]
  N values: 14, 22, 30 (gives 6-matchday validation windows)
Total folds: 3 seasons × 3 splits = 9 folds
```

**Aggregation**:

Report mean ± std across 9 folds for: RPS, Brier, F1 per class, log loss,
accuracy.

**Held-out generalization** (Section 3 / Q3 Option C maintained):

After CV-based selection, retrain on full 2021–2023 corpus and evaluate on
2024-25 holdout.

**Improvement classification (refined with n=9)**:

| Verdict | Rule |
|---------|------|
| **Real** | CV mean delta > 1 std AND holdout delta > 0.005 |
| **Ambiguous** | One condition met, other not |
| **Regression** | Either metric degrades beyond noise threshold |

**Merge implications**:

- **"Real"** or **"Ambiguous"** verdicts → ticket MAY merge (no degradation
  detected).
- **"Regression"** verdict → ticket BLOCKS merge per §1 quality gate protocol.

Rationale: an "Ambiguous" verdict means the improvement is hard to attribute
between CV and holdout signals, but there's no measurable harm. A ticket that
lands without measurable improvement is not a problem in itself — Phase 2 has
5 tickets, and we expect cumulative gains, not per-ticket gains. The hard
rule is: don't ship regressions.

**Strengths of this combination**:

- Within-season CV: statistical power for ticket-level decisions (n=9 vs n=2).
- Held-out 2024-25: season-level generalization assurance.
- Both signals combined: catches different failure modes.

**Implementation note**:

`WalkForwardSplit` replaces the simpler `SeasonSplit`. Estimated 60–80 lines of
well-tested code instead of 30–40. Morning of work.

**Open questions for T2.1 design phase** (parked, not meta-spec decisions):

- Do folds 1–3 in early-season produce consistently higher variance?
- Should we use slightly different N values per league (Bundesliga = 34
  matchdays, not 38)?
- Worth comparing 3×6 to 4×5 or 2×9 parametrizations on actual data.

---

## Section 3 — Quality Gate Enforcement (defense in depth)

**Gates** (from `TrainingConfig`, all measured on CV mean):

- `max_rps`: 0.21
- `max_brier`: 0.22
- `min_draw_f1`: 0.25 (diagnostic-only post-amendment 2026-05-04 — see §1.6 and "Diagnostic-only treatment" below)

### Tier 1 — `train.py` exit gate (primary defense)

- After CV + holdout eval, check all gates against CV mean.
- Save artifacts (for inspection).
- `sys.exit(1)` on any failure with verbose output (explicit threshold and
  actual value per failure).
- Fires on every manual retrain.

### Tier 2 — `tests/test_quality_gates.py` (safety net)

- Asserts CV mean from `eval_ensemble.json` against gates.
- Skips if eval JSON missing (same pattern as `test_no_leakage.py`).
- Validates schema (`cv.aggregate` block present, top-level `gates` block
  present, `schema_version == "cv_report.v1"`).
- Warns if eval JSON > 30 days old (warn, not fail).
- Runs in CI in seconds, on every push.

### Tier 3 — `predict.py` warning hook (last-chance)

- Loads eval, logs prominent warning if gates breached.
- Does NOT block predictions (would be too disruptive at deploy time).
- Logs prominently for monitoring; optional alerting hook for Phase 6+.

### Diagnostic-only treatment of `min_draw_f1` (amendment 2026-05-04)

The `min_draw_f1 ≥ 0.25` gate is reclassified to diagnostic-only post-T2.2
closure (see §1.6 and `MODEL_REVIEW.md §6.8`).

- **Tier 1**: reports the actual `draw_f1` value alongside `max_rps` /
  `max_brier`, but does NOT `sys.exit(1)` if only `min_draw_f1` fails.
  Tier 1 exit is gated on `max_rps` and `max_brier` only.
- **Tier 2**: keeps the threshold present in `TrainingConfig` and visible
  in `CVReport.gates` for regression detection, but skips the `min_draw_f1`
  assertion (with inline comment citing §6.8).
- **Tier 3**: unchanged — warning-only.
- **Re-promotion to merge-blocking requires meta-spec amendment.**

Implementation of these Tier 1 / Tier 2 changes is out of scope for this
amendment (which is meta-spec only); the change is owned by the next ticket
that retrains the model under the post-T2.2 baseline.

---

## Section 4 — Module Layout & Shared Interfaces

### Files created in T2.1 (consumed by every later ticket)

```
backend/evaluation/splits.py     # NEW — WalkForwardSplit class (Section 2 spec)
backend/evaluation/cv.py         # NEW — orchestration:
                                 #   run_walk_forward(model_factory, X, y, splitter)
                                 #   returns CVReport (per-fold + aggregate metrics)
backend/evaluation/cv_report.py  # NEW — CVReport dataclass + JSON serialization
                                 #   schema described in Section 5
```

### Modified in T2.1 (becomes the new training entrypoint)

```
backend/models/train.py          # MODIFIED — replaces single-split with walk-forward:
                                 #   1. Build WalkForwardSplit over 2021–2023
                                 #   2. For each fold: fit each enabled model, calibrate, eval
                                 #   3. Aggregate per-fold metrics → CVReport
                                 #   4. Retrain on full 2021–2023 → eval on locked 2024 holdout
                                 #   5. Tier 1 gate check; sys.exit(1) on breach
                                 #   6. Save artifacts only if gate passes (T2.2 strict mode)
```

### Per-ticket additions (after T2.1)

| Ticket | New files | Modified files | Why |
|--------|-----------|----------------|-----|
| T2.2 | `backend/training/draw_handling.py` (skeleton: `resample`, `class_sample_weights`); `backend/tools/validate_smote_classweight_composition.py` (6×6 ablation harness); `backend/data/output/smote_classweight_ablation.json` (evidence JSON) | `requirements.txt` (`imbalanced-learn`); model `fit()` signatures (`sample_weight` plumbing) | Ablation evidence; design superseded — see §1.6 |
| T2.3 | `backend/features/pi_ratings.py` | `features/build.py` (register), `feature_config.yaml` (toggle) | New feature builder |
| T2.4 | `backend/models/catboost_model.py` | `models/train.py` (register), `model_config.yaml` (config block) | New base model |
| T2.5 | `backend/training/oof_generator.py` — `generate_oof(model_factories, X, y, splitter) → oof_df`. Orchestrates OOF prediction generation across CV folds; reuses `WalkForwardSplit` from T2.1. | `backend/models/train.py` — adds stacking branch:<br>`if config.ensemble.method == 'stacking':`<br>`  oof = generate_oof(...)`<br>`  meta = LogisticRegression().fit(oof, y)`<br>`  base_models = [f().fit(X, y) for f in factories]`<br>`  model = StackingEnsemble(base_models, meta)` | `backend/models/ensemble.py::StackingEnsemble` — **inference-time only**. Holds `{base_models, meta_learner}`. `predict_proba()`: base preds → meta preds. **No training logic.** |

T2.5 explicitly does NOT live entirely inside `ensemble.py`. Training-loop
orchestration belongs in `oof_generator.py` + `train.py`.

### Shared interface contracts (locked by T2.1)

#### 1. `WalkForwardSplit.split()`

```python
class WalkForwardSplit:
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: pd.Series | None = None,  # ignored — see note
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_idx, val_idx) pairs for within-season walk-forward CV.

        DEVIATION FROM sklearn.BaseCrossValidator:
            sklearn's convention is that group structure flows in via `groups`
            (1-D array). Within-season walk-forward needs two-dimensional
            ordering — season AND matchday — which a 1-D array can't carry.
            So we read directly from X['season'] and X['matchday']. The
            `groups` parameter is kept only so this slots into sklearn-style
            tooling that calls `splitter.split(X, y, groups)`. It is ignored.

        Required X columns:
            - 'season' (e.g., '2021-22'): used to partition folds across seasons
            - 'matchday' (int 1..N): used for within-season ordering
        """
```

#### 2. `CVReport` dataclass

- Per-fold: `{fold_id, season, n_train, n_val, train_matchdays, val_matchdays,
  class_distribution_val, guards, metrics: {rps, brier, log_loss, accuracy,
  draw_f1, draw_precision, draw_recall}}`.
- Aggregate: `{mean, std, min, max}` over all 9 folds for each metric.
- Holdout: `{season, snapshot_locked_at, n_test, match_id_hash,
  match_ids_first_5, metrics}` — see Section 5.
- `.to_json()` matches schema in Section 5.

#### 3. `run_walk_forward(model_factory, X, y, groups, splitter) → CVReport`

`model_factory` is a **callable returning a fresh untrained model** — required
so each fold gets a clean fit, no state bleed. This is the function T2.4/T2.5
plug into when adding new base models.

#### 4. Tier 1 gate signature

`train.py` calls `cv_report.assert_gates(training_config)` after the report is
built. Raises `QualityGateFailure` with verbose per-metric breakdown; `main()`
catches and `sys.exit(1)`.

### No backwards-compat for the single-split mode

Walk-forward replaces the existing `_time_split`. Reasoning: keeping both
paths means every later ticket maintains two test surfaces, doubling the
testing burden across 5 tickets. Single-split is recoverable from git history
if ever needed.

### Feature schema versioning (NEW contract)

A single source-of-truth constant, bumped each time the feature set changes:

```python
# backend/features/build.py
FEATURE_SCHEMA_VERSION = "2.0"  # Phase 1 baseline. Bumped per ticket below.
```

**Bumping cadence (locked)**:

| Phase 2 ticket | Bumps version to | What changed |
|----------------|------------------|--------------|
| T2.1 | (no bump) | CV change is training-only, feature set unchanged |
| T2.2 | (no bump — closed before feature-add commit; see §1.6) | Three planned features did not land |
| T2.3 | `"2.1"` | Adds 4–5 pi-rating features (cadence shifted post-T2.2 closure: was `"2.2"`) |
| T2.4 | (no bump) | New base model, no new features |
| T2.5 | (no bump) | Ensemble change only |

**Artifact contract** (T2.1 introduces this; every later ticket maintains it):

```python
artifact = {
    'model': fitted_pipeline,
    'feature_schema_version': FEATURE_SCHEMA_VERSION,
    'feature_columns': feature_cols,           # ordered list
    'trained_at': datetime.now(timezone.utc).isoformat(),
    # ... plus existing fields
}
```

**Load-time validation** (T2.1 adds this guard to `predict.py`):

```python
if artifact['feature_schema_version'] != FEATURE_SCHEMA_VERSION:
    raise IncompatibleModelError(
        f"Model trained with feature schema {artifact['feature_schema_version']}, "
        f"current schema is {FEATURE_SCHEMA_VERSION}. Retrain required."
    )
```

This ships with T2.1 — paying the small upfront cost when nothing requires it
yet, so the contract is in place before T2.2's first bump exercises it.

---

## Section 5 — CVReport JSON Schema

Locked schema for `eval_ensemble.json` (and any other `eval_*.json` produced
under Phase 2). The Tier 2 quality-gates test validates against this shape.

```json
{
  "schema_version": "cv_report.v1",
  "model_name": "ensemble",
  "generated_at": "2026-04-27T15:42:00+00:00",
  "feature_schema_version": "2.0",

  "cv": {
    "splitter": "WalkForwardSplit",
    "n_folds": 9,
    "config": {
      "cv_seasons": ["2021-22", "2022-23", "2023-24"],
      "splits_per_season": 3,
      "fold_breakpoints_matchday": [14, 22, 30],
      "val_window_size": 6
    },

    "folds": [
      {
        "fold_id": 0,
        "season": "2021-22",
        "n_train": 140,
        "n_val": 60,
        "train_matchdays": "1..14",
        "val_matchdays": "15..20",
        "class_distribution_val": {"H": 28, "D": 16, "A": 16},
        "guards": {
          "n_val_below_threshold": false,
          "n_val_threshold": 50,
          "min_class_count_below_threshold": false,
          "min_class_count_threshold": 5,
          "min_class_observed": {"class": "D", "count": 16}
        },
        "metrics": {
          "rps": 0.2071,
          "brier": 0.2008,
          "log_loss": 1.0884,
          "accuracy": 0.5167,
          "draw_f1": 0.182,
          "draw_precision": 0.241,
          "draw_recall": 0.146
        }
      }
    ],

    "aggregate": {
      "rps":      {"mean": 0.2078, "std": 0.0042, "min": 0.2009, "max": 0.2156},
      "brier":    {"mean": 0.2014, "std": 0.0038, "min": 0.1953, "max": 0.2087},
      "log_loss": {"mean": 1.0901, "std": 0.0151, "min": 1.0712, "max": 1.1142},
      "accuracy": {"mean": 0.5172, "std": 0.0091, "min": 0.5009, "max": 0.5301},
      "draw_f1":        {"mean": 0.196, "std": 0.024, "min": 0.151, "max": 0.231},
      "draw_precision": {"mean": 0.250, "std": 0.031, "min": 0.198, "max": 0.298},
      "draw_recall":    {"mean": 0.162, "std": 0.027, "min": 0.117, "max": 0.207}
    }
  },

  "holdout": {
    "season": "2024-25",
    "snapshot_locked_at": "2026-04-27T15:42:00+00:00",
    "n_test": 1752,
    "match_id_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "match_ids_first_5": ["match_001", "match_002", "match_003", "match_004", "match_005"],
    "metrics": {
      "rps": 0.2069,
      "brier": 0.2006,
      "log_loss": 1.0872,
      "accuracy": 0.5171,
      "draw_f1": 0.0860,
      "draw_precision": 0.231,
      "draw_recall": 0.053
    }
  },

  "gates": {
    "config": {
      "max_rps": 0.21,
      "max_brier": 0.22,
      "min_draw_f1": 0.25
    },
    "checked_against": "cv.aggregate",
    "results": {
      "max_rps":     {"threshold": 0.21, "actual": 0.2078, "passed": true},
      "max_brier":   {"threshold": 0.22, "actual": 0.2014, "passed": true},
      "min_draw_f1": {"threshold": 0.25, "actual": 0.196,  "passed": false}
    },
    "all_passed": false
  },

  "calibration": {
    "method": "isotonic",
    "cv_folds": 5,
    "ece_per_class": {"home": 0.018, "draw": 0.041, "away": 0.022}
  },

  "feature_importances": [
    {"feature": "elo_diff", "importance": 0.087}
  ]
}
```

### Key design choices

1. **`schema_version: "cv_report.v1"`** is independent from
   `feature_schema_version`. Bumped only if the JSON shape itself changes.
   Phase 2 tickets do NOT bump it; only meta-changes do.

2. **`gates.checked_against: "cv.aggregate"`** is explicit — not
   `holdout.metrics`. Per Section 3, gates fire on CV mean. Holdout is the
   trust signal but not the gate input.

3. **`gates.results` is fine-grained per metric** — not a single bool. Tier 1
   verbose error output reads from this structure: "min_draw_f1 failed:
   threshold 0.25, actual 0.196." Removes any "which gate broke?" guesswork.

4. **`folds[].train_matchdays` / `val_matchdays`** as human-readable strings
   (`"1..14"`, `"15..20"`). The canonical row indices are reconstructible from
   `WalkForwardSplit` config.

5. **`calibration.ece_per_class`** is added (Phase 1 had only a flat summary).
   Per-class ECE is what makes "draws are systematically over/under-predicted"
   diagnosable — exactly what T2.2 targets, and the failure mode the metric
   guards against (SMOTE/class-weight tuning that increases draw F1 but also
   increases draw ECE).

6. **`feature_schema_version`** at top level — same field as the model
   artifact (Section 4), so eval JSON and saved model agree at load time.

### Per-fold guards

- `n_val_below_threshold`: true if `n_val < 50`. Surfaces in JSON, does NOT
  fail the run.
- `min_class_count_below_threshold`: true if any class has < 5 samples in val
  fold (especially for draws — fold with 2 draws produces meaningless
  draw F1).
- `min_class_observed`: names which class triggered (almost always 'D'
  early-season).

Aggregate-level guards live in `WalkForwardSplit` code, not duplicated in
schema.

### Holdout snapshot lock

T2.1 commits `backend/data/models/holdout_snapshot.json` (sorted match_ids +
SHA256 hash + timestamp). ~50KB. Bumping it requires a deliberate decision
recorded in `MODEL_REVIEW.md`.

**Why filter rather than fail-by-default**: as 2024-25 progresses, real new
matches will exist. Hard fail would mean every Phase 2 ticket waits for end
of season — no good. The filter lets new live data exist for *prediction*
purposes while keeping the *evaluation* surface fixed. The hard fail only
triggers if matches *disappear* (data drift / re-ingestion bug), which IS a
problem worth blocking on.

```python
# In train.py, before evaluating on holdout:
holdout_df = features_df[features_df["season"] == "2024-25"]
holdout_df = holdout_df[holdout_df["result"].notna()]    # Completed only
holdout_df = holdout_df.sort_values("match_id")

current_hash = sha256(",".join(holdout_df["match_id"]).encode()).hexdigest()

ref = json.loads((MODELS_DIR / "holdout_snapshot.json").read_text())

if current_hash != ref["match_id_hash"]:
    holdout_df = holdout_df[holdout_df["match_id"].isin(ref["match_ids"])]
    if len(holdout_df) != ref["n_test"]:
        raise HoldoutDriftError(
            f"Cannot reconstruct snapshot: live corpus has {len(holdout_df)} of "
            f"{ref['n_test']} required matches. Reconcile before proceeding."
        )
```

### Tier 2 test contract

```python
# tests/test_quality_gates.py
def test_quality_gates():
    eval_path = OUTPUT_DIR / "eval_ensemble.json"
    if not eval_path.exists():
        pytest.skip("eval_ensemble.json missing - run training first")

    report = json.loads(eval_path.read_text())

    # Schema validation
    assert report["schema_version"] == "cv_report.v1"
    assert "cv" in report and "aggregate" in report["cv"]
    assert "gates" in report

    # Staleness warning (warns, doesn't fail)
    age = datetime.now(tz) - datetime.fromisoformat(report["generated_at"])
    if age > timedelta(days=30):
        warnings.warn(f"Eval JSON is {age.days} days old - consider retraining")

    # Hard gate
    assert report["gates"]["all_passed"], (
        f"Quality gates breached:\n" + format_gate_failures(report["gates"]["results"])
    )
```

### Backward-compat note

T2.1's first run *replaces* the Phase 1 single-split JSON. The Tier 2 test
hard-fails on a Phase 1 JSON because `schema_version` won't match — that's
intentional, forces a retraining run before any Phase 2 work proceeds.

---

## Section 6 — Dependencies, DoD, Phase 4 Prep

### Dependency additions

Each ticket adds its own deps; pinned with upper bounds matching Phase 1
hygiene.

| Ticket | Library | `requirements.txt` line | Rationale |
|--------|---------|------------------------|-----------|
| T2.1 | (none) | — | Stdlib + pandas/numpy/sklearn already present |
| T2.2 | `imbalanced-learn` | `imbalanced-learn>=0.12,<1` | SMOTE + ImbPipeline. Compatible with sklearn ≥ 1.2. |
| T2.3 | `penaltyblog` | `penaltyblog>=1.0,<2` | Provides `PiRatingSystem`. Permissive license, maintained. *If* the dep proves heavy or unmaintained mid-T2.3, fallback is hand-rolling pi-ratings (~80 lines) — flagged as in-ticket decision, not a meta-spec one. |
| T2.4 | `catboost` | `catboost>=1.2,<2` | Native categorical handling. ~150 MB install but Python-only wheels are clean. |
| T2.5 | (none) | — | sklearn `LogisticRegression` is the meta-learner; already a transitive dep. |

**Dev dependency** (added once, in T2.1):

- `scipy>=1.10,<2` — pin explicitly because we'll use it for confidence-
  interval calcs in `aggregate.std`.

### Phase 2 Definition of Done

- [x] **T2.6 ✅ committed** — `010a711` (`docs(t2.6): document Phase 2 non-goals`)
- [x] **T2.1**:
  - [x] `WalkForwardSplit` produces 9 folds (3 seasons × 3 splits, matchday
        breakpoints `[14, 22, 30]`, val window 6) *(see `MODEL_REVIEW.md §6.7` for parametrization deviation)*
  - [x] `CVReport` schema (`cv_report.v1`) lands as documented in §5
  - [x] Holdout snapshot file `backend/data/models/holdout_snapshot.json` committed
  - [x] `feature_schema_version = "2.0"` constant lives in `backend/features/build.py`
  - [x] Tier 1 gate active in `train.py` (`sys.exit(1)` on breach)
  - [x] Tier 2 gate test `tests/test_quality_gates.py` lands
  - [x] Tier 3 warning hook in `predict.py`
  - [x] All 19 existing tests still green; leakage gate still passes
- [x] **T2.2**: Closed as ablation evidence (PR #2, merged 2026-05-02). Original three-mechanism design superseded; see §1.6 and `MODEL_REVIEW.md §6.8` for the load-bearing retrospective. The `min_draw_f1 ≥ 0.25` gate is reclassified to diagnostic-only (see §3). Re-promotion to merge-blocking requires meta-spec amendment.
- [ ] **T2.3**:
  - [ ] Pi-rating features added; `feature_schema_version` bumped to `"2.1"` (cadence shifted post-T2.2 closure; see §1.6)
  - [ ] Folded RPS satisfies §2's "real or ambiguous" improvement classification
        (regression blocks merge per §1 quality gate protocol)
  - [ ] **SHAP shows pi-rating features collectively account for ≥ 5% of total
        feature importance** (sum of `|SHAP value|` across pi-rating columns ÷
        sum across all features). Rationale: published research (Bunker et al.
        2024) shows pi-ratings comparable to or better than Elo. If they have
        < 5% contribution, the pi-rating builder is likely buggy. This
        threshold catches implementation bugs without over-constraining.
        Iterate or investigate before merging.
- [ ] **T2.4**:
  - [ ] CatBoost in stack with own model_config block
  - [ ] All three base models (XGB / LGBM / CatBoost) train cleanly per fold
  - [ ] Folded RPS satisfies §2's "real or ambiguous" improvement classification
        (regression blocks merge per §1 quality gate protocol)
- [ ] **T2.5**:
  - [ ] `oof_generator.py` + `StackingEnsemble` rewrite + `train.py` stacking
        branch all in
  - [ ] Stacked ensemble does NOT regress vs weighted-average ensemble per §2's
        improvement classification:
    - Either: stacked CV mean RPS ≤ weighted-average CV mean RPS, OR
    - (stacked − weighted-average) < 1 std across folds.

    Stacking is expected to improve (research shows 2–5%); ties or marginal
    improvement within noise are acceptable. Guards against: meta-learner
    overfitting on small CV samples degrading the ensemble.
  - [ ] No sklearn deprecation warnings (the `multi_class` arg is gone from
        the rewrite)
- [ ] **Cross-cutting**:
  - [ ] `MODEL_REVIEW.md` updated with new metrics + final folded baseline
  - [ ] `IMPLEMENTATION_PLAN_v2.md` updated with completion status (mirrors
        Phase 1 tracking)
  - [ ] **Phase 2 closure retro in `MODEL_REVIEW.md`**, addressing:
    1. Final folded baseline vs Phase 1 baseline (table).
    2. Per-ticket impact attribution: which ticket moved which metric most?
    3. Surprises: anything that didn't work as expected, or worked better.
    4. Carryovers: known issues deferred to Phase 4 (with ticket numbers).
    5. Quality gate history: were any tickets close to failing? Which?

    Structured 1-page document. More useful for Phase 3 planning and future
    retrospection than "we did stuff, it was fine."

### Final commit-set shape for Phase 2

| Status | Commit | Subject |
|--------|--------|---------|
| ✅ DONE | `010a711` | `docs(t2.6): document Phase 2 non-goals` |
| ✅ DONE | `c4dc674` | `fix(ingestion): derive matchday from date order` |
| ✅ DONE | `168786b` | `feat(t2.1): empirical CV parametrization harness` |
| ✅ DONE | `69b3356` | `feat(t2.1): WalkForwardSplit + matchday validators` |
| ✅ DONE | `d78c261` | `feat(t2.1): CVReport schema (cv_report.v1)` |
| ✅ DONE | `7774bc9` | `feat(t2.1): exceptions + Tier 2/3 quality gates` |
| ✅ DONE | `b845e7c` | `feat(t2.1): wire walk-forward CV into train.py` |
| ✅ DONE | `ce7b739` | `chore(t2.1): seed holdout snapshot` |
| ✅ DONE | `048759f` | `docs(t2.1): baseline shift retrospective` |
| ✅ DONE | `446b7c6` | `docs(t2.2): draw-class handling design` |
| ✅ DONE | `ff25e56` | `feat(t2.2): add imbalanced-learn + draw_handling.py skeleton` |
| ✅ DONE | `2845e5f` | `feat(t2.2): SMOTE+class_weight ablation harness + sample_weight plumbing` |
| ✅ DONE | `b5cb683` | `fix(t2.2): persist ablation evidence JSON on HarnessFailure` |
| ✅ DONE | `15078e3` | `fix(t2.2): rename failure_reason -> harness_failure_reason in evidence JSON` |
| ✅ DONE | `37ae2b0` | `chore(t2.2a): land 6x6 ablation evidence JSON (HarnessFailure)` |
| ✅ DONE | `44ba448` | `docs(t2.2): three-mechanism ablation retrospective and path forward` |
| ✅ DONE | `29fb556` | `docs(t2.2): supersede T2.2 implementation plan; move to plans/superseded/` |
| pending | (T2.3) | `feat(t2.3): pi-ratings alongside Elo` |
| pending | (T2.4) | `feat(t2.4): CatBoost base model` |
| pending | (T2.5) | `feat(t2.5): stacking meta-learner` |
| pending | (closure) | `docs: Phase 2 closure metrics + retro` |

---

## Cross-references

- `IMPLEMENTATION_PLAN_v2.md` § Phase 2 — original ticket descriptions (this
  meta-spec refines and locks them).
- `MODEL_REVIEW.md` § 6.6 — Phase 2 non-goals (T2.6, committed).
- `MODEL_REVIEW.md` § 0 — Phase 1 baseline.
- `backend/config/schema.py::TrainingConfig` — gate thresholds live as class
  defaults; meta-spec adds enforcement.
- `tests/test_no_leakage.py` — leakage gate kept green after every retraining.
