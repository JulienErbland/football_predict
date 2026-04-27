# Metrics Convention Changelog

## 2026-04-26 — T0.1d Form-Feature Alignment Correctness Fix

> ⚠️ **This is a correctness fix, not a performance fix.** The numbers in this
> entry are NOT directly comparable to the pre-T0.1d baseline — both metric
> blocks evaluate on the same 2024 holdout, but the model behind each block was
> trained on a different feature table. They measure two different models.

**Bug**: `backend/features/form.py` ended every rolling-stat assignment with
`rolled[col].values` after a `groupby(...).rolling().mean()` chain.
`groupby + rolling` returns rows ordered by **team group**, but
`all_team_matches` (and the local `df` in `_rolling_team_stats`) is sorted by
**date**. The `.values` strips the index and assigns positionally, scattering
each team's rolling values into rows belonging to other teams. Effect was
present in **both training (one batch call) and inference (per-match call)**,
but the row-ordering dependence meant the two paths produced *different* wrong
answers for the same match.

**The fix** (`backend/features/form.py`, build_form_features + _rolling_team_stats):

```python
# Before
all_team_matches[f"w{w}_win_rate"] = rolled["is_win"].values   # positional

# After
all_team_matches[f"w{w}_win_rate"] = rolled["is_win"]           # index-aligned
```

Twelve assignments in `build_form_features`, eight more in the dead-code
`_rolling_team_stats` helper (kept consistent with the live path).

**Detection**: `tools/dataset_health_check.py` §17–§18 plus
`backend/tests/test_form_alignment.py`. Manual ground truth for Arsenal's last
10 PL matches (window=3 PPG) was computed by walking the team's matches
chronologically and accumulating points. Pre-fix the batch path diverged from
ground truth by up to **1.333 PPG** (67% relative error on individual rows);
post-fix divergence is **0.000000** for both batch and per-match.

**Verification**:
- `backend/tests/test_form_alignment.py::test_arsenal_w3_ppg_matches_manual_ground_truth`
  pins the ground-truth values so any regression of the alignment bug fails the
  suite immediately. Failed pre-fix, passes post-fix.
- All 19 tests in `backend/tests/` pass.
- `tools/dataset_health_check.py` writes `docs/DATASET_HEALTH_CHECK.md` with
  the full Arsenal triple-comparison; rerun after any change to `form.py`.

**Metrics — pre-fix vs post-fix (2024 holdout, ensemble; literature conv.)**:

| Metric    | Pre-T0.1d (form bug) | Post-T0.1d (form aligned) |
|-----------|----------------------|---------------------------|
| brier     | 0.2005               | 0.2006                    |
| rps       | 0.2065               | 0.2069                    |
| log_loss  | 1.1014               | 1.0872                    |
| accuracy  | 0.5126               | 0.5171                    |
| n_samples | 1752                 | 1752                      |

Numbers landed in the same neighborhood — GBMs absorbed the alignment noise as
weak features. The point of T0.1d is **truthfulness**, not Δmetrics: each form
column now genuinely represents the team it claims to.

**Scope discipline**:
- Touched: `form.py`, `tests/test_form_alignment.py`, regenerated
  `features.parquet`, `ensemble.pkl`, `xgboost.pkl`, `lightgbm.pkl`,
  `feature_cols.pkl`, `eval_ensemble.json`.
- Backed up to `backend/data/_backup_pre_t01d/` for rollback.
- **Not touched**: any other feature builder, model code, predict pipeline,
  evaluation logic, or front-end.

**Known follow-up — same anti-pattern in `xg_features.py`**:
`backend/features/xg_features.py:83-88` has the identical
`rolled[col].values` pattern. **Did not affect production** because the xG
builder requires StatsBomb data which isn't wired up — every build emits
`"xG features skipped — StatsBomb data unavailable"` and the columns are
absent from `features.parquet`. Fix the same way when xG data is integrated
(tracked in MODEL_REVIEW.md §Known Issues).

---

## 2026-04-24 — T0.2 Normalization to Literature Convention

**Changed**:
- `brier_score()`: now returns **mean** across K=3 classes (was: sum across classes).
  - Old range: [0, 2]. New range: [0, 1].
  - Random 3-class baseline ≈ 0.222 (was ≈ 0.667).
  - Published competitive band: 0.18–0.22.
- `rps()`: now returns **sum / (K-1)** over the K-1 thresholds (was: raw sum).
  - Old range: [0, K-1] = [0, 2]. New range: [0, 1].
  - Published competitive band: 0.19–0.23.

**Unchanged**:
- `log_loss_score()` — already in literature convention. Random 3-class baseline ≈ ln(3) ≈ 1.099.
- `accuracy()` — argmax accuracy, unchanged.

**Why**:
The Phase 0 audit (`MODEL_REVIEW.md` §0.2) found that repo-internal Brier and RPS were
~3× and ~2× the values quoted in academic football-prediction papers (Constantinou & Fenton
2012, Hubáček et al. 2019). The drift came from summing instead of averaging across the
K classes / K-1 thresholds. This made it impossible to compare our model's quality
against published baselines, and made the eventual T1.4 quality gate (`max_rps`,
`max_brier`) impossible to set against any external reference.

**Impact on persisted artifacts**:
- `backend/data/output/eval_*.json` files produced **before 2026-04-24** use the old
  sum-based convention. Multiply old Brier by ~1/3 and old RPS by ~1/2 to compare
  against the new convention values.
- `backend/data/output/eval_phase1_baseline.json` already carries both conventions
  side-by-side under `metrics_repo_convention` and `metrics_literature_convention` —
  the literature_convention block is the source of truth going forward.
- `backend/data/output/eval_ensemble.json` was regenerated on 2026-04-24 with the new
  convention as part of T0.2.

**Verification**:
- `backend/tests/test_metrics.py` pins down hand-computable values (Brier = 1/9 and
  RPS = 1/18 on the canonical two-sample fixture) so the old convention cannot be
  reintroduced without the test suite failing.
- Phase 1 baseline (2024 holdout, ensemble): Brier = 0.2005, RPS = 0.2065 — both
  inside the published competitive band, confirming the model itself is fine and the
  pre-T0.2 numbers were a presentation bug, not a quality issue.

**Downstream consumers** (all updated 2026-04-24):
- `tools/dashboard.py` — now imports `brier_score` and `rps` from `evaluation.metrics`
  (previously had local duplicate definitions that bypassed this module). Streamlit
  metric cards show the new ranges in their tooltips.
- `backend/evaluation/report.py` — uses `evaluate_predictions()`, propagates automatically.
- `MODEL_REVIEW.md` — §0.2 baseline table updated to show only literature values.
- `IMPLEMENTATION_PLAN.md` — T1.4 quality gates (`max_rps=0.21`, `max_brier=0.22`) are
  set against the new convention.
