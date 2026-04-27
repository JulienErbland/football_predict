# Football Prediction SaaS — Complete Implementation Plan

> **Version**: 1.1 (Post-Audit Revision)
> **Last Updated**: 2026-04-24
> **Status**: Phase 0 complete; Phase 1 ready to execute
> **Timeline**: ~14–16 weeks (3–4 months at 5–10 hrs/week)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Strategic Vision](#strategic-vision)
3. [Business Model & Go-to-Market](#business-model--go-to-market)
4. [Technical Architecture](#technical-architecture)
5. [Phase 0: Ground Truth Audit](#phase-0-ground-truth-audit) ✅ COMPLETE
6. [Phase 1: Foundation & Critical Fixes](#phase-1-foundation--critical-fixes) ← **START HERE**
7. [Phase 2: ML Upgrades (RPS-Impact Ordered)](#phase-2-ml-upgrades-rps-impact-ordered)
8. [Phase 3: Narrative Generator + Weather](#phase-3-narrative-generator--weather)
9. [Phase 4: Supabase Foundation](#phase-4-supabase-foundation)
10. [Phase 5: FastAPI Premium API](#phase-5-fastapi-premium-api)
11. [Phase 6: Stripe + Tier Gating + Launch](#phase-6-stripe--tier-gating--launch)
12. [Phase 7: V2 Stacked Ensemble (Optional)](#phase-7-v2-stacked-ensemble-optional)
13. [Non-Negotiable Production Checklist](#non-negotiable-production-checklist)
14. [Appendix: Research Foundation](#appendix-research-foundation)

---

## Executive Summary

**What we're building**: A freemium football (soccer) match prediction SaaS for Top 5 European leagues.

**Free tier**: Win/Draw/Loss probabilities + narrative explanations ("Arsenal favored due to pi-rating edge, 3 days extra rest, and momentum over last 5 matches"). Goal: SEO + viral social cards.

**Premium tier** (~€9.99–14.99/mo): Everything in free + live bookmaker odds comparison, Expected Value (EV) highlighting, "upset radar," historical backtest ROI, Kelly stake sizing.

**Current state**: Working ML pipeline (XGBoost + LightGBM ensemble, nightly GitHub Actions → static JSON → Vercel), but production predictions are broken due to team-name join mismatch. Audit complete. Ready to fix and upgrade.

**Strategy**: Hybrid architecture — keep static-JSON free tier (fast, cheap, already works), add Supabase + FastAPI **alongside** for Premium (user accounts, live odds, backtest). Improve model surgically: walk-forward CV, draw-class handling (SMOTE + threshold tuning), pi-ratings, CatBoost, stacking meta-learner.

**Success metrics**:
- RPS < 0.21 on held-out test (competitive with 2024 research benchmarks)
- Draw F1 ≥ 0.25 (deployment gate — current model fails this)
- Premium tier shows positive backtest ROI at EV ≥ 5%

---

## Strategic Vision

### The Moat

Accuracy alone doesn't sell — bookmaker odds already imply ~67% accuracy on EPL matches. Our moat is:

1. **Explainability for free users** — SHAP-driven narratives that make predictions feel earned, not black-box
2. **EV discovery for paid users** — surfacing value bets where our model's probabilities exceed bookmaker-implied odds
3. **Transparency** — public prediction tracker showing hit rate, RPS, calibration; competitors hide their misses

### Positioning

**Not a tipster service.** We're an analytics tool. This matters for French/EU gambling law compliance. No guaranteed outcomes, 18+ age gate, responsible gambling links prominent, legal review before Premium launch.

---

## Business Model & Go-to-Market

### Tier Structure

| Tier | Price | Features | Conversion Trigger |
|------|-------|----------|-------------------|
| **Free** | €0 | Top 5 leagues, next 7 days predictions, probabilities + confidence, narrative explainer (fatigue/momentum/weather badges), form graphs, head-to-head | Shareable "fixture breakdown" cards (SVG), SEO via per-match landing pages |
| **Premium** | €9.99–14.99/mo | Everything in Free + live bookmaker odds, EV column, upset radar (high-confidence long-odds), Kelly stake sizing, historical backtest, all leagues beyond Top 5 | Blurred "EV edge" panel on free fixture pages; "Upgrade to see which side and by how much" |
| **Pro** (optional) | €29.99/mo | Everything in Premium + API access, raw probability exports, custom feature weights, multi-match parlay EV, CSV downloads | Thin market but high LTV |

### Go-to-Market: The Narrative Flywheel

1. **Pre-matchday content**: Auto-generate shareable fixture cards ("Liverpool vs Man City: Fatigue Edge LIV +2 days, Momentum MCI +1.3 xGD") → Twitter/X + Reddit r/soccer
2. **Post-matchday accountability**: Public prediction tracker ("7/10 this weekend, RPS 0.188") → builds trust faster than marketing spend
3. **SEO long tail**: Every fixture gets a landing page with narrative ("Why Arsenal are favored — Matchday 14") → cumulative, compounding
4. **Free-to-premium conversion**: Right before kickoff, show blurred EV panel → convert at moment of highest intent

---

## Technical Architecture

### Hybrid Model (Free + Premium)

```
┌─────────────────────────────────────────────────────────────────┐
│                         FREE TIER                                │
│  Nightly GitHub Actions → predictions.json → Vercel             │
│  (Static, fast, nearly free, already exists)                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  │ writes predictions to both
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREMIUM TIER (NEW)                          │
│  Supabase (users, subscriptions, odds_snapshots, value_picks)   │
│  + FastAPI (Vercel Python serverless or Fly.io)                 │
│  + Hourly odds refresh → EV calculation → backtest engine       │
└─────────────────────────────────────────────────────────────────┘
```

### Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Frontend | Next.js 14+ (App Router) + TypeScript + Tailwind + shadcn/ui | You already have this; it's production-ready |
| Free-tier backend | Python 3.11+, XGBoost/LightGBM/CatBoost, GitHub Actions cron | Already working; we're upgrading in-place |
| Premium backend | Supabase (Postgres + Auth) + FastAPI | Managed DB + auth, scales trivially |
| Payments | Stripe Checkout + webhooks | Industry standard |
| Deployment | Vercel (frontend + Python serverless) | Single project, no CORS, shared env vars |
| ML libraries | scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn (SMOTE), shap, penaltyblog (pi-ratings) | Research-backed, production-ready |
| Data sources | football-data.org v4, StatsBomb (xG), Transfermarkt (squad), The Odds API | Already integrated minus weather |

### Key Architectural Decisions

1. **Static JSON stays** — don't rebuild what works; Premium is additive
2. **Time-series leakage prevention** — season-aware CV, chronological rating updates, `cutoff` param on every feature function, property-based tests
3. **Draw handling** — SMOTE in training pipeline, class weights, calibrated threshold θ_D ≠ 0.33
4. **Hybrid metrics** — RPS primary (literature-standard), Brier for calibration, MCC for multiclass discrimination, macro-F1 to surface draw collapse

---

## Phase 0: Ground Truth Audit

**Status**: ✅ **COMPLETE** (2026-04-24)

### Key Findings

1. **Production predictions are broken** — team-name join mismatch between `football_data_csv.py` ("Arsenal") and `football_data.py` ("Arsenal FC") causes every upcoming match to get default features → identical predictions (0.387/0.232/0.381, `home_elo=1500`) for all matches
2. **Reported metrics are untrustworthy** — measured on internally-consistent training data, but don't reflect the broken real-world pipeline
3. **Infrastructure gaps confirmed** — no tests directory, smoke test clobbers production artifacts, metrics use non-standard conventions, dependencies unpinned
4. **All 12 upgrade targets MISSING or PARTIAL** — walk-forward CV, SMOTE, pi-ratings, CatBoost, stacking, narratives, weather, Supabase, FastAPI, hourly odds, backtest, leakage tests

### What We Can Trust

- Pipeline runs end-to-end (smoke test passed)
- XGBoost + LightGBM ensemble trains and serializes correctly
- Existing code has thoughtful leakage-prevention patterns (`shift(1).rolling()`, chronological Elo) but no mechanical enforcement

### What We Cannot Trust (Until T0.1 Completes)

- Any RPS/Brier/F1 numbers from the audit
- Claims about "competitive with literature"
- The specific Draw F1 = 0.067 finding (might be real or artifact)

**Next action**: Phase 1, starting with T0.1 (fix the join, rebuild dataset, establish true baseline).

---

## Phase 1: Foundation & Critical Fixes

**Goal**: Fix production bug, establish trustworthy metrics baseline, add guardrails to prevent regressions, create test infrastructure for Phase 2 work.

**Duration**: ~3 weeks

---

### T0.1 🔴 CRITICAL — Team-name reconciliation

**Priority**: Do this first, before anything else.

**Why**: Production predictions are completely broken. Every upcoming match gets default features due to team-name mismatch.

**Impact on metrics**: The audit's RPS/Brier/F1 numbers are not trustworthy until this is fixed. After T0.1, re-run full evaluation on a clean held-out season to establish the true baseline for Phase 2.

**Scope**:
1. Create `backend/ingestion/name_normalizer.py` with canonical mapping:
   ```python
   TEAM_NAME_MAP = {
       # Premier League
       "Arsenal FC": "Arsenal",
       "Manchester United FC": "Man United",
       "Newcastle United FC": "Newcastle",
       # ... cover all teams in 5 leagues
   }
   
   def normalize_team_name(name: str) -> str:
       """Canonical name used everywhere in the system."""
       return TEAM_NAME_MAP.get(name, name)
   ```
2. Apply normalization in every ingestion function:
   - `football_data.py:fetch_matches` → normalize before writing to parquet
   - `football_data.py:fetch_upcoming` → normalize before returning
   - `football_data_csv.py` → normalize after CSV read
   - `odds/fetcher.py` → normalize team names in odds data
3. **Rebuild entire dataset from scratch**:
   - Delete `backend/data/processed/all_matches.parquet`
   - Delete `backend/data/features/features.parquet`
   - Re-run ingestion with normalized names
   - Re-run feature engineering
4. Create regression test `backend/tests/test_name_consistency.py`:
   ```python
   def test_upcoming_teams_exist_in_historical():
       historical = pd.read_parquet("data/processed/all_matches.parquet")
       upcoming = fetch_upcoming()
       historical_teams = set(historical['home_team']) | set(historical['away_team'])
       upcoming_teams = set(upcoming['home_team']) | set(upcoming['away_team'])
       assert upcoming_teams.issubset(historical_teams)
   ```
5. Retrain from clean dataset
6. Generate new eval metrics on 2024 holdout
7. Document new numbers in `MODEL_REVIEW.md` as "Phase 1 baseline" — **this is the real starting point for Phase 2**
8. Regenerate `predictions.json` and spot-check 5 fixtures (probabilities must vary, Elo ratings must be team-specific)

**Definition of Done**:
- [ ] Name normalizer covers all teams in 5 target leagues
- [ ] Historical dataset rebuilt with canonical names
- [ ] Upcoming predictions show team-specific Elo (not all 1500)
- [ ] Probabilities vary between fixtures
- [ ] New eval metrics from clean 2024 holdout in `MODEL_REVIEW.md`
- [ ] Regression test passes; added to CI

---

### T0.2 ✅ DONE (2026-04-24) — Metrics convention alignment

**Why**: `evaluation/metrics.py` returned `brier_score` as sum-over-3-classes (≈3× literature) and `rps` as raw sum-over-2-thresholds. The plan's gates (`max_rps: 0.21`, `max_brier: 0.22`) are in literature convention. Must align.

**Adopted Option A** — normalised to literature convention. Gate values in T1.4 stand unchanged.

```python
def brier_score(y_true, y_proba):
    one_hot = np.zeros_like(y_proba); one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean((y_proba - one_hot) ** 2))           # mean across N×K, not sum

def rps(y_true, y_proba):
    # ... cumsum logic ...
    return float(np.mean(per_sample) / (num_classes - 1))     # divide by K-1
```

**Outcome on 2024 holdout** (regenerated `eval_ensemble.json`):
- Brier = **0.2005**, RPS = **0.2065** — both inside the published competitive band, confirming that the pre-T0.2 numbers (0.6014, 0.4130) were a presentation bug, not a model-quality problem.

**Definition of Done** — all checked:
- [x] `metrics.py` normalised + module docstring documents the convention
- [x] `backend/evaluation/METRICS_CHANGELOG.md` records the migration and impact on persisted artifacts
- [x] `backend/tests/test_metrics.py` pins down hand-computable values (Brier=1/9, RPS=1/18) so the old convention can't be reintroduced silently — 9/9 tests pass
- [x] `tools/dashboard.py` imports from `evaluation.metrics` (was: local duplicate definitions); metric tooltips show literature-convention ranges
- [x] `eval_ensemble.json` regenerated with new convention
- [x] `MODEL_REVIEW.md` §0.2 shows only literature values + historical note
- [x] T1.4 quality gates annotated against the new convention (see below)

---

### T1.0 🟡 REFACTOR — Reconcile existing `CLAUDE.md`

A `CLAUDE.md` exists on disk (untracked). Plan assumed it didn't.

**Scope**:
1. Diff existing vs plan-specified contents
2. Merge, with plan's rules taking precedence on conflicts
3. Commit

---

### T1.1 🔵 NEW — Test scaffolding

No `backend/tests/` exists. Phase 2+ gates require pytest infrastructure.

**Scope**:
1. Create `backend/tests/` with `__init__.py`, `conftest.py`
2. Add `pytest`, `pytest-cov`, `hypothesis` to new `requirements-dev.txt`
3. Add CI workflow `.github/workflows/tests.yml` running on PR
4. Seed with `test_config_loads.py` to verify infrastructure

---

### T1.2 🔴 CRITICAL — Sandbox the smoke test

`tools/pipeline_test.py` overwrites production artifacts. Audit had to manually backup/restore.

**Scope**:
1. Accept `--sandbox-dir` arg (default `tmp_smoke/`)
2. All paths honor sandbox override
3. Add `tmp_smoke/` to `.gitignore`
4. Add CI job running smoke test in sandbox mode

**Definition of Done**:
- [ ] `pipeline_test.py` doesn't touch `backend/data/models/`
- [ ] Smoke test runs in CI without side-effects

---

### T1.3 🔵 NEW — Property-based leakage test

**File**: `backend/tests/test_no_leakage.py`

```python
from hypothesis import given, strategies as st
import pytest

@given(perturbation=st.text(min_size=1))
def test_feature_vector_unchanged_by_post_kickoff_data(perturbation):
    """
    For a completed historical match, features computed with cutoff=kickoff
    must return identical values whether or not post-kickoff data exists.
    """
    # 1. Pick a completed match
    match = get_random_completed_match()
    cutoff = match.kickoff
    
    # 2. Compute features
    features_before = build_feature_vector(match.id, cutoff)
    
    # 3. Inject fake post-kickoff data (modify match stats)
    inject_fake_post_match_data(match.id, perturbation)
    
    # 4. Recompute features with same cutoff
    features_after = build_feature_vector(match.id, cutoff)
    
    # 5. Must be identical
    assert features_before == features_after
```

**This test is the most important quality gate in the repo.** If it ever fails, the model has silent leakage.

Run with ≥100 hypothesis examples in CI.

---

### T1.4 🟡 REFACTOR — Typed config accessors

`backend/config/loader.py` already loads YAML → dicts. Add typed Pydantic accessors alongside.

**File**: `backend/config/schema.py`

```python
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    class_weight_home: float = 1.0
    class_weight_draw: float = 2.5
    class_weight_away: float = 1.2
    draw_threshold_min: float = 0.18
    draw_threshold_max: float = 0.32
    min_draw_f1: float = 0.25
    # Quality gates — both expressed in the LITERATURE convention adopted in T0.2.
    # Brier here is mean across K=3 classes (range [0, 1]); RPS is the per-sample
    # cumulative score divided by K-1 (range [0, 1]). The Phase 1 baseline lands
    # at brier=0.2005 / rps=0.2065 on the 2024 holdout, so these gates are a
    # tight "do no harm" floor — Phase 2 work that regresses past them fails CI.
    max_rps: float = 0.21          # competitive band 0.19–0.23
    max_brier: float = 0.22        # competitive band 0.18–0.22
    smote_k_neighbors: int = 5
    cv_seasons: int = 3

class FeatureConfig(BaseModel):
    ewma_form_span: int = 5
    elo_k: int = 32
    pi_rating_lambda: float = 0.035
    pi_rating_gamma: float = 0.7
    enable_pi_ratings: bool = False
    enable_weather: bool = False

# ... etc
```

Keep backward compatibility — existing code reads dicts; add `from backend.config.schema import TrainingConfig; cfg = TrainingConfig(**model_config())` as new pattern.

---

### T1.5 🟡 REFACTOR — Dependency hygiene

**Issues**:
- No upper bounds, no lockfile
- `polars>=0.20` declared but never imported
- Runtime and dev deps mixed

**Scope**:
1. Remove `polars` from `requirements.txt`
2. Pin exact versions OR adopt `uv lock` / `pip-compile`
3. Split: `requirements.txt` (runtime) vs `requirements-dev.txt` (streamlit, plotly, shap, pytest, hypothesis, ruff, mypy)
4. Document in README

---

### T1.6 🟡 REFACTOR — Commit uncommitted working tree

12 modified files, 4 deleted frontend components, 14+ untracked at audit time.

**Scope**: Review with user, stage logical groups ("frontend redesign", "ingestion CSV", "docs"), commit. Clean baseline for Phase 2.

---

### Phase 1 Definition of Done

- [ ] T0.1: Team join fixed; live predictions vary; new baseline metrics documented
- [x] T0.2: Metrics convention aligned across repo + plan (2026-04-24)
- [ ] T1.0: `CLAUDE.md` committed
- [ ] T1.1: Test scaffolding + CI
- [ ] T1.2: Smoke test sandboxed
- [ ] T1.3: Leakage test runs in CI (≥100 examples)
- [ ] T1.4: Typed config accessors
- [ ] T1.5: Dependencies pinned/locked; dev/runtime split
- [ ] T1.6: Working tree clean

**Only then proceed to Phase 2.**

---

## Phase 2: ML Upgrades (RPS-Impact Ordered)

**Goal**: Improve model quality with surgical upgrades ranked by expected RPS impact. Each ticket produces a measurable delta vs Phase 1 baseline.

**Duration**: ~4–5 weeks

**Critical gate**: Draw F1 ≥ 0.25 on held-out test. If not met after T2.2, iterate on SMOTE/class weights before moving on.

---

### T2.1 🔵 NEW — Season-aware walk-forward CV

**Why first**: Current evaluation uses single train/test split (test=2024). Numbers are likely optimistic. Need honest baseline before claiming any "improvement."

**File**: `backend/evaluation/splits.py`

```python
class SeasonSplit:
    """
    Walk-forward splits over seasons.
    Example with 5 seasons and 3 CV folds:
      Fold 1: train={2021,2022}, val=2023
      Fold 2: train={2021,2022,2023}, val=2024
      Fold 3: train={2021,2022,2023,2024}, val=2025
    Final test: most recent season (held out from CV entirely).
    """
    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits
    
    def split(self, X, y, groups):
        # groups = season labels
        unique_seasons = sorted(set(groups))
        for i in range(self.n_splits):
            train_seasons = unique_seasons[:-(self.n_splits - i)]
            val_season = unique_seasons[-(self.n_splits - i)]
            train_idx = [j for j, s in enumerate(groups) if s in train_seasons]
            val_idx = [j for j, s in enumerate(groups) if s == val_season]
            yield train_idx, val_idx
```

**Refactor**: `backend/models/train.py` to use `SeasonSplit` instead of single split. Report mean ± std of RPS across folds. Mean becomes new baseline.

**Expected impact**: Your Phase 1 baseline RPS may get worse (more honest). This is good — it's the truth.

---

### T2.2 🟡 REFACTOR — Draw-class handling

**Why second**: Biggest user-trust issue. Models that never predict draws lose credibility fast.

**Changes to `backend/models/train.py`**:

1. **SMOTE in pipeline**:
   ```python
   from imblearn.pipeline import Pipeline as ImbPipeline
   from imblearn.over_sampling import SMOTE
   
   pipe = ImbPipeline([
       ('smote', SMOTE(sampling_strategy={'D': 'auto'}, k_neighbors=5)),
       ('model', XGBoostModel(...))
   ])
   pipe.fit(X_train, y_train)
   ```

2. **Class weights**:
   ```python
   # XGBoost
   sample_weights = np.array([
       config.class_weight_home if y == 'H' else
       config.class_weight_draw if y == 'D' else
       config.class_weight_away
       for y in y_train
   ])
   model.fit(X_train, y_train, sample_weight=sample_weights)
   
   # LightGBM (native support)
   model = LGBMClassifier(class_weight={
       'H': config.class_weight_home,
       'D': config.class_weight_draw,
       'A': config.class_weight_away
   })
   ```

3. **Draw threshold calibration**:
   ```python
   # After training, on OOF predictions:
   def optimize_draw_threshold(y_true, y_pred_proba):
       best_f1 = 0
       best_theta = 0.33
       for theta in np.arange(0.18, 0.32, 0.01):
           y_pred = apply_threshold(y_pred_proba, theta_D=theta)
           f1 = f1_score(y_true, y_pred, average='macro')
           if f1 > best_f1:
               best_f1 = f1
               best_theta = theta
       return best_theta
   
   theta_D = optimize_draw_threshold(y_val, model.predict_proba(X_val))
   # Store theta_D in model artifact
   ```

4. **At inference**:
   ```python
   probs = model.predict_proba(X)
   # Don't just argmax — use calibrated threshold
   if probs[1] > theta_D and probs[1] > probs[0] and probs[1] > probs[2]:
       pred = 'D'
   else:
       pred = 'H' if probs[0] > probs[2] else 'A'
   ```

**Add three draw-specific features** (`backend/features/context.py`):
- `h2h_draw_rate_last_5`: fraction of last 5 H2H that were draws
- `elo_diff_abs`: abs(home_elo - away_elo) — low values → draw-prone
- `defensive_match_indicator`: 1 if both teams have below-median xGA over last 10

**Deployment gate**: Held-out Draw F1 ≥ 0.25. If not met, iterate on SMOTE k_neighbors and class weights before merging.

---

### T2.3 🔵 NEW — Pi-ratings alongside Elo

**Why third**: Research literature shows pi-ratings outperform Elo on RPS. Cheap to add; keep Elo too.

**File**: `backend/features/pi_ratings.py`

```python
from penaltyblog.ratings import PiRatingSystem

def build_pi_ratings(matches: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """
    Compute pi-ratings for every team up to cutoff.
    Must be called chronologically (sorted by date).
    """
    system = PiRatingSystem(
        lambda_param=config.pi_rating_lambda,
        gamma=config.pi_rating_gamma
    )
    
    ratings = {}
    for _, match in matches[matches['date'] < cutoff].iterrows():
        # Update after match completes
        if pd.notna(match['home_goals']):
            system.update(
                match['home_team'],
                match['away_team'],
                match['home_goals'],
                match['away_goals'],
                home_game=True
            )
    
    return pd.DataFrame([
        {
            'team': team,
            'as_of': cutoff,
            'home_rating': system.get_rating(team, home=True),
            'away_rating': system.get_rating(team, home=False)
        }
        for team in system.teams
    ])
```

**Add to `backend/features/build.py`**:
- `home_pi_home`, `home_pi_away` (home team's ratings)
- `away_pi_home`, `away_pi_away` (away team's ratings)
- `pi_rating_delta` = `home_pi_home - away_pi_away` (key derived signal)

**Config toggle**:
```yaml
# feature_config.yaml
pi_ratings:
  enabled: true
  lambda: 0.035
  gamma: 0.7
```

Keep Elo enabled. The model decides which to trust via SHAP.

---

### T2.4 🔵 NEW — CatBoost base model

**Why**: Research shows CatBoost + rating features is state-of-the-art. Handles categoricals natively (useful for future referee, weather category).

**File**: `backend/models/catboost_model.py`

```python
from catboost import CatBoostClassifier
from .base import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(
            iterations=kwargs.get('iterations', 500),
            depth=kwargs.get('depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.05),
            loss_function='MultiClass',
            random_seed=42,
            verbose=False,
            **kwargs
        )
    
    def fit(self, X, y, **fit_params):
        # CatBoost accepts class_weights via sample_weight
        self.model.fit(X, y, **fit_params)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
```

**Add to `model_config.yaml`**:
```yaml
catboost:
  enabled: true
  weight: 0.33  # will be re-normalized with xgboost and lightgbm
  iterations: 500
  depth: 6
  learning_rate: 0.05
```

---

### T2.5 🟡 REFACTOR — Stacking meta-learner

**Why**: Replace fixed weights (0.60 XGB, 0.40 LGBM) with learned stacking. Research shows 2–5% RPS improvement.

**File**: `backend/models/stacking.py` (new)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from .base import BaseModel

class StackedEnsemble(BaseModel):
    """
    Level-0: XGBoost, LightGBM, CatBoost (each trained with CV → OOF preds)
    Level-1: LogisticRegression on OOF probabilities
    """
    def __init__(self, base_models: list[BaseModel], cv_splitter):
        self.base_models = base_models
        self.cv_splitter = cv_splitter
        self.meta_learner = LogisticRegression(
            max_iter=1000,
            random_state=42
        )  # NOTE: sklearn 1.8+ — do NOT pass multi_class arg (deprecated)
    
    def fit(self, X, y):
        # Generate OOF predictions from each base model
        oof_preds = []
        for model in self.base_models:
            oof = cross_val_predict(
                model, X, y,
                cv=self.cv_splitter,
                method='predict_proba'
            )
            oof_preds.append(oof)
        
        # Stack OOF probs as features for meta-learner
        X_meta = np.hstack(oof_preds)
        self.meta_learner.fit(X_meta, y)
        
        # Retrain base models on full data for inference
        for model in self.base_models:
            model.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        # All base models predict → stack → meta-learner
        base_preds = [model.predict_proba(X) for model in self.base_models]
        X_meta = np.hstack(base_preds)
        return self.meta_learner.predict_proba(X_meta)
```

**Update `train.py`**: instantiate `StackedEnsemble` if `config.stacking.enabled=true`.

**Config**:
```yaml
# model_config.yaml
stacking:
  enabled: true
  use_calibrated_oof: true  # apply isotonic before stacking
```

Keep `EnsembleModel` (fixed weights) for A/B comparison during rollout.

---

### T2.6 🟢 KEEP — Explicit non-goals

**Do NOT do** (per "partial upgrades" decision):
- Bivariate Poisson as a feature
- Neural networks / LSTMs / Transformers
- Player embeddings
- Bayesian hierarchical models
- Fixing disabled Dixon-Coles Poisson

Document in `MODEL_REVIEW.md` so future-you doesn't waste time.

---

### Phase 2 Definition of Done

- [ ] Walk-forward CV is default; mean RPS across folds documented as Phase 2 baseline
- [ ] Draw F1 ≥ 0.25 on held-out season (hard gate)
- [ ] Pi-ratings integrated; SHAP summary shows they're used
- [ ] CatBoost trained and in stack
- [ ] Stacked ensemble RPS ≤ weighted-average ensemble (non-regression)
- [ ] `MODEL_REVIEW.md` updated with new metrics + explicit non-goals

---

## Phase 3: Narrative Generator + Weather

**Goal**: Make free tier genuinely differentiated vs competitors. Narratives + badges are what make fixture cards shareable.

**Duration**: ~2–3 weeks

---

### T3.1 🔵 NEW — SHAP → narrative generator

**File**: `backend/output/narrative.py`

```python
NARRATIVE_TEMPLATES = {
    "pi_rating_delta": {
        "positive": "{home} rate {abs_value:.1f} points higher than {away} on our rating system.",
        "negative": "{away} come in with a {abs_value:.1f}-point rating advantage.",
    },
    "elo_diff": {
        "positive": "{home} have an Elo advantage of {value:.0f} points.",
        "negative": "{away} are favored by {abs_value:.0f} Elo points.",
    },
    "form_last_5_home": {
        "positive": "{home} are on {value:.0f} points from their last 5 matches.",
        "negative": "{home} have struggled recently with only {value:.0f} points from 5.",
    },
    "rest_days_delta": {
        "positive": "{home} have had {value:.0f} more days of rest than {away}.",
        "negative": "{away} are better rested by {abs_value:.0f} days.",
    },
    "ewma_xgd_home": {
        "positive": "{home} have outperformed xG by {value:+.1f} per match over their last 5 games.",
        "negative": "{home} are underperforming xG by {abs_value:.1f} per match recently.",
    },
    # ... one template per top feature
}

def generate_narrative(
    shap_values: dict,
    features: dict,
    teams: tuple[str, str],
    top_k: int = 3
) -> list[dict]:
    """
    Returns list of narrative bullets, structured:
      [
        {
          "feature": "pi_rating_delta",
          "direction": "positive",
          "sentence": "Arsenal rate 0.4 points above Chelsea...",
          "impact": 0.12
        },
        ...
      ]
    Frontend styles these.
    """
    home, away = teams
    
    # Sort features by absolute SHAP value
    ranked = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]
    
    narratives = []
    for feature, shap_val in ranked:
        if feature not in NARRATIVE_TEMPLATES:
            continue
        
        direction = "positive" if shap_val > 0 else "negative"
        template = NARRATIVE_TEMPLATES[feature][direction]
        
        value = features[feature]
        abs_value = abs(value)
        
        sentence = template.format(
            home=home,
            away=away,
            value=value,
            abs_value=abs_value
        )
        
        narratives.append({
            "feature": feature,
            "direction": direction,
            "sentence": sentence,
            "impact": round(shap_val, 3)
        })
    
    return narratives
```

**Extend `predictions.json` schema**:
```json
{
  "match_id": "...",
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "probs": {"H": 0.52, "D": 0.26, "A": 0.22},
  "narrative": [
    {
      "feature": "pi_rating_delta",
      "sentence": "Arsenal rate 0.4 points above Chelsea on our rating system.",
      "impact": 0.12
    },
    {
      "feature": "rest_days_delta",
      "sentence": "Chelsea are better rested by 3 days.",
      "impact": -0.04
    },
    {
      "feature": "form_last_5_home",
      "sentence": "Arsenal are on 11 points from their last 5 matches.",
      "impact": 0.07
    }
  ]
}
```

**Manual quality review**: Generate narratives for 50 sample predictions, review for naturalness, fix awkward templates.

---

### T3.2 🔵 NEW — OpenWeatherMap integration

**Files**: `backend/ingestion/weather.py`, `backend/features/weather_features.py`

**Stadium coordinates**: `backend/config/stadiums.yaml`
```yaml
stadiums:
  Arsenal: {lat: 51.5549, lon: -0.1084}  # Emirates Stadium
  Chelsea: {lat: 51.4817, lon: -0.1910}  # Stamford Bridge
  # ... all teams in 5 leagues
  # Fallback: city centroid if stadium unknown
```

**Ingestion** (`weather.py`):
```python
import httpx
from datetime import datetime, timedelta

async def fetch_weather_forecast(
    lat: float,
    lon: float,
    target_time: datetime
) -> dict:
    """
    Fetch OpenWeatherMap forecast for target_time.
    For upcoming matches, target = kickoff - 2 hours.
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": settings.openweather_api_key,
        "units": "metric"
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()
    
    # Find forecast closest to target_time
    forecasts = data['list']
    closest = min(
        forecasts,
        key=lambda f: abs(
            datetime.fromtimestamp(f['dt']) - target_time
        )
    )
    
    return {
        "temperature_c": closest['main']['temp'],
        "precipitation_mm": closest.get('rain', {}).get('3h', 0),
        "wind_speed_ms": closest['wind']['speed'],
        "conditions": closest['weather'][0]['main']
    }
```

**Features** (`weather_features.py`):
```python
def build_weather_features(match: dict) -> dict:
    weather = match.get('weather', {})
    
    # Raw features
    temp = weather.get('temperature_c', 15)  # default temperate
    precip = weather.get('precipitation_mm', 0)
    wind = weather.get('wind_speed_ms', 0)
    
    # Derived: weather impact score (0–1, higher = more extreme)
    temp_extreme = min(abs(temp - 15) / 20, 1)  # optimal ~15°C
    precip_impact = min(precip / 10, 1)
    wind_impact = min(wind / 15, 1)
    
    impact = (temp_extreme + precip_impact + wind_impact) / 3
    
    return {
        'temperature_c': temp,
        'precipitation_mm': precip,
        'wind_speed_ms': wind,
        'weather_impact_score': impact
    }
```

**Historical weather**: SKIP for V1. OpenWeather History API is paid. Add `TODO(v2)`.

**Config toggle**:
```yaml
# feature_config.yaml
weather:
  enabled: true
  lookhead_hours: 2
```

**Extend `predictions.json`**:
```json
"conditions": {
  "temperature_c": 8.2,
  "precipitation_mm": 2.1,
  "wind_speed_ms": 6.5,
  "summary": "Light rain, gusty wind — expect a tighter game."
}
```

**Narrative integration**: when `weather_impact_score > 0.6`, include a weather sentence in top-3 bullets.

---

### T3.3 🟢 KEEP — Fatigue already exists

`backend/features/context.py` already has `rest_days` and `congestion`. Confirm they appear in narrative templates.

**Add to `predictions.json`**:
```json
"badges": {
  "fatigue_edge": {"team": "home", "days_advantage": 3},
  "momentum_edge": {"team": "away", "form_delta_ppg": 0.8},
  "weather_flag": true
}
```

These badges are social-shareable atoms — what makes a screenshot worth retweeting.

---

### Phase 3 Definition of Done

- [ ] Narratives generated for every prediction (3 bullets each)
- [ ] Weather features integrated; SHAP shows non-zero contribution
- [ ] `predictions.json` extended with `narrative` + `badges`
- [ ] Frontend updated to render narratives and badges (minimal change)
- [ ] 50 sample narratives manually reviewed for quality

---

## Phase 4: Supabase Foundation

**Goal**: Add Premium-tier infrastructure without breaking free tier.

**Duration**: ~2 weeks

**Only start after Phase 3 deployed and free tier looks good.**

---

### T4.1 🔵 NEW — Supabase schema

**File**: `supabase/migrations/0001_initial_schema.sql`

```sql
-- Users and subscriptions
create table users (
  id uuid primary key references auth.users(id) on delete cascade,
  email text unique not null,
  tier text not null default 'free' check (tier in ('free','premium','pro')),
  stripe_customer_id text unique,
  created_at timestamptz default now()
);

-- Odds snapshots (hourly, Premium-only data)
create table odds_snapshots (
  id bigserial primary key,
  match_external_id text not null,   -- matches match_id in predictions.json
  bookmaker text not null,
  fetched_at timestamptz not null,
  odds_home numeric(6,3) not null,
  odds_draw numeric(6,3) not null,
  odds_away numeric(6,3) not null,
  unique (match_external_id, bookmaker, fetched_at)
);
create index idx_odds_match_fetched on odds_snapshots(match_external_id, fetched_at desc);

-- Value picks: materialized view refreshed after each odds pull
create table value_picks (
  id bigserial primary key,
  match_external_id text not null,
  computed_at timestamptz not null,
  best_bookmaker text not null,
  outcome char(1) not null check (outcome in ('H','D','A')),
  model_prob numeric(5,4) not null,
  best_odds numeric(6,3) not null,
  ev numeric(6,4) not null,
  kelly numeric(5,4) not null,
  confidence_tier text not null,
  unique (match_external_id, outcome, computed_at)
);
create index idx_value_picks_ev on value_picks(ev desc) where ev > 0;

-- Backtest results (Premium Pro tier)
create table backtest_runs (
  id bigserial primary key,
  model_version text not null,
  from_date date not null,
  to_date date not null,
  min_ev_threshold numeric(4,3) not null,
  stake_strategy text not null,       -- 'flat' | 'kelly_fractional'
  total_bets int not null,
  hit_rate numeric(5,4) not null,
  roi numeric(6,4) not null,
  max_drawdown numeric(6,4) not null,
  details_json jsonb,
  created_at timestamptz default now()
);
```

**Why no `matches`, `teams`, `predictions` tables?** They live in static JSON. Premium endpoints read `predictions.json` (small, cacheable) and join with Supabase tables by `match_external_id`.

**RLS**: Keep simple. `users` is user-scoped. `odds_snapshots` + `value_picks` readable by all authenticated; tier check at API layer (more flexible).

---

### T4.2 🔵 NEW — Supabase client

**File**: `backend/db/supabase_client.py`

```python
from supabase import create_client
from functools import lru_cache

@lru_cache
def get_client():
    return create_client(
        settings.supabase_url,
        settings.supabase_service_key
    )
```

**Pydantic models**: `backend/db/models.py` mirroring each table.

```python
from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    id: str
    email: str
    tier: str
    stripe_customer_id: str | None
    created_at: datetime

class OddsSnapshot(BaseModel):
    match_external_id: str
    bookmaker: str
    fetched_at: datetime
    odds_home: float
    odds_draw: float
    odds_away: float

# ... etc
```

Every read/write goes through typed accessors, never raw dicts.

---

### T4.3 🔵 NEW — Hourly odds ingestion

**File**: `.github/workflows/odds_refresh.yml`

```yaml
name: Hourly Odds Refresh
on:
  schedule:
    - cron: '0 * * * *'  # every hour

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r backend/requirements.txt
      - run: python backend/odds/refresh.py
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
```

**File**: `backend/odds/refresh.py`

```python
import json
from pathlib import Path
from backend.db.supabase_client import get_client
from backend.odds.fetcher import fetch_odds

def refresh_odds():
    # Load upcoming matches from predictions.json
    predictions = json.loads(
        Path("frontend/public/data/predictions.json").read_text()
    )
    
    upcoming_match_ids = [p['match_id'] for p in predictions['matches']]
    
    supabase = get_client()
    
    for match_id in upcoming_match_ids:
        odds_data = fetch_odds(match_id)
        
        for bookmaker, book_odds in odds_data.items():
            supabase.table('odds_snapshots').insert({
                'match_external_id': match_id,
                'bookmaker': bookmaker,
                'fetched_at': datetime.now(),
                'odds_home': book_odds['home'],
                'odds_draw': book_odds['draw'],
                'odds_away': book_odds['away']
            }).execute()
    
    recompute_value_picks()

if __name__ == '__main__':
    refresh_odds()
```

**Cost management**: The Odds API free tier is 500 req/month. Hourly × ~50 matches/week = ~8,500/month → need paid tier. OR filter to only next 48h to stay near free limit. Decision in `odds_config.yaml`.

---

### T4.4 🔵 NEW — Value-pick recomputer

**File**: `backend/odds/value_picks.py`

```python
def recompute_value_picks():
    predictions = load_predictions_json()
    latest_odds = get_latest_odds_per_match()  # from Supabase
    
    supabase = get_client()
    
    for match in predictions:
        for outcome in ['H', 'D', 'A']:
            best_book, best_odds = find_best_odds(
                latest_odds[match['match_id']],
                outcome
            )
            
            model_prob = match['probs'][outcome]
            ev = model_prob * best_odds - 1
            kelly = kelly_fraction(model_prob, best_odds)
            
            if ev > config.min_ev_threshold:
                supabase.table('value_picks').upsert({
                    'match_external_id': match['match_id'],
                    'computed_at': datetime.now(),
                    'best_bookmaker': best_book,
                    'outcome': outcome,
                    'model_prob': model_prob,
                    'best_odds': best_odds,
                    'ev': ev,
                    'kelly': kelly,
                    'confidence_tier': classify_confidence(model_prob, ev)
                }).execute()
```

---

### Phase 4 Definition of Done

- [ ] Supabase project provisioned; migration applied
- [ ] `odds_snapshots` populating hourly via workflow
- [ ] `value_picks` recomputed after each odds refresh
- [ ] EV > 0 picks surfaced in table
- [ ] Odds API cost projection documented; paid tier decision made

---

## Phase 5: FastAPI Premium API

**Goal**: Expose Premium-tier data via authenticated REST endpoints.

**Duration**: ~2 weeks

---

### T5.1 🔵 NEW — FastAPI app scaffold

**Directory**: `backend/api/`

```
backend/api/
├── main.py              # FastAPI app
├── deps.py              # Auth middleware, tier check, DB injection
├── routes/
│   ├── health.py
│   ├── odds.py          # /fixtures/{match_id}/odds
│   ├── value_picks.py   # /value-picks?min_ev=0.05
│   └── backtest.py      # /backtest?from=&to=&min_ev=
└── schemas.py           # Pydantic request/response models
```

**File**: `backend/api/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import health, odds, value_picks, backtest

app = FastAPI(title="Football Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

app.include_router(health.router)
app.include_router(odds.router)
app.include_router(value_picks.router)
app.include_router(backtest.router)
```

**Auth via Supabase JWT** (`deps.py`):

```python
from fastapi import Depends, HTTPException, Header
from jose import jwt
from backend.config import settings

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    
    token = authorization.split(" ")[1]
    
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"]
        )
        user_id = payload.get("sub")
        # Fetch user from Supabase
        user = get_user_by_id(user_id)
        return user
    except:
        raise HTTPException(401, "Invalid token")

def require_tier(min_tier: str):
    TIER_RANK = {'free': 0, 'premium': 1, 'pro': 2}
    
    def dep(user = Depends(get_current_user)):
        if TIER_RANK[user.tier] < TIER_RANK[min_tier]:
            raise HTTPException(403, "Upgrade required")
        return user
    return dep
```

---

### T5.2 🔵 NEW — Deployment target

**Option A (recommended)**: Vercel Python serverless under `/api/` path

```
api/
├── health.py        # handles GET /api/health
├── value_picks.py   # handles GET /api/value-picks
└── ...
```

Each file exports handler compatible with Vercel Python runtime.

**Option B**: Deploy FastAPI to Fly.io or Railway behind `api.yourdomain.com`

Decide once, implement consistently.

---

### T5.3 🔵 NEW — Endpoint set for launch

| Endpoint | Tier | Purpose |
|----------|------|---------|
| `GET /api/health` | public | Model version, last odds refresh timestamp |
| `GET /api/fixtures/{id}/odds` | premium | Current odds from all books + EV per outcome |
| `GET /api/value-picks?min_ev=&league=` | premium | Current value picks sorted by EV |
| `GET /api/backtest?from=&to=&min_ev=` | pro | Historical ROI simulation |

**Example**: `routes/value_picks.py`

```python
from fastapi import APIRouter, Depends
from backend.api.deps import require_tier
from backend.db.supabase_client import get_client

router = APIRouter()

@router.get("/value-picks", dependencies=[Depends(require_tier('premium'))])
def list_value_picks(min_ev: float = 0.05, league: str = None):
    supabase = get_client()
    
    query = supabase.table('value_picks').select('*').gte('ev', min_ev)
    
    if league:
        query = query.eq('league', league)
    
    results = query.order('ev', desc=True).execute()
    
    return {"picks": results.data}
```

---

### Phase 5 Definition of Done

- [ ] FastAPI deployed (Vercel or Fly/Railway)
- [ ] JWT auth works with Supabase tokens
- [ ] Tier gating tested (free user gets 403 on premium endpoints)
- [ ] p95 latency for `/api/value-picks` < 500ms
- [ ] OpenAPI spec exports to `docs/openapi.json`

---

## Phase 6: Stripe + Tier Gating + Launch

**Goal**: Payment processing, user tier management, public launch readiness.

**Duration**: ~2 weeks

**Critical**: Legal review before opening Premium signups.

---

### T6.1 🔵 NEW — Stripe integration

**Stripe Checkout** (embedded, not hosted pages):

```typescript
// frontend
import { loadStripe } from '@stripe/stripe-js';

const stripe = await loadStripe(process.env.NEXT_PUBLIC_STRIPE_KEY);

const handleUpgrade = async () => {
  const response = await fetch('/api/create-checkout-session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tier: 'premium' })
  });
  
  const { sessionId } = await response.json();
  await stripe.redirectToCheckout({ sessionId });
};
```

**Webhook endpoint** (`backend/api/routes/stripe_webhook.py`):

```python
@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    event = stripe.Webhook.construct_event(
        payload, sig_header, settings.stripe_webhook_secret
    )
    
    if event['type'] == 'customer.subscription.created':
        customer_id = event['data']['object']['customer']
        # Update user tier to 'premium'
        update_user_tier(customer_id, 'premium')
    
    elif event['type'] == 'customer.subscription.deleted':
        customer_id = event['data']['object']['customer']
        update_user_tier(customer_id, 'free')
    
    return {"status": "ok"}
```

**Customer portal**: Stripe's embedded portal for self-serve cancellation.

**Test mode first**; production keys only after end-to-end tested.

---

### T6.2 🔵 NEW — Frontend paywall

**Single source of truth**: `users.tier` from auth session.

```typescript
// components/PremiumPanel.tsx
const { user } = useAuth();

if (user.tier === 'free') {
  return (
    <div className="blur-sm relative">
      <div className="absolute inset-0 flex items-center justify-center">
        <Button onClick={handleUpgrade}>
          Upgrade to see EV and value bets
        </Button>
      </div>
      {/* Blurred placeholder content */}
    </div>
  );
}

// Premium/Pro: full panel
return <OddsComparisonPanel />;
```

---

### T6.3 🔵 NEW — Public prediction tracker

**The most important trust-building page.**

Auto-generated from completed predictions:

```typescript
// app/tracker/page.tsx
export default function PredictionTracker() {
  const stats = usePredictionStats(); // from predictions.json + completed matches
  
  return (
    <div>
      <h1>Prediction Tracker</h1>
      
      <StatsCard
        title="Last 30 Days"
        accuracy={stats.last_30.accuracy}
        rps={stats.last_30.rps}
        drawF1={stats.last_30.draw_f1}
      />
      
      <LeagueBreakdown data={stats.by_league} />
      
      <CalibrationPlot data={stats.calibration} />
      
      <WeeklyLog predictions={stats.weekly} />
    </div>
  );
}
```

Host at `/tracker`. Keep it honest — bad weeks show as bad weeks. Transparency converts skeptics.

---

### T6.4 🟢 CRITICAL — Launch checklist (legal + operational)

**Before opening Premium publicly**:

- [ ] **18+ age gate** on signup
- [ ] **Responsible gambling links** prominent on every Premium page
  - France: mention Autorité Nationale des Jeux (ANJ)
  - Link to https://www.joueurs-info-service.fr/
- [ ] **Terms of Service + Privacy Policy** reviewed by lawyer familiar with French gambling-adjacent services (Strasbourg-based — you need this)
- [ ] **Positioning as analytics tool, not tipster service** — no guaranteed outcomes, clear disclaimers
- [ ] **Rate limiting** on API (use `slowapi` + Vercel IP-based limits)
- [ ] **Sentry** or equivalent error tracking
- [ ] **Cookie banner** with granular consent (GDPR)
- [ ] **Stripe payment** tested with real card in test mode
- [ ] **Prediction tracker** live and populated with ≥30 days history
- [ ] **Email opt-in** for marketing (separate from transactional emails)

**This is not optional.** French/EU gambling law is strict. A lawyer costs €1k–3k for this review; a regulatory fine costs €10k–100k+. Pay for the lawyer.

---

### Phase 6 Definition of Done

- [ ] End-to-end: signup → payment → tier upgrade → Premium access works in prod
- [ ] Cancellation flow works
- [ ] Prediction tracker public and accurate
- [ ] Legal review completed; ToS + privacy policy live
- [ ] All launch checklist items green

**Only then market publicly.**

---

## Phase 7: V2 Stacked Ensemble (Optional)

**This phase is OPTIONAL.** Only start if:
1. Premium tier is live and has users
2. You want to squeeze another 2–3% RPS improvement
3. You have bandwidth for 3–4 more weeks of ML work

Otherwise, skip and focus on marketing, content, or new features.

---

### T7.1 🔵 NEW — Full stacked ensemble

Expand on Phase 2 T2.5 stacking:

**Level-0**: XGBoost + LightGBM + CatBoost + SVM (RBF) + Bivariate Poisson (predictions used as features)

**Level-1**: Logistic Regression meta-learner on OOF probabilities

All Level-0 models use SMOTE + class weights. Meta-learner does not (it sees calibrated probabilities).

---

### T7.2 🔵 NEW — Backtest engine

**File**: `backend/evaluation/backtest.py`

```python
def run_backtest(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    min_ev: float = 0.05,
    stake_strategy: str = 'flat'
) -> dict:
    """
    Walk-forward backtest.
    For each match: would we bet? At what stake? Outcome?
    """
    bets = []
    
    for _, pred in predictions.iterrows():
        match_odds = odds[odds['match_id'] == pred['match_id']]
        
        for outcome in ['H', 'D', 'A']:
            model_prob = pred[f'prob_{outcome}']
            best_odds = match_odds[f'odds_{outcome}'].max()
            
            ev = model_prob * best_odds - 1
            
            if ev >= min_ev:
                if stake_strategy == 'flat':
                    stake = 1.0
                elif stake_strategy == 'kelly_fractional':
                    kelly = kelly_fraction(model_prob, best_odds)
                    stake = kelly * 0.25  # quarter-Kelly
                
                actual_outcome = pred['result']
                profit = stake * (best_odds - 1) if outcome == actual_outcome else -stake
                
                bets.append({
                    'match_id': pred['match_id'],
                    'outcome': outcome,
                    'stake': stake,
                    'odds': best_odds,
                    'ev': ev,
                    'profit': profit
                })
    
    bets_df = pd.DataFrame(bets)
    
    return {
        'total_bets': len(bets),
        'hit_rate': (bets_df['profit'] > 0).mean(),
        'roi': bets_df['profit'].sum() / bets_df['stake'].sum(),
        'max_drawdown': compute_drawdown(bets_df['profit'].cumsum()),
        'sharpe': compute_sharpe(bets_df['profit']),
        'by_league': bets_df.groupby('league').agg({'profit': 'sum', 'stake': 'sum'})
    }
```

**Deployment gate**: Backtest must show **positive ROI at EV ≥ 5%** across 3 seasons. If not, do NOT market Premium as a betting tool.

---

### Phase 7 Definition of Done

- [ ] V2 stacked model outperforms Phase 2 ensemble by ≥2% RPS
- [ ] Backtest shows positive ROI at EV ≥ 5% across 3 seasons
- [ ] V2 model deployed to production
- [ ] Backtest results surfaced in Premium Pro tier

---

## Non-Negotiable Production Checklist

**Before any public marketing / Premium signups**:

### Model Quality Gates

- [ ] Draw F1 ≥ 0.25 (hard gate from Phase 2)
- [ ] RPS < 0.21 on held-out test (competitive with literature)
- [ ] Calibration: Brier + reliability diagrams reviewed
- [ ] Leakage test suite passes with ≥1000 hypothesis examples

### Security & Privacy

- [ ] All secrets in env, none in code (run `trufflehog` or similar)
- [ ] Rate limiting on API
- [ ] Error tracking (Sentry) wired up
- [ ] HTTPS everywhere
- [ ] GDPR-compliant cookie banner

### Legal & Compliance (France/EU)

- [ ] 18+ age gate on Premium signup
- [ ] Responsible gambling disclaimers on all Premium pages
- [ ] Link to ANJ (Autorité Nationale des Jeux)
- [ ] Terms of Service reviewed by lawyer
- [ ] Privacy Policy reviewed by lawyer
- [ ] Positioned as analytics tool, not tipster
- [ ] No guaranteed outcomes language anywhere

### Operational

- [ ] Public prediction tracker live (≥30 days history)
- [ ] Stripe payment flow tested end-to-end
- [ ] Cancellation flow works
- [ ] Email opt-in separate from transactional
- [ ] Support email staffed (even if it's just you)

**Do not skip legal review.** It's the difference between a sustainable business and a regulatory nightmare.

---

## Appendix: Research Foundation

### Key Papers Informing This Plan

The ML approach in this plan is grounded in 2024–2026 academic research:

1. **Bunker, Yeung & Fujii (2024)** — arXiv:2403.07669  
   CatBoost + pi-ratings achieves RPS 0.1925, best on goals-only datasets

2. **Wong et al. (2025)** — Decision Analytics Journal  
   Weather features + fatigue/momentum improve accuracy; ensemble stacking adds 2–5%

3. **Macrì-Demartino et al. (2025)** — arXiv preprint  
   Bayesian dynamic models with commensurate priors handle team strength non-stationarity

4. **International Journal of Computer Science in Sport (2024)**  
   Multinomial logistic regression > ordinal for draws; proportional odds assumption violated

5. **Nature Scientific Reports (2025)**  
   SMOTE evaluation across 30 variants confirms consistent minority-class improvement

### Why These Specific Choices

| Decision | Research Backing |
|----------|------------------|
| Pi-ratings over Elo alone | Constantinou & Fenton; Bunker et al. — measurably better RPS |
| CatBoost in the stack | Multiple 2024 benchmarks show CatBoost + ratings is SOTA for tabular |
| SMOTE for draws | Nature SR 2025 comprehensive eval; mitigates structural class imbalance |
| Walk-forward CV | Prevents look-ahead bias; standard in sports forecasting literature |
| RPS as primary metric | Soccer prediction challenge standard; punishes confident wrong predictions |
| Stacked ensemble over weighted avg | Empirically shown to add 2–5% in ensemble papers |
| No deep learning for pre-match | DL underperforms GBDTs on tabular match-level data per 2024 surveys |

### What We're NOT Doing (And Why)

- **Bivariate Poisson as main model** — useful as a feature, but GBDTs beat it head-to-head on RPS
- **Player embeddings** — data-intensive, marginal gains vs team-level features
- **Bayesian hierarchical models** — excellent for uncertainty quantification but slower to productionize than GBDTs
- **Neural nets / LSTMs** — underperform on tabular pre-match data; useful only for in-game event sequences
- **Fixing Dixon-Coles Poisson** — signature incompatibility with matrix interface; not worth the refactor given GBDT dominance

---

## Execution Timeline Summary

| Phase | Duration | Start After | Main Risk |
|-------|----------|-------------|-----------|
| 0 — Audit | 1 week | — | Skipping it ✅ COMPLETE |
| 1 — Foundation | 3 weeks | Phase 0 | T0.1 team-name join fix complexity |
| 2 — ML upgrades | 4–5 weeks | Phase 1 clean baseline | Draw F1 ≥ 0.25 gate may require iteration |
| 3 — Narratives + weather | 2–3 weeks | Phase 2 | Narrative template quality needs manual review |
| 4 — Supabase | 2 weeks | Phase 3 deployed | Odds API cost decision |
| 5 — FastAPI | 2 weeks | Phase 4 | Vercel Python serverless quirks |
| 6 — Stripe + launch | 2 weeks | Phase 5 | Legal review slippage (don't skip) |
| 7 — V2 ensemble (optional) | 3–4 weeks | Phase 6 live | Optional; skip if focusing on growth |

**Total to Premium launch**: ~14–16 weeks (3–4 months at 5–10 hrs/week)

---

## What to Tell Claude Code Next

Paste this exactly when starting Phase 1:

> Read this complete implementation plan (`IMPLEMENTATION_PLAN.md`), then `AUDIT.md`, then `PROJECT.md`. Execute **Phase 1, T0.1 only** — fix the team-name join, rebuild the dataset with canonical names, retrain from scratch, measure true baseline metrics on 2024 holdout, and document in `MODEL_REVIEW.md`. Do not proceed to T0.2 or any other ticket without my explicit approval. Stop after T0.1 and report the new baseline numbers.

After each phase's Definition of Done is met, release the next phase explicitly. Do not skip gates.

---

**End of Implementation Plan**

Version 1.1 — 2026-04-24  
This is a living document. Update after each phase completion.
