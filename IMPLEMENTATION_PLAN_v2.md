# Football Prediction Platform — Product-First Implementation Plan

> **Version**: 2.1 (Progress sync — T0.1d landed)
> **Last Updated**: 2026-04-27
> **Philosophy**: Build a genuinely great free product first. Monetize only after the product is loved and there's evidence people will pay.
>
> **Status**: Phase 0 ✅; Phase 1 in progress
> - ML correctness chain: T0.1 ✅ · T0.2 ✅ · T0.1d ✅ (2026-04-26) · T0.1c 🚧 partial (standings cached, full speedup pending)
> - Infra hygiene chain: T1.0 🟡 partial (file exists, untracked) · T1.1 🟡 partial (tests dir live, no dev-reqs/CI yet) · T1.2–T1.6 ⏳ not started
>
> **Strategy**:
> - **Months 1–3**: Build the best free football prediction site that exists (ML quality + narratives + working frontend)
> - **Months 4–6**: Validate, grow, listen to users, accumulate trust
> - **Month 6+**: Monetize ONLY when there's evidence demand exists

---

## Why Product-First, Not SaaS-First

The original plan (v1.1) front-loaded SaaS infrastructure (Supabase, FastAPI, Stripe, tier gating) before validating the product was worth paying for. **This was wrong**. Successful side projects work the opposite way:

1. **Strava** — free training log for years before Premium
2. **Notion** — free for individuals before Teams pricing
3. **Obsidian** — free forever; paid sync added only after community proved it
4. **Pinpoint** — free predictions before any monetization

The lesson: **monetize from a position of strength, not hope**. If the free product is great, premium becomes obvious. If it isn't, no amount of Stripe integration will save it.

This plan reflects that. Phases 1–4 are pure product work. Business mechanics come later, **and only if validated**.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Product Vision](#product-vision)
3. [Technical Architecture](#technical-architecture)
4. [Phase 0: Ground Truth Audit](#phase-0-ground-truth-audit) ✅ COMPLETE
5. [Phase 1: Foundation & Critical Fixes](#phase-1-foundation--critical-fixes) ← IN PROGRESS
6. [Phase 2: ML Quality Upgrades](#phase-2-ml-quality-upgrades)
7. [Phase 3: The "Why" Layer (Narratives + Weather)](#phase-3-the-why-layer-narratives--weather)
8. [Phase 4: Frontend Polish & Public Launch](#phase-4-frontend-polish--public-launch)
9. [Phase 5: Validate & Grow (Months 4–6)](#phase-5-validate--grow-months-46)
10. [Phase 6: Monetization (Conditional on Phase 5)](#phase-6-monetization-conditional-on-phase-5)
11. [Production Quality Gates](#production-quality-gates)
12. [Appendix: Research Foundation](#appendix-research-foundation)

---

## Executive Summary

**What we're building**: A football match prediction site for Top 5 European leagues that's genuinely **better** than competitors at explaining *why* predictions are what they are.

**Differentiation** (free tier):
- Calibrated H/D/A probabilities (not just "Team X to win")
- Narrative explanations from SHAP values ("Arsenal favored due to 0.4 pi-rating edge, 3 days extra rest, 2.1 xGD over last 5")
- Fatigue / Momentum / Weather badges (shareable, social-native)
- Public prediction tracker (transparency = trust)

**Quality bar before launch**:
- RPS < 0.21 on held-out test (competitive with research literature)
- Draw F1 ≥ 0.25 (deployment gate — current model fails this)
- Predictions vary meaningfully across fixtures (current production bug fixed in T0.1)

**Monetization decision**: Deferred until we have ≥1,000 weekly active users and evidence (interviews, surveys, behavioral data) that there's demand for paid features.

---

## Product Vision

### What "great" looks like

A user lands on the site Saturday morning. In 30 seconds they:
1. See the day's fixtures with calibrated H/D/A probabilities
2. Click into Arsenal vs Chelsea, see *why* the model favors Arsenal in plain language
3. Notice a Fatigue badge ("Chelsea +2 days rest"), find it interesting
4. Screenshot the fixture card and share it on Twitter/Reddit
5. Come back next week to see if last week's predictions were right

That's the product. No paywalls. No friction. Just genuinely useful + shareable + honest.

### What we're NOT building (yet)

- ❌ User accounts
- ❌ Subscriptions
- ❌ "Premium" features
- ❌ Stripe integration
- ❌ Tier gating
- ❌ FastAPI backend
- ❌ Database for users
- ❌ Email marketing automation

These are deferred to Phase 6, and **only built if Phase 5 shows the product has traction**.

### What we ARE building

- ✅ A model that actually works (and we can prove it)
- ✅ Narratives that make predictions feel earned
- ✅ A frontend that looks polished
- ✅ A public prediction tracker (accountability builds trust)
- ✅ Shareable fixture cards (organic distribution)
- ✅ SEO landing pages per fixture (compounding traffic)
- ✅ Static JSON pipeline that costs ~$0/month

---

## Technical Architecture

### Current state (post-Phase 0)

```
GitHub Actions (nightly cron)
        │
        ├── football-data.org v4 → matches
        ├── StatsBomb → xG
        ├── Transfermarkt → squads
        └── (Phase 3) OpenWeatherMap → weather
                │
                ▼
        Feature Engineering (Elo, form, H2H, context, ratings)
                │
                ▼
        ML Pipeline (XGBoost + LightGBM ensemble, isotonic calibration)
                │
                ▼
        predictions.json (static file)
                │
                ▼
        Vercel (Next.js frontend) → users
```

**Cost**: ~$0/month (GitHub Actions free tier + Vercel free tier).

### What stays (Phases 1–4)

This entire architecture stays. We're improving the model and the frontend, not rebuilding infrastructure.

### What might be added later (Phase 6, conditional)

- Supabase (only if user accounts become necessary)
- FastAPI (only if dynamic data like live odds becomes necessary)
- Stripe (only if monetization is validated)

**These are not commitments.** They're options to be exercised based on user behavior.

---

## Phase 0: Ground Truth Audit

**Status**: ✅ **COMPLETE** (2026-04-24)

### Key Findings

1. **T0.1 production bug fixed** — Team-name join was broken; predictions are now varying correctly
2. **T0.2 metrics aligned** — Brier 0.2005, RPS 0.2065 (literature convention)
3. **T0.1d critical bug fixed (2026-04-26)** — `backend/features/form.py` had a positional indexing bug that scattered each team's rolling stats into other teams' rows; now resolved with index-aligned assignment, regression test (`test_form_alignment.py`) pinned to manual ground truth, and pre-fix artifacts backed up to `backend/data/_backup_pre_t01d/`
4. **Infrastructure is sound** — Ingestion, storage, evaluation pipeline, frontend deployment all working

### Phase 1 Work Identified

- T0.1 ✅ Team-name normalization (2026-04-24)
- T0.2 ✅ Metrics convention alignment (2026-04-24)
- T0.1d ✅ **Form feature correctness fix** (2026-04-26) — bug fixed, anti-pattern audit done, model retrained, predictions regenerated
- T0.1c 🚧 **Predict.py performance optimization** — partial: standings cache implemented; ~12 min for 50 matches, full <60s target not yet met
- T1.0 🟡 partial — `CLAUDE.md` exists in working tree but is untracked; reconciliation/commit pending
- T1.1 🟡 partial — `backend/tests/` directory exists with 19 passing tests (`test_form_alignment.py`, `test_metrics.py`, `test_name_consistency.py`); still missing `requirements-dev.txt` split and `.github/workflows/tests.yml`
- T1.2 ⏳ Sandbox smoke test
- T1.3 ⏳ Property-based leakage test
- T1.4 ⏳ Typed config accessors
- T1.5 ⏳ Dependency hygiene
- T1.6 ⏳ Working tree cleanup

---

## Phase 1: Foundation & Critical Fixes

**Goal**: Production bug fixes, trustworthy baseline, infrastructure for Phase 2+.

**Duration**: ~3 weeks (some tickets already complete)

**Critical**: Phase 1 must complete before Phase 2 begins. ML improvements can't be measured against a broken baseline.

---

### T0.1 ✅ COMPLETE (2026-04-24) — Team-name reconciliation

Done. 130 canonical names, regression test (`test_name_consistency.py`), corpus rebuilt, predictions varying correctly.

---

### T0.2 ✅ COMPLETE (2026-04-24) — Metrics convention alignment

Done. Literature convention adopted (Brier mean across classes; RPS normalized by K-1). Regression tests in `test_metrics.py` pin Brier=1/9 and RPS=1/18 on the canonical fixture so the old convention can't be reintroduced silently. All docs updated.

---

### T0.1d ✅ COMPLETE (2026-04-26) — Form feature correctness fix

**The bug** (now resolved): `backend/features/form.py` used `.values` to assign rolling-window outputs after `groupby().rolling()`. Returns rows in team-grouped order; `.values` stripped the index and assigned positionally to a date-sorted DataFrame, scattering each team's rolling stats into other teams' rows. Both training data and `predictions.json` were affected.

**Evidence pre-fix**: Manual ground-truth comparison for Arsenal showed batch and per-match paths diverged from correct values by up to **1.333 PPG** (≈67% relative error on individual rows).

**The fix**: 20 assignments in `form.py` changed from `rolled[col].values` (positional) to `rolled[col]` (index-aligned). 12 in the live `build_form_features` path, 8 in the dead-code `_rolling_team_stats` helper for consistency.

**Verification**:
- `backend/tests/test_form_alignment.py` pins Arsenal's last-10 PPG ground truth at 1e-9 — fails pre-fix, passes post-fix
- `tools/dataset_health_check.py` §17–18 reports `max |batch − manual| = 0.000000` and `max |per_match − manual| = 0.000000`
- All 19 tests in `backend/tests/` pass
- Pre-fix artifacts archived under `backend/data/_backup_pre_t01d/` for rollback

**Anti-pattern audit completed**:
| File | Status |
|------|--------|
| `backend/features/form.py` | ✅ FIXED |
| `backend/features/xg_features.py:83-88` | ⚠️ Same bug, **NOT in production** — StatsBomb data isn't wired up so the builder skips every run. Tracked in `MODEL_REVIEW.md` §6.5. Fix when xG data is integrated. |
| `backend/features/context.py`, `ratings.py` (Elo), `squad_features.py`, `tactical.py` | ✅ clean — no `groupby().rolling().mean().values` pattern |

**Phase 1 Baseline — pre-fix vs post-fix (2024 holdout, ensemble, literature convention)**:

| Metric    | Pre-T0.1d (form bug) | **Post-T0.1d (canonical baseline)** |
|-----------|----------------------|-------------------------------------|
| brier     | 0.2005               | **0.2006**                          |
| rps       | 0.2065               | **0.2069**                          |
| log_loss  | 1.1014               | **1.0872**                          |
| accuracy  | 0.5126               | **0.5171**                          |
| n_samples | 1752                 | 1752                                |

Numbers landed in the same neighborhood — GBMs absorbed the alignment noise as weak features. **The point of T0.1d is truthfulness, not Δmetrics**: each form column now genuinely represents the team it claims to.

> ⚠️ **Do not compare post-T0.1d metrics directly to pre-T0.1d numbers** — both blocks evaluate on the same 2024 holdout, but the model behind each block was trained on a different feature table. They measure two different models. The Post-T0.1d row is the canonical Phase 1 reference for Phase 2 work.

**Definition of Done** — all checked:
- [x] Form values match manual ground truth for Arsenal (1e-9)
- [x] Regression test added and passing (`test_form_alignment.py`)
- [x] Same-pattern audit completed; xg_features.py followup tracked in `MODEL_REVIEW.md` §6.5
- [x] Model retrained on corrected features
- [x] New baseline documented (`METRICS_CHANGELOG.md`, `MODEL_REVIEW.md`)
- [x] `predictions.json` regenerated (50 matches, 45 distinct probability tuples)

---

### T0.1c 🚧 PARTIAL — Predict.py performance optimization

**Status**: Unblocked by T0.1d (parity now well-defined). Standings cache infrastructure landed during T0.1c earlier work. Remaining indexes (H2H, form, Elo) and the lookup refactor are not yet wired in. Current end-to-end run is **~12 min for 50 matches**; target is <60s.

**Goal**: 10–20× speedup (13 min → < 1 min for 50 matches).

**Approach**: Pre-compute indexes once before the prediction loop:
- ✅ Standings index: `{(league, date): {team: position}}` — implemented (`build_context_features` accepts a cache)
- ⏳ H2H index: `{(team_a, team_b): {h2h_stats}}`
- ⏳ Form index: `{(team, date): {form_stats}}` — now correct after T0.1d
- ⏳ Elo index: `{(team, date): elo_state}`

Replace O(n) DataFrame filters in `_build_feature_row` with O(1) dict lookups.

**Definition of Done**:
- [x] Standings cache implemented
- [ ] H2H, form, Elo indexes implemented
- [ ] `_build_feature_row` fully refactored to lookups
- [ ] Parity test passes against T0.1d corrected baseline (1e-6)
- [ ] Performance: < 60 sec for 50 matches
- [ ] Spot-check 3 predictions manually

---

### T1.0 🟡 PARTIAL — Reconcile existing `CLAUDE.md`

A `CLAUDE.md` exists on disk **but is untracked** (per `git status`). Diff against plan-specified content. Merge with plan rules taking precedence on conflicts. Commit.

---

### T1.1 🟡 PARTIAL — Test scaffolding

`backend/tests/` exists with `__init__.py` and 3 test files (`test_form_alignment.py`, `test_metrics.py`, `test_name_consistency.py`); 19 tests pass. Still missing:
- `requirements-dev.txt` split (`pytest`, `pytest-cov`, `hypothesis`, `streamlit`, `plotly`, `shap`)
- `.github/workflows/tests.yml` CI workflow (only `predict.yml` exists today)
- `conftest.py` for shared fixtures (currently each test loads its own data)

---

### T1.2 — Sandbox the smoke test

`tools/pipeline_test.py` currently overwrites production artifacts. Refactor to accept `--sandbox-dir` (default `tmp_smoke/`). Add CI job running smoke test in sandbox mode.

---

### T1.3 — Property-based leakage test

`backend/tests/test_no_leakage.py` with `hypothesis`: for a completed match, computing features with `cutoff=kickoff` must return identical values whether or not post-kickoff data exists. Run with ≥100 examples in CI.

**This is the single most important quality gate in the repo.**

---

### T1.4 — Typed config accessors

Add `backend/config/schema.py` with Pydantic models for `TrainingConfig`, `FeatureConfig`, etc. Keep backward-compatible dict reads.

---

### T1.5 — Dependency hygiene

Pin versions or adopt `uv lock` / `pip-compile`. Split `requirements.txt` (runtime) vs `requirements-dev.txt` (streamlit, plotly, shap, pytest, etc.). Remove unused `polars`.

---

### T1.6 — Commit working tree

Review uncommitted changes (12 modified, 14+ untracked at audit time). Stage in logical groups. Clean baseline for Phase 2.

---

### Phase 1 Definition of Done

- [x] T0.1 ✅ (2026-04-24)
- [x] T0.2 ✅ (2026-04-24)
- [x] T0.1d: form bug fixed, all `.values` patterns audited, model retrained, new baseline documented (2026-04-26)
- [ ] T0.1c: predict.py < 60s for 50 matches, parity passes against T0.1d baseline (standings cache landed; H2H/form/Elo lookup refactor pending)
- [ ] T1.0 commit `CLAUDE.md`
- [ ] T1.1 finish (`requirements-dev.txt`, `tests.yml` CI workflow, `conftest.py`)
- [ ] T1.2–T1.6 complete
- [ ] CI green on every commit
- [x] **New trustworthy baseline metrics documented** (Post-T0.1d: brier 0.2006 / rps 0.2069 / log_loss 1.0872 / accuracy 0.5171) — this is the reference point for all Phase 2 improvements

**Only then proceed to Phase 2.**

---

## Phase 2: ML Quality Upgrades

**Goal**: Improve model quality with surgical, RPS-impact-ordered upgrades. Each ticket measured against T0.1d baseline.

**Duration**: ~4–5 weeks

**Critical gate**: Draw F1 ≥ 0.25 on held-out test. If not met after T2.2, iterate before moving on.

**Note**: Post-T0.1d, your draw F1 may be meaningfully better than 0.086 (form features were a key draw signal). Re-measure first; the gap to 0.25 may be smaller than expected.

---

### T2.1 — Season-aware walk-forward CV

Current evaluation uses single train/test split. Replace with `SeasonSplit` doing walk-forward CV. Report mean ± std of RPS across folds. Mean becomes new baseline.

**File**: `backend/evaluation/splits.py`

```python
class SeasonSplit:
    """
    Walk-forward over seasons.
    Fold N: train={S1..Sk-1}, val=Sk
    Final test: most recent season (held out from CV).
    """
```

**Expected**: Reported RPS may get worse (more honest). This is good.

---

### T2.2 — Draw-class handling

**Three-pronged attack**:

1. **SMOTE in pipeline**:
   ```python
   from imblearn.pipeline import Pipeline as ImbPipeline
   from imblearn.over_sampling import SMOTE
   
   pipe = ImbPipeline([
       ('smote', SMOTE(sampling_strategy={'D': 'auto'}, k_neighbors=5)),
       ('model', XGBoostModel(...))
   ])
   ```

2. **Class weights**: `{'H': 1.0, 'D': 2.5, 'A': 1.2}` (tunable in config)

3. **Draw threshold calibration**: grid-search θ_D ∈ [0.18, 0.32] on validation, optimize macro-F1. Store θ_D in model artifact.

**Add 3 draw-specific features** to `backend/features/context.py`:
- `h2h_draw_rate_last_5`
- `elo_diff_abs` (low values → draw-prone)
- `defensive_match_indicator` (both teams below median xGA over last 10)

**Deployment gate**: Held-out Draw F1 ≥ 0.25.

---

### T2.3 — Pi-ratings alongside Elo

Research literature shows pi-ratings outperform Elo on RPS. Cheap to add.

**File**: `backend/features/pi_ratings.py` using `penaltyblog.ratings.PiRatingSystem`.

Features:
- `home_pi_home`, `home_pi_away`, `away_pi_home`, `away_pi_away`
- `pi_rating_delta` = `home_pi_home - away_pi_away`

Keep Elo. Model picks via SHAP.

---

### T2.4 — CatBoost base model

Third tree model in the ensemble. Handles categoricals natively (useful for future referee, weather category features).

`backend/models/catboost_model.py` extending `BaseModel`. Same pattern as `XGBoostModel`/`LGBMModel`.

---

### T2.5 — Stacking meta-learner

Replace fixed weights with learned stacking.

**Level-0**: XGBoost, LightGBM, CatBoost (each trained with CV → OOF predictions)  
**Level-1**: Logistic Regression on OOF probabilities

`backend/models/stacking.py`. Note: sklearn 1.8+ removed the deprecated `multi_class` arg from LogisticRegression — don't pass it.

---

### T2.6 — Explicit non-goals

**Do NOT do** (per partial-upgrades decision):
- Bivariate Poisson as feature
- Neural networks / LSTMs / Transformers
- Player embeddings
- Bayesian hierarchical models
- Fixing disabled Dixon-Coles Poisson

Document in `MODEL_REVIEW.md` so future-you doesn't waste time.

---

### Phase 2 Definition of Done

- [ ] Walk-forward CV is default
- [ ] Draw F1 ≥ 0.25 (hard gate)
- [ ] Pi-ratings integrated, used by SHAP
- [ ] CatBoost in stack
- [ ] Stacked ensemble RPS ≤ weighted-average ensemble (non-regression)
- [ ] `MODEL_REVIEW.md` updated with new metrics + non-goals

---

## Phase 3: The "Why" Layer (Narratives + Weather)

**Goal**: This is what makes the product genuinely differentiated. Without narratives, you're another prediction site. With them, you're explanatory journalism backed by ML.

**Duration**: ~2–3 weeks

---

### T3.1 — SHAP → narrative generator

**File**: `backend/output/narrative.py`

```python
NARRATIVE_TEMPLATES = {
    "pi_rating_delta": {
        "positive": "{home} rate {abs_value:.1f} points higher than {away} on our rating system.",
        "negative": "{away} come in with a {abs_value:.1f}-point rating advantage.",
    },
    "rest_days_delta": {
        "positive": "{home} have had {value:.0f} more days of rest.",
        "negative": "{away} are better rested by {abs_value:.0f} days.",
    },
    "ewma_xgd_home": {
        "positive": "{home} have outperformed xG by {value:+.1f} per match recently.",
        "negative": "{home} are underperforming xG by {abs_value:.1f}.",
    },
    # ... template per top-feature
}

def generate_narrative(shap_values, features, teams, top_k=3):
    """Returns top-k narrative bullets, structured for frontend."""
```

**Manual quality review**: Generate narratives for 50 sample predictions, fix awkward templates. This is content quality work, not just code.

**Schema extension** in `predictions.json`:
```json
"narrative": [
  {"feature": "pi_rating_delta", "sentence": "...", "impact": 0.12},
  {"feature": "rest_days_delta", "sentence": "...", "impact": -0.04}
]
```

---

### T3.2 — OpenWeatherMap integration

`backend/ingestion/weather.py` + `backend/features/weather_features.py`.

Stadium coordinates in `backend/config/stadiums.yaml`. Fetch forecast 2h before kickoff.

Features: `temperature_c`, `precipitation_mm`, `wind_speed_ms`, `weather_impact_score` (0–1 derived).

Skip historical weather (paid API). Mark as `TODO(v2)`.

When `weather_impact_score > 0.6`, narrative includes a weather sentence.

---

### T3.3 — Badges (the social-shareable atom)

Extend `predictions.json`:
```json
"badges": {
  "fatigue_edge": {"team": "home", "days_advantage": 3},
  "momentum_edge": {"team": "away", "form_delta_ppg": 0.8},
  "weather_flag": true
}
```

These are the screenshot-worthy elements. Frontend renders them as small visual chips on fixture cards.

---

### Phase 3 Definition of Done

- [ ] Narratives generated for every prediction (3 bullets each)
- [ ] Weather features integrated, SHAP shows non-zero contribution
- [ ] `predictions.json` has `narrative` + `badges`
- [ ] 50 sample narratives manually reviewed
- [ ] Frontend renders both (next phase)

---

## Phase 4: Frontend Polish & Public Launch

**Goal**: A frontend that looks polished enough that people genuinely enjoy using it and want to share it.

**Duration**: ~3–4 weeks

---

### T4.1 — Match Card component (the atom)

The single most important UI decision. This component appears everywhere.

**Required elements**:
- Team names + logos
- Probability bar (H/D/A) — visual, not just numbers
- Confidence indicator
- Top narrative bullet (truncated)
- Fatigue/Momentum/Weather badges
- Hover state showing more detail
- Click → match detail page

**Design system tokens**: typography scale, color palette (semantic: green=H, amber=D, red=A, distinct accent for "interesting"), spacing grid.

---

### T4.2 — Matchday Dashboard (home page)

List of upcoming fixtures grouped by league/date. Filterable. Match Card as row unit.

**This is the page users land on.** It must look great in 5 seconds.

---

### T4.3 — Match Detail Page

Expanded view:
- Full probability breakdown
- All 3 narrative bullets
- Form graph (last 10 matches, both teams)
- Head-to-head history
- Weather forecast (if applicable)
- xG trend chart
- SHAP feature importance (collapsible "Advanced" section)

**SEO**: This page is the long-tail traffic engine. Each fixture URL is a landing page. Title: "Arsenal vs Chelsea — Matchday 14 Prediction & Analysis". H1 + meta description from narrative.

---

### T4.4 — Public Prediction Tracker

`/tracker` page. The trust-building cornerstone.

- Last 7/30/90 days: accuracy, RPS, calibration
- By-league breakdown
- Reliability plot
- Weekly prediction log (immutable, public)

**Honest is better than impressive.** Bad weeks show. Calibration plot beats vanity stats.

---

### T4.5 — Shareable Fixture Cards (SVG generator)

For each upcoming fixture, generate a shareable image:
- 1200×630 (Twitter/Open Graph optimal)
- Team logos, probabilities, key narrative bullet, badges
- Watermarked with site URL

Backend: small SVG/PNG generator (matplotlib or `pillow` with template). Save to `frontend/public/og/{match_id}.png`.

Frontend: per-fixture page sets `<meta property="og:image">` to this URL. Twitter, LinkedIn, Discord, Slack all auto-render the card when shared.

**This is your distribution flywheel.** Every share is free marketing.

---

### T4.6 — Onboarding (lightweight)

Optional for v1. If included:
- 3 screens max
- Pick favorite team(s)
- Pick leagues to follow
- Show this weekend's breakdown for picked teams

No login required. State stored in localStorage.

---

### T4.7 — Public Launch

When all above complete:
- Custom domain (Vercel deploy)
- Open Graph + Twitter Cards meta on every page
- Sitemap + robots.txt
- Submit to Google Search Console
- Soft launch: post to r/soccer, /r/MachineLearning, Hacker News (Show HN)
- Email a few football analytics Twitter accounts for feedback

**No paywalls. No signups. Just a great free product.**

---

### Phase 4 Definition of Done

- [ ] Match Card component locked
- [ ] Dashboard, Match Detail, Tracker pages live
- [ ] Shareable cards generating per fixture
- [ ] SEO metadata + sitemap
- [ ] Public domain live
- [ ] Posted to at least 2 communities for feedback

---

## Phase 5: Validate & Grow (Months 4–6)

**Goal**: Prove the product is worth using before deciding whether to monetize.

**Duration**: 2–3 months of operation, not coding

**This phase is mostly NOT coding.** It's listening, measuring, iterating.

---

### T5.1 — Analytics

Add **privacy-respecting** analytics:
- **Plausible** (recommended) or **Umami** — GDPR-friendly, no cookies
- Events: fixture views, narrative reads, share clicks, tracker views
- Per-league engagement breakdown

Avoid Google Analytics — too invasive for what we need.

---

### T5.2 — User research

Talk to actual users. Concrete tactics:
- **Reddit DMs**: "I built this — would 10 mins of feedback help?" to people who upvote
- **Twitter polls**: "What would make this prediction site genuinely useful for you?"
- **Onsite micro-survey** (after 3 visits): "What would you want to see that's missing?"
- **5-star feedback widget**: 1 click, no form

Goal: 20+ qualitative conversations over 2 months.

---

### T5.3 — Iteration on the free product

Based on analytics + feedback, iterate:
- Which leagues actually get traffic? (Maybe drop one, add a different one)
- Which narrative templates land? (Refine the dud ones)
- Which badges drive shares? (Double down on those)
- Which prediction outcomes get the most engagement? (Underdog wins? Big derby calls?)

**Ship weekly updates.** Visible momentum builds trust.

---

### T5.4 — Content engine

Once the model is good and the tracker is honest, start a content pipeline:
- Weekly "Predictions Recap" post: "How did our model do? Best calls, worst misses, what we learned"
- Pre-matchday "Fixtures to Watch": "These 3 fixtures have the highest model uncertainty"
- Monthly "Model Performance Report"

Hosted on the same site (`/blog/`). SEO compounds. Twitter thread for each post.

---

### T5.5 — Validation gate

After 3 months of Phase 5, ask honestly:

**Quantitative checks**:
- ≥ 1,000 weekly active users? (or some equivalent threshold for your market)
- Median session > 90 seconds?
- Repeat visit rate > 30%?
- Organic search traffic growing month-over-month?

**Qualitative checks**:
- Have ≥3 users explicitly asked for premium features unprompted?
- Are users sharing fixture cards organically?
- Has the tracker page been screenshotted in any tweets/posts?
- Do you get unsolicited "this is great" messages?

**Decision tree**:

| Signal | Action |
|--------|--------|
| Quant ✅ + Qual ✅ | Proceed to Phase 6 (monetize) |
| Quant ✅ + Qual ❌ | Keep growing free; don't monetize yet |
| Quant ❌ + Qual ✅ | Niche product; consider B2B/API instead of consumer subscriptions |
| Quant ❌ + Qual ❌ | Either pivot product or accept it as portfolio piece |

**Don't monetize prematurely.** A free product with 500 dedicated users is more valuable than a paid product with 10 reluctant ones.

---

## Phase 6: Monetization (Conditional on Phase 5)

**Trigger**: Only enter Phase 6 if Phase 5 validation gate passes.

**Duration**: ~6–8 weeks

**Critical**: This phase introduces real complexity (auth, payments, legal). Don't start unless validation is clear.

---

### T6.1 — Decide the offer

Based on Phase 5 user research, decide what people want to pay for. Likely candidates:

**Option A: Bettor-focused** (highest LTV but legal complexity)
- Live bookmaker odds comparison
- Expected Value (EV) highlights
- Backtest ROI history
- Kelly stake sizing
- Price: €9.99–14.99/mo
- **Requires**: Legal review (especially France/EU), 18+ gate, responsible gambling compliance

**Option B: Analytics-focused** (lower LTV, simpler)
- Historical model performance per team/league
- Custom alerts ("notify when Arsenal has 70%+ win prob")
- API access for personal projects
- Multi-league coverage beyond Top 5
- Price: €4.99–9.99/mo
- **Requires**: User accounts, basic API auth

**Option C: B2B/API only** (if quant signal is strong but consumer signal weak)
- API access for fantasy/betting/media products
- Per-call pricing or tier subscriptions
- Price: €49–499/mo
- **Requires**: Solid uptime, SLAs, documentation

Pick ONE based on validated demand. Don't try to do all three.

---

### T6.2 — Backend infrastructure (only what's needed)

**For Option A or B**:
- Supabase: users, subscriptions, (Option A: odds_snapshots, value_picks)
- FastAPI: Premium endpoints
- Stripe: subscriptions + webhook → updates `users.tier`
- Vercel: existing frontend + Python serverless functions for API

Schemas + code patterns same as v1.1 plan, but only build what's needed for the chosen option.

---

### T6.3 — Tier gating

Single source of truth: `users.tier` from Supabase auth session.

Frontend:
- Free: full access to current free product
- Premium: unlock the additional content based on chosen Option

No retroactive paywalls. Existing free features stay free forever.

---

### T6.4 — Stripe integration

Stripe Checkout (embedded). Webhook updates user tier. Customer portal for self-serve cancellation.

Test mode end-to-end. Production keys only after thorough QA.

---

### T6.5 — Legal & compliance (FR/EU)

**Mandatory** before opening Premium signups:

- [ ] Lawyer review of ToS + Privacy Policy (Strasbourg-based, familiar with FR gambling-adjacent services)
- [ ] **Option A only**: 18+ age gate, responsible gambling links (link to ANJ + Joueurs Info Service), positioning as analytics not tipster
- [ ] GDPR-compliant cookie banner, granular consent
- [ ] DPA available on request
- [ ] Email opt-in separate from transactional

**Cost**: €1k–3k for legal review. Mandatory. Do not skip.

---

### T6.6 — Launch

When all above complete + validated:
- Soft launch to email list (people who joined waitlist during Phase 5)
- Public announcement: blog post + Twitter thread + Reddit
- 30-day money-back guarantee
- Founders' price for first 100 subscribers (€1–2/mo cheaper, lifetime locked)

---

### Phase 6 Definition of Done

- [ ] Validated offer chosen and built
- [ ] Stripe end-to-end working
- [ ] Legal review complete
- [ ] First paying customer 🎉
- [ ] Cancellation flow tested
- [ ] Customer support email staffed (even if just you)

---

## Production Quality Gates

**Hard gates** that must be green for launch:

### Model quality (any phase)
- [ ] Draw F1 ≥ 0.25 (Phase 2 gate)
- [ ] RPS < 0.21 normalized (Phase 2 gate)
- [ ] Calibration: Brier + reliability diagrams reviewed quarterly
- [ ] Leakage test passes ≥1000 hypothesis examples
- [ ] Public prediction tracker accurate

### Operational
- [ ] All secrets in env, none in code (`trufflehog` scan)
- [ ] Sentry or equivalent error tracking
- [ ] Rate limiting on any API
- [ ] HTTPS everywhere

### Legal (Phase 6 only)
- [ ] 18+ gate (Option A)
- [ ] Responsible gambling links (Option A)
- [ ] ToS + Privacy reviewed by lawyer
- [ ] GDPR cookie banner

---

## Honest Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| 0 — Audit | 1 week | ✅ COMPLETE |
| 1 — Foundation + critical fixes | 3 weeks | 🚧 IN PROGRESS (T0.1, T0.2, T0.1d ✅; T0.1c partial; T1.0–T1.6 pending) |
| 2 — ML upgrades | 4–5 weeks | ⏳ Blocked by Phase 1 |
| 3 — Narratives + weather | 2–3 weeks | ⏳ |
| 4 — Frontend + public launch | 3–4 weeks | ⏳ |
| **Free product live** | **~13 weeks (~3 months)** | |
| 5 — Validate + grow | 2–3 months operating | ⏳ |
| 6 — Monetize (conditional) | 6–8 weeks | ⏳ Only if Phase 5 validates |
| **Premium launch (if at all)** | **~6 months from now** | |

---

## Appendix: Research Foundation

[Same as v1.1 — Bunker et al. 2024 (CatBoost + pi-ratings), Wong et al. 2025 (ensemble stacking), Macrì-Demartino et al. 2025 (Bayesian dynamic), International Journal of CS in Sport 2024 (multinomial > ordinal for draws), Nature SR 2025 (SMOTE evaluation)]

### Why Product-First Reflects the Research

Research shows football prediction is **hard**. RPS 0.19 is competitive; 0.15 is essentially impossible without data sources retail products don't have. The marginal improvement from any ML upgrade is small.

But the marginal improvement from a **good narrative** is massive — the difference between a number and an explanation is what users remember. So product investment beats further model investment beyond a competitive baseline.

This plan reflects that reality: get to a competitive ML baseline (Phase 2), then put 3× more effort into the explanation layer (Phase 3 + 4) than into squeezing extra RPS percentage points.

---

## How to Use This Document

1. **This is the active source of truth.** `IMPLEMENTATION_PLAN.md` (v1.1) is kept as a historical reference only — its task IDs (T0.1 / T0.2 / T0.1d / T1.x) match this plan, so progress recorded under either ID is the same work.
2. **Update after each phase.** Mark tickets complete. Document deviations. Note what worked / didn't.
3. **The validation gate (end of Phase 5) is real.** Don't skip it because Phase 6 sounds exciting. Premature monetization kills more side projects than slow growth.
4. **Phase 6 is conditional, not committed.** If validation fails, you have a great portfolio piece + free product, not a dead startup.

---

**End of Plan**

Version 2.1 — Progress sync (T0.1d landed) — 2026-04-27
Version 2.0 — Product-First Restructure — 2026-04-26
