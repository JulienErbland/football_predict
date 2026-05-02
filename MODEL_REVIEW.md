# Model Review — Findings & Improvement Roadmap

> Based on a full code review + survey of recent research (2024–2026).  
> No code changes — ideas only. Ordered by expected impact vs. implementation effort.

---

## 0. Phase 1 Baseline (post-T0.1, 2026-04-24)

> ⚠️ **Superseded by T0.1d (2026-04-26)** — the form-feature alignment fix
> (`backend/features/form.py`) re-trained the ensemble on truthful rolling
> stats. New holdout numbers are documented in
> `backend/evaluation/METRICS_CHANGELOG.md` under T0.1d. The numbers in §0.2
> below are kept as the historical reference point; **do not compare T0.1d
> numbers to them directly** — the underlying feature table is different, so
> they measure different models on the same holdout.

The audit (`docs/AUDIT.md`, 2026-04-24) surfaced a production bug: the upcoming-fixture ingestion path (football-data.org API) returned long-form team names ("Arsenal FC") while the historical corpus (football-data.co.uk CSVs) used short-form ("Arsenal"). In `backend/output/predict.py`, every team-keyed feature lookup therefore cold-started on defaults. The pre-fix `predictions.json` (50 upcoming matches) contained only **5 distinct probability tuples** — 18 matches sharing the canonical degenerate vector `(0.387, 0.232, 0.381)`, a further 31 sharing four near-cold-start vectors, and only 1 match emerging with a real team-specific probability.

T0.1 resolved this by introducing `backend/ingestion/name_normalizer.py` — a canonical alias map that every ingestion source (football-data.org, football-data.co.uk CSVs, The Odds API) calls at its output boundary. The training corpus was re-ingested through the normalised path, feature engineering re-run, and the ensemble retrained from a clean dataset. Regression test: `backend/tests/test_name_consistency.py` (8 tests, all passing).

These are the **numbers you should compare every subsequent ML change against**.

### 0.1 Corpus and split

| | |
|-|-|
| Training corpus | 7 156 matches × 71 feature columns |
| Leagues | PL, PD, BL1, SA, FL1 (top-5 European) |
| Seasons | 2021/22, 2022/23, 2023/24, 2024/25 |
| Train / held-out split | seasons 2021 + 2022 + 2023 → train (5 404 rows); season 2024 → test (1 752 rows) |
| Class balance (train) | ~43.4% home / 25.4% draw / 31.2% away |
| Class balance (test) | 42.0% home (736) / 24.9% draw (437) / 33.1% away (579) |
| Ensemble | XGBoost (w=0.60) + LightGBM (w=0.40), isotonic-calibrated per base model |
| Training artefacts | `backend/data/models/{ensemble,xgboost,lightgbm,feature_cols}.pkl`, 2026-04-24 |

### 0.2 Headline metrics on 2024 holdout

| Metric | Phase 1 baseline | Range / reference |
|--------|-----------------:|-------------------|
| Accuracy | **0.5126** | random ≈ 0.33; bookmaker ≈ 0.54; published best 0.52–0.56 |
| Brier score | **0.2005** | range [0, 1]; published competitive band 0.18–0.22 |
| Ranked Probability Score (RPS) | **0.2065** | range [0, 1]; published competitive band 0.19–0.23 |
| Log-loss | 1.1014 | random 3-class ≈ 1.099 (ln 3) |
| Macro-F1 | 0.4168 | — |
| n_samples | 1752 | 2024-season holdout |

All metrics above use the literature convention (Brier = mean across K=3 classes; RPS = sum-over-thresholds divided by K-1). See `backend/evaluation/METRICS_CHANGELOG.md` for the migration history.

> **Historical note.** Pre-T0.2 (before 2026-04-24), `evaluation/metrics.py` returned the un-normalised forms — Brier 0.6014 and RPS 0.4130 on this same holdout. Multiplying the new values by K=3 and K-1=2 respectively reproduces those legacy numbers. Eval JSON artifacts produced before that date carry the old convention; `eval_phase1_baseline.json` records both side-by-side under `metrics_repo_convention` (legacy) and `metrics_literature_convention` (current).

### 0.3 Per-class breakdown

| Class | Precision | Recall | F1 | argmax picks |
|-------|-----------|--------|------|--------------|
| H (home win) | 0.5563 | 0.6916 | 0.6166 | 915 / 1752 (52.2%) |
| **D (draw)** | **0.2857** | **0.0503** | **0.0856** | **77 / 1752 (4.4%)** |
| A (away win) | 0.4829 | 0.6339 | 0.5482 | 760 / 1752 (43.4%) |

Confusion matrix (rows = true, cols = predicted):

|        | pred H | pred D | pred A |
|--------|-------:|-------:|-------:|
| true H | 509    | 29     | 198    |
| true D | 220    | 22     | 195    |
| true A | 186    | 26     | 367    |

### 0.4 Draw probability distribution

The model rarely emits a draw probability above 0.30. 85% of the test set sits in the 0.20–0.30 draw-prob band, where it is well-calibrated *on average* but effectively constant per-sample — argmax almost never picks draw because either home or away always exceeds the squashed draw value.

| p_draw bin | n | avg pred | actual rate |
|-----------:|--:|---------:|------------:|
| [0.10, 0.20) | 54 | 0.177 | 0.241 |
| [0.20, 0.25) | **1 063** | 0.234 | 0.259 |
| [0.25, 0.30) | 416 | 0.269 | 0.190 |
| [0.30, 0.35) | 119 | 0.318 | 0.370 |
| [0.35, 0.40) | 76 | 0.370 | 0.276 |
| [0.40, 0.50) | 21 | 0.428 | 0.238 |

### 0.5 Comparison to pre-fix audit

The aggregate numbers barely moved relative to the audit's pre-fix measurements (`docs/AUDIT.md §0.1.4`):

| Metric | Pre-fix (audit) | Phase 1 baseline | Δ |
|--------|-----------------:|------------------:|----:|
| Accuracy | 0.5211 | 0.5126 | −0.0085 |
| Brier (sum, pre-T0.2) | 0.5997 | 0.6014 | +0.0017 |
| RPS (sum, pre-T0.2) | 0.4117 | 0.4130 | +0.0013 |
| Log-loss | 1.0617 | 1.1014 | +0.0397 |
| Macro-F1 | 0.4156 | 0.4168 | +0.0012 |
| F1 draw | 0.0667 | 0.0856 | +0.0189 |

> The Brier and RPS rows above use the pre-T0.2 sum convention so the comparison stays apples-to-apples with the audit numbers (which were measured before the convention fix). The audit doc cannot be retroactively renormalised. Going forward, only literature-convention values appear in §0.2 and in `eval_*.json` files.

The deltas are small because **the bug never corrupted training**. The historical corpus used CSV short-form names internally consistently, so the feature pipeline and the model both saw a coherent world during fit and holdout evaluation. The bug lived at inference time, in `predict.py`, where the API-style upcoming row was appended to a CSV-style historical frame. The pre-fix numbers were therefore an honest measurement of what the ensemble had learned — what was dishonest was the production `predictions.json` those same weights then emitted. T0.1 fixes the inference path; training/holdout metrics move only by retraining noise (different XGBoost early-stopping iteration: 41 now, unknown-but-different at audit time).

### 0.6 What this baseline tells us for Phase 2

1. **Aggregate headroom**: normalised RPS 0.2065 is already inside the literature "expert band" of 0.19–0.23, and per-class Brier 0.2005 is at the published-model ceiling. Phase 2 gains will be small on aggregate metrics unless the draw class is fixed.
2. **Draw class is the binding constraint**. Draw F1 = 0.0856 is catastrophically bad; argmax picks draw only 4.4% of the time despite 24.9% of matches actually being draws. This is exactly what T2.2 (SMOTE + class weights + threshold tuning) is scoped to fix, and the `min_draw_f1: 0.25` gate stands.
3. **Home/Away are solid**. F1 ≈ 0.62 / 0.55 on the non-draw classes matches published benchmarks. Changes that regress those while raising draw F1 are a net loss.

### 0.7 Production inference verification

Post-fix spot-check over the full 50-match upcoming slate (2026-04-24): **43 distinct probability tuples** (vs. 5 pre-fix), with Elo values plausibly ranked — Bayern Munich 1753, Real Madrid 1748, Liverpool 1721, Barcelona 1761, Inter 1738, Man City/United mid-table, Leeds 1375 (relegated → low). Only 5 sides default to cold-start Elo=1500 and they are exactly the teams listed in `_KNOWN_NEW_ENTRANTS` (Paris FC, Pisa, Sunderland, Oviedo, Hamburger SV). Sample probabilities: Real Madrid vs Girona = (0.682, 0.248, 0.070), Marseille vs Metz = (0.853, 0.147, 0.000), Atalanta vs Juventus = (0.430, 0.213, 0.357) — exactly the kind of variation that was missing before.

> ⚠️ Cold-start caveat. Five clubs currently in the 2025/26 season were not in the 2021–2024 corpus and therefore still default to cold-start feature values at inference: **Sunderland** (PL), **Oviedo** (PD), **Hamburger SV** (BL1), **Pisa** (SA), **Paris FC** (FL1). This is *correct* behaviour for teams with no training history; it is listed explicitly in the regression test (`_KNOWN_NEW_ENTRANTS`) so any new unmapped alias surfaces as a test failure rather than silently re-introducing the production bug. Ingesting the partial 2025/26 season later will give these clubs real feature values without needing another T0.1-style intervention.

---

## 1. Performance ceiling — what to expect

Before anything else: football is fundamentally noisy. Even a perfect model cannot predict every match because ~30–40% of outcomes are driven by genuine randomness (deflections, individual moments, referee decisions). The academic literature consistently reports these ceilings for the best models:

| Metric | Random baseline | Betting odds | Best published models |
|--------|----------------|--------------|----------------------|
| Accuracy | 33% | ~54% | 52–56% |
| RPS | 0.37 | ~0.20 | 0.195–0.215 |
| Brier score | 0.67 | ~0.57 | ~0.55 |

**Implication:** chasing accuracy above ~55% with match-level features is likely overfitting. The real gains come from better calibration (getting probabilities right, not just picking winners) and better value bet identification. A well-calibrated 53% model beats a poorly-calibrated 55% model for betting purposes.

---

## 2. Missing features

### 2.1 Player availability — highest priority gap

The current model knows nothing about who is actually playing. A team missing its top striker is indistinguishable from the same team at full strength. This is the single most impactful missing signal.

**What to add:**
- Injury/suspension flag for each team's top-3 most influential players (by market value or rating)
- Total injured market value as a fraction of full squad value
- First-choice goalkeeper availability flag (goalkeepers have outsized defensive impact)
- Suspension count from yellow card accumulations

**Evidence:** Multiple recent papers confirm player availability explains significant variance that aggregate team stats miss. The *"From Players to Champions"* paper (arXiv 2505.01902) explicitly found player-level features outperform team-level features in generalisation across tournaments.

**Data source:** TransferMarkt injury pages (already scraped in the codebase but not used as features). FBref also publishes injury reports.

---

### 2.2 xG overperformance / underperformance delta

The current xG features are rolling averages of expected goals. What's missing is the **gap between actual goals and expected goals** — a regression-to-the-mean signal.

- A team scoring 2.1 goals/game but only generating 1.3 xG/game is almost certainly due to regress
- A team conceding fewer goals than their xGA suggests has an unusually good goalkeeper (possibly sustainable) or is getting lucky (unsustainable)

**Features to add:**
- `home_goals_minus_xg_w10` — rolling difference between actual and expected goals scored
- `away_goals_allowed_minus_xga_w10` — same on the defensive side
- Big Chance conversion rate (chances with xG > 0.4 that result in a goal)

**Evidence:** The Frontiers paper *"Bayes-xG"* (2024) and the PMC paper on event-sequence xG both show that plain xG averages leave substantial predictive signal on the table. The overperformance delta is already partially exploited by sharp bookmakers.

---

### 2.3 Season-to-season trajectory

A team currently 3rd that finished 6th last season is on an upward trajectory. A team currently 8th that won the title last year is declining. Rolling form captures within-season momentum but misses cross-season trends entirely.

**Features to add:**
- `position_delta`: current league position minus end-of-last-season position
- `elo_delta_since_season_start`: Elo change since the season began
- Season-over-season points delta (this season points pace vs. last season final points)

---

### 2.4 European competition fatigue

A team playing Champions League Thursday → Premier League Sunday has less recovery time (partially captured by rest days) **and** a rotation incentive. Coaches rest key players for league games when European qualification is not at risk.

**Features to add:**
- Binary flag: played European match in last 4 days
- Combined flag: European match in last 4 days AND team is safe in the league (rotation likely)
- Cumulative European matches played this season (fatigue accumulates)

---

### 2.5 Managerial change flag

Teams that recently changed managers behave erratically in the first 3–5 matches. There is well-documented evidence of a "new manager bounce" — a temporary improvement in results driven by motivational reset, before regressing as opponents adapt to the new tactics.

**Feature to add:**
- `home_new_manager` / `away_new_manager`: binary flag, "manager changed in last 5 matches"
- Optionally: matches since manager took over (the bounce decays over ~10 games)

---

### 2.6 Derby / rivalry flag

Derbies produce systematically different distributions: more draws, more red cards, more low-scoring matches, and more upsets. Current form and Elo are poor predictors because motivational intensity overrides quality differences.

A hardcoded list of known rivalries (Manchester Derby, El Clásico, Derby della Madonnina, etc.) with a binary flag would improve calibration on these outlier matches.

---

### 2.7 Weather at the stadium

Wind speed, rain, and cold temperatures suppress goal-scoring and shift outcomes toward draws. Research estimates extreme weather reduces total goals by 10–20%. This effect is large enough to matter for probability calibration.

**Data source:** any weather API using stadium GPS coordinates and match kickoff time. Historical data is freely available from Open-Meteo.

**Features to add:**
- Wind speed (km/h) at kickoff
- Precipitation (mm)
- Temperature (°C)
- A composite "bad weather" binary flag (wind > 50 km/h OR heavy rain)

---

### 2.8 Home advantage granularity

The current model treats home advantage as a global constant implicit in the home/away form split. But home advantage varies enormously by team and by context.

**Features to add:**
- Team-specific home win rate (current season)
- Stadium attendance rate (% of capacity) — a proxy for crowd intensity
- "Empty stadium" flag (relevant for COVID-era data and could matter in future)

The natural experiment from COVID matches (played behind closed doors) showed home advantage dropped to near zero, providing strong causal evidence that crowd noise is the primary mechanism.

---

### 2.9 Opening bookmaker odds as features (with caveats)

This is philosophically debatable given the project's design principle of keeping odds out of training. However, there is strong academic evidence that bookmaker odds — especially from sharp bookmakers like Pinnacle — are the single best publicly available signal for match outcomes.

**The key distinction:**
- **Opening odds** (set before sharp money moves the line) = bookmaker's prior, relatively independent of the model
- **Closing odds** (just before kickoff) = market consensus, highly correlated with any good model's output

Using **opening odds only** as features represents the bookmaker's expert prior. The model then tries to find residual signal *beyond* what the market already knows. The arXiv paper *"The Evolution of Football Betting: A Machine Learning Approach"* (2403.16282) documents this approach as viable and profitable.

**Risk:** if you use closing odds as features during training, you are training the model to mimic the market, which eliminates any edge. Opening odds are safer.

---

### 2.10 Lineup-adjusted quality (player ratings aggregated)

Rather than using squad value as a proxy, summing individual player ratings across the announced lineup gives a more accurate match-day quality estimate.

**Data sources:** FIFA ratings (in EA FC datasets, freely available), FBref's progressive stats, or Wyscout/InStat player scores.

**Features to add:**
- Sum of FIFA overall ratings of the starting XI
- Variance of lineup ratings (a team of 11 average players vs. 8 stars + 3 weak links)
- Rating gap between starting XI and typical XI (measures how much rotation occurred)

The PLOS ONE paper *"A framework of interpretable match results prediction with FIFA ratings and team formation"* showed that FIFA-rating-based features matched or outperformed raw result-based features.

---

## 3. Model architecture improvements

### 3.1 CatBoost as a third model — quick win

CatBoost handles categorical features natively without any encoding. Team names, referee, formation — instead of label-encoding these integers, CatBoost learns optimal categorical splits directly from raw strings. This consistently produces meaningfully better performance on sports data where team identity is itself predictive (certain teams systematically overperform or underperform their feature vector).

The xGFootball Club benchmark (linked below) shows CatBoost competitive with or better than XGBoost on football data. Adding it to the ensemble at ~25% weight gives diversity with minimal code complexity.

**Effort:** low. CatBoost has the same scikit-learn interface as XGBoost/LightGBM.

---

### 3.2 Stacking meta-learner instead of fixed weights

The current ensemble uses fixed weights (60/40). A logistic regression meta-learner trained on out-of-fold (OOF) predictions learns dynamically when to trust XGBoost more vs. LightGBM — for example, it might learn that LightGBM is better calibrated late in the season, or for derbies.

The infrastructure already exists in the codebase (`StackingEnsemble` class in `ensemble.py`). It needs:
1. OOF predictions generated during training via k-fold
2. A held-out meta-training set (kept separate from the test set)

**Effort:** medium. Requires refactoring the training loop.

---

### 3.3 Walk-forward cross-validation

The current evaluation trains on 2021–2023 and tests on 2024. This gives a single noisy estimate of generalisation. One lucky/unlucky season can make the model look better or worse than it is.

Walk-forward CV runs multiple folds:
```
Fold 1: train 2021      → test 2022
Fold 2: train 2021–2022 → test 2023
Fold 3: train 2021–2023 → test 2024
```

Average the metrics across folds. This gives a far more reliable estimate and catches overfitting to a specific season.

**Effort:** low. Add a loop around the existing train/test logic.

---

### 3.4 TabTransformer or TabNet (future, if dataset grows)

The NeurIPS 2023 paper *"When Do Neural Nets Outperform Boosted Trees on Tabular Data?"* found that neural nets beat GBDTs on tabular data primarily when:
1. Dataset size > 50,000 rows
2. Features have complex non-linear compound interactions

At ~10k matches, we are below both thresholds. **However**, if player-level features are added (one row per player per match), the dataset size changes the calculus entirely.

TabTransformer applies self-attention across features — it can learn interactions like "high-Elo team + away game + 3 days rest + derby" as a compound effect without requiring deep tree branching.

**Verdict:** not worth it now. Revisit if data grows to 50k+ rows or if player embeddings are added.

---

### 3.5 Poisson model — fix or drop

The Poisson (Dixon-Coles) model is currently excluded from the ensemble because its `predict_proba()` interface expects team name tuples rather than feature vectors. Two paths forward:

**Option A — Fix the interface:** wrap the Poisson model so it extracts team names from the feature row and routes internally. This lets it rejoin the ensemble and contribute its statistical baseline (especially useful in low-data regimes early in a season when rolling features are sparse).

**Option B — Use standalone for sanity-checking:** keep it excluded from the ensemble but run it separately to cross-check predictions. If the ensemble says 70% home win but Poisson says 45%, that mismatch is worth investigating before betting.

---

## 4. Data pipeline improvements

### 4.1 FBref as an additional data source

FBref (Sports Reference) provides free, detailed match-level and player-level stats including:
- Progressive passes, progressive carries, pressures, PPDA
- Player match ratings and participation minutes
- Expected goals with shot-level detail

The coverage overlaps with StatsBomb for the top leagues but extends to more competitions. Combining both would increase xG data coverage significantly.

### 4.2 Understat for xG

Understat provides free match-level xG data for the top 6 European leagues going back to 2014. It is more practical than StatsBomb (which requires API or open-data parsing) for bulk historical xG fetching. The data includes xG, xGA, and deep/open-play shot breakdowns.

---

## 5. What the research confirms we are already doing right

- **XGBoost + LightGBM ensemble:** the dominant approach in recent benchmarks. No published architecture consistently outperforms this on match-level tabular data at this scale.
- **Time-based train/test split:** critical and correctly implemented. Random splits systematically inflate reported performance in sports prediction due to form and Elo leakage.
- **Isotonic calibration:** the right choice over Platt scaling for multi-class sports outcomes.
- **RPS as primary metric:** academically endorsed as the best metric for ordered 3-class outcomes. Accuracy alone is misleading.
- **Odds isolation from training:** sound design. Using closing odds as training features produces a model that mimics the market rather than outperforming it.

---

## 6. Priority summary

| # | Idea | Effort | Expected impact |
|---|------|--------|----------------|
| 1 | Player availability / injury features | Medium | **Very high** |
| 2 | xG overperformance delta | Low | High |
| 3 | Walk-forward cross-validation | Low | High (evaluation quality) |
| 4 | CatBoost as third ensemble model | Low | Medium-high |
| 5 | Season-to-season trajectory features | Low | Medium |
| 6 | European competition fatigue flag | Low | Medium |
| 7 | Managerial change flag | Low | Medium |
| 8 | Stacking meta-learner | Medium | Medium |
| 9 | Opening odds as features | Low | High (but philosophically debatable) |
| 10 | Weather features | Medium | Low-medium |
| 11 | Derby / rivalry flag | Low | Low-medium |
| 12 | Home advantage granularity | Low | Low-medium |
| 13 | FIFA lineup ratings | High (data sourcing) | High (if data quality is good) |
| 14 | Fix Poisson model interface | Medium | Low (marginal ensemble gain) |
| 15 | TabTransformer / neural net | Very high | High (only if dataset > 50k rows) |

---

## 6.5 Known Issues (open)

| ID | File | Issue | Impact today | Fix when |
|----|------|-------|--------------|----------|
| T0.1d-FOLLOW | `backend/features/xg_features.py:83-88` | Same `rolled[col].values` positional-assignment anti-pattern that T0.1d fixed in `form.py`. `groupby(...).rolling().mean()` returns rows in team-grouped order; `.values` strips the index and assigns positionally to a date-sorted frame, scattering each team's xG rolling stats into rows of other teams. | **None in production.** The xG builder requires StatsBomb data which isn't wired up; every `features.build` run logs `"xG features skipped — StatsBomb data unavailable"` and the columns are absent from `features.parquet`. | Before integrating any StatsBomb / xG data source. The fix is identical: drop `.values` so pandas aligns by index (one-line change × four assignments). Add a regression test mirroring `tests/test_form_alignment.py`. |

---

## 6.6 Phase 2 Non-Goals (T2.6)

These directions were **explicitly considered and rejected** for Phase 2. They are
documented here so future work — human or AI-assisted — does not re-litigate them
mid-ticket. Each is rejected for a specific reason; do not pursue them as part of
T2.1–T2.5 even if they look like a "while I'm here" easy win.

| Non-goal | Why rejected for Phase 2 |
|----------|--------------------------|
| **Bivariate Poisson as a feature** | Marginal RPS gain on top of existing Elo + form + H2H + pi-ratings (T2.3). Cost of correct implementation (Dixon-Coles dependence parameter, EM fitting, calibration sanity) is high. Revisit only if Phase 4 expanded corpus shifts the cost-benefit. |
| **Neural networks / LSTMs / Transformers** | Tabular-data literature (cf. *When Do Neural Nets Outperform Boosted Trees…*, NeurIPS 2023) shows GBTs win at our scale (~5k–10k matches). Boosted trees + stacking (T2.5) is where the marginal RPS lives. NN revisit gated on >2k-match dataset growth, already documented in `model_config.yaml`. |
| **Player embeddings** | Requires reliable lineup data (currently absent — see §2.10) and a much larger corpus to learn meaningful embeddings without overfitting. Phase 4+ ingestion work, not Phase 2. |
| **Bayesian hierarchical models** | High implementation cost (Stan/PyMC), slow inference, no clear RPS ceiling advantage over GBTs at this scale. Useful for *interpretability* of latent strength parameters — out of scope for the quality-upgrade phase. |
| **Fixing the disabled Dixon-Coles Poisson** | The placeholder Poisson model in `backend/models/poisson_model.py` is currently disabled (`enabled: false` in `model_config.yaml`) and not in the ensemble. Reactivating it requires reconciling its `predict_proba` interface with the calibrated GBT pipeline — net negative ROI when CatBoost (T2.4) is filling the "third base model" slot. Leave disabled. |

If a Phase 2 implementation step starts drifting toward any of these, **stop and
flag scope creep** rather than implementing. The expected behaviour during
T2.3-T2.5 is to point at this section instead of debating the non-goal again.

---

## 6.7 T2.1 Baseline Shift Retrospective

T2.1 (walk-forward CV + locked holdout + quality gates) shipped on
2026-04-29 across seven atomic commits on `phase2/t2.1-walk-forward-cv`.
This section anchors the Phase 2 baseline and documents how the
single-split Phase 1 metrics map onto T2.1's CV-mean + holdout pair.

### Phase 1 → T2.1 metric shift

| Metric    | Phase 1 (single-split) | T2.1 CV mean | T2.1 holdout |
|-----------|------------------------|--------------|--------------|
| RPS       | 0.2069                 | 0.2099       | 0.2079       |
| Brier     | 0.2006                 | 0.2027       | 0.2019       |
| Log loss  | 1.0872                 | 1.0584       | 1.0236       |
| Accuracy  | 0.5171                 | 0.5031       | 0.5183       |
| Draw F1   | 0.0860                 | 0.0392       | 0.0634       |

Source: `backend/data/output/eval_ensemble.json` (commit 6's first run,
re-confirmed after commit 7's snapshot verification path).

### Component decomposition

The shift is the sum of two effects (per design §2):

- **Component A — methodology change.** Single-split → walk-forward CV
  with locked holdout. Dominates RPS / Brier / Accuracy.
- **Component B — `season_stage` recovered.** Was silently NaN→0 in
  Phase 1 due to the matchday=0 bug; now varies in [0, 1] post-T2.1
  commit 1. Expected small-but-non-zero contribution.

Component decomposition is **not separately measured.** The combined
shift on aggregate metrics is small (RPS +0.0030, Brier +0.0021) and
within the per-fold std band (RPS std=0.0129 from commit 2's harness),
so the decomposition is not load-bearing for T2.2's design. Re-running
the old single-split path on corrected matchday data is parked as
optional analysis — revisit if Phase 2 metric attributions become
ambiguous.

### Notable observations

1. **Holdout draw_f1 (0.0634) > CV mean draw_f1 (0.0392).** The 2024-25
   season may carry a slightly stronger draw-prediction signal than the
   2021-2023 average. Worth re-examining after T2.2 lands SMOTE +
   class-weight handling — if the gap inverts, current fold structure
   is over-pessimistic on draws.

2. **Holdout RPS (0.2079) < CV mean RPS (0.2099).** The model
   generalises slightly better to a full 2024-25 holdout than to
   within-CV mid-season slices. The (2, 9) folds validate at
   matchdays 10-18 and 26-34 — both away from end-of-season run-ins,
   which may be inherently more predictable. Difference is ~1×
   per-fold std, so likely noise rather than signal.

3. **Empirical (2, 9) chosen over the meta-spec's (3, 6) default.**
   Commit 2's harness rejected (3, 6) on the mean-RPS sanity band —
   its early-season-2021 fold trains on only 248 rows and produces
   rps=0.33, dragging the (3, 6) mean to 0.2246 (outside the ±0.01
   band of Phase 1's 0.2069). (2, 9)'s 9-matchday warmup avoids this.
   Decision recorded in
   `backend/data/output/cv_parametrization_validation.json`.

4. **Production runtime matches the harness within 0.0003 RPS.**
   Commit 2 predicted CV mean RPS = 0.2096; commit 6 measured 0.2099.
   The same `train_calibrated_models()` helper is shared between
   harness and runtime to keep them aligned — drift here would
   silently invalidate the empirical rationale that locked (2, 9).

### Implementation deviations from design

Three structural deviations during implementation, all sound and
documented in their respective commit messages:

1. **`exceptions.py` shipped in commit 3** (design assigned commit 5).
   `splits.py` needed `FootballPredictError` as a base; defining it
   locally would have forced commit 5 to refactor it out. Splitting
   class definition (commits 3 and 4) from policy wiring (commit 5)
   keeps each commit self-contained.

2. **`evaluation/cv.py` shipped in commit 6** (design assigned commit 3).
   Fold orchestration is a training-time concern, not a splitter
   primitive. cv.py's only consumer is train.py, which lands in commit
   6 — putting cv.py with its consumer keeps the diff coherent.

3. **Tier 1 wiring landed in commit 6** (design assigned commit 5).
   Tier 1 (`cv_report.assert_gates()` in train.py) needs the cv_report
   *instance* constructed in commit 6's orchestration. Commit 5 ships
   only Tiers 2 (`test_quality_gates.py`) and 3 (`predict.py`), which
   read the eval JSON from disk and don't depend on a live cv_report.

### Forward references

- **Phase 2 baseline locked**: CV mean RPS=0.2099, Brier=0.2027,
  draw_f1=0.0392.
- **All Phase 2 ticket improvements measured against this CV mean**
  (per meta-spec §3 improvement classification).
- **T2.2 target**: draw_f1 ≥ 0.25 on CV mean. Current gap: +0.21
  improvement needed via SMOTE + class-weighted training + threshold
  calibration.

### Quality gate state at T2.1 close

| Gate | Threshold | CV mean | Holdout | Status |
|---|---|---|---|---|
| max_rps | 0.21 | 0.2099 | 0.2079 | PASS |
| max_brier | 0.22 | 0.2027 | 0.2019 | PASS |
| min_draw_f1 | 0.25 | 0.0392 | 0.0634 | **FAIL** |

The `min_draw_f1` failure is **intentional and expected** per design §6.
T2.1 ships the gate machinery; T2.2 ships the gate-passing model. Per
meta-spec §1.5 override conditions, the failure is recorded with:

- **(a) Failure cause documented**: Phase 1's known catastrophic draw F1
  (§0.3 of this document, plus §0.6 observation 2 about no draws
  predicted across the holdout).
- **(b) Follow-up ticket opened**: T2.2 — SMOTE + class weights +
  threshold calibration.
- **(c) Justification recorded**: this section.

T2.1 is the gate-shipping ticket; T2.2 is the gate-passing ticket.
First green training run arrives with T2.2.

---

## 6.8 T2.2 Three-Mechanism Ablation: Measured Negative Result and Path Forward

T2.2 (SMOTE + class weights + calibrated draw threshold θ_D) was scoped on 2026-04-30 across 13 atomic commits on `phase2/t2.2-draw-class-handling`. Tasks 1–2 shipped (draw-handling primitives, ablation harness). Task 3 ran the harness on schema-2.0 features and produced a `HarnessFailure` — no SMOTE × class-weight cell cleared the margin filter (`mean.rps ≤ 0.205 AND mean.brier ≤ 0.215` across all 6 walk-forward folds). T2.2 did not ship the gate-passing model; this section documents what was measured, what the data implies about the design, and what a successor ticket would need to test.

The single number to anchor on: **cross-fold `std.rps = 0.0102`** (from cell 2's fold-results, schema 2.0). Every margin gap below is reported in σ-units of this fold-to-fold variance — without that anchor, "0.0037 above margin" reads as either negligible or fatal depending on the reader. With it: 0.36σ.

### Ablation methodology

- **Harness:** `tools/validate_smote_classweight_composition.py`, mirroring T2.1's `validate_cv_parametrization.py` pattern.
- **Grid:** 6 cells = {`off`, `partial_70`, `auto`} × {`(1.0, 1.0, 1.0)`, `(1.0, 2.5, 1.2)`}.
- **Per-cell loop:** walk-forward CV (6 folds, locked at T2.1's `(n_splits=2, vw=9)` parametrization), SMOTE-resample on `(X_tr, y_tr)`, class-derived `sample_weight` into the same `train_calibrated_models()` path as production CV (no shortcuts; verified call-site equivalence pre-run).
- **Decision rule (locked in design §2.2):** keep cells with `mean.rps ≤ 0.205 AND mean.brier ≤ 0.215`; pick `argmax(mean.draw_f1)`; tiebreak `argmin(std.draw_f1)`. No passing cell → `HarnessFailure`, no auto-tune fallback.
- **Wall-clock:** 46s on the full 6×6. Design's 30–60 min runtime budget was ~100× too conservative; a one-fold smoke calibrated this pre-run.

### 6×6 ablation result

Source: `backend/data/output/smote_classweight_ablation.json` (`schema_version: smote_cw_ablation.v1`, `feature_schema_version: 2.0`, `winner: null`, `harness_failure_reason` populated).

| Cell | SMOTE | Weights | mean.rps | mean.brier | mean.draw_f1 ± std | margin? |
|---:|:--|:--|---:|---:|:---|:--|
| 1 | off | (1.0, 1.0, 1.0) | 0.2099 | 0.2027 | 0.039 ± 0.031 | FAIL (rps + 0.0049 = 0.48σ) |
| 2 | off | (1.0, 2.5, 1.2) | **0.2087** | **0.2022** | 0.092 ± 0.043 | FAIL (rps + 0.0037 = 0.36σ) |
| 3 | partial_70 | (1.0, 1.0, 1.0) | 0.2101 | 0.2026 | 0.081 ± 0.050 | FAIL (rps + 0.0051 = 0.50σ) |
| 4 | partial_70 | (1.0, 2.5, 1.2) | 0.2097 | 0.2028 | **0.110** ± 0.056 | FAIL (rps + 0.0047 = 0.46σ) |
| 5 | auto | (1.0, 1.0, 1.0) | 0.2123 | 0.2056 | 0.099 ± 0.054 | FAIL (rps + 0.0073 = 0.72σ) |
| 6 | auto | (1.0, 2.5, 1.2) | 0.2100 | 0.2033 | 0.105 ± 0.039 | FAIL (rps + 0.0050 = 0.49σ) |

Cell 2 has the harness-best `mean.rps` (0.2087); cell 4 has the harness-best `mean.draw_f1` (0.110). They are different cells with different "best" metrics — the headline failure is on rps in every cell, not on draw_f1.

Brier passes margin in all six cells (~0.012 of headroom on the worst cell). The bottleneck is rps.

### Mechanism analysis

1. **Class weights work, modestly.** Cells 1 vs 2 (`off`, weights toggled): `mean.rps` improves 0.2099 → 0.2087 (Δ = −0.0012, 0.12σ); `mean.draw_f1` improves 0.039 → 0.092 (+0.053 absolute, +136% relative). Weights are the only mechanism producing a measurable rps improvement — but the improvement is well within fold noise, and the resulting cell still fails margin.

2. **SMOTE adds small marginal effect on draw_f1, no effect on RPS within fold noise.** Cells 2 vs 4 (`off` → `partial_70`, weighted): `mean.draw_f1` 0.092 → 0.110 (+0.018, +20% relative) — real on the discrete metric, defensible. `mean.rps` 0.2087 → 0.2097 (+0.001, 0.10σ — noise). `auto` SMOTE (cells 5–6) trades calibration for draw_f1 with the largest rps cost (cell 5 worst in class). The "three-pronged attack" framing of the design is empirically a "weights-led, SMOTE-marginal" framing.

3. **`partial_70` no-ops on early CV folds.** The training sets in folds with thin season-2021 history already exceed the `0.7 × n_home` draw count, hitting the no-op guard in `draw_handling.resample()` (sharpened comment in commit `b5cb683`). Fold-0 evidence: cell 3 ≡ cell 1, cell 4 ≡ cell 2 to four decimals. The 6-fold mean diverges from this — `partial_70` does fire on later folds — but the cumulative effect is small (cells 1 vs 3, uniform weights: rps 0.2099 → 0.2101, draw_f1 0.039 → 0.081). Future ablations of partial-rate SMOTE should verify firing-rate per fold rather than relying on the 6-fold mean alone.

4. **Some folds are not amenable to draw recovery regardless of mechanism.** Cell 2 fold-2 produces `draw_f1 = 0.018` (effectively no draws predicted), well below the cell mean of 0.092. Per-fold fluctuation includes folds where the training corpus offers no learnable draw signal — the mechanism choice is irrelevant on those folds. T2.2B's design should consider whether walk-forward fold construction is itself a confound, not just the loss/sampling layer.

5. **The rps cluster is tight: 0.2087–0.2123 across all six cells, range 0.0036.** The margin filter draws a line at 0.205; the closest cell is 0.36σ above it. No combination of these three knobs moves rps below 0.205 — and crucially, the cells differ within roughly ±0.36σ of each other on rps, not "all the same RPS" but "all within fold noise of each other on RPS". Mechanisms here do not add to a meaningful rps reduction; they redistribute mass at the discrete classifier boundary without recalibrating the underlying probability distribution.

### Schema-2.1 sanity probe

T2.2's plan ordered SMOTE/weights ablation (Task 3) before adding three new draw-specific features (Task 8: `elo_diff_abs`, `h2h_draw_rate_last_5`, `defensive_match_indicator`). Path B of the meta-spec re-open conversation hypothesized that the Task 3 → Task 8 ordering masked a feature-driven rps improvement that would have closed the 0.0037 margin gap. A locked-pre-data decision rule was applied:

- `cell-2 schema-2.1 mean.rps ≤ 0.205` → Path B (re-order plan, features before ablation).
- `mean.rps ∈ (0.205, 0.207]` → ambiguous, Path A on discipline grounds.
- `mean.rps > 0.207` → Path A (features don't close the gap).

**Methodology.** Three new features added to the working tree (`backend/features/elo.py`, `form.py`, `context.py`); not committed. `backend/data/features/features.parquet` rebuilt at 7156 rows × 74 feature columns (vs schema-2.0's 71). Cell 2 (`off` + (1.0, 2.5, 1.2), the harness-best on rps) re-run on the full 6 folds via monkey-patched `_CELLS = [cell_2]` against the schema-2.1 parquet. Working tree reverted (`git checkout backend/features/{elo,form,context}.py`); parquet restored from a pre-experiment backup; clean state verified by re-loading the parquet and confirming the three new columns are absent.

**Result.**

| Metric | Schema 2.0 cell 2 | Schema 2.1 cell 2 | Δ |
|---|---:|---:|---:|
| `mean.rps` | 0.20872 | 0.20872 | +0.00002 |
| `mean.brier` | 0.20217 | 0.20229 | +0.00012 |
| `mean.draw_f1` | 0.092 | 0.109 | +0.017 (+18% rel.) |
| `std.rps` (across 6 folds) | 0.01022 | 0.01086 | +0.0006 |

Per-fold rps schema 2.1: `[0.2267, 0.2167, 0.2056, 0.2099, 0.1932, 0.2004]`.

**Decision.** `0.20872 > 0.207` → Path A. Features close 0.00002 of the 0.0037 rps gap (0.5% of the gap, 0.002σ of fold noise). The result is decisively above the 0.207 threshold and triggers Path A as locked.

**Interpretation.** The new features improve discrete classification at the argmax boundary (+18% relative draw_f1) but do not shift probability mass on this corpus. RPS measures the cumulative-distribution mismatch over ordered outcomes; Brier measures squared probability error. Neither moves when the underlying probability distribution is unchanged — and the new features evidently leave the distribution near-fixed while flipping a small number of borderline argmax decisions. One plausible cause: information overlap. `elo_diff_abs` is a deterministic transform of the existing `elo_difference`; `defensive_match_indicator` is a binary threshold of `home_w10_avg_ga` and `away_w10_avg_ga`, both already in the feature set; `h2h_draw_rate_last_5` is a count-window variant of the existing 5-year `h2h_draw_rate`. The boosted ensembles were extracting these signals implicitly via tree splits before the features were named.

This probe is the load-bearing piece of evidence behind Path A. Without it, the retrospective could not honestly distinguish "the design's three mechanisms are insufficient" from "the design's three mechanisms are insufficient AND the planned Task 8 features would have closed the gap". With it, the second clause is empirically rejected on this corpus.

### Why the three-mechanism design is empirically insufficient

The design assumed three mechanisms acting in concert would lift `min_draw_f1` past 0.25 while keeping `max_rps ≤ 0.21` and `max_brier ≤ 0.22`. The data shows:

1. The mechanisms do not jointly reduce rps below the 0.205 margin (best 0.2087, all six cells within ±0.36σ of each other).
2. The mechanisms move discrete metrics (draw_f1) but not probability-calibration metrics (rps, brier) — they redistribute classifier decisions at boundaries, not probability mass.
3. The new draw-specific features planned for Task 8 do not change this pattern; they amplify the discrete-vs-calibration split rather than closing it.

The gate is rps-bounded; the mechanisms act on classification-decision space; the planned features act in the same space. Therefore the design's mechanisms cannot reach the gate, regardless of cell selection or task ordering.

### Path forward

T2.2 as designed does not ship. A successor ticket — to be scoped after a meta-spec amendment formalizes the redefinition of T2.2's Definition of Done — would need to test mechanisms that act on probability mass directly. Three candidates, framed as the questions they answer:

- **Focal loss** — *Does down-weighting easy correctly-classified examples concentrate model gradient on hard / minority-class examples enough to recover draw signal in the probability distribution, not just at the argmax boundary?*
- **Weighted Brier as the optimization target** (not just metric) — *Does directly optimizing the gate metric, rather than relying on log-loss as a proxy with class weights as a corrective, close the rps gap?*
- **Ordinal regression** — *Is the structural fact that H/D/A is ordered (confusing H↔D is "less wrong" than confusing H↔A) the missing inductive bias? RPS rewards this ordering; current cross-entropy does not.*

These are candidates, not commitments. The successor ticket's first question should be empirical-feasibility before mechanism selection; see methodology lessons below.

T2.3 and beyond are unaffected and can proceed in parallel. T2.1's gate machinery remains intact; the holdout snapshot is sealed; the `min_draw_f1 ≥ 0.25` failure is recorded under §1.5 override conditions of the meta-spec (failure cause documented, follow-up work scoped via meta-spec amendment, justification recorded — this section).

### Methodology lessons

The retrospective owes one procedural observation. T2.2's brainstorm and 13-commit plan were locked before any empirical-feasibility probe. A single-cell smoke run on representative data — ~2 minutes of compute — would have surfaced the rps cluster around 0.21 and the 0.0037 gap to margin before 13 commits of plumbing were specced. This is generalizable beyond T2.2:

> **Future Phase 2 (and beyond) brainstorms should include an empirical-feasibility probe — at minimum, a single-cell run on representative data — before locking a multi-commit plan.** The probe answers a Q1 of the form "is the achievable floor on the gate metric within reach of the planned mechanisms?" before any mechanism-selection or sequencing decisions are made.

For the meta-spec amendment governing T2.2's successor ticket, the recommended Q1 is: *what is the achievable RPS floor on this corpus, and what mechanism class is required to reach it?* Mechanism scoping follows that answer; it does not lead it.

### Quality gate state at T2.2 close

| Gate | Threshold | T2.1 CV mean (carried forward) | T2.2 status |
|---|---|---|---|
| max_rps | 0.21 | 0.2099 | PASS (carried) |
| max_brier | 0.22 | 0.2027 | PASS (carried) |
| min_draw_f1 | 0.25 | 0.0392 | **FAIL (unchanged from T2.1)** |

T2.2 ships ablation evidence and a measured negative result. The gate-passing ticket remains open; its scope is governed by the forthcoming meta-spec amendment.

---

## 7. Sources

- [A predictive analytics framework for soccer match outcome forecasting (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/pii/S2772662224001413)
- [Data-driven prediction of soccer outcomes using enhanced ML/DL (Springer, 2024)](https://link.springer.com/article/10.1186/s40537-024-01008-2)
- [Large-Scale In-Game Outcome Forecasting using Axial Transformer (arXiv, 2025)](https://arxiv.org/abs/2511.18730)
- [Evaluating soccer prediction: deep learning & GBT feature optimization (Springer/ML, 2024)](https://link.springer.com/article/10.1007/s10994-024-06608-w)
- [Predicting Football Match Outcomes Using Event Data and ML (Training Ground Guru, 2025)](https://trainingground.guru/wp-content/uploads/2025/12/Predicting_Football_Match_Outcomes_Using_Event_Data_and_Machine_Learning_Algorithms.pdf)
- [From Players to Champions: Generalizable ML for Match Prediction (arXiv, 2025)](https://arxiv.org/html/2505.01902v1)
- [Which Machine Learning Models Perform Best for Football? (xGFootball Club)](https://thexgfootballclub.substack.com/p/which-machine-learning-models-perform)
- [Bayes-xG: player and position correction on xG using Bayesian hierarchical approach (Frontiers, 2024)](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1348983/full)
- [Predicting goal probabilities with improved xG using event sequences (PMC, 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11524524/)
- [A framework of interpretable match prediction with FIFA ratings and formation (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0284318)
- [The Betting Odds Rating System: using soccer forecasts to forecast soccer (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5988281/)
- [The Evolution of Football Betting: A Machine Learning Approach (arXiv, 2024)](https://arxiv.org/pdf/2403.16282)
- [When Do Neural Nets Outperform Boosted Trees on Tabular Data? (arXiv/NeurIPS, 2023)](https://arxiv.org/abs/2305.02997)
- [Can simple models predict football and beat the odds? Bundesliga (Sage, 2026)](https://journals.sagepub.com/doi/10.1177/22150218261416681)
- [Predicting sport event outcomes using deep learning (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453701/)
