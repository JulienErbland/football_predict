# Phase 0 — Ground Truth Audit

- **Date**: 2026-04-24
- **Auditor**: Claude Code (read-only; no implementation code changed)
- **Git**: branch `main`, HEAD `3831e96` — working tree is dirty (many uncommitted edits to backend/ and frontend/, plus untracked `CLAUDE.md`, `UPGRADE_PLAN.md`, `MODEL_REVIEW.md`, `RECAP.md`, `backend/data/`, `tools/*.py`, etc.). See "Git state snapshot" below.
- **Python env**: `backend/.venv` → Python 3.12.3
- **Phase status**: 0 only. STOPPED as instructed. No code changed.

This document answers 0.1 (current-state verification) and 0.2 (gap analysis). A short "Additional observations" section flags issues outside the formal Phase 0 scope that materially affect later phases.

---

## 0.1 Current state verification

### 0.1.1 Repo tree

Excludes `node_modules`, `__pycache__`, `.venv`, `.next`, `.git`, `data/raw`, `data/processed`, `data/features`, `data/models`.

```
.
├── .env                            (gitignored — contains API keys)
├── .env.example
├── .github/workflows/predict.yml   (daily cron: ingest → features → predict → commit → deploy)
├── .gitignore
├── .vercel/                        (Vercel link metadata)
├── CLAUDE.md                       (untracked — workflow guardrails)
├── CLAUDE_CODE_PROMPT.md           (untracked)
├── MODEL_REVIEW.md                 (untracked — research review, 15 papers)
├── PROJECT.md                      (architecture + setup + run docs)
├── README.md
├── RECAP.md                        (untracked — theory doc)
├── UPGRADE_PLAN.md                 (the plan this audit services)
├── backend/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── feature_config.yaml     (feature-group toggles + windows)
│   │   ├── loader.py               (YAML loader, ${ENV} resolver, lru_cache singletons)
│   │   ├── model_config.yaml       (model weights, hyperparams, eval config)
│   │   └── settings.yaml           (leagues, seasons, bookmakers, paths)
│   ├── data/                       (untracked, outputs present — see 0.1.4)
│   │   ├── features/features.parquet
│   │   ├── models/{ensemble,xgboost,lightgbm,feature_cols}.pkl
│   │   ├── output/{predictions,predictions_test}.json
│   │   ├── processed/all_matches.parquet
│   │   └── raw/{football_data,odds}/…
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              (brier, log_loss, RPS, accuracy, calibration_summary)
│   │   └── report.py               (JSON eval report writer)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build.py                (orchestrator; calls every enabled builder)
│   │   ├── context.py              (rest days, congestion, season stage, league position, referee)
│   │   ├── elo.py                  (running Elo, K=32, initial 1500, expected-home)
│   │   ├── form.py                 (rolling form windows + 5y H2H)
│   │   ├── squad_features.py       (Transfermarkt squad value / age / injuries — optional)
│   │   ├── tactical.py             (formation pairs + archetype scores — optional)
│   │   └── xg_features.py          (StatsBomb xG / xGA / PPDA — optional)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── football_data.py        (football-data.org v4 client — 10 req/min sleep)
│   │   ├── football_data_csv.py    (untracked — free football-data.co.uk CSV alternative)
│   │   ├── statsbomb.py            (statsbombpy wrapper; xG, formations, PPDA)
│   │   └── transfermarkt.py        (BeautifulSoup scraper, 3s sleep, parquet cached)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 (BaseModel ABC + CalibratedModel wrapper)
│   │   ├── ensemble.py             (weighted-average ensemble + unused StackingEnsemble class)
│   │   ├── lgbm_model.py
│   │   ├── neural_net.py           (PyTorch MLP stub — disabled)
│   │   ├── poisson_model.py        (Dixon-Coles — disabled per predict_proba signature mismatch)
│   │   ├── train.py                (orchestrator; time-split; note: Poisson/NN fit() are placeholder)
│   │   └── xgboost_model.py
│   ├── odds/
│   │   ├── __init__.py
│   │   ├── fetcher.py              (The Odds API v4 client; daily per-league parquet cache)
│   │   └── value.py                (implied_probability, remove_vig, compute_edge, kelly_fraction)
│   ├── output/
│   │   ├── __init__.py
│   │   └── predict.py              (end-to-end inference → predictions.json)
│   └── requirements.txt
├── frontend/                        (Next.js; many recent rewrites, uncommitted)
│   ├── AGENTS.md                    ("This is NOT the Next.js you know" — check node_modules docs)
│   ├── CLAUDE.md → AGENTS.md
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   ├── methodology/page.tsx
│   │   └── page.tsx
│   ├── components/                  (Hero3D, HomeView, LeagueView, MatchView, MergedSite, Primitives)
│   ├── eslint.config.mjs
│   ├── lib/{fixtures,teams}.ts
│   ├── next.config.ts
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.mjs
│   ├── public/{data/predictions.json, *.svg, favicon…}
│   ├── tsconfig.json
│   └── types/predictions.ts
├── the-odds-api.json                (committed copy of an Odds API response — ~100 KB, should probably be gitignored)
├── tools/
│   ├── dashboard.py                 (Streamlit analysis dashboard, 4 tabs, ~47 KB)
│   ├── generate_predictions.py      (one-shot real-odds prediction script)
│   ├── make_demo_predictions.py     (demo-data JSON writer for free tier)
│   └── pipeline_test.py             (end-to-end smoke test — see 0.1.3)
└── vercel.json
```

### 0.1.2 Module inventory — `backend/` and `tools/`

#### `backend/config/`

| File | One-line description |
|------|----------------------|
| `__init__.py` | Empty package marker |
| `loader.py` | Loads `settings.yaml`, `feature_config.yaml`, `model_config.yaml` with `${ENV_VAR}` substitution from `.env`; exposes three `@lru_cache` singletons (`settings()`, `feature_config()`, `model_config()`) and rewrites relative `paths.*` to absolute paths via `_REPO_ROOT`. |
| `settings.yaml` | API keys, 5 leagues (PL/PD/BL1/SA/FL1), seasons `[2020..2024]` (current=2024), bookmakers, markets, data paths. |
| `feature_config.yaml` | Toggles `form` (windows 3/5/10), `xg`, `squad`, `tactics`, `context`, `head_to_head` (5y), `elo` (K=32, initial=1500). |
| `model_config.yaml` | Poisson disabled; XGBoost (w=0.60, ES=50, 500 trees, lr=0.05, depth=6); LightGBM (w=0.40, 500 trees, 63 leaves); NN disabled; isotonic calibration (5-fold); `test_seasons=[2024]`; primary metric `brier_score`. |

#### `backend/ingestion/`

| File | One-line description |
|------|----------------------|
| `football_data.py` | v4 client: `fetch_matches`, `fetch_standings`, `fetch_upcoming` (next 10), `fetch_lineups`; 6.5 s sleep between calls (10 req/min free tier); concat to `all_matches.parquet`. |
| `football_data_csv.py` | *(untracked)* Free alternative: downloads football-data.co.uk CSVs, hashes synthetic `match_id`/`team_id`, writes same schema — `matchday=0`. |
| `statsbomb.py` | `statsbombpy` wrapper: per-match xG/shots/SoT/PPDA and lineup formations; saved as `xg_{comp}_{season}.parquet` and `formations_{comp}_{season}.parquet`. |
| `transfermarkt.py` | BeautifulSoup scraper with browser UA, 3 s sleep, parquet cache; writes `{LEAGUE}_squad_values.parquet`, per-team `players_{id}.parquet`, `injuries_{id}.parquet`. |

#### `backend/features/`

| File | One-line description |
|------|----------------------|
| `elo.py` | Running Elo (K=32, init 1500), chronological, emits `home_elo/away_elo/elo_difference/elo_expected_home`; also `get_current_ratings()` for inference. |
| `form.py` | `build_form_features` (rolling win/draw/loss/PPG/GF/GA/GD/clean-sheet over windows 3/5/10, using `groupby.shift(1).rolling(w).mean()` to prevent leakage) and `build_h2h_features` (5-year reversible lookup). |
| `xg_features.py` | Rolling xG/xGA/PPDA over windows 5/10 from StatsBomb parquet; returns `None` (graceful skip) if absent. |
| `squad_features.py` | Join-by-team-name Transfermarkt squad value, avg age, injury count; returns `None` if absent. |
| `tactical.py` | Formation ID encoding (12 known), pair-hash, historical pair-wise win rates, and pace/aerial/technical archetype scores from lineup positions; returns `None` if absent. |
| `context.py` | Rest days, 30-day congestion, season stage, referee label-encoding, chronological league-position computation (expensive Python loop), relegation-pressure and title-race flags. |
| `build.py` | Orchestrator: loads `all_matches.parquet`, calls every enabled builder, writes `features.parquet`. Optional builders log a warning and skip when input data is absent. |

#### `backend/models/`

| File | One-line description |
|------|----------------------|
| `base.py` | `BaseModel` ABC (fit/predict_proba/save/load), `.calibrate()` returns `CalibratedModel` — a wrapper that fits per-class `IsotonicRegression(out_of_bounds="clip")` and renormalises to sum-to-1. |
| `xgboost_model.py` | `multi:softprob` multi-class, `hist` tree method, XGBoost 3.x API (early_stopping on constructor, not `.fit`). |
| `lgbm_model.py` | `LGBMClassifier` with `early_stopping` callback; re-wraps inputs in a DataFrame with stored `feature_names_` on predict to avoid LightGBM feature-name warnings. |
| `poisson_model.py` | Dixon-Coles: MLE for attack/defence/home-adv/ρ via `scipy.optimize.L-BFGS-B`; `predict_proba(X)` expects `(home_team, away_team)` rows (incompatible with the matrix interface — **disabled in the ensemble**). |
| `neural_net.py` | PyTorch MLP stub (BatchNorm + Dropout + CE + Adam). Only enabled if dataset > 2000 rows; fit-call commented out in `train.py`. |
| `ensemble.py` | `EnsembleModel`: weighted-average over trained models, normalised over *enabled* models; renormalises rows post-sum. `StackingEnsemble`: LR meta-learner — class present, **not instantiated anywhere in the current training flow**. |
| `train.py` | Loads features, time-splits by `test_seasons`, uses last 10% of train as val for early stopping, fits XGB + LGBM, calibrates each via isotonic, builds `EnsembleModel`, saves artefacts to `data/models/`. Poisson and NN `fit()` blocks are explicit placeholders. |

#### `backend/evaluation/`

| File | One-line description |
|------|----------------------|
| `metrics.py` | `brier_score` (sum over 3 classes, **not** mean), `log_loss_score`, `rps` (sum over 2 thresholds, not divided by K-1), `accuracy`, `evaluate_predictions`, `calibration_summary` (10-bin per-class). |
| `report.py` | Writes `eval_{model_name}.json` to `data/output/` with metrics + calibration + top-50 feature importances. |

#### `backend/odds/`

| File | One-line description |
|------|----------------------|
| `fetcher.py` | The Odds API v4 client; per-league sport-key table; per-league per-day parquet cache; returns empty DF on request failure. |
| `value.py` | `implied_probability`, `remove_vig` (vig-stripped probs that sum to 1), `compute_edge`, `kelly_fraction` (capped at 25 %), `find_value_bets` (tier thresholds: high≥10 %, medium≥7 %, default low). |

#### `backend/output/`

| File | One-line description |
|------|----------------------|
| `predict.py` | Inference pipeline: load ensemble + feature_cols artefact → for each league fetch upcoming (football-data.org) + odds → append upcoming-match row to historical `all_matches.parquet` → recompute Elo/form/H2H/context → extract the just-built row by `match_id` → run ensemble → compute odds comparison + value bets → write `predictions.json`. |

#### `tools/`

| File | One-line description |
|------|----------------------|
| `dashboard.py` | Streamlit analysis dashboard (4 tabs: Performance / Feature Importance / Match Inspector / Upcoming Matches); ~47 KB; SHAP force plots (two-layer unwrap), calibration curves, feature violin plots. |
| `generate_predictions.py` | One-shot real-pipeline predictor with live Odds API calls. |
| `make_demo_predictions.py` | Generates a demo `predictions.json` from odds-only data for the free tier when no trained model is ready. |
| `pipeline_test.py` | End-to-end synthetic smoke test (270 matches × 10 teams × 3 seasons, 6 steps). Note: writes to the *same* paths as production (`data/processed/`, `data/features/`, `data/models/`) — running it without backup clobbers the real trained artefacts. |

### 0.1.3 Smoke-test results

Command (as specified in the plan):

```bash
backend/.venv/bin/python3 tools/pipeline_test.py
```

**All 6 steps passed**, total runtime ≈ 10 s. Full log in `/tmp/pipeline_test_run.log`.

| Step | Outcome | Notes |
|------|---------|-------|
| 1 — Import all modules | OK | 12 modules |
| 2 — Synthetic data | OK | 270 rows generated |
| 3 — Feature engineering | OK | 71 feature columns (Elo, Form, H2H, Context) |
| 4 — Train XGB+LGBM ensemble | OK | XGB 50 est, LGBM 14 est (early stopped), saved ensemble artefact |
| 5 — Evaluation metrics | OK | Brier 0.6648, LogLoss 1.0869, RPS 0.5034, Accuracy 43.3 % (synthetic data, not meaningful) |
| 6 — Predictions + Σ prob = 1 | OK | 5 mock predictions, all probabilities sum to 1 |

**Warnings / caveats observed**:
- None from the test itself.
- ⚠️ *Process side-effect*: Steps 2–4 overwrite `data/processed/all_matches.parquet`, `data/features/features.parquet`, and the `data/models/` artefacts. To preserve the current production model and historical data, I backed those files up to `/tmp/football_predict_audit_backup_36427/` before running and restored them afterwards. Re-running the smoke test in the normal workflow will destroy the real trained model. Worth noting for Phase 1 — the smoke test should either use a sandbox path or the production train should commit versioned artefacts.

### 0.1.4 Current model quality

**Artefacts on disk** (at audit time, before running the smoke test):

| Path | Size | Last modified |
|------|------|---------------|
| `backend/data/models/ensemble.pkl` | 1 697 312 B | 2026-04-09 21:30 |
| `backend/data/models/xgboost.pkl` (calibrated) | 1 096 140 B | 2026-04-09 21:30 |
| `backend/data/models/lightgbm.pkl` (calibrated) | 602 711 B | 2026-04-09 21:30 |
| `backend/data/models/feature_cols.pkl` | 1 350 B | 2026-04-09 21:30 |
| `backend/data/features/features.parquet` | 741 971 B | 2026-04-09 21:16 |
| `backend/data/processed/all_matches.parquet` | 117 636 B | 2026-04-09 19:02 |
| `backend/data/output/predictions.json` | 96 604 B | 2026-04-10 00:33 |

**Training corpus** (from `features.parquet`):
- **7 156 rows × 84 columns** (71 feature columns + 13 meta columns).
- **Leagues**: BL1, FL1, PD, PL, SA (top-5 European).
- **Seasons**: 2021, 2022, 2023, 2024.
- **Class balance**: 43.4 % home win (0), 25.4 % draw (1), 31.2 % away win (2) — draw rate consistent with literature (~25 %).
- **Rows per (league, season)** — Bundesliga 306 per season (18-team league); FL1 drops from 380 to 306 after 2022 (real 20-team → 18-team Ligue 1 reform).

**Train / test split used by the saved ensemble** (per `model_config.yaml: test_seasons=[2024]`):
- Train: **5 404 rows** (2021 + 2022 + 2023, `result notna`)
- Test: **1 752 rows** (2024 only)
- Test class counts: H=736, D=437, A=579

**Ensemble composition**: `xgboost` (w=0.600) + `lightgbm` (w=0.400). Poisson disabled. Weights normalised over enabled models.

**Held-out 2024 test metrics** (ensemble with isotonic calibration):

| Metric | Value | Note |
|--------|-------|------|
| Accuracy | **0.5211** (52.11 %) | vs always-predict-home baseline of 42.0 % |
| `brier_score` | **0.5997** | non-standard convention: sum across 3 classes (≈ 3× per-class Brier → per-class ≈ 0.200 — in-line with literature) |
| `log_loss` | **1.0617** | random-3-class baseline ≈ 1.099; so only a small edge on log-loss |
| `rps` | **0.4117** | non-standard convention: sum over 2 thresholds, not divided by (K-1)=2 → normalised ≈ 0.206 (literature expert range 0.19–0.23) |
| macro F1 | **0.4156** | |
| F1 home (0) | **0.6200** | recall 0.7038, precision 0.5540 |
| F1 **draw** (1) | **0.0667** | recall 0.0366, precision 0.3721 — **argmax picks draw only 2.5 % of the time** |
| F1 away (2) | **0.5602** | recall 0.6546, precision 0.4897 |

**Confusion matrix** (rows = true, cols = argmax predicted):

| true\pred | H | D | A |
|-----------|---|---|---|
| H | 518 | 13 | 205 |
| D | 231 | 16 | 190 |
| A | 186 | 14 | 379 |

**Calibration highlights** (bins with ≥ 20 samples):
- Home: well-calibrated from 0.20–0.80 (avg_pred within ±0.05 of actual rate).
- Away: well-calibrated 0.20–0.70; over-confident in the 0.70–0.80 bucket (pred 0.74 vs actual 0.625, n=40).
- **Draw: probabilities are squashed into a narrow band.** 85 % of the test set falls in the `[0.20, 0.30)` draw-prob bin (n=1 492, avg_pred 0.246, actual rate 0.245 — so well-calibrated, but effectively constant). The model almost never emits a draw probability > 0.30. This is the classic "draw underestimation" failure the plan targets — argmax can almost never choose draw because home or away always beat the squashed draw value.

**Summary — the existing numbers are the baseline for Phase 2**:
- Home/Away: ~0.62 / 0.56 F1, competitive with published models.
- Draw: **F1 ≈ 0.07 is catastrophically bad**; the current model is essentially a binary home-vs-away classifier with a flat draw prior.
- Reported Brier and RPS scale differently from the literature. See **Additional observations** for the implication for the plan's `min_rps=0.21` gate.

### 0.1.5 Dependency audit

Declared in `backend/requirements.txt` (all `>=` minima, no upper bounds, no lockfile):

```
pandas>=2.0          numpy>=1.24          polars>=0.20
pyarrow>=14.0        scikit-learn>=1.4    xgboost>=2.0
lightgbm>=4.0        scipy>=1.11          requests>=2.31
beautifulsoup4>=4.12 lxml>=4.9            statsbombpy>=1.1
pyyaml>=6.0          python-dotenv>=1.0   tqdm>=4.66
loguru>=0.7
```

**Installed in `backend/.venv` today** (selected; from `pip freeze`):

| Declared | Installed | Status |
|----------|-----------|--------|
| `pandas>=2.0` | **3.0.2** | ⚠️ major-version bump past minimum; fine but API deltas possible |
| `numpy>=1.24` | **2.4.4** | ⚠️ numpy 2 — any remaining 1.x-only code would crash; none observed |
| `scikit-learn>=1.4` | **1.8.0** | ⚠️ note: scikit-learn 1.8 emits a `FutureWarning` for `LogisticRegression(multi_class=...)` — `StackingEnsemble` in `models/ensemble.py` passes `multi_class="multinomial"`; if it gets instantiated in Phase 2 T2.5 it will need that arg removed. |
| `xgboost>=2.0` | **3.2.0** | OK (early-stopping-on-constructor pattern already honoured in `xgboost_model.py`) |
| `lightgbm>=4.0` | **4.6.0** | OK |
| `scipy>=1.11` | **1.17.1** | OK |
| `requests>=2.31` | **2.33.1** | OK |
| `beautifulsoup4>=4.12` | **4.14.3** | OK |
| `lxml>=4.9` | **6.0.2** | OK |
| `statsbombpy>=1.1` | **1.17.0** | OK |
| `pyarrow>=14.0` | **23.0.1** | OK |
| `polars>=0.20` | **not installed** | ⚠️ declared but absent; no `polars` import found in code — safe to drop from `requirements.txt`. |
| `pyyaml>=6.0` | **6.0.3** | OK |
| `python-dotenv>=1.0` | **1.2.2** | OK |
| `tqdm>=4.66` | **4.67.3** | OK |
| `loguru>=0.7` | **0.7.3** | OK |

**Extras present but not declared** (installed ad-hoc; add to `requirements.txt` or a `requirements-dev.txt`):
- `streamlit==1.56.0`, `plotly==6.6.0`, `shap==0.51.0` — used by `tools/dashboard.py`.
- `altair`, `pydeck`, `watchdog`, `GitPython`, `requests-cache`, `cloudpickle`, `typeguard`, `inflect`, `numba`, `llvmlite`, `narwhals` — transitive deps of the above.

**Missing for the plan's later phases** (confirmed absent via `pip freeze`):
- `imbalanced-learn` / `imblearn` (SMOTE for T2.2) — **missing**
- `hypothesis` (leakage test T1.3) — **missing**
- `catboost` (T2.4) — **missing**
- `fastapi`, `supabase`, `stripe` (Phases 4–6) — **missing**
- `openmeteo` / weather client (T3.2) — **missing**
- `torch` (optional NN; not used) — **missing** (fine per decision)

**Staleness / unpinned risk**:
- `requirements.txt` has **no upper bounds** and **no lock file** — a fresh install could silently upgrade across major versions at any time. Nothing is "stale" (>1 year old) in the currently installed environment, but the manifest offers zero reproducibility guarantee. A Phase 1 hardening step (without expanding the plan) could pin exact versions or add `pip-compile` output.

---

## 0.2 Gap analysis against target

| Area | Target | Status | Evidence |
|------|--------|--------|----------|
| Walk-forward cross-validation | Season-aware expanding-window CV | **MISSING** | `backend/models/train.py:70` uses a single season-based split (`test_seasons=[2024]`); no `SeasonSplit` class; `backend/evaluation/splits.py` does not exist. |
| Draw-class handling | SMOTE + class weights + threshold tuning | **MISSING** | No SMOTE (`imblearn` not installed); XGBoost/LightGBM are not passed `class_weight` params; no θ_D threshold tuning; argmax inference only. Observed Draw F1 = 0.07 on 2024, Draw recall = 3.7 % — well below the 0.25 gate. |
| Pi-ratings | Pi-ratings as additional feature alongside Elo | **MISSING** | No `features/pi_ratings.py`; only Elo is implemented (`features/elo.py`). |
| Stacking ensemble | LR meta-learner over base models | **PARTIAL** | `models/ensemble.py` defines a `StackingEnsemble` class with an LR meta-learner, but it is **never instantiated** — `train.py` only builds `EnsembleModel` (weighted average). `use_calibrated_oof` config key does not exist. Also: the hard-coded `multi_class="multinomial"` arg will break on the currently installed scikit-learn 1.8. |
| CatBoost base model | Third tree model in ensemble | **MISSING** | `catboost` not installed; no `models/catboost_model.py`. |
| Narrative generator | SHAP → human-readable explanations | **MISSING** | SHAP is used inside `tools/dashboard.py` for plots, but no narrative-template generator; `backend/output/narrative.py` does not exist; `predictions.json` has no `narrative` field. |
| Weather features | OpenWeatherMap integration | **MISSING** | No `backend/ingestion/weather.py`; no `backend/features/weather_features.py`; no `config/stadiums.yaml`. |
| Supabase backend | DB schema + auth + API | **MISSING** | No `supabase/` directory; no migration files; `supabase` package not installed. |
| FastAPI service | Premium-tier endpoints | **MISSING** | No `backend/api/`; `fastapi` not installed. |
| Live odds refresh | Hourly odds snapshots | **MISSING** | `backend/odds/fetcher.py` exists and caches per-day, but there is no hourly refresh job, no snapshot table, and no cron workflow (`.github/workflows/predict.yml` runs daily, inference only). |
| Backtest engine | Walk-forward ROI simulation | **MISSING** | No backtest module; `odds/value.py` has per-match Kelly/edge computation only, not a historical simulator. |
| User tiers | Free / Premium / Pro with Stripe | **MISSING** | No auth, no billing, no `users` schema; Stripe not integrated. |
| Leakage tests | Property-based tests asserting no post-kickoff data affects pre-match features | **MISSING** | No `backend/tests/` directory exists at all; `hypothesis` not installed. Per-builder leakage prevention is *documented in-code* (`shift(1).rolling(w)`, `cutoff` by date, chronological Elo), but nothing mechanically enforces it. |

Summary counts: **13 target areas → 1 PARTIAL, 12 MISSING, 0 HAVE**.

---

## 0.3 Additional observations

These are outside the formal Phase 0 checklist but are directly load-bearing for the plan. Flagging now so Phase 1/2 scope can absorb them.

1. **Production `predictions.json` is degenerate — all Premier League matches currently return `0.387 / 0.232 / 0.381` probabilities with `home_elo = 1500` (the initial rating)** (see `backend/data/output/predictions.json` and `frontend/public/data/predictions.json`). The saved ensemble is not broken — spot-check on a real 2024 test row emits varied probs (`H=0.639 D=0.240 A=0.121`). The root cause is a **team-name join mismatch**: `ingestion/football_data_csv.py` writes names like "Arsenal" (bare) while `ingestion/football_data.py`'s `fetch_upcoming` returns "Arsenal FC" (with suffix). In `backend/output/predict.py` the upcoming-match row is appended to the CSV-based historical DataFrame and the team lookups fall back to defaults for every feature that is team-keyed (Elo, Form, H2H). Every match therefore gets an identical feature vector → identical probabilities. Either the ingestion sources must be reconciled to a single canonical name, or a normalisation layer must sit between. This is user-visible and should be high priority regardless of the plan.

2. **Metrics convention in `evaluation/metrics.py` is non-standard.** `brier_score` returns the *sum* over K=3 classes (not the mean); `rps` returns the *sum* over K-1=2 thresholds (not the (K-1)-normalised form). This is fine as long as the same function is used to evaluate every candidate model, but:
   - The plan's deployment gate `max_rps: 0.21` (Phase 1 T1.2 typed config) is expressed in the **literature convention** (RPS ÷ 2). The current ensemble's value in that convention is ≈ 0.206 — already inside the gate — but 0.412 in this repo's convention. A Phase 1 note should fix this: either rename the metric, divide the result, or rewrite the gate value.
   - Same issue for Brier: literature "per-class Brier ≈ 0.20" corresponds to this repo's 0.60.

3. **Training pipeline has "placeholder" fit() blocks for Poisson and NN.** `models/train.py:115-132, 174-188` — both are commented out. Nothing breaks today because both models are disabled in `model_config.yaml`. Keep them disabled (matches plan's T2.6 non-goals) but note that anyone re-enabling them will hit a no-op.

4. **Daily workflow (`.github/workflows/predict.yml`) is a single cron that runs inference and commits `predictions.json` to git.** Any Phase 4+ work that moves to an hourly odds refresh needs a second workflow; the existing one should keep running (free tier stays on static JSON per the plan's invariant).

5. **`tools/pipeline_test.py` writes to the same `data/models/` path as production.** A good Phase 1 hardening, parallel to T1.3's leakage test, is to route the smoke test through a per-run sandbox dir (e.g. `tmp_smoke/`) so running it doesn't clobber the real trained artefacts. This would also mean the test could run in CI without side-effects. Not in the plan today — I flag it because I had to back up/restore the real artefacts to complete this very audit.

6. **No `backend/tests/` directory exists.** Phase 1 T1.3 (leakage test) and Phase 2 gates need `pytest` + `hypothesis` scaffolding; both are currently absent. Not a gap in the plan's gap-analysis row — captured already — but worth noting that the first Phase 1 ticket may need to add `pytest`/`hypothesis` to `requirements.txt` (or a `requirements-dev.txt`) and create the directory layout.

7. **`StackingEnsemble` (currently dormant) uses `LogisticRegression(multi_class="multinomial")` which is deprecated in scikit-learn 1.8** (installed). Phase 2 T2.5 will need to drop the arg when it starts wiring stacking in.

8. **`requirements.txt` declares `polars>=0.20` but polars is not imported anywhere** (grep-confirmed). Safe to delete from the manifest in Phase 1 config/deps hygiene.

9. **Frontend invariant** (`frontend/AGENTS.md`): the repo's Next.js version has breaking changes vs. training-data Next.js — the AGENTS doc tells any agent to consult `node_modules/next/dist/docs/` before editing. This is not a backend concern for Phase 0, but later phases that touch the frontend (T3.x badges, T5.x API surface, T6.x paywall) should honour that rule.

---

## Git state snapshot (HEAD `3831e96` on `main`)

Summary (uncommitted at audit start; the audit did not modify any of these):

- Modified (12): `.gitignore`, `PROJECT.md`, `backend/config/loader.py`, `backend/config/model_config.yaml`, `backend/models/lgbm_model.py`, `backend/models/train.py`, `backend/models/xgboost_model.py`, `backend/odds/fetcher.py`, `backend/output/predict.py`, `frontend/app/{globals.css,layout.tsx,methodology/page.tsx,page.tsx}`, `frontend/package.json`, `frontend/package-lock.json`, `frontend/public/data/predictions.json`.
- Deleted (4 frontend components): `LeagueFilter.tsx`, `MatchCard.tsx`, `PredictionsDashboard.tsx`, `ValueBetBadge.tsx`.
- Untracked (incl. plan docs and data): `CLAUDE.md`, `CLAUDE_CODE_PROMPT.md`, `MODEL_REVIEW.md`, `RECAP.md`, `UPGRADE_PLAN.md`, `backend/data/`, `backend/ingestion/football_data_csv.py`, `frontend/components/{Hero3D,HomeView,LeagueView,MatchView,MergedSite,Primitives}.tsx`, `frontend/lib/`, `the-odds-api.json`, `tools/{dashboard,generate_predictions,pipeline_test}.py`.

The plan's Phase 1 T1.1 (`CLAUDE.md`) already exists on disk but is **untracked** — it has not yet been committed. The plan's stated "Phase 1 — create `CLAUDE.md` guardrails" ticket is therefore already partially done; the open question for Phase 1 will be whether the existing file (`football_predict/CLAUDE.md`) matches the plan's specified contents, or whether the plan's version should supersede it.

---

## STOP — awaiting human review

Per the plan: "After producing the audit, do not proceed. Wait for human review."

No implementation code was touched during this audit. The production model artefacts were backed up to `/tmp/football_predict_audit_backup_36427/` before running the smoke test and restored bit-for-bit afterwards (verified by an inference spot-check: `H=0.639 D=0.240 A=0.121`, consistent with the pre-audit state).

Ready for your review and green-light on Phase 1 scope.
