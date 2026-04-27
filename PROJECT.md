# Football ML Predictions — Project Documentation

## What this is

A full-stack football match prediction system that uses a weighted ensemble of XGBoost and LightGBM (with isotonic calibration) to forecast 1X2 outcomes (home win / draw / away win) for the top 5 European leagues. Predictions are pre-generated nightly as static JSON by a Python backend pipeline, then served via a Next.js dashboard deployed on Vercel. The dashboard also surfaces value bet opportunities by comparing model probabilities against live bookmaker odds.

A Streamlit analysis dashboard (`tools/dashboard.py`) provides interactive model interpretation: feature importances, SHAP values, calibration curves, and per-match prediction vs. actual breakdowns.

**Pipeline status: fully operational.** Training, evaluation, prediction, and the analysis dashboard all run end-to-end.

---

## Architecture diagram

```
football_predict/
├── .github/
│   └── workflows/
│       └── predict.yml          # Daily cron: ingest → features → predict → commit → Vercel redeploy
│
├── backend/
│   ├── config/
│   │   ├── settings.yaml        # Leagues, seasons, bookmakers, API keys, data paths
│   │   ├── feature_config.yaml  # Toggle feature groups on/off, set rolling windows
│   │   ├── model_config.yaml    # Model weights, hyperparams, ensemble method, eval metrics
│   │   │                        #   XGBoost: weight=0.60 | LightGBM: weight=0.40 | Poisson: disabled
│   │   └── loader.py            # YAML loader with ${ENV_VAR} resolution, lru_cache singletons
│   │
│   ├── ingestion/
│   │   ├── football_data.py     # football-data.org v4 client: matches, standings, upcoming, lineups
│   │   ├── statsbomb.py         # StatsBomb open data: xG, PPDA, formations, lineups
│   │   └── transfermarkt.py     # Transfermarkt scraper: squad values, injuries, player list
│   │
│   ├── features/
│   │   ├── elo.py               # Running Elo ratings (K=32), Elo-implied probability
│   │   ├── form.py              # Rolling form stats (3/5/10 games) + H2H features
│   │   ├── xg_features.py       # Rolling xG for/against, PPDA (optional: StatsBomb)
│   │   ├── squad_features.py    # Squad value, avg age, injury count (optional: Transfermarkt)
│   │   ├── tactical.py          # Formation encoding, matchup win rates (optional: lineups)
│   │   ├── context.py           # Rest days, congestion, season stage, league position, referee
│   │   └── build.py             # Master orchestrator: loads all builders, merges, saves Parquet
│   │
│   ├── models/
│   │   ├── base.py              # Abstract BaseModel + calibration interface
│   │   ├── poisson_model.py     # Dixon-Coles Poisson (disabled — predict_proba interface mismatch)
│   │   ├── xgboost_model.py     # XGBoost multi:softprob classifier (early_stopping in constructor)
│   │   ├── lgbm_model.py        # LightGBM multiclass classifier
│   │   ├── neural_net.py        # STUB ONLY — disabled; see note below
│   │   ├── ensemble.py          # Weighted average ensemble
│   │   └── train.py             # Training orchestrator: split → train → calibrate → save
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # brier_score, log_loss, RPS, accuracy, calibration_summary
│   │   └── report.py            # Saves per-model JSON eval report to data/output/
│   │
│   ├── odds/
│   │   ├── fetcher.py           # The Odds API v4 client with daily Parquet caching
│   │   │                        #   PL key: "soccer_epl" (not soccer_england_premier_league)
│   │   └── value.py             # implied_probability, remove_vig, compute_edge, kelly_fraction
│   │
│   ├── output/
│   │   └── predict.py           # Full pipeline: upcoming → features → predict → odds → JSON
│   │
│   ├── data/
│   │   ├── raw/                 # Raw Parquet files per source (gitignored)
│   │   ├── processed/           # all_matches.parquet (gitignored)
│   │   ├── features/            # features.parquet (gitignored)
│   │   ├── models/              # ensemble.pkl, feature_cols.pkl (gitignored)
│   │   └── output/              # predictions.json, predictions_test.json (included in git)
│   │
│   └── requirements.txt
│
├── frontend/                    # Next.js dashboard deployed on Vercel
│   └── ...
│
├── tools/
│   ├── dashboard.py             # Streamlit analysis dashboard (4 tabs — see below)
│   ├── pipeline_test.py         # End-to-end smoke test with synthetic data (no API needed)
│   └── generate_predictions.py  # One-shot prediction script (uses real Odds API)
│
├── RECAP.md                     # Theory doc: features, maths, models, evaluation (no code)
├── MODEL_REVIEW.md              # Research review: 15 papers, 10 missing features, improvement roadmap
├── .env.example
├── .gitignore
├── vercel.json
└── PROJECT.md
```

---

## Why no PyTorch?

With ~10,000 matches across 5 leagues (roughly 2,000 per season), this is firmly tabular ML territory. XGBoost and LightGBM consistently outperform neural nets on structured tabular data at this scale. Adding PyTorch would introduce a heavy dependency, GPU complexity, and longer training times for likely *worse* results.

`neural_net.py` is a commented stub. Revisit if the dataset ever grows beyond ~50,000 rows, or if player-level embeddings (FBref data) become part of the feature set.

---

## Analysis dashboard

`tools/dashboard.py` is a Streamlit app for interactive model interpretation. Run with:

```bash
backend/.venv/bin/streamlit run tools/dashboard.py
```

**Tab 1 — Model Performance**
- Metric cards: Brier score, Log loss, RPS, Accuracy (train vs. test split)
- Confusion matrix and calibration curves
- Predicted probability distribution by actual outcome class

**Tab 2 — Feature Importance**
- XGBoost / LightGBM native feature importance bar chart
- On-demand SHAP global summary (samples 500 matches, ~10–30s)
- Violin plots: feature distribution by outcome class
- Scatter: compare any two features across all matches

**Tab 3 — Match Inspector (Predictions vs Reality)**
- Filters: league, season, "correct only" / "wrong only", minimum confidence
- Overview table of all matches: predicted vs. actual, colour-coded green/red
- Summary stats: accuracy, avg confidence when correct vs. wrong
- Deep dive on any selected match: probability bar (colour shows correct/wrong/actual),
  full team comparison table (Elo, form, H2H, context), full feature table with
  home/away row highlighting, SHAP force plot (after Tab 2 computation)

**Tab 4 — Upcoming Matches**
- League filter and value-bets-only toggle
- Per-match expandable cards: probabilities, value bets, odds comparison, key feature breakdown

---

## Smoke test (no API needed)

`tools/pipeline_test.py` generates 270 synthetic matches (10 teams × 3 seasons, round-robin),
runs all feature builders, actually trains XGBoost + LightGBM, evaluates metrics, and writes
`predictions_test.json` — without any external API call. All 6 steps complete in under 60 seconds.

```bash
backend/.venv/bin/python3 tools/pipeline_test.py
```

Use this to verify that a code change hasn't broken the pipeline before committing.

---

## Setup instructions

### Prerequisites
- Python 3.12+
- Node.js 18+ (frontend only)
- API keys: football-data.org (free) and The Odds API (free, 500 req/month)

### 1. Clone and configure environment
```bash
git clone https://github.com/JulienErblandEPFL/football_predict
cd football_predict
cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Set up Python environment
```bash
python -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
# Dashboard extras (if not already in requirements.txt):
backend/.venv/bin/pip install streamlit plotly shap
```

### 3. Verify the pipeline works (no API key needed)
```bash
backend/.venv/bin/python3 tools/pipeline_test.py
# All 6 steps should print OK
```

### 4. Launch the analysis dashboard
```bash
backend/.venv/bin/streamlit run tools/dashboard.py
# Opens at http://localhost:8501
# Sidebar shows which data files are loaded (features / models / predictions)
```

### 5. Set up the frontend
```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

---

## How to run the full pipeline

Run these commands in order. The full pipeline (historical ingestion → features → training → prediction) takes 30–90 minutes.

```bash
cd backend

# Step 1: Ingest historical match data (~2h for 5 leagues × 4 seasons due to API rate limits)
python -m ingestion.football_data --leagues PL,PD,BL1,SA,FL1 --seasons 2021,2022,2023,2024

# Step 2: Ingest StatsBomb xG data (optional — improves model, not required)
python -m ingestion.statsbomb --competitions 2,11

# Step 3: Ingest Transfermarkt squad values (optional)
python -m ingestion.transfermarkt --leagues PL,PD,BL1,SA,FL1

# Step 4: Build the feature table
python -m features.build

# Step 5: Train all models
python -m models.train

# Step 6: Run predictions
python -m output.predict --matchday next

# Step 7: Copy predictions to frontend (after Task 11)
cp data/output/predictions.json ../frontend/public/data/predictions.json
```

**Hardware requirements:** 8 GB RAM minimum.  
**Expected runtime:** Step 1 is the bottleneck (~2h due to API rate limits). Steps 4–5 take ~20 min.

---

## How to add a new league

1. **`backend/config/settings.yaml`** — add entry to `leagues:`:
   ```yaml
   - { code: PPL, name: Primeira Liga, country: Portugal }
   ```
2. **`backend/ingestion/transfermarkt.py`** — add slug to `_LEAGUE_SLUGS`:
   ```python
   "PPL": ("primeira-liga", "PO1"),
   ```
3. **`backend/odds/fetcher.py`** — add sport key to `_LEAGUE_KEYS`:
   ```python
   "PPL": "soccer_portugal_primeira_liga",
   ```
4. Re-run the pipeline — the new league appears automatically.

---

## How to add a new feature

1. Create `backend/features/my_feature.py` with a function that takes the matches DataFrame and returns an enriched DataFrame (or `None` on failure).
2. Add a toggle in **`backend/config/feature_config.yaml`**:
   ```yaml
   my_feature:
     enabled: true
   ```
3. Import and call your builder in **`backend/features/build.py`** following the existing pattern.
4. Re-run `python -m features.build` — the new columns appear automatically.

---

## How to add a new model

1. Create `backend/models/my_model.py` extending `BaseModel` and implement `fit()` and `predict_proba()`.
2. Add config in **`backend/config/model_config.yaml`**:
   ```yaml
   my_model:
     enabled: true
     weight: 0.15
   ```
3. Import and instantiate in **`backend/models/train.py`** following the XGBoost/LightGBM pattern.
4. The `EnsembleModel` picks it up automatically — weights are normalised over enabled models.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FOOTBALL_DATA_API_KEY` | Yes | football-data.org v4 API key. Free tier: 10 req/min. |
| `THE_ODDS_API_KEY` | Yes (for odds) | The Odds API key. Free tier: 500 req/month. Degrades gracefully if missing. |

Both must also be added as **GitHub repository secrets** for the CI workflow.

---

## Output format — predictions.json schema

```json
{
  "generated_at": "2026-04-08T06:00:00Z",
  "matches": [
    {
      "match_id": 12345,
      "league": "Premier League",
      "date": "2026-04-12",
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "prediction": {
        "home_win": 0.52,
        "draw": 0.26,
        "away_win": 0.22,
        "predicted_outcome": "home_win",
        "confidence": "medium"
      },
      "odds_comparison": [{
        "bookmaker": "betclic",
        "home_odds": 1.85,
        "draw_odds": 3.50,
        "away_odds": 4.20,
        "home_implied": 0.541,
        "draw_implied": 0.286,
        "away_implied": 0.238,
        "home_edge": -0.021,
        "draw_edge": -0.026,
        "away_edge": -0.018
      }],
      "value_bets": [{
        "bookmaker": "pinnacle",
        "outcome": "home_win",
        "model_prob": 0.52,
        "bookmaker_odds": 2.10,
        "implied_prob": 0.476,
        "edge": 0.044,
        "kelly": 0.083,
        "confidence_tier": "medium"
      }]
    }
  ]
}
```

---

## Deployment

### Vercel (frontend)
- Connected to GitHub, auto-deploys on every push to `main`
- `vercel.json` configures the build from `frontend/`
- No environment variables needed — reads committed `predictions.json`

### GitHub Actions (daily pipeline)
- Runs at 6am UTC daily (inference only — not training)
- Commits updated `predictions.json` → triggers Vercel redeploy
- Trained `.pkl` files are committed manually and assumed to exist in CI

### Training
Train manually (locally), commit the `.pkl` files, push. Retrain when:
- Several months of new data have accumulated
- New features or ensemble weights are changed
- Calibration curves drift significantly

---

## Known limitations & resolved issues

### Active limitations
- **Small dataset**: ~10,000 matches total. XGBoost/LightGBM are well-suited; neural nets are not at this scale.
- **Draw probability**: All models underestimate draws — a known failure mode. Isotonic calibration helps but doesn't fully solve it.
- **Team name matching**: Odds API and football-data.org use different name conventions. Matching is case-insensitive substring — may fail for some clubs.
- **StatsBomb coverage**: Only covers selected competitions and seasons. xG features are missing for most matches.
- **No live updates**: Predictions generated once daily. Late team news not reflected.
- **Odds isolation**: The model has no knowledge of market prices and can't detect when the market has already priced in information the features missed.
- **Poisson model disabled**: Dixon-Coles model's `predict_proba()` expects `(home_team, away_team)` tuples rather than feature vectors — incompatible with the ensemble's matrix interface. Currently excluded; weights redistributed to XGBoost (0.60) and LightGBM (0.40).

### Resolved issues
- **XGBoost 3.x API change**: `early_stopping_rounds` must be passed to the `XGBClassifier()` constructor, not to `.fit()`. Fixed in `backend/models/xgboost_model.py`.
- **Premier League Odds API key**: The correct sport key is `soccer_epl`, not `soccer_england_premier_league`. Fixed in `backend/odds/fetcher.py` and `tools/generate_predictions.py`.
- **SHAP "model type not supported" error**: The ensemble stores `CalibratedModel` wrappers under each key, not raw `XGBoostModel` instances. The SHAP code was only unwrapping one level (`CalibratedModel → XGBoostModel`) instead of two (`→ XGBClassifier`), then passing the wrapper to `TreeExplainer`. Fixed in `tools/dashboard.py` by drilling `calibrated.model._model` and calling `.get_booster()` for reliable multi-class SHAP support.

---

## Reference documents

| File | Purpose |
|------|---------|
| `RECAP.md` | Theory-only explanation of every feature group (formulas), the 3 models, ensemble weighting, calibration, and evaluation metrics |
| `MODEL_REVIEW.md` | Research review citing 15 papers (2024–2026): performance ceiling, 10 missing features ranked by impact, 5 architecture improvements |
| `PROJECT.md` | This file — architecture, setup, run commands, known issues |

---

## Roadmap

### High priority (model quality)
- **Player availability**: Injury/suspension flags — highest-impact missing feature per literature
- **xG delta features**: Over/underperformance vs. expected goals (luck-adjusted form)
- **Poisson model fix**: Refactor `predict_proba()` to accept feature vectors → re-enable in ensemble
- **Walk-forward cross-validation**: Replace single train/test split with expanding-window CV

### Medium priority
- **Stacking ensemble**: Learn weights from OOF predictions instead of fixed weights
- **CatBoost**: Add as third tree model (handles categoricals natively, strong on tabular)
- **Season-to-season trajectory**: Momentum signal across seasons, not just within

### Lower priority / future
- **Player embeddings**: Replace formation archetypes with learned player-level embeddings (FBref)
- **Market-implied features**: Use odds movements as a separate signal (not in training)
- **Live odds alerts**: Telegram/Slack bot when a new value bet crosses the edge threshold
- **Automatic retraining**: Trigger when test-set RPS degrades beyond a threshold
- **More leagues**: Eredivisie, Primeira Liga, Scottish Premiership
- **Over/under market**: Predict total goals (2.5 line)
- **Better injury data**: Structured injury APIs (e.g. SportRadar) instead of Transfermarkt scraping
