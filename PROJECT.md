# Football ML Predictions — Project Documentation

## What this is

A full-stack football match prediction system that uses a weighted ensemble of Poisson regression, XGBoost, and LightGBM to forecast 1X2 outcomes (home win / draw / away win) for the top 5 European leagues. Predictions are pre-generated nightly as static JSON by a Python backend pipeline, then served via a Next.js 16 dashboard deployed on Vercel. The dashboard also surfaces value bet opportunities by comparing model probabilities against live bookmaker odds from The Odds API, computing edge and Kelly fractions for each bookmaker × outcome combination.

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
│   │   ├── tactical.py          # Formation encoding, matchup win rates, player archetypes
│   │   ├── context.py           # Rest days, congestion, season stage, league position, referee
│   │   └── build.py             # Master orchestrator: loads all builders, merges, saves Parquet
│   │
│   ├── models/
│   │   ├── base.py              # Abstract BaseModel + CalibratedModel with isotonic calibration
│   │   ├── poisson_model.py     # Dixon-Coles Poisson regression (statistical baseline)
│   │   ├── xgboost_model.py     # XGBoost multi:softprob classifier
│   │   ├── lgbm_model.py        # LightGBM multiclass classifier
│   │   ├── neural_net.py        # PyTorch MLP (disabled; activate for >2000 match datasets)
│   │   ├── ensemble.py          # Weighted average ensemble + stacking variant
│   │   └── train.py             # Training orchestrator: split → train → calibrate → save
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # brier_score, log_loss, RPS, accuracy, calibration_summary
│   │   └── report.py            # Saves per-model JSON eval report to data/output/
│   │
│   ├── odds/
│   │   ├── fetcher.py           # The Odds API v4 client with daily Parquet caching
│   │   └── value.py             # implied_probability, remove_vig, compute_edge, kelly_fraction
│   │
│   ├── output/
│   │   └── predict.py           # Full pipeline: upcoming → features → predict → odds → JSON
│   │
│   ├── data/
│   │   ├── raw/                 # Raw Parquet files per source (gitignored)
│   │   ├── processed/           # all_matches.parquet (gitignored)
│   │   ├── features/            # features.parquet (gitignored)
│   │   ├── models/              # Trained .pkl files (gitignored)
│   │   └── output/              # predictions.json, eval_*.json (included in git)
│   │
│   └── requirements.txt         # Python deps: pandas, sklearn, xgboost, lightgbm, etc.
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Home dashboard (async Server Component, reads predictions.json)
│   │   ├── methodology/
│   │   │   └── page.tsx         # Explanation of model, features, value bet formula, disclaimer
│   │   ├── layout.tsx           # Root layout with metadata
│   │   └── globals.css          # Tailwind base styles
│   │
│   ├── components/
│   │   ├── PredictionsDashboard.tsx  # Client component: league filter + match list
│   │   ├── MatchCard.tsx             # Probability bar, badge, expandable odds table
│   │   ├── LeagueFilter.tsx          # Tab bar for filtering by league
│   │   └── ValueBetBadge.tsx         # Compact badge: bookmaker, outcome, edge%, kelly%
│   │
│   ├── types/
│   │   └── predictions.ts       # TypeScript interfaces for predictions.json schema
│   │
│   └── public/
│       └── data/
│           └── predictions.json # Committed predictions (updated daily by CI)
│
├── .env.example                 # Template for required environment variables
├── .gitignore                   # Excludes data/, .venv/, .pkl, but includes output/*.json
├── vercel.json                  # Vercel build config pointing to frontend/
└── PROJECT.md                   # This file
```

---

## Setup instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
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
```

### 3. Set up the frontend
```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

The frontend already has a sample `predictions.json` and will render immediately.

---

## How to train the model

Run these commands in order. The full pipeline (historical ingestion → features → training) takes 30–60 minutes on a modern CPU.

```bash
cd backend

# Step 1: Ingest historical match data (rate-limited: ~2h for 5 leagues × 4 seasons)
python -m ingestion.football_data --leagues PL,PD,BL1,SA,FL1 --seasons 2021,2022,2023,2024

# Step 2: Ingest StatsBomb xG data (optional — improves model but not required)
python -m ingestion.statsbomb --competitions 2,11

# Step 3: Ingest Transfermarkt squad values (optional)
python -m ingestion.transfermarkt --leagues PL,PD,BL1,SA,FL1

# Step 4: Build the feature table
python -m features.build

# Step 5: Train all models and build the ensemble
# Edit models/train.py and uncomment the model.fit() blocks first
python -m models.train

# Step 6: Run predictions
python -m output.predict --matchday next

# Step 7: Copy predictions to frontend
cp data/output/predictions.json ../frontend/public/data/predictions.json
```

**Hardware requirements:** 8 GB RAM minimum, 16 GB recommended for LightGBM on full dataset.  
**Expected runtime:** Step 1 is the bottleneck (~2h due to API rate limits). Steps 4–5 take ~30 min.

> **Note:** `models/train.py` contains placeholder comments where `model.fit()` calls would go. Uncomment those blocks to actually train. This was intentional per project setup constraints.

---

## How to add a new league

1. **`backend/config/settings.yaml`** — add a new entry to `leagues:`:
   ```yaml
   - { code: PPL, name: Primeira Liga, country: Portugal }
   ```
2. **`backend/ingestion/transfermarkt.py`** — add the league slug to `_LEAGUE_SLUGS`:
   ```python
   "PPL": ("primeira-liga", "PO1"),
   ```
3. **`backend/odds/fetcher.py`** — add the sport key to `_LEAGUE_KEYS`:
   ```python
   "PPL": "soccer_portugal_primeira_liga",
   ```
4. **`frontend/public/data/predictions.json`** — the new league appears automatically once the pipeline runs.

---

## How to add a new feature

1. Create `backend/features/my_feature.py` implementing a function that takes the matches DataFrame and returns an enriched DataFrame (or `None` on failure).
2. Add a toggle in **`backend/config/feature_config.yaml`**:
   ```yaml
   my_feature:
     enabled: true
     # any parameters
   ```
3. Import and call your builder in **`backend/features/build.py`** following the pattern of existing optional builders.
4. The new columns automatically appear in the training feature set on the next `python -m features.build` run.

---

## How to add a new model

1. Create `backend/models/my_model.py` extending `BaseModel` from `models/base.py` and implement `fit()` and `predict_proba()`.
2. Add config in **`backend/config/model_config.yaml`**:
   ```yaml
   my_model:
     enabled: true
     weight: 0.15
     # hyperparameters
   ```
3. Import and instantiate your model in **`backend/models/train.py`** following the pattern of the XGBoost/LightGBM blocks.
4. The `EnsembleModel` picks it up automatically — weights are normalised over enabled models.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FOOTBALL_DATA_API_KEY` | Yes | football-data.org v4 API key. Free tier: 10 req/min, covers all 5 leagues. |
| `THE_ODDS_API_KEY` | Yes (for odds) | The Odds API key. Free tier: 500 req/month. Odds features degrade gracefully without it. |

Both must also be added as **GitHub repository secrets** for the CI workflow to work.

---

## Output format — predictions.json schema

```json
{
  "generated_at": "2026-04-08T06:00:00Z",   // ISO 8601 UTC timestamp
  "matches": [
    {
      "match_id": 12345,                      // football-data.org match ID
      "league": "Premier League",
      "date": "2026-04-12",                   // ISO date (YYYY-MM-DD)
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "prediction": {
        "home_win": 0.52,                     // Model probability [0,1]
        "draw": 0.26,
        "away_win": 0.22,
        "predicted_outcome": "home_win",      // Argmax outcome
        "confidence": "medium"                // high|medium|low based on max_prob
      },
      "odds_comparison": [{
        "bookmaker": "betclic",
        "home_odds": 1.85,                    // Decimal odds
        "draw_odds": 3.50,
        "away_odds": 4.20,
        "home_implied": 0.541,               // Vig-normalised implied probability
        "draw_implied": 0.286,
        "away_implied": 0.238,
        "home_edge": -0.021,                 // model_prob - implied_prob
        "draw_edge": -0.026,
        "away_edge": -0.018
      }],
      "value_bets": [{
        "bookmaker": "pinnacle",
        "outcome": "home_win",
        "model_prob": 0.52,
        "bookmaker_odds": 2.10,
        "implied_prob": 0.476,
        "edge": 0.044,                       // > 0.05 to qualify as value bet
        "kelly": 0.083,                      // Full Kelly fraction (scale down for safety)
        "confidence_tier": "medium"          // high(≥10%)|medium(≥7%)|low(≥5%) edge
      }]
    }
  ]
}
```

---

## Deployment

### Vercel (frontend)
- Vercel is connected to this GitHub repo and auto-deploys on every push to `main`
- `vercel.json` at the repo root configures the build to run from `frontend/`
- No environment variables needed by the frontend — it reads the committed `predictions.json`

### GitHub Actions (daily pipeline)
- `.github/workflows/predict.yml` runs at 6am UTC daily
- After committing updated predictions, the push triggers Vercel to redeploy
- Required secrets in GitHub repo settings: `FOOTBALL_DATA_API_KEY`, `THE_ODDS_API_KEY`
- The trained model `.pkl` files must be committed to the repo — the CI assumes they already exist

### Training workflow
The model is trained **manually** (locally or on a rented GPU/CPU), committed, then pushed. The daily CI only runs inference, not training. Retrain when:
- Several months of new data have accumulated
- You add new features or change the ensemble weights
- Calibration curves drift significantly from the diagonal

---

## Known limitations

- **Small dataset**: ~1,500–2,000 matches per season across 5 leagues. XGBoost/LightGBM perform well but neural nets are likely to underfit at this scale.
- **Draw probability**: All models underestimate draw probability — a known failure mode in football prediction. Isotonic calibration helps but doesn't fully solve it.
- **Team name matching**: Odds API and football-data.org use different team name conventions. The current matching is case-insensitive substring match — it may fail for some clubs.
- **StatsBomb coverage**: StatsBomb open data only covers certain competitions and seasons. xG features are missing for most matches, reducing model power.
- **Odds isolation**: Odds are only used post-prediction. The model has no knowledge of market prices, which means it can't detect when the market has already priced in information the features missed (injuries, weather, etc.).
- **No live updates**: Predictions are generated once daily. Late team news (injuries, suspensions announced after 6am UTC) is not reflected.

---

## Roadmap

- **Player embeddings**: Replace formation archetypes with learned player-level embeddings (FBref data)
- **Market-implied features**: Use odds movements as features (not in training — a separate "odds shift" signal)
- **Live odds alerts**: Telegram/Slack bot that pings when a new value bet crosses the edge threshold
- **Automatic retraining**: Trigger retraining when test-set RPS degrades beyond a threshold
- **More leagues**: Add Eredivisie, Primeira Liga, Scottish Premiership
- **Over/under market**: Extend the pipeline to predict total goals (2.5 line)
- **Confidence-weighted ensemble**: Learn ensemble weights from OOF predictions (stacking) instead of fixed weights
- **Better injury data**: Use structured injury APIs (e.g. SportRadar) instead of Transfermarkt scraping
