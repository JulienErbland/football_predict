# Football ML Predictions — Project Documentation

## What this is

A full-stack football match prediction system that uses an ML ensemble (Poisson + XGBoost + LightGBM) to predict 1X2 outcomes for the top 5 European leagues. Predictions are pre-generated as static JSON and served via a Next.js dashboard on Vercel, which also surfaces value bet opportunities by comparing model probabilities against bookmaker odds.

---

## Current Status

| Task | Description | Status |
|------|-------------|--------|
| 1 | Repo & environment setup | ✅ Done |
| 2 | Backend config system | ✅ Done |
| 3 | Data ingestion: football-data.org | ✅ Done |
| 4 | Data ingestion: StatsBomb | ✅ Done |
| 5 | Data ingestion: Transfermarkt | ✅ Done |
| 6 | Feature engineering | ✅ Done |
| 7 | Models | ✅ Done |
| 8 | Evaluation | ✅ Done |
| 9 | Odds fetcher & value bet detector | ✅ Done |
| 10 | Prediction pipeline | ✅ Done |
| 11 | Frontend: Next.js app on Vercel | ✅ Done |
| 12 | GitHub Actions CI + Vercel integration | ✅ Done |
| 13 | Final PROJECT.md pass | ⬜ Pending |

---

## Task 1 — Repo & Environment Setup

### What was built
- Monorepo directory structure: `backend/` (Python ML pipeline) + `frontend/` (Next.js app)
- `backend/requirements.txt` with all Python dependencies
- `.env.example` documenting required API keys
- `.gitignore` excluding data dirs, model binaries, and secrets — but **including** `data/output/*.json` and `frontend/public/data/*.json` (predictions served to frontend)
- `vercel.json` wiring Vercel to build from `frontend/`
- Next.js 14 app scaffolded with TypeScript, Tailwind CSS, and App Router

### How to run
```bash
# Backend Python environment
python -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt

# Copy and fill in API keys
cp .env.example .env

# Frontend dev server
cd frontend && npm install && npm run dev
```

### Known limitations / TODOs
- No ML code yet — tasks 2–10 pending
- Frontend is scaffolded but has no content yet

---

## Task 2 — Backend Config System

### What was built
- `backend/config/settings.yaml` — leagues, seasons, bookmakers, API key placeholders, data paths
- `backend/config/feature_config.yaml` — toggleable feature groups (form, xG, squad, tactics, context, H2H, Elo) with per-group windows and parameters
- `backend/config/model_config.yaml` — model weights, hyperparameters, ensemble method, calibration strategy, evaluation metrics
- `backend/config/loader.py` — resolves `${ENV_VAR}` placeholders from `.env`, exposes `settings()`, `feature_config()`, `model_config()` as `lru_cache` singletons

### Key decisions
- `${ENV_VAR}` pattern in YAML keeps API keys out of source control while still having a single readable config file
- `lru_cache` on loaders means YAML is parsed exactly once per process — no global mutable state
- `isotonic` calibration chosen over Platt scaling because it's non-parametric and performs better on multi-class problems
- RPS (Ranked Probability Score) chosen as secondary metric — it accounts for the ordered nature of 1X2 outcomes

---

## Task 12 — GitHub Actions CI + Vercel Integration

### What was built
- `.github/workflows/predict.yml` — daily prediction pipeline at 6am UTC (+ manual trigger)

### CI workflow steps
1. Checkout repo
2. Install Python deps (`pip install -r backend/requirements.txt`)
3. Fetch current season match data (`ingestion.football_data`)
4. Rebuild feature table (`features.build`)
5. Run predictions (`output.predict --matchday next`)
6. Copy `predictions.json` → `frontend/public/data/`
7. Commit & push → triggers Vercel auto-deploy

### Required GitHub secrets
| Secret | Description |
|--------|-------------|
| `FOOTBALL_DATA_API_KEY` | football-data.org API key (free tier) |
| `THE_ODDS_API_KEY` | The Odds API key (500 req/month free tier) |

### How Vercel + GitHub Actions work together
- Vercel is connected to this repo and auto-deploys on every push to `main`
- The daily GitHub Actions job commits updated `predictions.json` → triggers a Vercel redeploy
- The model itself is trained **manually** (run `python -m models.train` locally, commit `.pkl` files, push)
- Note: Vercel Cron Jobs were considered but rejected — they run Node.js/Edge runtimes; this pipeline requires Python

---

## How to train the model

*(Full section will be written in Task 13. Placeholder:)*

```bash
# 1. Ingest data
cd backend
python -m ingestion.football_data --leagues PL,PD,BL1,SA,FL1 --seasons 2021,2022,2023,2024

# 2. Build features
python -m features.build

# 3. Train models
python -m models.train
```

Expected runtime: ~30–60 min on a modern laptop (CPU-only). No GPU required.
Hardware requirements: 8 GB RAM minimum, 16 GB recommended for LightGBM on full dataset.
