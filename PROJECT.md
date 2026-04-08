# Football ML Predictions — Project Documentation

## What this is

A full-stack football match prediction system that uses an ML ensemble (Poisson + XGBoost + LightGBM) to predict 1X2 outcomes for the top 5 European leagues. Predictions are pre-generated as static JSON and served via a Next.js dashboard on Vercel, which also surfaces value bet opportunities by comparing model probabilities against bookmaker odds.

---

## Current Status

| Task | Description | Status |
|------|-------------|--------|
| 1 | Repo & environment setup | ✅ Done |
| 2 | Backend config system | ⬜ Pending |
| 3 | Data ingestion: football-data.org | ⬜ Pending |
| 4 | Data ingestion: StatsBomb | ⬜ Pending |
| 5 | Data ingestion: Transfermarkt | ⬜ Pending |
| 6 | Feature engineering | ⬜ Pending |
| 7 | Models | ⬜ Pending |
| 8 | Evaluation | ⬜ Pending |
| 9 | Odds fetcher & value bet detector | ⬜ Pending |
| 10 | Prediction pipeline | ⬜ Pending |
| 11 | Frontend: Next.js app on Vercel | ⬜ Pending |
| 12 | GitHub Actions CI + Vercel integration | ⬜ Pending |
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
