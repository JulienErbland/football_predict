"""
End-to-end pipeline smoke test.

Runs the full pipeline with synthetic data — no API calls, no waiting for
football-data.org rate limits. Proves all modules can be imported, feature
builders work, models train, and predictions.json is written correctly.

Run from repo root:
    backend/.venv/bin/python3 tools/pipeline_test.py

By default all artifacts (synthetic matches, features, models, predictions)
land under ``tmp_smoke/`` at the repo root so the run can never clobber the
production corpus in ``backend/data/``. To target a different location pass
``--sandbox-dir PATH``. The legacy "write to production paths" behaviour is
reachable with ``--sandbox-dir backend/data`` but is no longer the default.

Expected: green checkmark on every step, final predictions_test.json written.

Note: pickle is used here only to load our own locally-generated model files,
matching the same pattern as the production pipeline.
"""
from __future__ import annotations

import argparse
import json
import pickle  # noqa: S403 — loads only locally-generated trusted model files
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


# ── helpers ──────────────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def fail(msg: str, exc: Exception) -> None:
    print(f"  \033[31m✗\033[0m {msg}")
    traceback.print_exc()
    sys.exit(1)


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── synthetic data ────────────────────────────────────────────────────────────

def _make_matches(n_teams: int = 10, seasons: list[int] = None) -> pd.DataFrame:
    """Generate a synthetic round-robin schedule — no API calls needed."""
    seasons = seasons or [2022, 2023, 2024]
    teams = [f"Team_{chr(65 + i)}" for i in range(n_teams)]
    rng = np.random.default_rng(42)

    rows = []
    match_id = 1
    base = date(2022, 8, 6)

    for season_idx, season in enumerate(seasons):
        season_start = base + timedelta(weeks=42 * season_idx)
        matchday = 0
        for h_idx, home in enumerate(teams):
            for a_idx, away in enumerate(teams):
                if home == away:
                    continue
                matchday += 1
                match_date = season_start + timedelta(days=matchday * 7 // n_teams)
                home_goals = int(rng.poisson(1.4))
                away_goals = int(rng.poisson(1.1))
                result = 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)
                rows.append({
                    "match_id": match_id,
                    "league": "Test League",
                    "season": season,
                    "matchday": matchday % 38 + 1,
                    "date": pd.Timestamp(match_date),
                    "home_team_id": h_idx + 1,
                    "home_team": home,
                    "away_team_id": a_idx + 1,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                    "referee": f"Referee_{(match_id % 5) + 1}",
                })
                match_id += 1

    return pd.DataFrame(rows)


# ── steps ─────────────────────────────────────────────────────────────────────

def step_imports() -> None:
    section("STEP 1 — Import all pipeline modules")
    mods = [
        ("config.loader", "settings, feature_config, model_config"),
        ("features.elo", "compute_elo"),
        ("features.form", "build_form_features, build_h2h_features"),
        ("features.context", "build_context_features"),
        ("features.xg_features", "build_xg_features"),
        ("features.squad_features", "build_squad_features"),
        ("models.base", "BaseModel"),
        ("models.xgboost_model", "XGBoostModel"),
        ("models.lgbm_model", "LGBMModel"),
        ("models.ensemble", "EnsembleModel"),
        ("odds.value", "remove_vig, compute_edge, kelly_fraction"),
        ("evaluation.metrics", "brier_score, log_loss_score, rps, accuracy"),
    ]
    for mod, symbols in mods:
        try:
            __import__(mod, fromlist=symbols.split(", "))
            ok(mod)
        except Exception as e:
            fail(mod, e)


def step_synthetic_data(processed_dir: Path) -> None:
    section("STEP 2 — Generate synthetic match data")
    matches = _make_matches(n_teams=10, seasons=[2022, 2023, 2024])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "all_matches.parquet"
    matches.to_parquet(out, index=False)
    ok(f"{len(matches)} matches saved → {out}")


def step_features(processed_dir: Path, features_dir: Path) -> None:
    section("STEP 3 — Feature engineering")
    from features.elo import compute_elo
    from features.form import build_form_features, build_h2h_features
    from features.context import build_context_features

    df = pd.read_parquet(processed_dir / "all_matches.parquet")

    for label, fn in [
        ("Elo", lambda d: compute_elo(d)),
        ("Form", lambda d: build_form_features(d)),
        ("H2H", lambda d: build_h2h_features(d)),
        ("Context", lambda d: build_context_features(d)),
    ]:
        try:
            df = fn(df)
            ok(f"{label} features")
        except Exception as e:
            fail(f"{label} features", e)

    features_dir.mkdir(parents=True, exist_ok=True)
    out = features_dir / "features.parquet"
    df.to_parquet(out, index=False)

    meta_cols = {
        "match_id", "league", "season", "date", "matchday",
        "home_team", "away_team", "home_team_id", "away_team_id",
        "referee", "home_goals", "away_goals", "result",
    }
    feat_cols = [c for c in df.columns if c not in meta_cols]
    ok(f"{len(feat_cols)} feature columns, {len(df)} rows → {out}")


def step_train(features_dir: Path, models_dir: Path) -> None:
    section("STEP 4 — Train XGBoost + LightGBM ensemble")
    from models.xgboost_model import XGBoostModel
    from models.lgbm_model import LGBMModel
    from models.ensemble import EnsembleModel

    df = pd.read_parquet(features_dir / "features.parquet")
    meta_cols = {
        "match_id", "league", "season", "date", "matchday",
        "home_team", "away_team", "home_team_id", "away_team_id",
        "referee", "home_goals", "away_goals", "result",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]

    train_df = df[df["season"].isin([2022, 2023]) & df["result"].notna()]
    test_df  = df[df["season"] == 2024].dropna(subset=["result"])

    X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train = train_df["result"].values.astype(int)

    val_cut = int(len(X_train) * 0.9)
    X_tr, y_tr   = X_train[:val_cut], y_train[:val_cut]
    X_val, y_val = X_train[val_cut:], y_train[val_cut:]

    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        xgb = XGBoostModel(n_estimators=50, max_depth=4)
        xgb.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, early_stopping_rounds=10)
        ok(f"XGBoost: {X_tr.shape[0]} train rows, {X_tr.shape[1]} features")
    except Exception as e:
        fail("XGBoost training", e)

    try:
        lgbm = LGBMModel(n_estimators=50, num_leaves=15)
        lgbm.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        ok(f"LightGBM: {X_tr.shape[0]} train rows, {X_tr.shape[1]} features")
    except Exception as e:
        fail("LightGBM training", e)

    try:
        ensemble = EnsembleModel(
            models={"xgboost": xgb, "lightgbm": lgbm},
            weights={"xgboost": 0.6, "lightgbm": 0.4},
        )
        ensemble.save(models_dir / "ensemble.pkl")
        ok(f"Ensemble saved → {models_dir / 'ensemble.pkl'}")
    except Exception as e:
        fail("Ensemble build/save", e)

    with open(models_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)  # noqa: S301 — locally-generated, trusted
    ok(f"feature_cols.pkl saved ({len(feature_cols)} columns)")


def step_evaluate(models_dir: Path, features_dir: Path) -> None:
    section("STEP 5 — Evaluation metrics")
    from evaluation.metrics import brier_score, log_loss_score, rps, accuracy
    from models.ensemble import EnsembleModel

    with open(models_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)  # noqa: S301 — trusted local file

    df = pd.read_parquet(features_dir / "features.parquet")
    test_df = df[df["season"] == 2024].dropna(subset=["result"])
    if len(test_df) == 0:
        ok("No test data (2024 in synthetic set) — skipping, that's fine")
        return

    X_test = test_df[feature_cols].fillna(0).values.astype(np.float32)
    y_test  = test_df["result"].values.astype(int)

    ensemble = EnsembleModel.load(models_dir / "ensemble.pkl")
    proba = ensemble.predict_proba(X_test)

    for label, fn in [
        ("Brier score", lambda: brier_score(y_test, proba)),
        ("Log loss",    lambda: log_loss_score(y_test, proba)),
        ("RPS",         lambda: rps(y_test, proba)),
        ("Accuracy",    lambda: f"{accuracy(y_test, proba):.1%}"),
    ]:
        try:
            val = fn()
            ok(f"{label}: {val if isinstance(val, str) else f'{val:.4f}'}")
        except Exception as e:
            fail(label, e)


def step_predict(models_dir: Path, features_dir: Path, output_dir: Path) -> None:
    section("STEP 6 — Predictions (mock upcoming, no API calls)")
    from models.ensemble import EnsembleModel
    from odds.value import remove_vig, compute_edge, kelly_fraction

    ensemble = EnsembleModel.load(models_dir / "ensemble.pkl")
    with open(models_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)  # noqa: S301

    df = pd.read_parquet(features_dir / "features.parquet")
    all_teams = df["home_team"].unique().tolist()

    upcoming = [
        {"home_team": all_teams[i], "away_team": all_teams[i + 1],
         "date": "2025-08-15", "match_id": 90000 + i, "league": "Test League"}
        for i in range(min(5, len(all_teams) - 1))
    ]

    X_dummy = np.zeros((1, len(feature_cols)), dtype=np.float32)
    matches_out = []

    for m in upcoming:
        proba  = ensemble.predict_proba(X_dummy)[0]
        p_home = float(proba[0])
        p_draw = float(proba[1])
        p_away = float(proba[2])
        max_p  = max(p_home, p_draw, p_away)
        predicted  = ["home_win", "draw", "away_win"][[p_home, p_draw, p_away].index(max_p)]
        confidence = "high" if max_p > 0.60 else ("medium" if max_p >= 0.45 else "low")

        h_o, d_o, a_o = 2.10, 3.40, 3.20
        impl_h, impl_d, impl_a = remove_vig(h_o, d_o, a_o)
        value_bets = []
        for outcome, mp, ip, odds in [
            ("home_win", p_home, impl_h, h_o),
            ("draw",     p_draw, impl_d, d_o),
            ("away_win", p_away, impl_a, a_o),
        ]:
            edge = compute_edge(mp, ip)
            if edge >= 0.05:
                k    = kelly_fraction(edge, odds)
                tier = "high" if edge >= 0.10 else ("medium" if edge >= 0.07 else "low")
                value_bets.append({
                    "bookmaker": "mock", "outcome": outcome,
                    "model_prob": round(mp, 3), "bookmaker_odds": odds,
                    "implied_prob": round(ip, 3), "edge": round(edge, 3),
                    "kelly": round(k, 3), "confidence_tier": tier,
                })

        matches_out.append({
            "match_id": m["match_id"],
            "league": m["league"],
            "date": m["date"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "prediction": {
                "home_win": round(p_home, 3),
                "draw": round(p_draw, 3),
                "away_win": round(p_away, 3),
                "predicted_outcome": predicted,
                "confidence": confidence,
            },
            "odds_comparison": [{
                "bookmaker": "mock",
                "home_odds": h_o, "draw_odds": d_o, "away_odds": a_o,
                "home_implied": round(impl_h, 3),
                "draw_implied": round(impl_d, 3),
                "away_implied": round(impl_a, 3),
                "home_edge": round(compute_edge(p_home, impl_h), 3),
                "draw_edge": round(compute_edge(p_draw, impl_d), 3),
                "away_edge": round(compute_edge(p_away, impl_a), 3),
            }],
            "value_bets": value_bets,
        })

    from datetime import datetime, timezone
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches": matches_out,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    ok(f"{len(matches_out)} predictions → {out_path}")

    for m in matches_out:
        total = m["prediction"]["home_win"] + m["prediction"]["draw"] + m["prediction"]["away_win"]
        assert abs(total - 1.0) < 0.01, f"Probs don't sum to 1 for {m['home_team']} vs {m['away_team']}"
    ok("All probabilities sum to 1.0 ✓")


# ── entry point ───────────────────────────────────────────────────────────────

def _sandbox_paths(sandbox_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return (processed, features, models, output) under ``sandbox_dir``.

    Mirrors the layout that ``settings.yaml`` uses under ``backend/data/`` so
    feature builders, train, evaluate, and predict all find their files in
    the same relative locations they do in production — just rooted under
    the sandbox instead of ``backend/data/``.
    """
    return (
        sandbox_dir / "processed",
        sandbox_dir / "features",
        sandbox_dir / "models",
        sandbox_dir / "output",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sandbox-dir",
        type=Path,
        default=ROOT / "tmp_smoke",
        help=(
            "Directory under which the smoke test writes synthetic data, "
            "features, models, and predictions. Defaults to <repo>/tmp_smoke "
            "so production data in backend/data/ is never overwritten. Pass "
            "'--sandbox-dir backend/data' to deliberately target production "
            "(legacy behaviour)."
        ),
    )
    args = parser.parse_args()

    sandbox = args.sandbox_dir.resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    processed_dir, features_dir, models_dir, output_dir = _sandbox_paths(sandbox)

    print("\n\033[1mFootball Predict — Pipeline Smoke Test\033[0m")
    print(f"Backend:  {ROOT / 'backend'}")
    print(f"Sandbox:  {sandbox}")

    step_imports()
    step_synthetic_data(processed_dir)
    step_features(processed_dir, features_dir)
    step_train(features_dir, models_dir)
    step_evaluate(models_dir, features_dir)
    step_predict(models_dir, features_dir, output_dir)

    print(f"\n{'═'*60}")
    print(f"  \033[1;32mALL STEPS PASSED\033[0m — pipeline is working end-to-end")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
