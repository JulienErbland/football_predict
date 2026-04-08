"""
Prediction pipeline — generates predictions.json for the frontend.

Workflow:
    1. Load ensemble model from data/models/ensemble.pkl (trusted local file)
    2. Load feature column order from data/models/feature_cols.pkl
    3. For each upcoming match per configured league:
        a. Build a feature row using the same feature builders as training
        b. Run through ensemble → (p_home, p_draw, p_away)
        c. Determine predicted outcome (argmax) and confidence label
        d. Fetch today's odds from The Odds API
        e. Compute edge/Kelly for each bookmaker × outcome
        f. Flag value bets (edge > 5%)
    4. Write data/output/predictions.json

The model is loaded from data/models/ensemble.pkl — this file must exist.
Train first: cd backend && python -m models.train

CLI usage:
    python -m output.predict --matchday next
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.loader import settings
from ingestion.football_data import FootballDataClient
from features.elo import compute_elo
from features.form import build_form_features, build_h2h_features
from features.context import build_context_features
from odds.fetcher import OddsFetcher
from odds.value import remove_vig, compute_edge, kelly_fraction


_META_COLS = {
    "match_id", "league", "season", "date", "matchday",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
}


def _confidence_label(max_prob: float) -> str:
    """
    Classify prediction confidence:
        high   — model is very sure (>60% for one outcome)
        medium — moderate confidence (45–60%)
        low    — uncertain outcome (<45%)
    """
    if max_prob > 0.60:
        return "high"
    elif max_prob >= 0.45:
        return "medium"
    return "low"


def _load_model(models_dir: Path):
    """Load the ensemble model from a trusted local pickle file."""
    import pickle  # noqa: PLC0415 — import here to keep the pickle usage localised
    ensemble_path = models_dir / "ensemble.pkl"
    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"Ensemble not found at {ensemble_path}. "
            "Run `python -m models.train` first."
        )
    with open(ensemble_path, "rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted local file, never from user input


def _load_feature_cols(models_dir: Path) -> list[str]:
    """Load the feature column order used during training."""
    import pickle  # noqa: PLC0415
    path = models_dir / "feature_cols.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"feature_cols.pkl not found at {path}. "
            "Run `python -m models.train` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301 — trusted local file


def _load_historical_matches(cfg: dict) -> pd.DataFrame | None:
    """Load all_matches.parquet for feature computation context."""
    path = Path(cfg["paths"]["processed"]) / "all_matches.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _build_feature_row(
    upcoming_match: pd.Series,
    historical_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Build a single feature row for an upcoming match.

    Appends the upcoming match (with null result) to historical data,
    then runs the feature builders and extracts the row for this match.
    This reuses the exact same pipeline as training — no leakage risk because
    the match result is null and won't be included in any rolling window.
    """
    match_row = upcoming_match.copy()
    match_row["result"] = np.nan
    match_row["home_goals"] = np.nan
    match_row["away_goals"] = np.nan

    combined = pd.concat(
        [historical_df, pd.DataFrame([match_row])],
        ignore_index=True,
    )
    combined["date"] = pd.to_datetime(combined["date"])

    # Run the feature builders (same pipeline as training)
    combined = compute_elo(combined)
    combined = build_form_features(combined)
    combined = build_h2h_features(combined)
    combined = build_context_features(combined)

    row = combined[combined["match_id"] == match_row["match_id"]]
    if row.empty:
        return np.zeros((1, len(feature_cols)), dtype=np.float32)

    X = row[feature_cols].fillna(0).values.astype(np.float32)
    return X


def predict(matchday: str = "next") -> dict:
    """
    Generate predictions for all upcoming matches across configured leagues.

    Returns the full predictions dict (also written to data/output/predictions.json).
    """
    cfg = settings()
    models_dir = Path(cfg["paths"]["models"])
    out_dir = Path(cfg["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble = _load_model(models_dir)
    feature_cols = _load_feature_cols(models_dir)

    historical_df = _load_historical_matches(cfg)
    client = FootballDataClient()
    odds_fetcher = OddsFetcher()

    all_matches = []

    for league in cfg["leagues"]:
        league_code = league["code"]
        league_name = league["name"]
        logger.info(f"Processing upcoming matches for {league_name}...")

        try:
            upcoming = client.fetch_upcoming(league_code)
        except Exception as e:
            logger.error(f"Failed to fetch upcoming matches for {league_code}: {e}")
            continue

        if upcoming.empty:
            logger.info(f"No upcoming matches for {league_code}")
            continue

        try:
            odds_df = odds_fetcher.fetch_upcoming_odds(league_code)
        except Exception as e:
            logger.warning(f"Odds fetch failed for {league_code}: {e}")
            odds_df = pd.DataFrame()

        for _, match in upcoming.iterrows():
            match_id = int(match["match_id"])
            home = match["home_team"]
            away = match["away_team"]
            match_date = match["date"]
            match_date_str = (
                match_date.date().isoformat()
                if hasattr(match_date, "date")
                else str(match_date)
            )

            try:
                if historical_df is not None:
                    X = _build_feature_row(match, historical_df, feature_cols)
                else:
                    X = np.zeros((1, len(feature_cols)), dtype=np.float32)
                proba = ensemble.predict_proba(X)[0]
            except Exception as e:
                logger.error(f"Prediction failed for {home} vs {away}: {e}")
                proba = np.array([1 / 3, 1 / 3, 1 / 3])

            p_home = float(proba[0])
            p_draw = float(proba[1])
            p_away = float(proba[2])
            max_prob = max(p_home, p_draw, p_away)
            outcome_map = {0: "home_win", 1: "draw", 2: "away_win"}
            predicted_outcome = outcome_map[int(np.argmax(proba))]
            confidence = _confidence_label(max_prob)

            odds_comparison = []
            value_bets = []

            if not odds_df.empty:
                match_odds = odds_df[
                    (odds_df["home_team"].str.lower() == home.lower()) &
                    (odds_df["away_team"].str.lower() == away.lower())
                ]
                for bm, bm_rows in match_odds.groupby("bookmaker"):
                    row = bm_rows.iloc[0]
                    h_o = row.get("home_odds")
                    d_o = row.get("draw_odds")
                    a_o = row.get("away_odds")
                    if any(pd.isna(x) for x in [h_o, d_o, a_o] if x is not None):
                        continue
                    h_o, d_o, a_o = float(h_o or 0), float(d_o or 0), float(a_o or 0)
                    if 0 in [h_o, d_o, a_o]:
                        continue
                    impl_h, impl_d, impl_a = remove_vig(h_o, d_o, a_o)
                    edge_h = compute_edge(p_home, impl_h)
                    edge_d = compute_edge(p_draw, impl_d)
                    edge_a = compute_edge(p_away, impl_a)

                    odds_comparison.append({
                        "bookmaker": bm,
                        "home_odds": round(h_o, 3),
                        "draw_odds": round(d_o, 3),
                        "away_odds": round(a_o, 3),
                        "home_implied": round(impl_h, 3),
                        "draw_implied": round(impl_d, 3),
                        "away_implied": round(impl_a, 3),
                        "home_edge": round(edge_h, 3),
                        "draw_edge": round(edge_d, 3),
                        "away_edge": round(edge_a, 3),
                    })

                    for outcome, model_p, impl_p, dec_odds in [
                        ("home_win", p_home, impl_h, h_o),
                        ("draw",     p_draw, impl_d, d_o),
                        ("away_win", p_away, impl_a, a_o),
                    ]:
                        edge = compute_edge(model_p, impl_p)
                        if edge < 0.05:
                            continue
                        kelly = kelly_fraction(edge, dec_odds)
                        tier = ("high" if edge >= 0.10
                                else ("medium" if edge >= 0.07 else "low"))
                        value_bets.append({
                            "bookmaker": bm,
                            "outcome": outcome,
                            "model_prob": round(model_p, 3),
                            "bookmaker_odds": dec_odds,
                            "implied_prob": round(impl_p, 3),
                            "edge": round(edge, 3),
                            "kelly": round(kelly, 3),
                            "confidence_tier": tier,
                        })

            all_matches.append({
                "match_id": match_id,
                "league": league_name,
                "date": match_date_str,
                "home_team": home,
                "away_team": away,
                "prediction": {
                    "home_win": round(p_home, 3),
                    "draw": round(p_draw, 3),
                    "away_win": round(p_away, 3),
                    "predicted_outcome": predicted_outcome,
                    "confidence": confidence,
                },
                "odds_comparison": odds_comparison,
                "value_bets": value_bets,
            })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches": all_matches,
    }

    out_path = out_dir / "predictions.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote {len(all_matches)} predictions → {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate match predictions")
    parser.add_argument("--matchday", default="next",
                        help="Which matchday to predict (default: next)")
    args = parser.parse_args()
    predict(matchday=args.matchday)


if __name__ == "__main__":
    main()
