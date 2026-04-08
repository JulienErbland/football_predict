"""
Value bet detection logic.

Key concepts:

implied_probability(odds):
    The bookmaker's raw probability = 1 / decimal_odds.
    Bookmakers build in a margin (vig/overround) so raw implied probs sum to > 1.

remove_vig(home_odds, draw_odds, away_odds):
    Divide each raw implied prob by the sum — this removes the vig and gives
    normalised implied probs that sum to 1. These are the bookmaker's "true"
    probability estimates.

edge:
    edge = model_prob - implied_prob (after vig removal)
    Positive edge means the model thinks the outcome is more likely than the bookmaker.
    We require edge > 0.05 (5%) to flag as a value bet.

kelly_fraction:
    Kelly criterion: f* = (b * p - q) / b
    where b = decimal_odds - 1, p = model_prob, q = 1 - p
    This is the optimal fraction of bankroll to bet to maximise log-growth.
    Practitioners typically use fractional Kelly (e.g. 0.25 * full Kelly) to
    reduce variance. We return full Kelly here; the caller can scale it down.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def implied_probability(decimal_odds: float) -> float:
    """Raw implied probability = 1 / decimal_odds. Doesn't account for vig."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def remove_vig(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
) -> tuple[float, float, float]:
    """
    Normalise bookmaker odds to remove the vig (overround).

    Raw implied probs sum to > 1 because the bookmaker takes a margin.
    Dividing by the sum gives fair-implied probs that sum to 1.
    """
    raw_home = implied_probability(home_odds)
    raw_draw = implied_probability(draw_odds)
    raw_away = implied_probability(away_odds)
    total = raw_home + raw_draw + raw_away
    if total == 0:
        return 0.0, 0.0, 0.0
    return raw_home / total, raw_draw / total, raw_away / total


def compute_edge(model_prob: float, implied_prob: float) -> float:
    """
    Edge = model's estimated probability minus bookmaker's fair implied probability.

    Positive = model thinks this outcome is underpriced (value bet opportunity).
    """
    return model_prob - implied_prob


def kelly_fraction(edge: float, decimal_odds: float) -> float:
    """
    Full Kelly fraction of bankroll.

    f* = (b * p - q) / b  where b = decimal_odds - 1

    Returns 0 if edge <= 0 (no bet recommended).
    Clipped at 0.25 (25% of bankroll max) for safety.
    """
    if edge <= 0 or decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1.0
    p = edge + (1.0 / decimal_odds)  # Reconstruct model_prob from edge + implied
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, min(f, 0.25))  # Cap at 25% for safety


def find_value_bets(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    min_edge: float = 0.05,
) -> pd.DataFrame:
    """
    Find value bets by comparing model probabilities to bookmaker odds.

    Args:
        predictions_df: must have columns home_team, away_team,
            p_home_win, p_draw, p_away_win
        odds_df: must have columns home_team, away_team, bookmaker,
            home_odds, draw_odds, away_odds
        min_edge: minimum edge to flag as value bet (default 5%)

    Returns DataFrame with columns:
        match, market, outcome, model_prob, bookmaker_odds,
        implied_prob, edge, kelly, confidence_tier
    """
    outcomes = [("home_win", "home_odds", "p_home_win"),
                ("draw",     "draw_odds",  "p_draw"),
                ("away_win", "away_odds",  "p_away_win")]

    rows = []
    # Match predictions to odds by team names
    for _, pred in predictions_df.iterrows():
        home, away = pred["home_team"], pred["away_team"]
        match_odds = odds_df[
            (odds_df["home_team"] == home) & (odds_df["away_team"] == away)
        ]
        for _, odds_row in match_odds.iterrows():
            bm = odds_row["bookmaker"]
            h_odds = odds_row.get("home_odds")
            d_odds = odds_row.get("draw_odds")
            a_odds = odds_row.get("away_odds")

            if None in [h_odds, d_odds, a_odds] or any(
                pd.isna(x) for x in [h_odds, d_odds, a_odds]
            ):
                continue

            impl_h, impl_d, impl_a = remove_vig(
                float(h_odds), float(d_odds), float(a_odds)
            )
            implied_map = {"home_win": impl_h, "draw": impl_d, "away_win": impl_a}
            odds_map = {"home_win": h_odds, "draw": d_odds, "away_win": a_odds}

            for outcome_name, _, prob_col in outcomes:
                model_p = float(pred.get(prob_col, 0))
                impl_p = implied_map[outcome_name]
                decimal_odds = float(odds_map[outcome_name])
                edge = compute_edge(model_p, impl_p)
                if edge < min_edge:
                    continue
                kelly = kelly_fraction(edge, decimal_odds)
                # Confidence tier based on edge magnitude
                if edge >= 0.10:
                    tier = "high"
                elif edge >= 0.07:
                    tier = "medium"
                else:
                    tier = "low"
                rows.append({
                    "match": f"{home} vs {away}",
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": bm,
                    "market": "h2h",
                    "outcome": outcome_name,
                    "model_prob": round(model_p, 4),
                    "bookmaker_odds": decimal_odds,
                    "implied_prob": round(impl_p, 4),
                    "edge": round(edge, 4),
                    "kelly": round(kelly, 4),
                    "confidence_tier": tier,
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)
