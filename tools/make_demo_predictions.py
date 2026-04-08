"""
Demo predictions generator — no trained model, no API key needed.

Reads the-odds-api.json (manually downloaded odds snapshot) and creates a
realistic predictions.json by:
  1. Extracting all football events with h2h odds
  2. Deriving "model probabilities" from the vig-removed implied odds + a small
     random perturbation (simulates a model that's slightly different from the market)
  3. Computing edges and Kelly fractions exactly as the real pipeline does
  4. Writing predictions.json to both backend/data/output/ and frontend/public/data/

This is for frontend layout testing only. Replace with the real pipeline once
football-data.org ingestion and model training are complete.

Usage:
    python tools/make_demo_predictions.py
    python tools/make_demo_predictions.py --seed 99  # different perturbation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add backend to sys.path so we can reuse odds/value.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

SPORT_LEAGUE_MAP = {
    "soccer_england_premier_league": "Premier League",
    "soccer_spain_la_liga": "La Liga",
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_uefa_europa_league": "UEFA Europa League",
    "soccer_uefa_champions_league": "UEFA Champions League",
    "soccer_turkey_super_league": "Süper Lig",
    "soccer_germany_liga3": "3. Liga",
    "soccer_saudi_arabia_pro_league": "Saudi Pro League",
    "soccer_greece_super_league": "Super League Greece",
    "soccer_netherlands_eredivisie": "Eredivisie",
    "soccer_portugal_primeira_liga": "Primeira Liga",
}

MIN_EDGE = 0.05  # 5% minimum edge to flag as value bet


def remove_vig(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float]:
    raw_h = 1.0 / home_odds
    raw_d = 1.0 / draw_odds
    raw_a = 1.0 / away_odds
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def compute_edge(model_p: float, implied_p: float) -> float:
    return model_p - implied_p


def kelly(edge: float, decimal_odds: float) -> float:
    if edge <= 0 or decimal_odds <= 1:
        return 0.0
    b = decimal_odds - 1.0
    p = edge + (1.0 / decimal_odds)
    q = 1.0 - p
    return max(0.0, min((b * p - q) / b, 0.25))


def perturb_probs(impl_h: float, impl_d: float, impl_a: float,
                   rng: random.Random, strength: float = 0.06) -> tuple[float, float, float]:
    """
    Simulate a model that differs from the market by up to ±strength per class.
    Renormalises so the three probabilities sum to 1.
    """
    ph = impl_h + rng.uniform(-strength, strength)
    pd = impl_d + rng.uniform(-strength * 0.5, strength * 0.5)  # Draw is harder to predict
    pa = impl_a + rng.uniform(-strength, strength)
    # Clip to [0.02, 0.96] and renormalise
    ph, pd, pa = max(ph, 0.02), max(pd, 0.02), max(pa, 0.02)
    total = ph + pd + pa
    return ph / total, pd / total, pa / total


def best_h2h_odds(bookmakers: list[dict]) -> dict[str, tuple[float, float, float, str]]:
    """
    Return a dict of bookmaker → (home_odds, draw_odds, away_odds) for h2h markets.
    Skips bookmakers missing any of the three outcomes.
    """
    result = {}
    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
            if len(outcomes) < 2:
                continue
            # Football h2h has 3 outcomes; other sports (cricket) have 2
            draw_keys = [k for k in outcomes if k.lower() == "draw"]
            if not draw_keys:
                continue  # Skip non-football h2h (no draw)
            draw_odds = outcomes[draw_keys[0]]
            teams = [k for k in outcomes if k.lower() != "draw"]
            if len(teams) < 2:
                continue
            result[bm["key"]] = (outcomes[teams[0]], draw_odds, outcomes[teams[1]])
    return result


def make_predictions(odds_path: Path, seed: int = 42) -> dict:
    rng = random.Random(seed)

    with open(odds_path) as f:
        raw = json.load(f)

    # Filter to football events only
    events = [e for e in raw if e.get("sport_key", "").startswith("soccer_")]
    print(f"Found {len(events)} football events in {odds_path.name}")

    matches = []
    for event in events:
        sport_key = event["sport_key"]
        league = SPORT_LEAGUE_MAP.get(sport_key, sport_key.replace("soccer_", "").replace("_", " ").title())
        home = event["home_team"]
        away = event["away_team"]
        date = event["commence_time"][:10]

        bm_odds = best_h2h_odds(event.get("bookmakers", []))
        if not bm_odds:
            print(f"  Skipping {home} vs {away} — no h2h odds with draw")
            continue

        # Use the market consensus (average implied probs) as the starting point
        all_implied = [remove_vig(h, d, a) for h, d, a in bm_odds.values()]
        avg_impl_h = sum(x[0] for x in all_implied) / len(all_implied)
        avg_impl_d = sum(x[1] for x in all_implied) / len(all_implied)
        avg_impl_a = sum(x[2] for x in all_implied) / len(all_implied)

        # "Model" probabilities = market consensus + small perturbation
        model_h, model_d, model_a = perturb_probs(avg_impl_h, avg_impl_d, avg_impl_a, rng)

        max_p = max(model_h, model_d, model_a)
        outcome_idx = [model_h, model_d, model_a].index(max_p)
        outcome_name = ["home_win", "draw", "away_win"][outcome_idx]
        confidence = "high" if max_p > 0.60 else ("medium" if max_p >= 0.45 else "low")

        # Build per-bookmaker comparison
        odds_comparison = []
        value_bets = []
        for bm_key, (h_o, d_o, a_o) in sorted(bm_odds.items()):
            impl_h, impl_d, impl_a = remove_vig(h_o, d_o, a_o)
            edge_h = compute_edge(model_h, impl_h)
            edge_d = compute_edge(model_d, impl_d)
            edge_a = compute_edge(model_a, impl_a)
            odds_comparison.append({
                "bookmaker": bm_key,
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
                ("home_win", model_h, impl_h, h_o),
                ("draw",     model_d, impl_d, d_o),
                ("away_win", model_a, impl_a, a_o),
            ]:
                edge = compute_edge(model_p, impl_p)
                if edge < MIN_EDGE:
                    continue
                k = kelly(edge, dec_odds)
                tier = "high" if edge >= 0.10 else ("medium" if edge >= 0.07 else "low")
                value_bets.append({
                    "bookmaker": bm_key,
                    "outcome": outcome,
                    "model_prob": round(model_p, 3),
                    "bookmaker_odds": dec_odds,
                    "implied_prob": round(impl_p, 3),
                    "edge": round(edge, 3),
                    "kelly": round(k, 3),
                    "confidence_tier": tier,
                })

        vb_count = len(value_bets)
        print(f"  {home} vs {away} ({league}): "
              f"H={model_h:.0%} D={model_d:.0%} A={model_a:.0%} — "
              f"{vb_count} value bet(s)")

        matches.append({
            "match_id": abs(hash(f"{home}{away}{date}")) % 100000,
            "league": league,
            "date": date,
            "home_team": home,
            "away_team": away,
            "prediction": {
                "home_win": round(model_h, 3),
                "draw": round(model_d, 3),
                "away_win": round(model_a, 3),
                "predicted_outcome": outcome_name,
                "confidence": confidence,
            },
            "odds_comparison": odds_comparison,
            "value_bets": value_bets,
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches": matches,
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate demo predictions from odds JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--odds", default="the-odds-api.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    odds_path = repo_root / args.odds

    if not odds_path.exists():
        print(f"ERROR: {odds_path} not found.")
        sys.exit(1)

    output = make_predictions(odds_path, seed=args.seed)

    # Write to both backend output and frontend public
    out1 = repo_root / "backend" / "data" / "output" / "predictions.json"
    out2 = repo_root / "frontend" / "public" / "data" / "predictions.json"
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)

    for out in [out1, out2]:
        with open(out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output['matches'])} matches → {out}")


if __name__ == "__main__":
    main()
