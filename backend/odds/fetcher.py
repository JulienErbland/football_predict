"""
The Odds API v4 fetcher.

Fetches upcoming match odds for configured bookmakers and markets.
Free tier allows 500 requests/month — we cache aggressively (one file per date).
If today's file already exists, we skip the API call entirely.

API docs: https://the-odds-api.com/lol-of-the-api/
Endpoint used: GET /v4/sports/{sport}/odds

League key mapping (The Odds API uses different codes from football-data.org):
    PL  → soccer_england_premier_league
    PD  → soccer_spain_la_liga
    BL1 → soccer_germany_bundesliga
    SA  → soccer_italy_serie_a
    FL1 → soccer_france_ligue_one

CRITICAL: odds data is NEVER used for training. It is only consumed in
output/predict.py for post-prediction value bet comparison.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from config.loader import settings

_BASE_URL = "https://api.the-odds-api.com/v4"

# Map our league codes to The Odds API sport keys
_LEAGUE_KEYS = {
    "PL":  "soccer_england_premier_league",
    "PD":  "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA":  "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
}


class OddsFetcher:
    """Client for The Odds API v4."""

    def __init__(self, api_key: str | None = None):
        cfg = settings()
        self.api_key = api_key or cfg["api_keys"]["the_odds_api"]
        self._raw_dir = Path(cfg["paths"]["raw"]) / "odds"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._bookmakers = cfg.get("bookmakers", [])

    def fetch_upcoming_odds(
        self,
        league_code: str,
        markets: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch upcoming match odds for a league from The Odds API.

        Returns DataFrame with:
            home_team, away_team, date, bookmaker, market,
            home_odds, draw_odds, away_odds

        Caches one Parquet file per league per date to avoid burning free quota.
        """
        if markets is None:
            markets = ["h2h"]

        sport_key = _LEAGUE_KEYS.get(league_code)
        if sport_key is None:
            logger.warning(f"No Odds API key for league {league_code}")
            return pd.DataFrame()

        today_str = date.today().isoformat()
        cache_path = self._raw_dir / f"odds_{league_code}_{today_str}.parquet"
        if cache_path.exists():
            logger.info(f"Cache hit — loading {cache_path}")
            return pd.read_parquet(cache_path)

        url = f"{_BASE_URL}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "bookmakers": ",".join(self._bookmakers) if self._bookmakers else None,
        }
        # Remove None params
        params = {k: v for k, v in params.items() if v is not None}

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Odds API request failed for {league_code}: {e}")
            return pd.DataFrame()

        data = resp.json()
        rows = []
        for event in data:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            commence = event.get("commence_time", "")
            for bm in event.get("bookmakers", []):
                bm_key = bm.get("key", "")
                for market in bm.get("markets", []):
                    if market.get("key") not in markets:
                        continue
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    draw_key = "Draw"
                    home_odds = outcomes.get(home)
                    away_odds = outcomes.get(away)
                    draw_odds = outcomes.get(draw_key)
                    rows.append({
                        "home_team": home,
                        "away_team": away,
                        "date": commence,
                        "bookmaker": bm_key,
                        "market": market.get("key", "h2h"),
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                    })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df.to_parquet(cache_path, index=False)
            logger.info(f"Saved odds ({len(df)} rows) → {cache_path}")
        return df
