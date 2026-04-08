"""
football-data.org ingestion client.

Fetches match results, standings, upcoming fixtures, and lineups from the
football-data.org v4 API. Free tier allows 10 requests/minute, so we sleep
6.5s between requests to stay safely under the limit.

All data is saved as Parquet to data/raw/football_data/.
After fetching all leagues/seasons, all match files are concatenated into
data/processed/all_matches.parquet for use by the feature pipeline.

CLI usage:
    python -m ingestion.football_data --leagues PL,PD --seasons 2021,2022,2023,2024
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from config.loader import settings

# Football-data.org v4 base URL
_BASE_URL = "https://api.football-data.org/v4"

# Result encoding: 0=home win, 1=draw, 2=away win (consistent with model target)
_RESULT_MAP = {"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}


class FootballDataClient:
    """Client for the football-data.org v4 REST API."""

    def __init__(self, api_key: str | None = None):
        cfg = settings()
        self.api_key = api_key or cfg["api_keys"]["football_data"]
        self._session = requests.Session()
        self._session.headers.update({"X-Auth-Token": self.api_key})
        # Rate limit: free tier = 10 req/min → sleep 6.5s between calls
        self._rate_limit_sleep = 6.5
        self._raw_dir = Path(cfg["paths"]["raw"]) / "football_data"
        self._processed_dir = Path(cfg["paths"]["processed"])
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a rate-limited GET request and return parsed JSON."""
        url = f"{_BASE_URL}/{endpoint}"
        logger.debug(f"GET {url} params={params}")
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self._rate_limit_sleep)
        return resp.json()

    def fetch_matches(self, league_code: str, season: int) -> pd.DataFrame:
        """
        Fetch completed match results for a league and season.

        Returns DataFrame with columns:
            match_id, league, season, matchday, date,
            home_team_id, home_team, away_team_id, away_team,
            home_goals, away_goals, result (0/1/2), referee
        """
        data = self._get(f"competitions/{league_code}/matches", params={"season": season})
        rows = []
        for m in data.get("matches", []):
            score = m.get("score", {})
            full = score.get("fullTime", {})
            home_g = full.get("home")
            away_g = full.get("away")
            # Only include finished matches with a recorded result
            if m["status"] != "FINISHED" or home_g is None or away_g is None:
                continue
            winner = score.get("winner", "")
            result = _RESULT_MAP.get(winner)
            if result is None:
                continue
            referee = ""
            if m.get("referees"):
                referee = m["referees"][0].get("name", "")
            rows.append({
                "match_id": m["id"],
                "league": league_code,
                "season": season,
                "matchday": m.get("matchday"),
                "date": pd.Timestamp(m["utcDate"]).date(),
                "home_team_id": m["homeTeam"]["id"],
                "home_team": m["homeTeam"]["name"],
                "away_team_id": m["awayTeam"]["id"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": int(home_g),
                "away_goals": int(away_g),
                "result": result,
                "referee": referee,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            out_path = self._raw_dir / f"{league_code}_{season}_matches.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved {len(df)} matches → {out_path}")
        return df

    def fetch_standings(self, league_code: str, season: int) -> pd.DataFrame:
        """
        Fetch full league table for a given season.

        Returns DataFrame with: position, team_id, team, played, won, drawn, lost,
            goals_for, goals_against, goal_diff, points.
        """
        data = self._get(f"competitions/{league_code}/standings", params={"season": season})
        rows = []
        for standing in data.get("standings", []):
            if standing.get("type") != "TOTAL":
                continue
            for entry in standing.get("table", []):
                rows.append({
                    "league": league_code,
                    "season": season,
                    "position": entry["position"],
                    "team_id": entry["team"]["id"],
                    "team": entry["team"]["name"],
                    "played": entry["playedGames"],
                    "won": entry["won"],
                    "drawn": entry["draw"],
                    "lost": entry["lost"],
                    "goals_for": entry["goalsFor"],
                    "goals_against": entry["goalsAgainst"],
                    "goal_diff": entry["goalDifference"],
                    "points": entry["points"],
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            out_path = self._raw_dir / f"{league_code}_{season}_standings.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved standings ({len(df)} teams) → {out_path}")
        return df

    def fetch_upcoming(self, league_code: str) -> pd.DataFrame:
        """
        Fetch the next 10 scheduled (unplayed) matches for a league.

        Returns same schema as fetch_matches but without home_goals/away_goals/result
        (those fields are NaN — they haven't been played yet).
        """
        data = self._get(f"competitions/{league_code}/matches", params={"status": "SCHEDULED"})
        rows = []
        matches = data.get("matches", [])[:10]
        for m in matches:
            rows.append({
                "match_id": m["id"],
                "league": league_code,
                "matchday": m.get("matchday"),
                "date": pd.Timestamp(m["utcDate"]),
                "home_team_id": m["homeTeam"]["id"],
                "home_team": m["homeTeam"]["name"],
                "away_team_id": m["awayTeam"]["id"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": None,
                "away_goals": None,
                "result": None,
                "referee": (m["referees"][0].get("name", "") if m.get("referees") else ""),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            out_path = self._raw_dir / f"{league_code}_upcoming.parquet"
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved {len(df)} upcoming matches → {out_path}")
        return df

    def fetch_lineups(self, match_id: int) -> dict:
        """
        Fetch lineup information for a finished match.

        Returns dict:
            {
                "home_formation": "4-3-3",
                "away_formation": "4-4-2",
                "home_players": [...],
                "away_players": [...],
            }
        Formations may be None if the API doesn't provide them.
        """
        data = self._get(f"matches/{match_id}")
        m = data.get("match", data)  # v4 wraps in "match" key
        home_lineup = m.get("homeTeam", {})
        away_lineup = m.get("awayTeam", {})
        return {
            "home_formation": home_lineup.get("formation"),
            "away_formation": away_lineup.get("formation"),
            "home_players": [p.get("name") for p in home_lineup.get("lineup", [])],
            "away_players": [p.get("name") for p in away_lineup.get("lineup", [])],
        }


def build_all_matches(raw_dir: Path, processed_dir: Path) -> pd.DataFrame:
    """
    Concatenate all league/season match Parquet files into one master file.

    Saved to data/processed/all_matches.parquet. This is the primary input
    for the feature engineering pipeline.
    """
    files = list(raw_dir.glob("*_matches.parquet"))
    if not files:
        logger.warning("No match Parquet files found — run ingestion first.")
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["league", "season", "date"]).reset_index(drop=True)
    out_path = processed_dir / "all_matches.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Built all_matches.parquet with {len(df)} rows → {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest football-data.org match data")
    parser.add_argument("--leagues", default="PL,PD,BL1,SA,FL1",
                        help="Comma-separated league codes")
    parser.add_argument("--seasons", default="2021,2022,2023,2024",
                        help="Comma-separated season years")
    args = parser.parse_args()

    league_codes = [c.strip() for c in args.leagues.split(",")]
    seasons = [int(s.strip()) for s in args.seasons.split(",")]

    client = FootballDataClient()
    for league in league_codes:
        for season in seasons:
            logger.info(f"Fetching {league} {season}...")
            try:
                client.fetch_matches(league, season)
                client.fetch_standings(league, season)
            except Exception as e:
                logger.error(f"Failed {league} {season}: {e}")

    cfg = settings()
    raw_dir = Path(cfg["paths"]["raw"]) / "football_data"
    processed_dir = Path(cfg["paths"]["processed"])
    build_all_matches(raw_dir, processed_dir)


if __name__ == "__main__":
    main()
