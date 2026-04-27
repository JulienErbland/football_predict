"""
football-data.co.uk CSV ingestion — free alternative to football-data.org API.

Downloads historical match CSVs (no API key required) and produces the same
Parquet schema as football_data.py so the rest of the pipeline is unaffected.

Output schema (matches football_data.py exactly):
    match_id, league, season, matchday, date,
    home_team_id, home_team, away_team_id, away_team,
    home_goals, away_goals, result (0=H, 1=D, 2=A), referee

Note: match_id is synthetic (hash of league+season+teams+date).
      home_team_id / away_team_id are synthetic (hash of team name).
      matchday is left as 0 (not provided by this source).

CLI usage (from backend/):
    python -m ingestion.football_data_csv --leagues PL,PD,BL1,SA,FL1 --seasons 2021,2022,2023,2024
"""

from __future__ import annotations

import argparse
import hashlib
import io
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from config.loader import settings
from ingestion.name_normalizer import normalize_columns

# football-data.co.uk league codes
_LEAGUE_MAP = {
    "PL":  "E0",   # Premier League
    "PD":  "SP1",  # La Liga
    "BL1": "D1",   # Bundesliga
    "SA":  "I1",   # Serie A
    "FL1": "F1",   # Ligue 1
}

# FTR column → result int
_RESULT_MAP = {"H": 0, "D": 1, "A": 2}

_BASE_URL = "https://www.football-data.co.uk/mmz4281"


def _season_slug(season: int) -> str:
    """Convert API-style season year to URL slug: 2021 → '2122'."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _team_id(name: str) -> int:
    """Deterministic synthetic team ID from name (stable across seasons)."""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def _match_id(league: str, season: int, date: str, home: str, away: str) -> int:
    key = f"{league}_{season}_{date}_{home}_{away}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def fetch_matches_csv(league_code: str, season: int, raw_dir: Path) -> pd.DataFrame:
    """
    Download one season CSV from football-data.co.uk and return a DataFrame
    with the same schema as FootballDataClient.fetch_matches().
    """
    csv_code = _LEAGUE_MAP.get(league_code)
    if csv_code is None:
        raise ValueError(f"Unknown league code: {league_code}. Valid: {list(_LEAGUE_MAP)}")

    slug = _season_slug(season)
    url = f"{_BASE_URL}/{slug}/{csv_code}.csv"
    logger.debug(f"GET {url}")

    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        logger.warning(f"No data for {league_code} {season} (404) — skipping.")
        return pd.DataFrame()
    resp.raise_for_status()

    # The CSVs sometimes have trailing commas → use python engine to be lenient
    raw = pd.read_csv(io.StringIO(resp.text), engine="python", on_bad_lines="skip")

    rows = []
    for _, m in raw.iterrows():
        ftr = str(m.get("FTR", "")).strip()
        result = _RESULT_MAP.get(ftr)
        if result is None:
            continue  # skip rows without a valid result (header duplicates, blanks)

        try:
            home_g = int(m["FTHG"])
            away_g = int(m["FTAG"])
        except (ValueError, KeyError):
            continue

        home = str(m.get("HomeTeam", "")).strip()
        away = str(m.get("AwayTeam", "")).strip()
        if not home or not away:
            continue

        # Parse date — CSVs use DD/MM/YY or DD/MM/YYYY
        raw_date = str(m.get("Date", "")).strip()
        try:
            date = pd.to_datetime(raw_date, dayfirst=True)
        except Exception:
            logger.warning(f"Unparseable date '{raw_date}' — skipping row.")
            continue

        referee = str(m.get("Referee", "")).strip()

        rows.append({
            "match_id":     _match_id(league_code, season, str(date.date()), home, away),
            "league":       league_code,
            "season":       season,
            "matchday":     0,
            "date":         date,
            "home_team_id": _team_id(home),
            "home_team":    home,
            "away_team_id": _team_id(away),
            "away_team":    away,
            "home_goals":   home_g,
            "away_goals":   away_g,
            "result":       result,
            "referee":      referee,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning(f"No finished matches parsed for {league_code} {season}.")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = normalize_columns(df)
    out_path = raw_dir / f"{league_code}_{season}_matches.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} matches → {out_path}")
    return df


def run(leagues: list[str], seasons: list[int]) -> None:
    cfg = settings()
    raw_dir = Path(cfg["paths"]["raw"]) / "football_data"
    processed_dir = Path(cfg["paths"]["processed"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    for league in leagues:
        for season in seasons:
            logger.info(f"Fetching {league} {season}...")
            try:
                df = fetch_matches_csv(league, season, raw_dir)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed {league} {season}: {e}")
            time.sleep(0.5)  # be polite — no strict rate limit but avoid hammering

    if not all_dfs:
        logger.error("No data fetched. Check league/season arguments.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = processed_dir / "all_matches.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info(f"Saved combined {len(combined)} matches → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical CSVs from football-data.co.uk")
    parser.add_argument("--leagues", default="PL,PD,BL1,SA,FL1",
                        help="Comma-separated league codes (PL,PD,BL1,SA,FL1)")
    parser.add_argument("--seasons", default="2021,2022,2023,2024",
                        help="Comma-separated season start years (e.g. 2021 = 2021/22)")
    args = parser.parse_args()

    leagues = [l.strip() for l in args.leagues.split(",")]
    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    run(leagues, seasons)


if __name__ == "__main__":
    main()
