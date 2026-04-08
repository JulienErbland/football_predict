"""
Transfermarkt scraping client.

Scrapes squad market values, injury lists, and player details from Transfermarkt
using requests + BeautifulSoup. No official API — be respectful:
  - 3s sleep between requests
  - Aggressive Parquet caching (skip re-fetch if file already exists)
  - Browser-like User-Agent header to avoid 403s

Supported league slugs (Transfermarkt's URL format):
    PL  → premier-league/startseite/wettbewerb/GB1
    PD  → laliga/startseite/wettbewerb/ES1
    BL1 → bundesliga/startseite/wettbewerb/L1
    SA  → serie-a/startseite/wettbewerb/IT1
    FL1 → ligue-1/startseite/wettbewerb/FR1

CLI usage:
    python -m ingestion.transfermarkt --leagues PL,PD,BL1
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

from config.loader import settings

_BASE_URL = "https://www.transfermarkt.com"

# Map our league codes to Transfermarkt URL slugs
_LEAGUE_SLUGS = {
    "PL":  ("premier-league", "GB1"),
    "PD":  ("laliga", "ES1"),
    "BL1": ("bundesliga", "L1"),
    "SA":  ("serie-a", "IT1"),
    "FL1": ("ligue-1", "FR1"),
}

# Mimic a real browser to avoid 403 responses
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_SLEEP_SEC = 3  # Polite delay between requests


def _raw_dir() -> Path:
    cfg = settings()
    path = Path(cfg["paths"]["raw"]) / "transfermarkt"
    path.mkdir(parents=True, exist_ok=True)
    return path


class TransfermarktScraper:
    """Scraper for Transfermarkt squad, injury, and player data."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _get_soup(self, url: str) -> BeautifulSoup:
        logger.debug(f"Scraping {url}")
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(_SLEEP_SEC)
        return BeautifulSoup(resp.content, "lxml")

    def fetch_squad_values(self, league_code: str) -> pd.DataFrame:
        """
        Fetch squad market values for all teams in a league.

        Returns DataFrame with:
            team_name, team_tm_id, squad_size, avg_age, total_value_eur_m
        """
        out_path = _raw_dir() / f"{league_code}_squad_values.parquet"
        if out_path.exists():
            logger.info(f"Cache hit — loading {out_path}")
            return pd.read_parquet(out_path)

        if league_code not in _LEAGUE_SLUGS:
            logger.warning(f"Unknown league code: {league_code}")
            return pd.DataFrame()

        slug, tm_code = _LEAGUE_SLUGS[league_code]
        url = f"{_BASE_URL}/{slug}/startseite/wettbewerb/{tm_code}"
        try:
            soup = self._get_soup(url)
        except Exception as e:
            logger.error(f"Failed to fetch squad values for {league_code}: {e}")
            return pd.DataFrame()

        rows = []
        # Main squad-value table is usually the first responsive table
        table = soup.find("table", class_="items")
        if table is None:
            logger.warning(f"No items table found for {league_code} at {url}")
            return pd.DataFrame()

        for row in table.find_all("tr", class_=["odd", "even"]):
            cols = row.find_all("td")
            if len(cols) < 5:
                continue
            try:
                # Extract team link to get TM id
                team_link = cols[0].find("a", href=True)
                team_name = team_link.get_text(strip=True) if team_link else cols[0].get_text(strip=True)
                team_href = team_link["href"] if team_link else ""
                # TM ID is the last numeric segment in the href
                team_tm_id = None
                parts = [p for p in team_href.split("/") if p.isdigit()]
                if parts:
                    team_tm_id = int(parts[-1])

                squad_size = _parse_int(cols[2].get_text(strip=True))
                avg_age = _parse_float(cols[3].get_text(strip=True))
                total_value = _parse_market_value(cols[-1].get_text(strip=True))
                rows.append({
                    "league": league_code,
                    "team_name": team_name,
                    "team_tm_id": team_tm_id,
                    "squad_size": squad_size,
                    "avg_age": avg_age,
                    "total_value_eur_m": total_value,
                })
            except Exception as e:
                logger.debug(f"Error parsing row: {e}")
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved squad values ({len(df)} teams) → {out_path}")
        return df

    def fetch_injuries(self, team_tm_id: int, team_name: str) -> pd.DataFrame:
        """
        Fetch current injury list for a team.

        Returns DataFrame with:
            team_name, player_name, position, injury_type, return_date
        """
        out_path = _raw_dir() / f"injuries_{team_tm_id}.parquet"
        if out_path.exists():
            logger.info(f"Cache hit — loading {out_path}")
            return pd.read_parquet(out_path)

        url = f"{_BASE_URL}/team/verletzungen/verein/{team_tm_id}"
        try:
            soup = self._get_soup(url)
        except Exception as e:
            logger.error(f"Failed to fetch injuries for team {team_tm_id}: {e}")
            return pd.DataFrame()

        rows = []
        table = soup.find("table", class_="items")
        if table is None:
            return pd.DataFrame()

        for row in table.find_all("tr", class_=["odd", "even"]):
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            try:
                player_name = cols[1].get_text(strip=True)
                position = cols[2].get_text(strip=True)
                injury_type = cols[3].get_text(strip=True)
                return_date = cols[-1].get_text(strip=True) if len(cols) > 4 else ""
                rows.append({
                    "team_name": team_name,
                    "player_name": player_name,
                    "position": position,
                    "injury_type": injury_type,
                    "return_date": return_date,
                })
            except Exception as e:
                logger.debug(f"Error parsing injury row: {e}")
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved injuries ({len(df)} players) for {team_name} → {out_path}")
        return df

    def fetch_squad_players(self, team_tm_id: int, team_name: str) -> pd.DataFrame:
        """
        Fetch full squad player list with age, nationality, position, and market value.

        Returns DataFrame with:
            team_name, player_name, age, nationality, position, market_value_eur_m
        """
        out_path = _raw_dir() / f"players_{team_tm_id}.parquet"
        if out_path.exists():
            logger.info(f"Cache hit — loading {out_path}")
            return pd.read_parquet(out_path)

        url = f"{_BASE_URL}/team/kader/verein/{team_tm_id}"
        try:
            soup = self._get_soup(url)
        except Exception as e:
            logger.error(f"Failed to fetch squad players for team {team_tm_id}: {e}")
            return pd.DataFrame()

        rows = []
        table = soup.find("table", class_="items")
        if table is None:
            return pd.DataFrame()

        for row in table.find_all("tr", class_=["odd", "even"]):
            cols = row.find_all("td")
            if len(cols) < 5:
                continue
            try:
                player_name = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                position = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                age = _parse_int(cols[3].get_text(strip=True)) if len(cols) > 3 else None
                nationality = cols[4].get_text(strip=True) if len(cols) > 4 else ""
                market_value = _parse_market_value(cols[-1].get_text(strip=True))
                rows.append({
                    "team_name": team_name,
                    "team_tm_id": team_tm_id,
                    "player_name": player_name,
                    "age": age,
                    "nationality": nationality,
                    "position": position,
                    "market_value_eur_m": market_value,
                })
            except Exception as e:
                logger.debug(f"Error parsing player row: {e}")
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved squad players ({len(df)}) for {team_name} → {out_path}")
        return df


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_float(s: str) -> float | None:
    """Parse a string like '26.3' or '26,3' to float."""
    try:
        return float(s.replace(",", ".").strip())
    except (ValueError, AttributeError):
        return None


def _parse_int(s: str) -> int | None:
    """Parse a string like '25' to int."""
    try:
        digits = "".join(c for c in s if c.isdigit())
        return int(digits) if digits else None
    except (ValueError, AttributeError):
        return None


def _parse_market_value(s: str) -> float | None:
    """
    Parse Transfermarkt market value strings like '€450m', '€34.50m', '€850k'.

    Returns value in millions of EUR.
    """
    if not s:
        return None
    s = s.strip().replace(",", ".").replace("€", "").replace(" ", "")
    multiplier = 1.0
    if s.endswith("m"):
        s = s[:-1]
        multiplier = 1.0
    elif s.endswith("k"):
        s = s[:-1]
        multiplier = 0.001  # Convert thousands to millions
    elif s.endswith("bn"):
        s = s[:-2]
        multiplier = 1000.0
    try:
        return float(s) * multiplier
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Scrape Transfermarkt squad data")
    parser.add_argument("--leagues", default="PL,PD,BL1",
                        help="Comma-separated league codes")
    args = parser.parse_args()

    league_codes = [c.strip() for c in args.leagues.split(",")]
    scraper = TransfermarktScraper()

    for league in league_codes:
        logger.info(f"Fetching squad values for {league}...")
        squad_df = scraper.fetch_squad_values(league)
        if squad_df.empty:
            logger.warning(f"No squad data for {league}")
            continue
        # For each team, fetch player list and injuries
        for _, team_row in squad_df.iterrows():
            tm_id = team_row.get("team_tm_id")
            name = team_row.get("team_name", "")
            if tm_id is None:
                continue
            logger.info(f"  Fetching players for {name} (id={tm_id})...")
            scraper.fetch_squad_players(int(tm_id), name)
            logger.info(f"  Fetching injuries for {name}...")
            scraper.fetch_injuries(int(tm_id), name)


if __name__ == "__main__":
    main()
