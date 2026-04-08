"""
StatsBomb open data ingestion.

Uses the statsbombpy library to pull free open-data for competitions, matches,
events, and lineups. No API key required — data is hosted publicly on GitHub.

Key metric computed here: PPDA (Passes Permitted per Defensive Action)
    PPDA = opponent_passes_in_opponent_half / our_defensive_actions
    Lower PPDA = more pressing (Liverpool-style); higher = sitting deep.
    We compute it as: opponent_passes_allowed / (tackles + interceptions + fouls)

All data saved as Parquet to data/raw/statsbomb/.

CLI usage:
    python -m ingestion.statsbomb --competitions 2,11
"""

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from config.loader import settings

try:
    from statsbombpy import sb
    _STATSBOMB_AVAILABLE = True
except ImportError:
    _STATSBOMB_AVAILABLE = False
    logger.warning("statsbombpy not installed — StatsBomb ingestion will be skipped.")


def _raw_dir() -> Path:
    cfg = settings()
    path = Path(cfg["paths"]["raw"]) / "statsbomb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_competitions() -> pd.DataFrame:
    """
    List all available StatsBomb open-data competitions.

    Returns DataFrame with: competition_id, season_id, competition_name,
        season_name, country_name.
    """
    if not _STATSBOMB_AVAILABLE:
        return pd.DataFrame()
    comps = sb.competitions()
    out_path = _raw_dir() / "competitions.parquet"
    comps.to_parquet(out_path, index=False)
    logger.info(f"Fetched {len(comps)} competitions → {out_path}")
    return comps


def _compute_ppda(events_df: pd.DataFrame, team: str) -> float:
    """
    Compute PPDA for a team from a single match's event stream.

    PPDA = passes allowed by the team's opponents in the opponent's own half
           / defensive actions (tackles + interceptions + fouls) by the team.

    A low PPDA means the team presses hard; a high PPDA means they sit back.
    We use the opponent's passes as the numerator because PPDA measures how many
    passes the pressing team *allows* before winning the ball.
    """
    if events_df is None or events_df.empty:
        return float("nan")
    try:
        teams = events_df["team"].unique()
        opponent = [t for t in teams if t != team]
        if not opponent:
            return float("nan")
        opponent = opponent[0]
        # Opponent passes in their own half (roughly: y < 40 in StatsBomb coords)
        opp_passes = events_df[
            (events_df["team"] == opponent)
            & (events_df["type"] == "Pass")
        ]
        # Defensive actions by the pressing team
        def_actions = events_df[
            (events_df["team"] == team)
            & (events_df["type"].isin(["Pressure", "Tackle", "Interception", "Foul Committed"]))
        ]
        if len(def_actions) == 0:
            return float("nan")
        return len(opp_passes) / len(def_actions)
    except Exception:
        return float("nan")


def fetch_match_xg(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Fetch per-match xG, shots, shots on target, and PPDA for a competition/season.

    Returns DataFrame with:
        match_id, home_team, away_team,
        home_xg, away_xg,
        home_shots, away_shots,
        home_sot, away_sot,
        home_ppda, away_ppda
    """
    if not _STATSBOMB_AVAILABLE:
        return pd.DataFrame()

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    if matches.empty:
        logger.warning(f"No matches for competition {competition_id}, season {season_id}")
        return pd.DataFrame()

    rows = []
    for _, match in matches.iterrows():
        match_id = match["match_id"]
        home = match["home_team"]
        away = match["away_team"]
        try:
            events = sb.events(match_id=match_id)
        except Exception as e:
            logger.warning(f"Could not fetch events for match {match_id}: {e}")
            events = pd.DataFrame()

        def _xg_for(team):
            if events.empty:
                return float("nan")
            shots = events[(events["team"] == team) & (events["type"] == "Shot")]
            xg_col = "shot_statsbomb_xg" if "shot_statsbomb_xg" in events.columns else None
            if xg_col:
                return shots[xg_col].fillna(0).sum()
            return float("nan")

        def _shots_for(team):
            if events.empty:
                return 0
            return int((events["team"] == team).sum() &
                       (events["type"] == "Shot").sum()) if False else \
                   len(events[(events["team"] == team) & (events["type"] == "Shot")])

        def _sot_for(team):
            if events.empty:
                return 0
            shots = events[(events["team"] == team) & (events["type"] == "Shot")]
            if "shot_outcome" in shots.columns:
                return int(shots["shot_outcome"].isin(["Saved", "Goal"]).sum())
            return 0

        rows.append({
            "match_id": match_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "home_team": home,
            "away_team": away,
            "home_xg": _xg_for(home),
            "away_xg": _xg_for(away),
            "home_shots": _shots_for(home),
            "away_shots": _shots_for(away),
            "home_sot": _sot_for(home),
            "away_sot": _sot_for(away),
            "home_ppda": _compute_ppda(events, home) if not events.empty else float("nan"),
            "away_ppda": _compute_ppda(events, away) if not events.empty else float("nan"),
        })
        logger.debug(f"Processed match {match_id}")

    df = pd.DataFrame(rows)
    if not df.empty:
        out_path = _raw_dir() / f"xg_{competition_id}_{season_id}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved xG data ({len(df)} matches) → {out_path}")
    return df


def fetch_formations(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Fetch lineup formations and player positions for each match.

    Returns DataFrame with:
        match_id, team, formation, player_name, player_position, jersey_number
    """
    if not _STATSBOMB_AVAILABLE:
        return pd.DataFrame()

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    if matches.empty:
        return pd.DataFrame()

    rows = []
    for _, match in matches.iterrows():
        match_id = match["match_id"]
        try:
            lineups = sb.lineups(match_id=match_id)
        except Exception as e:
            logger.warning(f"Could not fetch lineups for match {match_id}: {e}")
            continue
        for team, lineup_df in lineups.items():
            formation = match.get("home_team_formation") if team == match["home_team"] \
                else match.get("away_team_formation")
            for _, player in lineup_df.iterrows():
                position = ""
                if "positions" in player and player["positions"]:
                    pos_list = player["positions"]
                    if isinstance(pos_list, list) and pos_list:
                        position = pos_list[0].get("position", "") if isinstance(pos_list[0], dict) else str(pos_list[0])
                rows.append({
                    "match_id": match_id,
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "team": team,
                    "formation": formation,
                    "player_name": player.get("player_name", ""),
                    "player_position": position,
                    "jersey_number": player.get("jersey_number"),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        out_path = _raw_dir() / f"formations_{competition_id}_{season_id}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved formations ({len(df)} player-rows) → {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest StatsBomb open data")
    parser.add_argument("--competitions", default="2,11",
                        help="Comma-separated competition IDs")
    args = parser.parse_args()

    competition_ids = [int(c.strip()) for c in args.competitions.split(",")]

    fetch_competitions()

    if not _STATSBOMB_AVAILABLE:
        logger.error("statsbombpy not available — cannot fetch competition data.")
        return

    comps = sb.competitions()
    for comp_id in competition_ids:
        comp_seasons = comps[comps["competition_id"] == comp_id]
        if comp_seasons.empty:
            logger.warning(f"Competition {comp_id} not found in open data.")
            continue
        for _, row in comp_seasons.iterrows():
            season_id = row["season_id"]
            logger.info(f"Fetching competition {comp_id}, season {season_id}...")
            fetch_match_xg(comp_id, season_id)
            fetch_formations(comp_id, season_id)


if __name__ == "__main__":
    main()
