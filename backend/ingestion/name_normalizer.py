"""
Team-name canonicalisation for the 5 target leagues.

The pipeline has three ingestion sources that use different conventions
for the same club:

    - football-data.co.uk CSVs  → short names  (canonical)   e.g. "Arsenal"
    - football-data.org API     → long names with suffix     e.g. "Arsenal FC"
    - The Odds API v4           → a third variant            e.g. "Nottingham Forest"

Any team-keyed feature (Elo, Form, H2H, context) joins on the string value
of ``home_team`` / ``away_team``. If the historical corpus holds "Arsenal"
but inference appends a row with "Arsenal FC", every lookup cold-starts to
defaults and every match gets an identical feature vector. This is the
production bug T0.1 targets.

This module declares a single canonical form (the CSV short name, because
the training corpus already uses it) and maps every known alias to it. The
normaliser is idempotent: a canonical name maps to itself.

Unknown names pass through unchanged. The regression test
``backend/tests/test_name_consistency.py`` asserts that every team found in
``fetch_upcoming()`` output, post-normalisation, is also present in the
historical frame — so any new alias surfaces there rather than silently
collapsing predictions.
"""

from __future__ import annotations

import pandas as pd

# ─── Canonical alias map ─────────────────────────────────────────────────────
# Keys are any variant we might see in the wild; values are the canonical
# short-form used by football-data.co.uk CSV ingestion (and therefore by the
# training corpus and every downstream team-keyed feature).
#
# Entries where the key equals the value exist explicitly so the test can
# enumerate "known-canonical teams" without re-deriving the set.

_ALIASES: dict[str, str] = {
    # ─── Premier League ──────────────────────────────────────────────────────
    "Arsenal": "Arsenal",
    "Arsenal FC": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Aston Villa FC": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "AFC Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brentford FC": "Brentford",
    "Brighton": "Brighton",
    "Brighton & Hove Albion FC": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Burnley FC": "Burnley",
    "Chelsea": "Chelsea",
    "Chelsea FC": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Crystal Palace FC": "Crystal Palace",
    "Everton": "Everton",
    "Everton FC": "Everton",
    "Fulham": "Fulham",
    "Fulham FC": "Fulham",
    "Ipswich": "Ipswich",
    "Ipswich Town FC": "Ipswich",
    "Leeds": "Leeds",
    "Leeds United FC": "Leeds",
    "Leeds United": "Leeds",
    "Leicester": "Leicester",
    "Leicester City FC": "Leicester",
    "Liverpool": "Liverpool",
    "Liverpool FC": "Liverpool",
    "Luton": "Luton",
    "Luton Town FC": "Luton",
    "Man City": "Man City",
    "Manchester City FC": "Man City",
    "Manchester City": "Man City",
    "Man United": "Man United",
    "Manchester United FC": "Man United",
    "Manchester United": "Man United",
    "Newcastle": "Newcastle",
    "Newcastle United FC": "Newcastle",
    "Newcastle United": "Newcastle",
    "Norwich": "Norwich",
    "Norwich City FC": "Norwich",
    "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest FC": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield United FC": "Sheffield United",
    "Southampton": "Southampton",
    "Southampton FC": "Southampton",
    "Sunderland": "Sunderland",
    "Sunderland AFC": "Sunderland",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Watford": "Watford",
    "Watford FC": "Watford",
    "West Ham": "West Ham",
    "West Ham United FC": "West Ham",
    "West Ham United": "West Ham",
    "Wolves": "Wolves",
    "Wolverhampton Wanderers FC": "Wolves",
    "Wolverhampton Wanderers": "Wolves",

    # ─── La Liga (PD) ────────────────────────────────────────────────────────
    "Alaves": "Alaves",
    "Deportivo Alavés": "Alaves",
    "Alavés": "Alaves",
    "Almeria": "Almeria",
    "UD Almería": "Almeria",
    "Ath Bilbao": "Ath Bilbao",
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "Ath Madrid": "Ath Madrid",
    "Club Atlético de Madrid": "Ath Madrid",
    "Atlético Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Barcelona": "Barcelona",
    "FC Barcelona": "Barcelona",
    "Betis": "Betis",
    "Real Betis Balompié": "Betis",
    "Real Betis": "Betis",
    "Cadiz": "Cadiz",
    "Cádiz CF": "Cadiz",
    "Celta": "Celta",
    "RC Celta de Vigo": "Celta",
    "Celta Vigo": "Celta",
    "Elche": "Elche",
    "Elche CF": "Elche",
    "Espanol": "Espanol",
    "RCD Espanyol de Barcelona": "Espanol",
    "Espanyol": "Espanol",
    "Getafe": "Getafe",
    "Getafe CF": "Getafe",
    "Girona": "Girona",
    "Girona FC": "Girona",
    "Granada": "Granada",
    "Granada CF": "Granada",
    "Las Palmas": "Las Palmas",
    "UD Las Palmas": "Las Palmas",
    "Leganes": "Leganes",
    "CD Leganés": "Leganes",
    "Levante": "Levante",
    "Levante UD": "Levante",
    "Mallorca": "Mallorca",
    "RCD Mallorca": "Mallorca",
    "Osasuna": "Osasuna",
    "CA Osasuna": "Osasuna",
    "Oviedo": "Oviedo",
    "Real Oviedo": "Oviedo",
    "Real Madrid": "Real Madrid",
    "Real Madrid CF": "Real Madrid",
    "Sevilla": "Sevilla",
    "Sevilla FC": "Sevilla",
    "Sociedad": "Sociedad",
    "Real Sociedad de Fútbol": "Sociedad",
    "Real Sociedad": "Sociedad",
    "Valencia": "Valencia",
    "Valencia CF": "Valencia",
    "Valladolid": "Valladolid",
    "Real Valladolid CF": "Valladolid",
    "Vallecano": "Vallecano",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Rayo Vallecano": "Vallecano",
    "Villarreal": "Villarreal",
    "Villarreal CF": "Villarreal",

    # ─── Bundesliga (BL1) ────────────────────────────────────────────────────
    "Augsburg": "Augsburg",
    "FC Augsburg": "Augsburg",
    "Bayern Munich": "Bayern Munich",
    "FC Bayern München": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    "Bielefeld": "Bielefeld",
    "DSC Arminia Bielefeld": "Bielefeld",
    "Bochum": "Bochum",
    "VfL Bochum 1848": "Bochum",
    "VfL Bochum": "Bochum",
    "Darmstadt": "Darmstadt",
    "SV Darmstadt 98": "Darmstadt",
    "Dortmund": "Dortmund",
    "Borussia Dortmund": "Dortmund",
    "BVB": "Dortmund",
    "Ein Frankfurt": "Ein Frankfurt",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Koln": "FC Koln",
    "1. FC Köln": "FC Koln",
    "1. FC Koln": "FC Koln",
    "1. FC Köln 1948": "FC Koln",
    "Freiburg": "Freiburg",
    "SC Freiburg": "Freiburg",
    "Greuther Furth": "Greuther Furth",
    "SpVgg Greuther Fürth": "Greuther Furth",
    "Hamburger SV": "Hamburger SV",
    "Hamburger SV 1887": "Hamburger SV",
    "Heidenheim": "Heidenheim",
    "1. FC Heidenheim 1846": "Heidenheim",
    "1. FC Heidenheim": "Heidenheim",
    "Hertha": "Hertha",
    "Hertha BSC": "Hertha",
    "Hoffenheim": "Hoffenheim",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "Holstein Kiel": "Holstein Kiel",
    "Leverkusen": "Leverkusen",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Bayer Leverkusen": "Leverkusen",
    "M'gladbach": "M'gladbach",
    "Borussia Mönchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "Mainz": "Mainz",
    "1. FSV Mainz 05": "Mainz",
    "FSV Mainz 05": "Mainz",
    "RB Leipzig": "RB Leipzig",
    "Schalke 04": "Schalke 04",
    "FC Schalke 04": "Schalke 04",
    "St Pauli": "St Pauli",
    "FC St. Pauli 1910": "St Pauli",
    "FC St. Pauli": "St Pauli",
    "St. Pauli": "St Pauli",
    "Stuttgart": "Stuttgart",
    "VfB Stuttgart": "Stuttgart",
    "Union Berlin": "Union Berlin",
    "1. FC Union Berlin": "Union Berlin",
    "Werder Bremen": "Werder Bremen",
    "SV Werder Bremen": "Werder Bremen",
    "Wolfsburg": "Wolfsburg",
    "VfL Wolfsburg": "Wolfsburg",

    # ─── Serie A (SA) ────────────────────────────────────────────────────────
    "Atalanta": "Atalanta",
    "Atalanta BC": "Atalanta",
    "Bologna": "Bologna",
    "Bologna FC 1909": "Bologna",
    "Cagliari": "Cagliari",
    "Cagliari Calcio": "Cagliari",
    "Como": "Como",
    "Como 1907": "Como",
    "Cremonese": "Cremonese",
    "US Cremonese": "Cremonese",
    "Empoli": "Empoli",
    "Empoli FC": "Empoli",
    "Fiorentina": "Fiorentina",
    "ACF Fiorentina": "Fiorentina",
    "Frosinone": "Frosinone",
    "Frosinone Calcio": "Frosinone",
    "Genoa": "Genoa",
    "Genoa CFC": "Genoa",
    "Inter": "Inter",
    "FC Internazionale Milano": "Inter",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "Juventus": "Juventus",
    "Juventus FC": "Juventus",
    "Lazio": "Lazio",
    "SS Lazio": "Lazio",
    "Lecce": "Lecce",
    "US Lecce": "Lecce",
    "Milan": "Milan",
    "AC Milan": "Milan",
    "Monza": "Monza",
    "AC Monza": "Monza",
    "Napoli": "Napoli",
    "SSC Napoli": "Napoli",
    "Parma": "Parma",
    "Parma Calcio 1913": "Parma",
    "Pisa": "Pisa",
    "AC Pisa 1909": "Pisa",
    "Pisa SC": "Pisa",
    "Roma": "Roma",
    "AS Roma": "Roma",
    "Salernitana": "Salernitana",
    "US Salernitana 1919": "Salernitana",
    "Sampdoria": "Sampdoria",
    "UC Sampdoria": "Sampdoria",
    "Sassuolo": "Sassuolo",
    "US Sassuolo Calcio": "Sassuolo",
    "Spezia": "Spezia",
    "Spezia Calcio": "Spezia",
    "Torino": "Torino",
    "Torino FC": "Torino",
    "Udinese": "Udinese",
    "Udinese Calcio": "Udinese",
    "Venezia": "Venezia",
    "Venezia FC": "Venezia",
    "Verona": "Verona",
    "Hellas Verona FC": "Verona",
    "Hellas Verona": "Verona",

    # ─── Ligue 1 (FL1) ───────────────────────────────────────────────────────
    "Ajaccio": "Ajaccio",
    "AC Ajaccio": "Ajaccio",
    "Angers": "Angers",
    "Angers SCO": "Angers",
    "Auxerre": "Auxerre",
    "AJ Auxerre": "Auxerre",
    "Bordeaux": "Bordeaux",
    "Girondins de Bordeaux": "Bordeaux",
    "Brest": "Brest",
    "Stade Brestois 29": "Brest",
    "Clermont": "Clermont",
    "Clermont Foot 63": "Clermont",
    "Le Havre": "Le Havre",
    "Le Havre AC": "Le Havre",
    "Lens": "Lens",
    "Racing Club de Lens": "Lens",
    "RC Lens": "Lens",
    "Lille": "Lille",
    "Lille OSC": "Lille",
    "LOSC Lille": "Lille",
    "Lorient": "Lorient",
    "FC Lorient": "Lorient",
    "Lyon": "Lyon",
    "Olympique Lyonnais": "Lyon",
    "Marseille": "Marseille",
    "Olympique de Marseille": "Marseille",
    "Olympique Marseille": "Marseille",
    "Metz": "Metz",
    "FC Metz": "Metz",
    "Monaco": "Monaco",
    "AS Monaco FC": "Monaco",
    "AS Monaco": "Monaco",
    "Montpellier": "Montpellier",
    "Montpellier HSC": "Montpellier",
    "Nantes": "Nantes",
    "FC Nantes": "Nantes",
    "Nice": "Nice",
    "OGC Nice": "Nice",
    "Paris SG": "Paris SG",
    "Paris Saint-Germain FC": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Paris Saint Germain": "Paris SG",
    "PSG": "Paris SG",
    "Paris FC": "Paris FC",
    "Reims": "Reims",
    "Stade de Reims": "Reims",
    "Rennes": "Rennes",
    "Stade Rennais FC 1901": "Rennes",
    "Stade Rennais": "Rennes",
    "St Etienne": "St Etienne",
    "AS Saint-Étienne": "St Etienne",
    "Saint-Étienne": "St Etienne",
    "Saint-Etienne": "St Etienne",
    "Strasbourg": "Strasbourg",
    "RC Strasbourg Alsace": "Strasbourg",
    "Toulouse": "Toulouse",
    "Toulouse FC": "Toulouse",
    "Troyes": "Troyes",
    "ESTAC Troyes": "Troyes",
}


def normalize_team(name: str | None) -> str:
    """Return the canonical team name for ``name``.

    Unknown names fall through unchanged after whitespace strip. This keeps
    the pipeline robust to newly-promoted teams that are not yet in the
    alias table — they cold-start on features, which is correct behaviour,
    and the regression test surfaces the new alias so it can be added.
    """
    if name is None:
        return ""
    cleaned = str(name).strip()
    if not cleaned:
        return cleaned
    return _ALIASES.get(cleaned, cleaned)


def normalize_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("home_team", "away_team"),
) -> pd.DataFrame:
    """Apply :func:`normalize_team` to each listed column in a DataFrame."""
    if df is None or df.empty:
        return df
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(normalize_team)
    return df


def canonical_team_names() -> set[str]:
    """Return the set of distinct canonical names the normaliser can produce."""
    return set(_ALIASES.values())
