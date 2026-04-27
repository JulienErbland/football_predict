"""Regression test for the T0.1 team-name join fix.

Guards against a reintroduction of the production bug where
``ingestion.football_data.fetch_upcoming()`` returns team names that do not
appear as ``home_team`` / ``away_team`` in the historical corpus, collapsing
every upcoming match to a cold-start feature vector and identical
probabilities.

The test covers two things independently:

1. **Normalizer fixed points** — every value in the alias table is itself a
   known key, so ``normalize_team(normalize_team(x)) == normalize_team(x)``
   for every registered alias.

2. **Upcoming ↔ historical alignment** — for every league and both API
   variants we've seen in the wild (football-data.org long-form and The Odds
   API short-form), the normalised name must exist in the historical
   ``all_matches.parquet`` OR be listed as a known new-entrant. New-entrants
   are teams that are legitimately playing in the current season but have no
   historical appearance in the 2021–2024 corpus (promoted clubs, first-time
   top-flight, etc.). They will cold-start their features; that is correct
   and expected — but we track them explicitly so that a *different* new
   name (a typo, an un-mapped alias) does not silently rejoin the bug class.

The test runs offline: it reads the local parquet and uses a hand-curated
fixture of known upcoming team names. It does not hit any network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from ingestion.name_normalizer import (  # noqa: E402
    _ALIASES,
    canonical_team_names,
    normalize_team,
)


# Teams that appear in current-season fixtures but did not play in 2021-2024.
# Every entry is a canonical name (the output of normalize_team, not the alias).
# Keeping this list small and curated forces any new cold-start team to be
# acknowledged rather than discovered in production.
_KNOWN_NEW_ENTRANTS: set[str] = {
    "Sunderland",      # Promoted to PL for 2025/26
    "Oviedo",          # Promoted to PD for 2025/26
    "Hamburger SV",    # Promoted to BL1 for 2025/26
    "Pisa",            # Promoted to SA for 2025/26
    "Paris FC",        # Promoted to FL1 for 2025/26 (distinct from Paris SG)
}


# A realistic offline fixture of team names that either ingestion source can
# return. Maintained by hand from inspection of saved upcoming parquet files
# and odds cache files. The test will flag any production drift away from
# these variants — if the football-data.org API renames a team, adding the
# new alias here and to the normaliser keeps them in sync.
_UPCOMING_VARIANTS: dict[str, list[str]] = {
    "PL": [
        "Arsenal FC", "AFC Bournemouth", "Brentford FC", "Brighton & Hove Albion FC",
        "Burnley FC", "Chelsea FC", "Crystal Palace FC", "Everton FC", "Fulham FC",
        "Leeds United FC", "Liverpool FC", "Manchester City FC", "Manchester United FC",
        "Newcastle United FC", "Nottingham Forest FC", "Sunderland AFC",
        "Tottenham Hotspur FC", "West Ham United FC", "Wolverhampton Wanderers FC",
        # Odds API variants
        "Brighton and Hove Albion", "Leeds United", "Manchester City",
        "Manchester United", "Newcastle United", "Nottingham Forest",
        "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers",
    ],
    "PD": [
        "Real Madrid CF", "Real Sociedad de Fútbol", "Elche CF", "FC Barcelona",
        "RCD Espanyol de Barcelona", "Sevilla FC", "Club Atlético de Madrid",
        "CA Osasuna", "Real Betis Balompié", "RCD Mallorca",
        "Rayo Vallecano de Madrid", "RC Celta de Vigo", "Real Oviedo",
        "Athletic Club", "Villarreal CF", "Levante UD", "Getafe CF",
        "Deportivo Alavés", "Girona FC",
        # Odds API variants
        "Alavés", "Athletic Bilbao", "Atlético Madrid", "Celta Vigo", "Espanyol",
        "Oviedo", "Rayo Vallecano", "Real Betis", "Real Sociedad",
    ],
    "BL1": [
        "FC Augsburg", "TSG 1899 Hoffenheim", "Borussia Dortmund", "Bayer 04 Leverkusen",
        "VfL Wolfsburg", "Eintracht Frankfurt", "RB Leipzig", "Borussia Mönchengladbach",
        "1. FC Heidenheim 1846", "1. FC Union Berlin", "FC St. Pauli 1910",
        "FC Bayern München", "1. FC Köln", "SV Werder Bremen", "VfB Stuttgart",
        "Hamburger SV", "1. FSV Mainz 05", "SC Freiburg",
        # Odds API variants
        "1. FC Heidenheim", "Bayer Leverkusen", "Borussia Monchengladbach",
        "FSV Mainz 05", "FC St. Pauli", "TSG Hoffenheim", "Union Berlin",
    ],
    "SA": [
        "AS Roma", "AC Pisa 1909", "Cagliari Calcio", "US Cremonese", "Torino FC",
        "Hellas Verona FC", "AC Milan", "Udinese Calcio", "Atalanta BC",
        "Juventus FC", "Genoa CFC", "US Sassuolo Calcio", "Parma Calcio 1913",
        "SSC Napoli", "Bologna FC 1909", "US Lecce", "Como 1907",
        "FC Internazionale Milano", "ACF Fiorentina", "SS Lazio",
        # Odds API variants
        "Hellas Verona", "Inter Milan", "Pisa",
    ],
    "FL1": [
        "Paris FC", "AS Monaco FC", "Olympique de Marseille", "FC Metz",
        "AJ Auxerre", "FC Nantes", "Stade Rennais FC 1901", "Angers SCO",
        "Toulouse FC", "Lille OSC", "OGC Nice", "Le Havre AC",
        "Olympique Lyonnais", "FC Lorient", "Racing Club de Lens",
        # Odds API variants
        "AS Monaco", "Paris Saint Germain", "RC Lens",
    ],
}


@pytest.fixture(scope="module")
def historical_names() -> set[str]:
    """Set of distinct canonical team names present in the historical corpus."""
    path = _BACKEND / "data" / "processed" / "all_matches.parquet"
    if not path.exists():
        pytest.skip(f"all_matches.parquet not found at {path}")
    df = pd.read_parquet(path)
    return set(df["home_team"]) | set(df["away_team"])


def test_normalizer_is_idempotent():
    """Applying the normaliser twice yields the same result as applying it once."""
    for alias in _ALIASES:
        once = normalize_team(alias)
        twice = normalize_team(once)
        assert once == twice, f"not idempotent: {alias!r} → {once!r} → {twice!r}"


def test_canonical_names_self_map():
    """Every canonical name in the value set is itself a valid alias key."""
    for canon in canonical_team_names():
        assert canon in _ALIASES, f"canonical name {canon!r} missing from alias keys"
        assert _ALIASES[canon] == canon, (
            f"canonical name {canon!r} does not self-map (maps to {_ALIASES[canon]!r})"
        )


def test_historical_corpus_is_canonical(historical_names):
    """Every team name in all_matches.parquet is a known canonical form."""
    canonical = canonical_team_names()
    leaked = historical_names - canonical
    assert not leaked, f"non-canonical names in historical corpus: {sorted(leaked)}"


@pytest.mark.parametrize("league", sorted(_UPCOMING_VARIANTS.keys()))
def test_upcoming_names_resolve_to_historical(league, historical_names):
    """After normalisation, upcoming team names are either in the historical
    corpus or explicitly listed as known new-entrants (promoted clubs)."""
    unresolved: list[tuple[str, str]] = []
    for variant in _UPCOMING_VARIANTS[league]:
        canon = normalize_team(variant)
        if canon in historical_names:
            continue
        if canon in _KNOWN_NEW_ENTRANTS:
            continue
        unresolved.append((variant, canon))
    assert not unresolved, (
        f"[{league}] variants not mapped to historical or new-entrants: {unresolved}"
    )
