"""
Live NBA game log fetcher via ESPN public API (no auth required).

Returns a DataFrame matching the player_features.parquet schema so it
plugs straight into the existing simulate() pipeline.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nba_gpt.config import DATA_CONFIG, INPUT_FEATURES

_ESPN_BASE  = "https://site.api.espn.com/apis"
_SEARCH     = f"{_ESPN_BASE}/search/v2"
_GAMELOG    = f"{_ESPN_BASE}/common/v3/sports/basketball/nba/athletes/{{athlete_id}}/gamelog"
_ROSTER     = f"{_ESPN_BASE}/site/v2/sports/basketball/nba/teams/{{team}}/roster"
_STANDINGS  = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 20
_SLEEP   = 0.5

# In-process cache: {team_abbr: avg_pts_allowed}
_DEF_RATING_CACHE: dict[str, float] = {}

TARGET_ROLL = ["points", "reboundsTotal", "assists", "steals", "blocks", "threePointersMade"]

ERA_BOUNDARIES = [
    (1947, 0), (1955, 1), (1970, 2),
    (1980, 3), (1992, 4), (2004, 5),
]


def _get(url: str, params: dict | None = None) -> dict:
    time.sleep(_SLEEP)
    r = requests.get(url, headers=_HEADERS, params=params or {}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _era_id(season_year: int) -> int:
    era = 0
    for start, eid in ERA_BOUNDARIES:
        if season_year >= start:
            era = eid
    return era


def get_team_def_ratings() -> dict[str, float]:
    """
    Return {team_abbr: avg_pts_allowed_per_game} for the current season.
    Fetched once per process run from ESPN standings; falls back to 109.0.
    """
    global _DEF_RATING_CACHE
    if _DEF_RATING_CACHE:
        return _DEF_RATING_CACHE

    try:
        data = _get(_STANDINGS)
        for conference in data.get("children", []):
            for entry in conference.get("standings", {}).get("entries", []):
                abbr = entry.get("team", {}).get("abbreviation", "")
                if not abbr:
                    continue
                for stat in entry.get("stats", []):
                    if stat.get("name") == "avgPointsAgainst":
                        _DEF_RATING_CACHE[abbr] = float(stat["value"])
                        break
    except Exception as e:
        print(f"  [def ratings] fetch failed ({e}), using league average")

    return _DEF_RATING_CACHE


def find_espn_athlete_id(player_name: str) -> tuple[int, str]:
    """Search ESPN for a player and return (espn_athlete_id, full_name)."""
    data = _get(_SEARCH, {"query": player_name, "sport": "basketball", "league": "nba"})
    contents = data.get("results", [{}])[0].get("contents", [])
    players = [c for c in contents if c.get("type") == "player"]
    if not players:
        raise ValueError(f"No ESPN player found matching '{player_name}'")

    hit = players[0]
    # extract numeric ID from web link  e.g. .../id/3945274/...
    web_link = hit.get("link", {}).get("web", "")
    m = re.search(r"/id/(\d+)/", web_link)
    if not m:
        raise ValueError(f"Could not parse ESPN athlete ID from link: {web_link}")
    return int(m.group(1)), hit["displayName"]


def _parse_gamelog(data: dict, espn_id: int, season_year: int) -> pd.DataFrame:
    """Parse an ESPN gamelog API response into a raw DataFrame."""
    labels = data.get("labels", [])
    events_meta = data.get("events", {})

    season_types = data.get("seasonTypes", [])
    reg = next(
        (st for st in season_types if "Regular" in st.get("displayName", "")),
        season_types[0] if season_types else None,
    )
    if reg is None:
        return pd.DataFrame()

    categories = reg.get("categories", [])
    best_cat = max(categories, key=lambda c: len(c.get("events", [])), default=None)
    if best_cat is None or not best_cat.get("events"):
        return pd.DataFrame()

    game_events = best_cat["events"]
    stat_idx = {lbl: i for i, lbl in enumerate(labels)}

    rows = []
    for ge in game_events:
        eid = ge["eventId"]
        stats_raw = ge.get("stats", [])
        meta = events_meta.get(eid, {})
        if not stats_raw or len(stats_raw) < len(labels):
            continue

        def _float(lbl: str, default: float = 0.0) -> float:
            idx = stat_idx.get(lbl)
            if idx is None:
                return default
            try:
                return float(str(stats_raw[idx]).split("-")[0])
            except (ValueError, TypeError):
                return default

        def _split(lbl: str) -> tuple[float, float]:
            idx = stat_idx.get(lbl)
            if idx is None:
                return 0.0, 0.0
            parts = str(stats_raw[idx]).split("-")
            try:
                return float(parts[0]), float(parts[1])
            except (ValueError, TypeError, IndexError):
                return 0.0, 0.0

        fg_made, fg_att   = _split("FG")
        fg3_made, fg3_att = _split("3PT")
        ft_made, ft_att   = _split("FT")

        at_vs    = str(meta.get("atVs", "@"))
        home     = 1.0 if "vs" in at_vs.lower() else 0.0
        opp_abbr = meta.get("opponent", {}).get("abbreviation", "")

        rows.append({
            "gameDateTimeEst":     str(meta.get("gameDate", "")),
            "points":              _float("PTS"),
            "reboundsTotal":       _float("REB"),
            "assists":             _float("AST"),
            "steals":              _float("STL"),
            "blocks":              _float("BLK"),
            "threePointersMade":   fg3_made,
            "numMinutes":          _float("MIN"),
            "fieldGoalsAttempted": fg_att,
            "fieldGoalsMade":      fg_made,
            "freeThrowsAttempted": ft_att,
            "freeThrowsMade":      ft_made,
            "turnovers":           _float("TO"),
            "plusMinusPoints":     0.0,
            "home":                home,
            "opp_abbr":            opp_abbr,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    df = df.dropna(subset=["gameDateTimeEst"]).sort_values("gameDateTimeEst").reset_index(drop=True)
    df["personId"] = espn_id
    df["era_id"]   = _era_id(season_year)
    return df


def fetch_player_live(
    player_name: str,
    season_year: int = 2024,
    n_games: int = 25,
) -> pd.DataFrame:
    """
    Fetch a player's recent games from ESPN and return a model-ready DataFrame.
    If the current season has fewer than n_games, pulls from the prior season too.

    Args:
        player_name:  Partial or full name.
        season_year:  NBA season start year (2024 = 2024-25).
        n_games:      How many recent games to return (need >= 20 for model).
    """
    espn_id, resolved_name = find_espn_athlete_id(player_name)
    print(f"  Found {resolved_name} (ESPN id={espn_id})")

    # Fetch current season
    data = _get(_GAMELOG.format(athlete_id=espn_id))
    df = _parse_gamelog(data, espn_id, season_year)

    # If not enough games, also pull previous season and prepend
    if len(df) < n_games and season_year > 2015:
        print(f"  Only {len(df)} games in {season_year}-{str(season_year+1)[-2:]}, fetching prior season...")
        prior_url = _GAMELOG.format(athlete_id=espn_id) + f"?season={season_year - 1}"
        try:
            prior_data = _get(prior_url)
            prior_df = _parse_gamelog(prior_data, espn_id, season_year - 1)
            if not prior_df.empty:
                df = pd.concat([prior_df, df], ignore_index=True).sort_values("gameDateTimeEst").reset_index(drop=True)
                print(f"  Combined: {len(df)} games total")
        except Exception as e:
            print(f"  Prior season fetch failed ({e}), continuing with {len(df)} games")

    if df.empty:
        raise ValueError(f"No parseable game rows for {resolved_name}")

    # rest_days
    df["rest_days"] = df["gameDateTimeEst"].diff().dt.days.fillna(2.0).clip(0, 14)

    # rolling 5-game averages
    for col in TARGET_ROLL:
        df[f"roll5_{col}"] = (
            df[col].rolling(5, min_periods=1).mean().shift(1).fillna(df[col].mean())
        )

    # game_pace proxy: player FGA * 2
    df["game_pace"] = df["fieldGoalsAttempted"] * 2.0

    # opp_pts_allowed_roll10: look up each opponent's season avg pts allowed.
    def_ratings = get_team_def_ratings()
    league_avg  = 109.0
    df["opp_pts_allowed_roll10"] = df["opp_abbr"].map(
        lambda abbr: def_ratings.get(abbr, league_avg)
    )

    # personId (use ESPN id as surrogate)
    df["personId"] = espn_id

    # era_id
    df["era_id"] = _era_id(season_year)

    # player_id_encoded — look up from existing NBA id_map if available.
    # ESPN ids differ from NBA personIds, so we try name-based fuzzy lookup first.
    df["player_id_encoded"] = 0
    id_map_path = DATA_CONFIG.player_id_map_path
    if id_map_path.exists():
        # Try to find the NBA personId from the nba_api static player list
        try:
            from nba_api.stats.static import players as nba_static
            matches = nba_static.find_players_by_full_name(player_name)
            if matches:
                nba_pid = str(matches[0]["id"])
                with open(id_map_path) as f:
                    id_map = json.load(f)
                enc = id_map.get(nba_pid)
                if enc is not None:
                    df["player_id_encoded"] = int(enc)
        except Exception:
            pass

    # ensure all INPUT_FEATURES present
    for col in INPUT_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.tail(n_games).reset_index(drop=True)
