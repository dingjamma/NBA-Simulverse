"""
Daily NBA picks pipeline.

1. Get today's NBA games from ESPN
2. Pull all player lines from Underdog Fantasy
3. Run NBA-GPT ensemble predictions for each player with a line
4. Rank by edge (model - line), pick top N
5. Log picks to logs/picks.csv

Usage:
  python scripts/16_daily_picks.py           # top 3 picks for tomorrow
  python scripts/16_daily_picks.py --top 5   # top 5
  python scripts/16_daily_picks.py --date 20260330  # specific date
"""
import argparse
import csv
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from live_feed.fetcher import (
    fetch_player_live, find_espn_athlete_id,
    _get, _GAMELOG, _parse_gamelog, INPUT_FEATURES,
)
from nba_gpt.config import MODEL_CONFIG, TARGET_STATS
from nba_gpt.data.dataset import load_norm_stats
from nba_gpt.simulation.engine import (
    _build_input_sequence, _build_context_tensor, ScenarioOverride,
)
from nba_gpt.simulation.ensemble import EnsemblePredictor

LOGS_DIR  = Path(__file__).parent.parent / "logs"
PICKS_CSV = LOGS_DIR / "picks.csv"

STAT_COL = {"Points": 0, "Rebounds": 1, "Assists": 2, "3-Pointers Made": 5}
STAT_MAP  = {"Points": "points", "Rebounds": "reboundsTotal",
             "Assists": "assists", "3-Pointers Made": "threePointersMade"}

ESPN_BASE  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
UD_API     = "https://api.underdogfantasy.com/v1/over_under_lines"
UD_HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

BLOWOUT_THRESHOLD = 12.0  # absolute spread points


def get_games(date_str: str) -> list[dict]:
    """Return list of {home, away, home_abbr, away_abbr, spread} for date_str (YYYYMMDD)."""
    r = requests.get(
        f"{ESPN_BASE}/scoreboard",
        params={"dates": date_str},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    r.raise_for_status()
    games = []
    for ev in r.json().get("events", []):
        comps = ev.get("competitions", [{}])[0]
        teams = comps.get("competitors", [])
        home      = next((t["team"]["displayName"]  for t in teams if t.get("homeAway") == "home"), "")
        away      = next((t["team"]["displayName"]  for t in teams if t.get("homeAway") == "away"), "")
        home_abbr = next((t["team"]["abbreviation"] for t in teams if t.get("homeAway") == "home"), "")
        away_abbr = next((t["team"]["abbreviation"] for t in teams if t.get("homeAway") == "away"), "")

        # Extract spread from ESPN odds if available
        spread = 0.0
        odds_list = comps.get("odds", [])
        if odds_list:
            raw = odds_list[0].get("spread", "0") or "0"
            try:
                spread = abs(float(str(raw).replace("±", "").strip()))
            except ValueError:
                spread = 0.0

        games.append({
            "home": home, "away": away,
            "home_abbr": home_abbr, "away_abbr": away_abbr,
            "spread": spread,
        })
    return games


def get_underdog_lines() -> pd.DataFrame:
    """Fetch all active NBA Underdog Fantasy lines. Returns DataFrame."""
    r = requests.get(UD_API, headers=UD_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()

    # Build lookup maps
    players     = {p["id"]: p for p in data.get("players", [])}
    appearances = {a["id"]: a for a in data.get("appearances", [])}
    games       = {g["id"]: g for g in data.get("games", [])}

    rows = []
    for line in data.get("over_under_lines", []):
        if line.get("status") != "active":
            continue

        ou        = line.get("over_under", {})
        app_stat  = ou.get("appearance_stat", {})
        stat      = app_stat.get("display_stat", "")
        if stat not in STAT_COL:
            continue

        line_val    = line.get("stat_value")
        if line_val is None:
            continue

        app_id  = app_stat.get("appearance_id", "")
        app     = appearances.get(app_id, {})
        player  = players.get(app.get("player_id", ""), {})

        if player.get("sport_id") != "NBA":
            continue

        first = player.get("first_name", "")
        last  = player.get("last_name", "")
        name  = f"{first} {last}".strip()
        if not name:
            continue

        # Determine home/away from game record
        match_id  = app.get("match_id")
        game      = games.get(match_id, {})
        team_id   = app.get("team_id", "")
        is_home   = game.get("home_team_id") == team_id
        game_abbr = game.get("abbreviated_title", "")  # e.g. "LAC @ MIL"

        rows.append({
            "name":      name,
            "team":      game_abbr,
            "is_home":   is_home,
            "stat":      stat,
            "line":      float(line_val),
            "match_id":  match_id,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep one row per player+stat (Underdog sometimes has alternate lines)
    return (
        df.groupby(["name", "stat"])
        .apply(lambda g: g.iloc[len(g) // 2])
        .reset_index(drop=True)
    )


def get_df_for_player(name: str) -> pd.DataFrame | None:
    """Fetch and sanitize a player's recent game log."""
    try:
        df = fetch_player_live(name, n_games=25)
        if len(df) < MODEL_CONFIG.sequence_length:
            espn_id, _ = find_espn_athlete_id(name)
            for yr in [2022, 2021, 2020]:
                prior_data = _get(_GAMELOG.format(athlete_id=espn_id) + f"?season={yr}")
                prior_df   = _parse_gamelog(prior_data, espn_id, yr)
                if not prior_df.empty:
                    df = pd.concat([prior_df, df], ignore_index=True)\
                           .sort_values("gameDateTimeEst").reset_index(drop=True)
                    if len(df) >= MODEL_CONFIG.sequence_length:
                        break
        if len(df) < MODEL_CONFIG.sequence_length:
            return None
        for col in INPUT_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df.tail(25).reset_index(drop=True)
    except Exception as e:
        print(f"    [{name}] fetch error: {e}")
        return None


# ---------------------------------------------------------------------------
# Model prediction
# ---------------------------------------------------------------------------

def run_prediction(
    df: pd.DataFrame,
    norm_stats: dict,
    predictor: EnsemblePredictor,
    is_home: bool,
    device: torch.device,
) -> np.ndarray:
    """Return (n_members, n_targets) raw stat samples."""
    override   = ScenarioOverride(home=is_home)
    input_seq, pid_enc, era_id = _build_input_sequence(df, norm_stats, MODEL_CONFIG.sequence_length)
    ctx        = _build_context_tensor(df, override, norm_stats)
    wm         = df.tail(MODEL_CONFIG.sequence_length)[TARGET_STATS].values.mean(0).astype("float32")
    pid_t      = torch.tensor([pid_enc], dtype=torch.long).to(device)
    era_t      = torch.tensor([era_id],  dtype=torch.long).to(device)
    return np.clip(np.stack([
        m(pid_t, era_t, input_seq.to(device), ctx.to(device)).float().detach().cpu().numpy()[0] + wm
        for m in predictor.models
    ]), 0, None)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

PICKS_COLS = [
    "date", "player", "stat", "line",
    "model_mean", "model_p25", "model_p75", "edge", "direction",
    "blowout",
    "result", "actual",   # filled in later by results script
]


def log_picks(picks: list[dict], game_date: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PICKS_CSV.exists()
    with open(PICKS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PICKS_COLS)
        if write_header:
            writer.writeheader()
        for pick in picks:
            writer.writerow({
                "date":       game_date,
                "player":     pick["player"],
                "stat":       pick["stat"],
                "line":       pick["line"],
                "model_mean": round(pick["model_mean"], 2),
                "model_p25":  round(pick["model_p25"],  2),
                "model_p75":  round(pick["model_p75"],  2),
                "edge":       round(pick["edge"], 2),
                "direction":  pick["direction"],
                "blowout":    pick.get("blowout", False),
                "result":     "",
                "actual":     "",
            })
    print(f"\nLogged {len(picks)} picks to {PICKS_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",  type=int, default=999, help="Max picks to log (default: unlimited)")
    parser.add_argument("--date", type=str, default=None,
                        help="Game date YYYYMMDD (default: today ET)")
    parser.add_argument("--min-edge", type=float, default=5.0,
                        help="Minimum |edge| to include a pick")
    args = parser.parse_args()

    if args.date:
        game_date = args.date
        display_date = f"{game_date[:4]}-{game_date[4:6]}-{game_date[6:]}"
    else:
        # ESPN buckets games by Eastern Time date. Use ET today so late-night
        # games (e.g. 10pm ET tipping into UTC tomorrow) stay in the right slate.
        ET = timezone(timedelta(hours=-4))  # EDT (UTC-4); adjust to -5 in winter
        today_et     = datetime.now(ET)
        game_date    = today_et.strftime("%Y%m%d")
        display_date = today_et.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"NBA Daily Picks — {display_date}")
    print(f"{'='*60}")

    # 1. Today's games + spreads
    print("\nFetching schedule...")
    games = get_games(game_date)
    if not games:
        print("No games found. Exiting.")
        return
    for g in games:
        flag = " [BLOWOUT RISK]" if g["spread"] >= BLOWOUT_THRESHOLD else ""
        print(f"  {g['away']} @ {g['home']}  spread={g['spread']:.1f}{flag}")

    # Build abbr -> spread lookup from ESPN games
    # key: frozenset of both team abbrs so we can match from either side
    spread_lookup: dict[frozenset, float] = {
        frozenset([g["home_abbr"], g["away_abbr"]]): g["spread"]
        for g in games
    }

    # 2. Underdog Fantasy lines — already filtered to NBA, already have is_home flag
    print("\nFetching Underdog Fantasy lines...")
    lines_df = get_underdog_lines()
    if lines_df.empty:
        print("No lines found. Exiting.")
        return
    print(f"  {len(lines_df)} lines ({lines_df['name'].nunique()} players)")

    # 3. Model setup
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_stats = load_norm_stats()
    predictor  = EnsemblePredictor(device=device)

    # 4. Score every player+stat line
    all_edges = []
    players_seen = {}

    for _, row in lines_df.iterrows():
        name = row["name"]
        stat = row["stat"]
        line = row["line"]

        # Get or fetch player df (cache per player)
        if name not in players_seen:
            print(f"  Fetching {name}...")
            players_seen[name] = get_df_for_player(name)

        df = players_seen[name]
        if df is None:
            continue

        # is_home already embedded by get_underdog_lines()
        is_home = bool(row.get("is_home", False))

        # Blowout risk: match game abbrs from team string e.g. "LAC @ MIL"
        game_str  = str(row.get("team", ""))
        abbrs     = [p.strip() for p in game_str.replace("@", " ").split() if p.strip()]
        spread    = spread_lookup.get(frozenset(abbrs), 0.0)
        blowout   = spread >= BLOWOUT_THRESHOLD

        try:
            samples = run_prediction(df, norm_stats, predictor, is_home, device)
            col     = samples[:, STAT_COL[stat]]
            mean    = float(col.mean())
            p25     = float(np.percentile(col, 25))
            p75     = float(np.percentile(col, 75))
            edge    = mean - line

            # Blowout games: always bet UNDER regardless of model direction
            direction = "UNDER" if blowout else ("OVER" if edge > 0 else "UNDER")

            if abs(edge) >= args.min_edge:
                all_edges.append({
                    "player":     name,
                    "stat":       stat,
                    "line":       line,
                    "model_mean": mean,
                    "model_p25":  p25,
                    "model_p75":  p75,
                    "edge":       edge,
                    "direction":  direction,
                    "blowout":    blowout,
                })
        except Exception as e:
            print(f"    [{name} {stat}] predict error: {e}")

    if not all_edges:
        print("\nNo edges found above threshold.")
        return

    # 5. Sort by absolute edge, take top N
    all_edges.sort(key=lambda x: abs(x["edge"]), reverse=True)
    top_picks = all_edges[:args.top]

    # 6. Print
    print(f"\n{'='*60}")
    print(f"TOP {args.top} PICKS — {display_date}")
    print(f"{'='*60}")
    print(f"  {'#':<3} {'Player':<24} {'Stat':<12} {'Line':>6}  {'Model':>7}  {'Edge':>7}  BET")
    print(f"  {'-'*68}")
    for i, p in enumerate(top_picks, 1):
        blowout_flag = " [BLOWOUT]" if p.get("blowout") else ""
        print(f"  {i:<3} {p['player']:<24} {p['stat']:<12} {p['line']:>6.1f}  "
              f"{p['model_mean']:>7.1f}  {p['edge']:>+7.1f}  {p['direction']}{blowout_flag}")

    print(f"\n  Parlay all {args.top}: {' + '.join(p['player'].split()[0] + ' ' + p['stat'] + ' ' + p['direction'] for p in top_picks)}")

    # 7. Log
    log_picks(top_picks, display_date)


if __name__ == "__main__":
    main()
