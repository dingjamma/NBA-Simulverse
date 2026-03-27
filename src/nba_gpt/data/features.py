"""
Feature engineering: derive rest_days, era_id, player_id_encoded,
opponent defensive quality, and game pace.
Compute and save normalization statistics.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

from nba_gpt.config import DATA_CONFIG, ERA_BOUNDARIES, INPUT_FEATURES


def assign_era(year: int) -> int:
    """Map a year to an era index (0-5)."""
    era = 0
    for i, (start_year, _) in enumerate(ERA_BOUNDARIES):
        if year >= start_year:
            era = i
    return era


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days since last game for each player. Cap at 30."""
    df = df.copy()
    df = df.sort_values(["personId", "gameDateTimeEst"])

    prev_date = df.groupby("personId")["gameDateTimeEst"].shift(1)
    delta = (df["gameDateTimeEst"] - prev_date).dt.days

    # First career game gets neutral value (7 days)
    delta = delta.fillna(7).clip(upper=30).astype(float)
    df["rest_days"] = delta
    return df


def encode_player_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """Map personId to contiguous integer index."""
    unique_ids = sorted(df["personId"].unique())
    id_map = {pid: idx for idx, pid in enumerate(unique_ids)}
    df = df.copy()
    df["player_id_encoded"] = df["personId"].map(id_map)
    return df, id_map


def compute_opponent_features(
    player_df: pd.DataFrame,
    team_stats_path: Path,
) -> pd.DataFrame:
    """
    Join per-game opponent defensive quality and pace to player rows.

    opp_pts_allowed_roll10: opponent team's rolling 10-game avg points allowed
                            before this game. A low value = tough defense.
    game_pace:              total game pace (possessions) from team stats.
                            High pace = more possessions = more stat opportunities.
    """
    ts = pd.read_csv(team_stats_path, low_memory=False)
    ts["gameDateTimeEst"] = pd.to_datetime(ts["gameDateTimeEst"], errors="coerce")
    ts = ts.dropna(subset=["gameDateTimeEst", "gameId", "teamId"])

    # Compute rolling 10-game points allowed per team (sorted by date)
    ts = ts.sort_values(["teamId", "gameDateTimeEst"])
    # "points allowed" = opponentScore for this team in this game
    ts["pts_allowed_roll10"] = (
        ts.groupby("teamId")["opponentScore"]
        .transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
        .fillna(ts["opponentScore"].mean())  # fallback to league avg for early games
    )

    # Pace: avg of both teams' pace this game (total_minutes / 48 * possessions proxy)
    # We approximate pace as total points scored / 2 normalized, but better:
    # Use total field goals attempted as a possessions proxy
    # For each game, take the sum of FGA for both teams as pace proxy
    game_pace = (
        ts.groupby("gameId")["fieldGoalsAttempted"]
        .sum()
        .reset_index()
        .rename(columns={"fieldGoalsAttempted": "game_pace"})
    )
    # Normalize: league average ~180 FGA/game; scale to ~1.0
    game_pace["game_pace"] = game_pace["game_pace"].clip(lower=50, upper=300).astype(float)

    # Build lookup: (gameId, opponentTeamId) -> pts_allowed_roll10
    # The opponent of the player's team = the team the player faces
    # In player_df: opponentteamName identifies the opponent
    # In ts: teamCity + teamName identifies the team
    # Best join: gameId + the team that IS the opponent of the player's team
    # player_df has: gameId, playerteamCity, playerteamName, opponentteamCity, opponentteamName
    # ts has: gameId, teamCity, teamName, pts_allowed_roll10

    ts_lookup = ts[["gameId", "teamCity", "teamName", "pts_allowed_roll10"]].copy()
    ts_lookup.columns = ["gameId", "opp_teamCity", "opp_teamName", "opp_pts_allowed_roll10"]

    # Join on gameId + opponent team identity
    player_df = player_df.merge(
        ts_lookup,
        left_on=["gameId", "opponentteamCity", "opponentteamName"],
        right_on=["gameId", "opp_teamCity", "opp_teamName"],
        how="left",
    )

    # Join pace
    player_df = player_df.merge(game_pace, on="gameId", how="left")

    # Fill missing values with league averages
    league_avg_pts_allowed = float(ts["opponentScore"].mean())
    player_df["opp_pts_allowed_roll10"] = (
        player_df["opp_pts_allowed_roll10"].fillna(league_avg_pts_allowed)
    )
    league_avg_pace = float(game_pace["game_pace"].mean())
    player_df["game_pace"] = player_df["game_pace"].fillna(league_avg_pace)

    return player_df


def compute_norm_stats(df: pd.DataFrame, train_cutoff: str) -> dict[str, Any]:
    """Compute per-feature mean and std on training data only."""
    train_df = df[df["gameDateTimeEst"] < pd.Timestamp(train_cutoff)]
    stats: dict[str, Any] = {}
    for feat in INPUT_FEATURES:
        if feat not in train_df.columns:
            continue
        col = train_df[feat].astype(float)
        mean = float(col.mean())
        std = float(col.std())
        # Prevent division by zero for binary features like 'home'
        if std < 1e-6:
            std = 1.0
        stats[feat] = {"mean": mean, "std": std}
    return stats


def run(
    input_path: Path | None = None,
    output_path: Path | None = None,
    norm_stats_path: Path | None = None,
    id_map_path: Path | None = None,
) -> pd.DataFrame:
    input_path = input_path or DATA_CONFIG.player_games_path
    output_path = output_path or DATA_CONFIG.player_features_path
    norm_stats_path = norm_stats_path or DATA_CONFIG.norm_stats_path
    id_map_path = id_map_path or DATA_CONFIG.player_id_map_path
    team_stats_path = DATA_CONFIG.raw_dir / "TeamStatistics.csv"

    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} rows")

    print("Computing rest_days...")
    df = compute_rest_days(df)

    print("Assigning era_id...")
    df["era_id"] = df["gameDateTimeEst"].dt.year.apply(assign_era)

    print("Encoding player IDs...")
    df, id_map = encode_player_ids(df)
    print(f"  {len(id_map):,} unique players encoded")

    print("Joining opponent defensive features and pace...")
    df = compute_opponent_features(df, team_stats_path)
    print(f"  opp_pts_allowed_roll10 nulls: {df['opp_pts_allowed_roll10'].isnull().sum()}")
    print(f"  game_pace nulls: {df['game_pace'].isnull().sum()}")

    print("Computing rolling 5-game target averages...")
    target_stats = ["points", "reboundsTotal", "assists", "steals", "blocks", "threePointersMade"]
    df = df.sort_values(["personId", "gameDateTimeEst"])
    for stat in target_stats:
        df[f"roll5_{stat}"] = (
            df.groupby("personId")[stat]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            .fillna(df[stat].mean())
        )

    print("Computing normalization stats (training data only)...")
    norm_stats = compute_norm_stats(df, DATA_CONFIG.val_season_start)

    # Select columns needed for modeling
    keep_cols = [
        "personId", "player_id_encoded", "gameId", "gameDateTimeEst",
        "era_id", "player_game_number",
    ] + INPUT_FEATURES

    for col in INPUT_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    df_out = df[keep_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"  Saved features to {output_path}")

    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved norm stats to {norm_stats_path}")

    with open(id_map_path, "w") as f:
        json.dump({str(k): v for k, v in id_map.items()}, f)
    print(f"  Saved player ID map to {id_map_path}")

    return df_out


if __name__ == "__main__":
    run()
