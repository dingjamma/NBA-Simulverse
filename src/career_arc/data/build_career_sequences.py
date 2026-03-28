"""
Aggregate game-level NBA data into player-season sequences for career arc modeling.

Pipeline:
  1. Load PlayerStatistics.csv and Players.csv from data/raw/
  2. Filter to Regular Season games only
  3. Group by (personId, season) to compute per-season averages/totals
  4. Assign era_id and team_id
  5. For each player with >= 3 seasons, build sliding windows of 5 seasons
  6. Compute auxiliary labels: breakout, decline, injury risk
  7. Save to data/processed/career_sequences.npz + career_norm_stats.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from career_arc.config import (
    RAW_DIR, PROCESSED_DIR,
    CAREER_STAT_FEATURES, CAREER_CONTEXT_FEATURES, TARGET_STATS,
    CAREER_ERA_BOUNDARIES, N_CAREER_ERAS,
    CAREER_MODEL_CONFIG,
)

# Minimum seasons for a player to be included (need at least seq_len + 1 for at least 1 window)
MIN_PLAYER_SEASONS = 3
# Minimum games in a season for it to count as a valid season row
MIN_GAMES_IN_SEASON = 10

REGULAR_SEASON_LABELS = {"Regular Season"}

# Output paths
CAREER_SEQUENCES_PATH = PROCESSED_DIR / "career_sequences.npz"
CAREER_NORM_STATS_PATH = PROCESSED_DIR / "career_norm_stats.json"


def _assign_era(year: int) -> int:
    """Return era_id for the given season start year."""
    era_id = 0
    for i, (start_year, _label) in enumerate(CAREER_ERA_BOUNDARIES):
        if year >= start_year:
            era_id = i
    return era_id


def _extract_season_year(date_str: str) -> int:
    """
    Extract the NBA season start year from a game date.
    NBA seasons starting in autumn belong to the year they start in.
    Dates from Jan-Jun belong to the previous year's season.
    """
    ts = pd.Timestamp(date_str)
    if ts.month >= 7:
        return ts.year
    return ts.year - 1


def _build_team_id_map(df: pd.DataFrame) -> dict[str, int]:
    """Build a deterministic team-name -> int mapping. Index 0 reserved for unknown."""
    teams = sorted(df["playerteamName"].dropna().unique())
    return {team: idx + 1 for idx, team in enumerate(teams)}


def _compute_per_season(
    player_games: pd.DataFrame,
    team_id_map: dict[str, int],
    players_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate a single player's game rows into per-season rows.

    Returns a DataFrame with one row per season, sorted chronologically.
    """
    rows = []
    for season_year, season_games in player_games.groupby("season_year"):
        n_games = len(season_games)
        if n_games < MIN_GAMES_IN_SEASON:
            continue

        fg_made = season_games["fieldGoalsMade"].sum()
        fg_att = season_games["fieldGoalsAttempted"].sum()
        fg3_made = season_games["threePointersMade"].sum()
        fg3_att = season_games["threePointersAttempted"].sum()
        ft_made = season_games["freeThrowsMade"].sum()
        ft_att = season_games["freeThrowsAttempted"].sum()

        row = {
            "season_year": int(season_year),
            "pts_per_game": float(season_games["points"].mean()),
            "reb_per_game": float(season_games["reboundsTotal"].mean()),
            "ast_per_game": float(season_games["assists"].mean()),
            "stl_per_game": float(season_games["steals"].mean()),
            "blk_per_game": float(season_games["blocks"].mean()),
            "fg_pct": float(fg_made / fg_att) if fg_att > 0 else 0.0,
            "fg3_pct": float(fg3_made / fg3_att) if fg3_att > 0 else 0.0,
            "ft_pct": float(ft_made / ft_att) if ft_att > 0 else 0.0,
            "minutes_per_game": float(season_games["numMinutes"].mean()),
            "games_played": float(n_games),
            "era_id": int(_assign_era(int(season_year))),
        }

        # Team id: use the team with most games that season
        most_common_team = season_games["playerteamName"].value_counts().idxmax()
        row["team_id"] = int(team_id_map.get(most_common_team, 0))

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    season_df = pd.DataFrame(rows).sort_values("season_year").reset_index(drop=True)

    # Derive age and years_in_league
    pid = player_games["personId"].iloc[0]
    meta = players_meta[players_meta["personId"] == pid]

    birth_year: float | None = None
    if not meta.empty and pd.notna(meta["birthDate"].iloc[0]):
        birth_ts = pd.Timestamp(meta["birthDate"].iloc[0])
        birth_year = float(birth_ts.year)

    first_season = float(season_df["season_year"].min())

    draft_round = 0.0
    draft_pick = 0.0
    if not meta.empty:
        dr = meta["draftRound"].iloc[0]
        dp = meta["draftNumber"].iloc[0]
        draft_round = float(dr) if pd.notna(dr) else 0.0
        draft_pick = float(dp) if pd.notna(dp) else 0.0

    ages = []
    yils = []
    for sy in season_df["season_year"]:
        yil = float(sy) - first_season
        if birth_year is not None:
            age = float(sy) - birth_year + 0.5  # mid-season approximation
        else:
            # Fallback: assume 19 years old as a rookie
            age = 19.0 + yil
        ages.append(age)
        yils.append(yil)

    season_df["age"] = ages
    season_df["years_in_league"] = yils
    season_df["draft_round"] = draft_round
    season_df["draft_pick"] = draft_pick

    return season_df


def _compute_labels(
    seasons: pd.DataFrame,
    target_idx: int,
) -> tuple[int, int, int]:
    """
    Compute auxiliary labels for the target season (index target_idx).

    breakout: pts_per_game increases > 20% vs previous season
    decline:  pts_per_game decreases > 15% vs previous season
    injury:   games_played < 50 in target season
    """
    if target_idx == 0:
        return 0, 0, int(seasons.iloc[0]["games_played"] < 50)

    prev_pts = float(seasons.iloc[target_idx - 1]["pts_per_game"])
    curr_pts = float(seasons.iloc[target_idx]["pts_per_game"])
    curr_games = float(seasons.iloc[target_idx]["games_played"])

    if prev_pts > 0:
        pct_change = (curr_pts - prev_pts) / prev_pts
        breakout = int(pct_change > 0.20)
        decline = int(pct_change < -0.15)
    else:
        breakout = 0
        decline = 0

    injury = int(curr_games < 50)
    return breakout, decline, injury


def build_career_sequences(
    raw_dir: Path = RAW_DIR,
    output_dir: Path = PROCESSED_DIR,
    seq_len: int = CAREER_MODEL_CONFIG.seq_len,
) -> dict[str, Any]:
    """
    Build sliding window sequences from raw game data.

    Returns a summary dict with counts and output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw game data
    stats_path = raw_dir / "PlayerStatistics.csv"
    players_path = raw_dir / "Players.csv"

    print(f"Loading {stats_path}...")
    games_df = pd.read_csv(stats_path, low_memory=False)
    print(f"  Loaded {len(games_df):,} rows")

    print(f"Loading {players_path}...")
    players_df = pd.read_csv(players_path, low_memory=False)
    print(f"  Loaded {len(players_df):,} players")

    # Filter to regular season only
    games_df = games_df[games_df["gameType"].isin(REGULAR_SEASON_LABELS)].copy()
    print(f"  After regular-season filter: {len(games_df):,} rows")

    # Parse dates, drop rows without valid dates
    games_df["gameDateTimeEst"] = pd.to_datetime(games_df["gameDateTimeEst"], errors="coerce")
    games_df = games_df.dropna(subset=["gameDateTimeEst"])

    # Assign season year
    games_df["season_year"] = games_df["gameDateTimeEst"].apply(
        lambda ts: ts.year if ts.month >= 7 else ts.year - 1
    )

    # Ensure numeric personId
    games_df["personId"] = pd.to_numeric(games_df["personId"], errors="coerce")
    games_df = games_df.dropna(subset=["personId"])
    games_df["personId"] = games_df["personId"].astype(int)

    players_df["personId"] = pd.to_numeric(players_df["personId"], errors="coerce")
    players_df = players_df.dropna(subset=["personId"])
    players_df["personId"] = players_df["personId"].astype(int)

    # Fill numeric game-level columns
    numeric_cols = [
        "points", "reboundsTotal", "assists", "steals", "blocks",
        "fieldGoalsAttempted", "fieldGoalsMade",
        "threePointersAttempted", "threePointersMade",
        "freeThrowsAttempted", "freeThrowsMade",
        "numMinutes",
    ]
    for col in numeric_cols:
        if col in games_df.columns:
            games_df[col] = pd.to_numeric(games_df[col], errors="coerce").fillna(0.0)

    # Build team id map
    team_id_map = _build_team_id_map(games_df)
    n_teams = len(team_id_map) + 1  # +1 for the 0=unknown slot
    print(f"  Teams found: {len(team_id_map)} (max team_id={max(team_id_map.values())})")

    # Build per-player season records
    print("Aggregating game data into player-seasons...")
    n_players_total = games_df["personId"].nunique()
    player_season_records: dict[int, pd.DataFrame] = {}

    for i, (pid, group) in enumerate(games_df.groupby("personId")):
        season_df = _compute_per_season(group, team_id_map, players_df)
        if len(season_df) >= MIN_PLAYER_SEASONS:
            player_season_records[int(pid)] = season_df

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n_players_total} players...")

    print(f"  Players with >= {MIN_PLAYER_SEASONS} seasons: {len(player_season_records):,}")

    # Build sliding window sequences
    print(f"Building sliding windows (seq_len={seq_len})...")
    stat_seqs_list = []
    ctx_seqs_list = []
    team_ids_list = []
    era_ids_list = []
    targets_list = []
    breakout_list = []
    decline_list = []
    injury_list = []
    player_ids_list = []

    for pid, season_df in player_season_records.items():
        n_seasons = len(season_df)
        # Need seq_len history + 1 target = seq_len + 1 minimum
        if n_seasons < seq_len + 1:
            continue

        stat_vals = season_df[CAREER_STAT_FEATURES].values.astype(np.float32)   # (n_s, 10)
        ctx_vals = season_df[CAREER_CONTEXT_FEATURES].values.astype(np.float32) # (n_s, 4)
        team_vals = season_df["team_id"].values.astype(np.int64)                 # (n_s,)
        era_vals = season_df["era_id"].values.astype(np.int64)                   # (n_s,)

        for i in range(seq_len, n_seasons):
            # Window: seasons [i-seq_len .. i-1] as input, season i as target
            stat_seqs_list.append(stat_vals[i - seq_len : i])     # (seq_len, 10)
            ctx_seqs_list.append(ctx_vals[i - seq_len : i])       # (seq_len, 4)
            team_ids_list.append(team_vals[i - seq_len : i])      # (seq_len,)
            era_ids_list.append(era_vals[i - seq_len : i])        # (seq_len,)
            targets_list.append(stat_vals[i])                      # (10,)

            bo, dec, inj = _compute_labels(season_df, i)
            breakout_list.append(bo)
            decline_list.append(dec)
            injury_list.append(inj)
            player_ids_list.append(pid)

    n = len(targets_list)
    print(f"  Total sequences: {n:,}")

    if n == 0:
        print("WARNING: No sequences generated. Check that data/raw/ files are present.")
        return {"n_sequences": 0}

    stat_seqs = np.stack(stat_seqs_list).astype(np.float32)    # (N, 5, 10)
    ctx_seqs = np.stack(ctx_seqs_list).astype(np.float32)      # (N, 5, 4)
    team_ids = np.stack(team_ids_list).astype(np.int64)         # (N, 5)
    era_ids = np.stack(era_ids_list).astype(np.int64)           # (N, 5)
    targets = np.stack(targets_list).astype(np.float32)         # (N, 10)
    breakout_labels = np.array(breakout_list, dtype=np.float32) # (N,)
    decline_labels = np.array(decline_list, dtype=np.float32)   # (N,)
    injury_labels = np.array(injury_list, dtype=np.float32)     # (N,)
    player_ids = np.array(player_ids_list, dtype=np.int64)      # (N,)

    # Clip team_ids to valid range (handle edge cases)
    max_team_id = CAREER_MODEL_CONFIG.n_teams - 1
    team_ids = np.clip(team_ids, 0, max_team_id)

    # Compute normalization stats over the full sequence stat data (flattened)
    print("Computing normalization statistics...")
    flat_stats = stat_seqs.reshape(-1, len(CAREER_STAT_FEATURES))  # (N*seq_len, 10)
    flat_ctx = ctx_seqs.reshape(-1, len(CAREER_CONTEXT_FEATURES))   # (N*seq_len, 4)

    norm_stats: dict[str, dict[str, float]] = {}
    for j, feat in enumerate(CAREER_STAT_FEATURES):
        col = flat_stats[:, j]
        norm_stats[feat] = {
            "mean": float(col.mean()),
            "std": float(col.std()) if col.std() > 1e-8 else 1.0,
        }
    for j, feat in enumerate(CAREER_CONTEXT_FEATURES):
        col = flat_ctx[:, j]
        norm_stats[feat] = {
            "mean": float(col.mean()),
            "std": float(col.std()) if col.std() > 1e-8 else 1.0,
        }

    # Save normalization stats
    norm_path = output_dir / "career_norm_stats.json"
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved norm stats to {norm_path}")

    # Save team_id_map for use during simulation
    team_map_path = output_dir / "career_team_id_map.json"
    with open(team_map_path, "w") as f:
        json.dump(team_id_map, f, indent=2)
    print(f"  Saved team id map to {team_map_path}")

    # Save sequences
    seq_path = output_dir / "career_sequences.npz"
    np.savez_compressed(
        seq_path,
        stat_seqs=stat_seqs,
        ctx_seqs=ctx_seqs,
        team_ids=team_ids,
        era_ids=era_ids,
        targets=targets,
        breakout_labels=breakout_labels,
        decline_labels=decline_labels,
        injury_labels=injury_labels,
        player_ids=player_ids,
    )
    size_mb = seq_path.stat().st_size / 1e6
    print(f"  Saved {n:,} sequences to {seq_path} ({size_mb:.1f} MB)")

    # Summary statistics for the console
    print("\nDataset summary:")
    for feat in CAREER_STAT_FEATURES:
        print(
            f"  {feat:<18}  mean={norm_stats[feat]['mean']:7.3f}  "
            f"std={norm_stats[feat]['std']:7.3f}"
        )
    print(f"\n  Breakout rate:    {breakout_labels.mean():.3f}")
    print(f"  Decline rate:     {decline_labels.mean():.3f}")
    print(f"  Injury rate:      {injury_labels.mean():.3f}")

    return {
        "n_sequences": n,
        "n_players": len(player_season_records),
        "sequences_path": str(seq_path),
        "norm_stats_path": str(norm_path),
        "team_map_path": str(team_map_path),
    }


if __name__ == "__main__":
    build_career_sequences()
