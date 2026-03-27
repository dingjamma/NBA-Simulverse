"""
Preprocess raw PlayerStatistics.csv into a clean parquet file.
Filters to Regular Season, removes DNPs, sorts chronologically.
"""
import pandas as pd
from pathlib import Path

from nba_gpt.config import DATA_CONFIG

REGULAR_SEASON_TYPES = {"Regular Season"}


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Regular season only
    df = df[df["gameType"].isin(REGULAR_SEASON_TYPES)].copy()

    # Coerce numeric columns that may have mixed types
    df["numMinutes"] = pd.to_numeric(df["numMinutes"], errors="coerce")

    # Drop DNPs (< 1 minute played)
    df = df[df["numMinutes"] >= 1.0].copy()

    # Drop rows with null game dates
    df = df[df["gameDateTimeEst"].notna()].copy()

    # Fill three-point stats with 0 for pre-1979 games (no 3pt line)
    for col in ["threePointersAttempted", "threePointersMade", "threePointersPercentage"]:
        df[col] = df[col].fillna(0).astype(float)

    # Clip obviously bad values (negative minutes, etc.)
    df["numMinutes"] = df["numMinutes"].clip(lower=0, upper=60)
    df["plusMinusPoints"] = df["plusMinusPoints"].fillna(0)

    # Sort chronologically per player
    df = df.sort_values(["personId", "gameDateTimeEst"]).reset_index(drop=True)

    # Assign per-player game sequence number (1-indexed)
    df["player_game_number"] = df.groupby("personId").cumcount() + 1

    return df


def validate(df: pd.DataFrame) -> None:
    dupes = df.duplicated(subset=["personId", "gameId"])
    assert not dupes.any(), f"Found {dupes.sum()} duplicate (personId, gameId) pairs"

    required_cols = ["points", "reboundsTotal", "assists", "steals", "blocks",
                     "threePointersMade", "numMinutes", "fieldGoalsAttempted",
                     "fieldGoalsMade", "freeThrowsAttempted", "freeThrowsMade",
                     "turnovers", "plusMinusPoints", "home"]
    for col in required_cols:
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} nulls after cleaning"


def run(raw_path: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
    raw_path = raw_path or DATA_CONFIG.raw_dir / "PlayerStatistics.csv"
    output_path = output_path or DATA_CONFIG.player_games_path

    print(f"Loading {raw_path}...")
    df = load_raw(raw_path)
    print(f"  Loaded {len(df):,} rows")

    print("Cleaning...")
    df = clean(df)
    print(f"  After cleaning: {len(df):,} rows")
    print(f"  Date range: {df['gameDateTimeEst'].min().date()} to {df['gameDateTimeEst'].max().date()}")
    print(f"  Unique players: {df['personId'].nunique():,}")
    print(f"  Unique games: {df['gameId'].nunique():,}")

    print("Validating...")
    validate(df)
    print("  Validation passed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")

    return df


if __name__ == "__main__":
    run()
