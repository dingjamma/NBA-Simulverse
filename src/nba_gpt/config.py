from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Era definitions: (start_year_inclusive, name)
ERA_BOUNDARIES = [
    (1947, "Pre-Shot Clock"),        # 0
    (1955, "Shot Clock Era"),        # 1
    (1977, "Showtime / Bird-Magic"), # 2
    (1994, "Jordan Era"),            # 3
    (2003, "Post-Jordan"),           # 4
    (2013, "Three-Point Revolution"),# 5
]
N_ERAS = len(ERA_BOUNDARIES)

# Features included in each game vector (input to model)
INPUT_FEATURES = [
    "points", "reboundsTotal", "assists", "steals", "blocks",
    "threePointersMade", "numMinutes",
    "fieldGoalsAttempted", "fieldGoalsMade",
    "freeThrowsAttempted", "freeThrowsMade",
    "turnovers", "plusMinusPoints",
    "home", "rest_days",
    "opp_pts_allowed_roll10",  # opponent's rolling avg points allowed (defensive quality)
    "game_pace",               # pace of the game (possessions)
    # Rolling 5-game averages of target stats — explicit prior, same signal XGBoost uses
    "roll5_points", "roll5_reboundsTotal", "roll5_assists",
    "roll5_steals", "roll5_blocks", "roll5_threePointersMade",
]
N_INPUT_FEATURES = len(INPUT_FEATURES)

# Stats we predict (subset of INPUT_FEATURES)
TARGET_STATS = ["points", "reboundsTotal", "assists", "steals", "blocks", "threePointersMade"]
N_TARGETS = len(TARGET_STATS)

# Explicit next-game context features — separate input to the model.
# These are the conditions of game 21 (the one being predicted).
# In simulation, these get overridden directly to model counterfactuals.
CONTEXT_FEATURES = [
    "rest_days",               # days since last game
    "home",                    # home/away
    "opp_pts_allowed_roll10",  # opponent defensive quality
    "game_pace",               # game pace (possessions)
    "numMinutes",              # actual/projected minutes
]
N_CONTEXT_FEATURES = len(CONTEXT_FEATURES)


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    player_games_path: Path = PROCESSED_DIR / "player_games.parquet"
    player_features_path: Path = PROCESSED_DIR / "player_features.parquet"
    norm_stats_path: Path = PROCESSED_DIR / "norm_stats.json"
    player_id_map_path: Path = PROCESSED_DIR / "player_id_map.json"
    train_sequences_path: Path = PROCESSED_DIR / "train_sequences.npz"
    val_sequences_path: Path = PROCESSED_DIR / "val_sequences.npz"
    test_sequences_path: Path = PROCESSED_DIR / "test_sequences.npz"
    sequence_length: int = 20
    min_player_games: int = 25
    # Only include training sequences where the TARGET game is post-2002.
    # Pre-2003 basketball has very different pace/3pt distributions.
    # Input windows can still include pre-2003 games as context.
    min_target_year: int = 1951  # no filter — use full history
    # Season boundaries (ISO date strings)
    val_season_start: str = "2023-07-01"
    test_season_start: str = "2024-07-01"  # 2024-25 season holdout


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 5
    d_ff: int = 768
    dropout: float = 0.1
    max_players: int = 8000
    n_eras: int = N_ERAS
    n_input_features: int = N_INPUT_FEATURES
    n_context_features: int = N_CONTEXT_FEATURES
    n_targets: int = N_TARGETS
    sequence_length: int = 20


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 80
    warmup_steps: int = 500
    patience: int = 15
    checkpoint_dir: Path = CHECKPOINT_DIR
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"


DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
TRAIN_CONFIG = TrainConfig()
