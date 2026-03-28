from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# Era definitions matching nba_gpt config but using career-arc semantics.
# Each tuple: (start_year_inclusive, label, era_id)
CAREER_ERA_BOUNDARIES = [
    (1947, "Pre-Shot Clock"),       # 0  pre-1955
    (1955, "Russell Era"),          # 1  1955-1969
    (1970, "ABA/Expansion"),        # 2  1970-1979
    (1980, "Magic/Bird"),           # 3  1980-1991
    (1992, "Jordan Era"),           # 4  1992-2003
    (2004, "Three-Point Revolution"), # 5  2004+
]
N_CAREER_ERAS = len(CAREER_ERA_BOUNDARIES)

CAREER_STAT_FEATURES = [
    "pts_per_game",
    "reb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "minutes_per_game",
    "games_played",
]
CAREER_CONTEXT_FEATURES = ["age", "years_in_league", "draft_round", "draft_pick"]

N_CAREER_STATS = len(CAREER_STAT_FEATURES)
N_CAREER_CONTEXT = len(CAREER_CONTEXT_FEATURES)

# Predict all 10 stat features
TARGET_STATS = CAREER_STAT_FEATURES


@dataclass(frozen=True)
class CareerModelConfig:
    n_stat_features: int = N_CAREER_STATS
    n_context_features: int = N_CAREER_CONTEXT
    n_teams: int = 45          # NBA teams + expansions + free agent slot (data has 39, padded)
    n_eras: int = N_CAREER_ERAS
    team_embed_dim: int = 16
    era_embed_dim: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    seq_len: int = 5           # 5 seasons of history
    n_targets: int = N_CAREER_STATS


@dataclass(frozen=True)
class CareerTrainConfig:
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 100
    patience: int = 15
    seed: int = 42
    checkpoint_dir: Path = CHECKPOINT_DIR / "career"
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    regression_weight: float = 1.0
    aux_weight: float = 0.3    # weight for auxiliary classification heads
    device: str = "cuda"
    num_workers: int = 4


CAREER_MODEL_CONFIG = CareerModelConfig()
CAREER_TRAIN_CONFIG = CareerTrainConfig()
