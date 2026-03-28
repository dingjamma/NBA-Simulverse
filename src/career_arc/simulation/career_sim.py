"""
Career arc simulation engine.

Given a player's last 5 recorded seasons, auto-regressively predicts
N future seasons and returns per-season stats plus breakout/decline/injury probs.
"""
import json
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from career_arc.config import (
    RAW_DIR, PROCESSED_DIR,
    CAREER_STAT_FEATURES, CAREER_CONTEXT_FEATURES, TARGET_STATS,
    CAREER_MODEL_CONFIG, CAREER_TRAIN_CONFIG,
    CAREER_ERA_BOUNDARIES,
)
from career_arc.model import CareerArcModel


CAREER_SEQUENCES_PATH = PROCESSED_DIR / "career_sequences.npz"
CAREER_NORM_STATS_PATH = PROCESSED_DIR / "career_norm_stats.json"
CAREER_TEAM_MAP_PATH = PROCESSED_DIR / "career_team_id_map.json"


@dataclass
class SeasonProjection:
    season_offset: int       # 1 = next season, 2 = two seasons out, etc.
    stats: dict[str, float]  # predicted stat values
    breakout_prob: float
    decline_prob: float
    injury_prob: float


@dataclass
class CareerSimResult:
    player_name: str
    n_seasons: int
    projections: list[SeasonProjection] = field(default_factory=list)

    def summary(self) -> str:
        stat_cols = CAREER_STAT_FEATURES
        header_stats = " ".join(f"{s[:7]:>8}" for s in stat_cols)
        lines = [
            f"\n{'='*100}",
            f"Career Projection: {self.player_name}",
            f"{'='*100}",
            f"{'Season':>8}  {header_stats}  {'Breakout':>9} {'Decline':>8} {'Injury':>7}",
            f"{'-'*100}",
        ]
        for proj in self.projections:
            label = f"N+{proj.season_offset}"
            stat_vals = " ".join(f"{proj.stats.get(s, 0.0):>8.2f}" for s in stat_cols)
            lines.append(
                f"{label:>8}  {stat_vals}  "
                f"{proj.breakout_prob:>9.3f} "
                f"{proj.decline_prob:>8.3f} "
                f"{proj.injury_prob:>7.3f}"
            )
        lines.append("=" * 100)
        return "\n".join(lines)


def _load_model(checkpoint_path: Path, device: torch.device) -> CareerArcModel:
    model = CareerArcModel(CAREER_MODEL_CONFIG).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_norm_stats(path: Path | None = None) -> dict[str, Any]:
    path = path or CAREER_NORM_STATS_PATH
    with open(path) as f:
        return json.load(f)


def _assign_era(year: int) -> int:
    """Return era_id for the given season start year."""
    era_id = 0
    for i, (start_year, _label) in enumerate(CAREER_ERA_BOUNDARIES):
        if year >= start_year:
            era_id = i
    return era_id


def _find_player_sequences(
    player_name: str,
    sequences_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Locate the last seq_len seasons for a player by name.

    Searches Players.csv for a name match, then finds the most recent
    5-season window from career_sequences.npz.

    Returns:
        stat_window:  (seq_len, 10) float32
        ctx_window:   (seq_len, 4)  float32
        team_window:  (seq_len,)    int64
        era_window:   (seq_len,)    int64
        person_id:    int  (not tier_id — removed)
    """
    raw_path = RAW_DIR / "Players.csv"
    stats_path = RAW_DIR / "PlayerStatistics.csv"
    players_df = pd.read_csv(raw_path, low_memory=False)
    players_df["personId"] = pd.to_numeric(players_df["personId"], errors="coerce")
    players_df = players_df.dropna(subset=["personId"])
    players_df["personId"] = players_df["personId"].astype(int)
    players_df["fullName"] = (
        players_df["firstName"].fillna("") + " " + players_df["lastName"].fillna("")
    ).str.strip()

    matches = players_df[
        players_df["fullName"].str.contains(player_name, case=False, na=False)
    ].drop_duplicates("personId")

    if matches.empty:
        raise ValueError(f"No player found matching '{player_name}'")

    if len(matches) > 1:
        # Pick the player with the most rows in the sequences file
        data = np.load(sequences_path)
        player_ids_arr = data["player_ids"]
        counts = {
            int(row["personId"]): int((player_ids_arr == int(row["personId"])).sum())
            for _, row in matches.iterrows()
        }
        best_id = max(counts, key=counts.get)
        matches = matches[matches["personId"] == best_id]

    person_id = int(matches.iloc[0]["personId"])

    # Load the sequences file and find this player's most recent window
    data = np.load(sequences_path)
    player_ids_arr = data["player_ids"]
    mask = player_ids_arr == person_id

    if not mask.any():
        raise ValueError(
            f"Player '{player_name}' (id={person_id}) has no sequences in "
            f"{sequences_path}. Run script 10 first."
        )

    # Use the last matching sequence (most recent 5 seasons)
    last_idx = np.where(mask)[0][-1]

    stat_window = data["stat_seqs"][last_idx]    # (seq_len, 10)
    ctx_window = data["ctx_seqs"][last_idx]      # (seq_len, 4)
    team_window = data["team_ids"][last_idx]     # (seq_len,)
    era_window = data["era_ids"][last_idx]       # (seq_len,)

    return stat_window, ctx_window, team_window, era_window, person_id


def _normalize_stat_window(
    raw_window: np.ndarray,
    norm_stats: dict[str, Any],
) -> np.ndarray:
    """Z-score normalize a (seq_len, 10) stat window."""
    means = np.array([norm_stats[f]["mean"] for f in CAREER_STAT_FEATURES], dtype=np.float32)
    stds = np.array([norm_stats[f]["std"] for f in CAREER_STAT_FEATURES], dtype=np.float32)
    return (raw_window - means) / stds


def _denormalize_stats(
    norm_pred: np.ndarray,
    norm_stats: dict[str, Any],
) -> np.ndarray:
    """Reverse z-score normalization on (10,) or (N, 10) array."""
    means = np.array([norm_stats[f]["mean"] for f in CAREER_STAT_FEATURES], dtype=np.float32)
    stds = np.array([norm_stats[f]["std"] for f in CAREER_STAT_FEATURES], dtype=np.float32)
    return norm_pred * stds + means


def simulate_career(
    player_name: str,
    n_seasons: int = 5,
    scenario: dict | None = None,
    checkpoint_path: Path | None = None,
    sequences_path: Path | None = None,
) -> CareerSimResult:
    """
    Simulate a player's career trajectory autoregressively.

    Args:
        player_name:     Partial or full player name (case-insensitive match).
        n_seasons:       Number of future seasons to project.
        scenario:        Optional dict with override values for context features.
                         Keys: age_offset (float), team_id (int).
                         If None, context is extrapolated naturally.
        checkpoint_path: Path to CareerArcModel checkpoint. Defaults to best.pt.
        sequences_path:  Path to career_sequences.npz. Defaults to processed dir.

    Returns:
        CareerSimResult with per-season projections.
    """
    checkpoint_path = checkpoint_path or (CAREER_TRAIN_CONFIG.checkpoint_dir / "best.pt")
    sequences_path = sequences_path or CAREER_SEQUENCES_PATH

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Run scripts/11_train_career.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    norm_stats = _load_norm_stats()

    # Find player's most recent 5-season window (raw, unnormalized stat values)
    raw_stat_window, raw_ctx_window, team_window, era_window, person_id = (
        _find_player_sequences(player_name, sequences_path)
    )

    # Resolve player name for output
    players_df = pd.read_csv(RAW_DIR / "Players.csv", low_memory=False)
    players_df["personId"] = pd.to_numeric(players_df["personId"], errors="coerce")
    players_df = players_df.dropna(subset=["personId"])
    players_df["personId"] = players_df["personId"].astype(int)
    players_df["fullName"] = (
        players_df["firstName"].fillna("") + " " + players_df["lastName"].fillna("")
    ).str.strip()
    name_row = players_df[players_df["personId"] == person_id]
    resolved_name = name_row["fullName"].iloc[0] if not name_row.empty else player_name

    # Normalize the initial window for model input
    norm_stat_window = _normalize_stat_window(raw_stat_window, norm_stats)  # (5, 10)

    ctx_means = np.array(
        [norm_stats[f]["mean"] for f in CAREER_CONTEXT_FEATURES], dtype=np.float32
    )
    ctx_stds = np.array(
        [norm_stats[f]["std"] for f in CAREER_CONTEXT_FEATURES], dtype=np.float32
    )
    norm_ctx_window = (raw_ctx_window - ctx_means) / ctx_stds  # (5, 4)

    # Autoregressive simulation
    projections: list[SeasonProjection] = []

    # Working copies (normalized) that grow autoregressively
    working_stat = norm_stat_window.copy()   # (5, 10)
    working_ctx = norm_ctx_window.copy()     # (5, 4)
    working_teams = team_window.copy()       # (5,)
    working_eras = era_window.copy()         # (5,)

    with torch.no_grad():
        for step in range(1, n_seasons + 1):
            stat_t = torch.tensor(working_stat, dtype=torch.float32).unsqueeze(0).to(device)
            ctx_t = torch.tensor(working_ctx, dtype=torch.float32).unsqueeze(0).to(device)
            team_t = torch.tensor(working_teams, dtype=torch.long).unsqueeze(0).to(device)
            era_t = torch.tensor(working_eras, dtype=torch.long).unsqueeze(0).to(device)

            stat_pred, breakout_prob, decline_prob, injury_prob = model(
                stat_t, ctx_t, team_t, era_t
            )

            # Denormalize the stat prediction
            pred_norm = stat_pred[0].float().cpu().numpy()      # (10,)
            pred_raw = np.clip(_denormalize_stats(pred_norm, norm_stats), 0.0, None)

            # Build projection for this step
            proj_stats = {feat: float(pred_raw[j]) for j, feat in enumerate(CAREER_STAT_FEATURES)}
            projections.append(SeasonProjection(
                season_offset=step,
                stats=proj_stats,
                breakout_prob=float(breakout_prob[0].cpu()),
                decline_prob=float(decline_prob[0].cpu()),
                injury_prob=float(injury_prob[0].cpu()),
            ))

            # Slide the window: drop oldest season, append predicted season
            new_norm_stat = pred_norm.reshape(1, -1)           # (1, 10)

            # Extrapolate context
            # CAREER_CONTEXT_FEATURES = [age, years_in_league, draft_round, draft_pick]
            last_ctx_raw = raw_ctx_window[-1].copy()           # (4,) raw context
            last_ctx_raw[0] += 1.0          # age += 1
            last_ctx_raw[1] += float(step)  # years_in_league += step
            if scenario and "age_offset" in scenario:
                last_ctx_raw[0] += float(scenario["age_offset"])
            # draft_round and draft_pick stay constant

            new_norm_ctx = ((last_ctx_raw - ctx_means) / ctx_stds).reshape(1, -1)

            # Era: derive from last era or keep same
            new_era = np.array([int(working_eras[-1])], dtype=np.int64)

            # Team: keep last team unless overridden
            new_team = working_teams[-1:]
            if scenario and "team_id" in scenario:
                new_team = np.array(
                    [min(int(scenario["team_id"]), CAREER_MODEL_CONFIG.n_teams - 1)],
                    dtype=np.int64,
                )

            working_stat = np.concatenate([working_stat[1:], new_norm_stat], axis=0)
            working_ctx = np.concatenate([working_ctx[1:], new_norm_ctx], axis=0)
            working_teams = np.concatenate([working_teams[1:], new_team], axis=0)
            working_eras = np.concatenate([working_eras[1:], new_era], axis=0)

    return CareerSimResult(
        player_name=resolved_name,
        n_seasons=n_seasons,
        projections=projections,
    )
