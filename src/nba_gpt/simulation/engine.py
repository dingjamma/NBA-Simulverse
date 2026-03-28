"""
Simulverse counterfactual engine.

Given a player's recent game history, simulate their next game under
hypothetical conditions (different minutes, matchup, rest, home/away).

Uncertainty backend (auto-selected):
  - Ensemble (preferred): load N independently trained models, spread = real uncertainty.
  - MC Dropout (fallback): N forward passes with dropout active, used before ensemble exists.
"""
import json
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nba_gpt.config import (
    DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG,
    INPUT_FEATURES, TARGET_STATS, CONTEXT_FEATURES,
)
from nba_gpt.data.dataset import load_norm_stats
from nba_gpt.model.transformer import NBAGPTModel


@dataclass
class ScenarioOverride:
    """
    Hypothetical conditions to inject for the next game.
    Any field left as None keeps the player's most recent value.
    """
    minutes: float | None = None          # projected minutes (numMinutes)
    home: bool | None = None              # True = home game
    rest_days: float | None = None        # days since last game
    opp_pts_allowed: float | None = None  # opponent's defensive quality (lower = tougher)
    game_pace: float | None = None        # game pace (total FGA proxy)


@dataclass
class SimulationResult:
    player_name: str
    scenario: ScenarioOverride
    n_samples: int
    # Per-stat distributions
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    p10: dict[str, float] = field(default_factory=dict)
    p25: dict[str, float] = field(default_factory=dict)
    p75: dict[str, float] = field(default_factory=dict)
    p90: dict[str, float] = field(default_factory=dict)
    # Raw samples for downstream analysis
    samples: np.ndarray | None = None    # (n_samples, n_targets)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Player: {self.player_name}",
            f"Scenario: {self.scenario}",
            f"Samples:  {self.n_samples}",
            f"{'='*60}",
            f"{'Stat':<20} {'Mean':>7} {'Std':>6} {'P10':>6} {'P25':>6} {'P75':>6} {'P90':>6}",
            f"{'-'*60}",
        ]
        for stat in TARGET_STATS:
            lines.append(
                f"{stat:<20} "
                f"{self.mean[stat]:>7.1f} "
                f"{self.std[stat]:>6.2f} "
                f"{self.p10[stat]:>6.1f} "
                f"{self.p25[stat]:>6.1f} "
                f"{self.p75[stat]:>6.1f} "
                f"{self.p90[stat]:>6.1f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def _enable_dropout(model: torch.nn.Module) -> None:
    """Set all dropout layers to train mode for MC sampling."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def _load_model(checkpoint_path: Path, device: torch.device) -> NBAGPTModel:
    model = NBAGPTModel(MODEL_CONFIG).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def _build_input_sequence(
    player_df: pd.DataFrame,
    norm_stats: dict[str, Any],
    seq_len: int = 20,
) -> tuple[torch.Tensor, int, int]:
    """
    Extract the player's last seq_len games as a normalized input tensor.
    Returns (input_tensor, player_id_encoded, era_id_of_last_game).
    """
    player_df = player_df.sort_values("gameDateTimeEst").tail(seq_len)
    if len(player_df) < seq_len:
        raise ValueError(
            f"Player only has {len(player_df)} games in features, need {seq_len}."
        )

    input_vals = player_df[INPUT_FEATURES].values.astype(np.float32)  # (seq_len, n_feat)

    # Normalize
    means = np.array([norm_stats[f]["mean"] for f in INPUT_FEATURES], dtype=np.float32)
    stds = np.array([norm_stats[f]["std"] for f in INPUT_FEATURES], dtype=np.float32)
    input_norm = (input_vals - means) / stds

    player_id_enc = int(player_df["player_id_encoded"].iloc[-1])
    era_id = int(player_df["era_id"].iloc[-1])

    return torch.tensor(input_norm).unsqueeze(0), player_id_enc, era_id  # (1, seq_len, n_feat)


def _build_context_tensor(
    player_df: pd.DataFrame,
    override: ScenarioOverride,
    norm_stats: dict[str, Any],
) -> torch.Tensor:
    """
    Build the next-game context tensor, applying scenario overrides.
    Baseline values come from the player's most recent game.
    """
    last_game = player_df.sort_values("gameDateTimeEst").iloc[-1]

    def norm(feat: str, val: float) -> float:
        return (val - norm_stats[feat]["mean"]) / norm_stats[feat]["std"]

    ctx = []
    for feat in CONTEXT_FEATURES:
        baseline_val = float(last_game[feat]) if feat in last_game.index else 0.0

        if feat == "rest_days" and override.rest_days is not None:
            val = override.rest_days
        elif feat == "home" and override.home is not None:
            val = float(override.home)
        elif feat == "opp_pts_allowed_roll10" and override.opp_pts_allowed is not None:
            val = override.opp_pts_allowed
        elif feat == "game_pace" and override.game_pace is not None:
            val = override.game_pace
        elif feat == "numMinutes" and override.minutes is not None:
            val = override.minutes
        else:
            val = baseline_val

        ctx.append(norm(feat, val))

    return torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)  # (1, n_context)


def simulate(
    player_name: str,
    override: ScenarioOverride | None = None,
    n_samples: int = 500,
    checkpoint_path: Path | None = None,
    features_path: Path | None = None,
) -> SimulationResult:
    """
    Run a counterfactual simulation for a player.

    Args:
        player_name:     Partial or full player name (case-insensitive match).
        override:        Hypothetical game conditions. None = baseline prediction.
        n_samples:       Number of Monte Carlo dropout samples.
        checkpoint_path: Path to model checkpoint. Defaults to best.pt.
        features_path:   Path to player_features.parquet. Defaults to DATA_CONFIG.

    Returns:
        SimulationResult with mean/std/percentile distributions per stat.
    """
    checkpoint_path = checkpoint_path or (TRAIN_CONFIG.checkpoint_dir / "best.pt")
    features_path = features_path or DATA_CONFIG.player_features_path
    override = override or ScenarioOverride()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    norm_stats = load_norm_stats()
    df = pd.read_parquet(features_path)

    # Find player
    id_map_path = DATA_CONFIG.player_id_map_path
    with open(id_map_path) as f:
        id_map = json.load(f)  # str(personId) -> encoded_idx

    # Match player name — features don't have names, so look up via raw data
    raw_path = DATA_CONFIG.raw_dir / "PlayerStatistics.csv"
    raw = pd.read_csv(raw_path, usecols=["personId", "firstName", "lastName"], low_memory=False)
    raw = raw.dropna(subset=["personId"]).drop_duplicates("personId")
    raw["personId"] = pd.to_numeric(raw["personId"], errors="coerce")
    raw = raw.dropna(subset=["personId"])
    raw["personId"] = raw["personId"].astype(int)
    raw["fullName"] = raw["firstName"].fillna("") + " " + raw["lastName"].fillna("")
    raw["fullName"] = raw["fullName"].str.strip()

    matches = raw[raw["fullName"].str.contains(player_name, case=False, na=False)]
    matches = matches.drop_duplicates("personId")
    if matches.empty:
        raise ValueError(f"No player found matching '{player_name}'")
    if len(matches) > 1:
        # Multiple personIds with same name (duplicates/historical entries).
        # Pick the one with the most game rows in the features dataset.
        game_counts = {
            int(row["personId"]): int((df["personId"] == int(row["personId"])).sum())
            for _, row in matches.iterrows()
        }
        best_id = max(game_counts, key=game_counts.get)
        matches = matches[matches["personId"] == best_id]

    person_id = int(matches.iloc[0]["personId"])
    resolved_name = matches.iloc[0]["fullName"]

    # Get player's feature rows
    player_df = df[df["personId"] == person_id]
    if len(player_df) < MODEL_CONFIG.sequence_length:
        raise ValueError(
            f"{resolved_name} has only {len(player_df)} games, need {MODEL_CONFIG.sequence_length}."
        )

    # Build input sequence (historical context)
    input_seq, player_id_enc, era_id = _build_input_sequence(
        player_df, norm_stats, seq_len=MODEL_CONFIG.sequence_length
    )

    # Build next-game context tensor with scenario override applied
    next_game_ctx = _build_context_tensor(player_df, override, norm_stats)

    # Compute window mean of target stats (baseline for delta reconstruction)
    last_games = player_df.sort_values("gameDateTimeEst").tail(MODEL_CONFIG.sequence_length)
    window_mean = last_games[TARGET_STATS].values.mean(axis=0).astype(np.float32)

    player_id_t = torch.tensor([player_id_enc], dtype=torch.long).to(device)
    era_id_t = torch.tensor([era_id], dtype=torch.long).to(device)
    input_seq_t = input_seq.to(device)
    next_game_ctx_t = next_game_ctx.to(device)

    # Try ensemble first (real uncertainty); fall back to MC dropout
    from nba_gpt.simulation.ensemble import EnsemblePredictor, ENSEMBLE_DIR
    use_ensemble = ENSEMBLE_DIR.exists() and any(ENSEMBLE_DIR.glob("seed_*/best.pt"))

    if use_ensemble:
        predictor = EnsemblePredictor(device=device)
        dist = predictor.predict_distribution(
            player_id_t, era_id_t, input_seq_t, next_game_ctx_t, window_mean
        )
        # Synthesize samples array from ensemble stats for consistent downstream API
        n_members = predictor.n_members
        member_actuals = np.stack([
            m(player_id_t, era_id_t, input_seq_t, next_game_ctx_t).float().detach().cpu().numpy()[0] + window_mean
            for m in predictor.models
        ])
        samples = np.clip(member_actuals, 0, None)  # (n_members, n_targets)
    else:
        # MC Dropout fallback
        model = _load_model(checkpoint_path, device)
        model.eval()
        _enable_dropout(model)

        samples_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred_delta = model(player_id_t, era_id_t, input_seq_t, next_game_ctx_t).float().cpu().numpy()
                pred_actual = pred_delta[0] + window_mean
                samples_list.append(pred_actual)
        samples = np.clip(np.stack(samples_list), 0, None)  # (n_samples, n_targets)

    # Build result
    result = SimulationResult(
        player_name=resolved_name,
        scenario=override,
        n_samples=n_samples,
        samples=samples,
    )
    for i, stat in enumerate(TARGET_STATS):
        col = samples[:, i]
        result.mean[stat] = float(col.mean())
        result.std[stat] = float(col.std())
        result.p10[stat] = float(np.percentile(col, 10))
        result.p25[stat] = float(np.percentile(col, 25))
        result.p75[stat] = float(np.percentile(col, 75))
        result.p90[stat] = float(np.percentile(col, 90))

    return result


def compare_scenarios(
    player_name: str,
    scenario_a: ScenarioOverride,
    scenario_b: ScenarioOverride,
    labels: tuple[str, str] = ("Scenario A", "Scenario B"),
    n_samples: int = 500,
    checkpoint_path: Path | None = None,
) -> None:
    """
    Run two scenarios and print a side-by-side comparison.
    The delta shows the causal impact of changing conditions.
    """
    print(f"\nRunning '{labels[0]}'...")
    result_a = simulate(player_name, scenario_a, n_samples, checkpoint_path)
    print(f"Running '{labels[1]}'...")
    result_b = simulate(player_name, scenario_b, n_samples, checkpoint_path)

    print(f"\n{'='*70}")
    print(f"Counterfactual: {result_a.player_name}")
    print(f"{'='*70}")
    print(f"{'Stat':<20} {labels[0]:>14} {labels[1]:>14} {'Delta':>10}")
    print(f"{'-'*70}")
    for stat in TARGET_STATS:
        a = result_a.mean[stat]
        b = result_b.mean[stat]
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(f"{stat:<20} {a:>14.2f} {b:>14.2f} {sign}{delta:>9.2f}")
    print("=" * 70)
