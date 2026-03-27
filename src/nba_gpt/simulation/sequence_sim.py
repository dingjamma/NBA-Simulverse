"""
Full sequence injection — Phase 2's real counterfactual engine.

Instead of tweaking one feature in one position, we construct a
synthetic multi-game sequence under hypothetical conditions and run
it forward through the model. Each game feeds into the next.

This answers questions like:
  "What does Steph's game 5 look like after a 4-game road trip?"
  "How does Luka perform game 3 into a back-to-back stretch?"
  "What happens to a player's stats after a week off, game by game?"

Process for an N-game scenario:
  1. Start from the player's real last 20 games as the seed window.
  2. For each hypothetical game in the scenario:
     a. Build next_game_ctx from the scenario for this game.
     b. Run model → predicted delta → reconstruct actual stats.
     c. Append the predicted game to the sequence window (slide forward).
     d. Repeat for game+1 using the updated window.

This is autoregressive simulation — each prediction becomes the history
for the next prediction.
"""
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
class GameCondition:
    """Conditions for one game in a synthetic sequence."""
    rest_days: float          # days since previous game
    home: bool                # home or away
    opp_pts_allowed: float    # opponent defensive quality
    game_pace: float          # game pace (total FGA proxy)
    minutes: float            # projected minutes


@dataclass
class SequenceScenario:
    """
    A multi-game hypothetical scenario.
    Each element of `games` is one future game to simulate.
    """
    games: list[GameCondition]
    description: str = ""


@dataclass
class SequenceSimResult:
    player_name: str
    scenario: SequenceScenario
    # Per-game predictions: list of {stat: value} dicts
    game_predictions: list[dict[str, float]] = field(default_factory=list)
    # Ensemble std per game per stat (None if single model)
    game_uncertainty: list[dict[str, float]] | None = None

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"Sequence simulation: {self.player_name}",
            f"Scenario: {self.scenario.description}",
            f"{'='*70}",
        ]
        header = f"{'Game':<6} " + "  ".join(f"{s[:5]:>7}" for s in TARGET_STATS)
        lines.append(header)
        lines.append("-" * 70)

        for i, preds in enumerate(self.game_predictions):
            row = f"{'G'+str(i+1):<6} "
            row += "  ".join(f"{preds[s]:>7.1f}" for s in TARGET_STATS)
            if self.game_uncertainty:
                unc = self.game_uncertainty[i]
                row += "   ± " + "/".join(f"{unc[s]:.2f}" for s in TARGET_STATS[:3])
            lines.append(row)

        lines.append("=" * 70)
        return "\n".join(lines)


def _load_model(checkpoint_path: Path, device: torch.device) -> NBAGPTModel:
    model = NBAGPTModel(MODEL_CONFIG).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _build_norm_context(
    condition: GameCondition,
    norm_stats: dict[str, Any],
) -> torch.Tensor:
    """Build a normalized context tensor from a GameCondition."""
    feat_map = {
        "rest_days": condition.rest_days,
        "home": float(condition.home),
        "opp_pts_allowed_roll10": condition.opp_pts_allowed,
        "game_pace": condition.game_pace,
        "numMinutes": condition.minutes,
    }
    ctx = [
        (feat_map[f] - norm_stats[f]["mean"]) / norm_stats[f]["std"]
        for f in CONTEXT_FEATURES
    ]
    return torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)  # (1, n_context)


def _pred_to_input_row(
    pred_stats: dict[str, float],
    condition: GameCondition,
    norm_stats: dict[str, Any],
    prev_row: np.ndarray,  # previous row in raw space for features not predicted
) -> np.ndarray:
    """
    Construct a full INPUT_FEATURES row for the predicted game.
    Uses predicted stats where available, condition for context features,
    and previous row values for everything else.
    """
    row = prev_row.copy()
    feat_idx = {f: i for i, f in enumerate(INPUT_FEATURES)}

    # Fill in predicted stats
    for stat in TARGET_STATS:
        if stat in feat_idx:
            row[feat_idx[stat]] = pred_stats[stat]

    # Fill in known conditions
    row[feat_idx["numMinutes"]] = condition.minutes
    row[feat_idx["home"]] = float(condition.home)
    row[feat_idx["rest_days"]] = condition.rest_days
    row[feat_idx["opp_pts_allowed_roll10"]] = condition.opp_pts_allowed
    row[feat_idx["game_pace"]] = condition.game_pace

    # Approximate derived stats from predicted main stats
    if "points" in feat_idx and "fieldGoalsAttempted" in feat_idx:
        # Rough FGA from points: assume ~1.1 pts/FGA
        row[feat_idx["fieldGoalsAttempted"]] = max(0, pred_stats.get("points", 0) / 1.1)
    if "points" in feat_idx and "fieldGoalsMade" in feat_idx:
        row[feat_idx["fieldGoalsMade"]] = max(0, pred_stats.get("points", 0) / 2.2)

    return row


def simulate_sequence(
    player_name: str,
    scenario: SequenceScenario,
    checkpoint_path: Path | None = None,
    features_path: Path | None = None,
) -> SequenceSimResult:
    """
    Run an autoregressive multi-game simulation for a player.

    Each game in the scenario is predicted in order, with the
    previous prediction feeding into the next game's history window.
    """
    checkpoint_path = checkpoint_path or (TRAIN_CONFIG.checkpoint_dir / "best.pt")
    features_path = features_path or DATA_CONFIG.player_features_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_stats = load_norm_stats()
    df = pd.read_parquet(features_path)

    # Resolve player
    raw_path = DATA_CONFIG.raw_dir / "PlayerStatistics.csv"
    raw = pd.read_csv(raw_path, usecols=["personId", "firstName", "lastName"],
                      low_memory=False)
    raw["personId"] = pd.to_numeric(raw["personId"], errors="coerce")
    raw = raw.dropna(subset=["personId"])
    raw["personId"] = raw["personId"].astype(int)
    raw["fullName"] = (raw["firstName"].fillna("") + " " + raw["lastName"].fillna("")).str.strip()
    raw = raw.drop_duplicates("personId")

    matches = raw[raw["fullName"].str.contains(player_name, case=False, na=False)]
    matches = matches.drop_duplicates("personId")
    if matches.empty:
        raise ValueError(f"No player found matching '{player_name}'")
    if len(matches) > 1:
        game_counts = {
            int(r["personId"]): int((df["personId"] == int(r["personId"])).sum())
            for _, r in matches.iterrows()
        }
        best_id = max(game_counts, key=game_counts.get)
        matches = matches[matches["personId"] == best_id]

    person_id = int(matches.iloc[0]["personId"])
    resolved_name = matches.iloc[0]["fullName"]

    player_df = df[df["personId"] == person_id].sort_values("gameDateTimeEst")
    if len(player_df) < MODEL_CONFIG.sequence_length:
        raise ValueError(f"{resolved_name}: only {len(player_df)} games, need {MODEL_CONFIG.sequence_length}.")

    # Initialize: seed window = last 20 real games (raw feature values)
    seed_rows = player_df.tail(MODEL_CONFIG.sequence_length)
    seed_raw = seed_rows[INPUT_FEATURES].values.astype(np.float32)   # (seq_len, n_feat)
    seed_targets = seed_rows[TARGET_STATS].values.astype(np.float32) # (seq_len, n_targets)

    player_id_enc = int(seed_rows["player_id_encoded"].iloc[-1])
    era_id = int(seed_rows["era_id"].iloc[-1])

    # Normalize helpers
    input_means = np.array([norm_stats[f]["mean"] for f in INPUT_FEATURES], dtype=np.float32)
    input_stds = np.array([norm_stats[f]["std"] for f in INPUT_FEATURES], dtype=np.float32)

    # Try ensemble; fall back to single model
    from nba_gpt.simulation.ensemble import EnsemblePredictor, ENSEMBLE_DIR
    use_ensemble = ENSEMBLE_DIR.exists() and any(ENSEMBLE_DIR.glob("seed_*/best.pt"))

    if use_ensemble:
        predictor = EnsemblePredictor(device=device)
        models = predictor.models
    else:
        models = [_load_model(checkpoint_path, device)]

    player_id_t = torch.tensor([player_id_enc], dtype=torch.long).to(device)
    era_id_t = torch.tensor([era_id], dtype=torch.long).to(device)

    # Rolling window in raw space
    window_raw = seed_raw.copy()        # (seq_len, n_feat)
    window_targets = seed_targets.copy()  # (seq_len, n_targets)

    game_predictions = []
    game_uncertainty = [] if use_ensemble else None

    for game_idx, condition in enumerate(scenario.games):
        # Build normalized input sequence
        seq_norm = (window_raw - input_means) / input_stds
        input_seq_t = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(device)

        # Build next-game context
        ctx_t = _build_norm_context(condition, norm_stats).to(device)

        # Window mean for delta reconstruction
        window_mean = window_targets.mean(axis=0)

        # Run all models
        all_deltas = []
        with torch.no_grad():
            for model in models:
                delta = model(player_id_t, era_id_t, input_seq_t, ctx_t).float().cpu().numpy()[0]
                all_deltas.append(delta)

        all_deltas = np.stack(all_deltas)  # (n_models, n_targets)
        mean_delta = all_deltas.mean(axis=0)
        pred_actual = np.clip(mean_delta + window_mean, 0, None)

        preds = {stat: float(pred_actual[i]) for i, stat in enumerate(TARGET_STATS)}
        game_predictions.append(preds)

        if use_ensemble and len(models) > 1:
            member_actuals = np.clip(all_deltas + window_mean, 0, None)
            unc = {stat: float(member_actuals[:, i].std()) for i, stat in enumerate(TARGET_STATS)}
            game_uncertainty.append(unc)

        # Slide window: drop oldest game, append predicted game
        predicted_row = _pred_to_input_row(preds, condition, norm_stats, window_raw[-1].copy())
        window_raw = np.vstack([window_raw[1:], predicted_row])
        window_targets = np.vstack([window_targets[1:], pred_actual])

        print(f"  Game {game_idx+1}: pts={preds['points']:.1f} reb={preds['reboundsTotal']:.1f} "
              f"ast={preds['assists']:.1f} 3pm={preds['threePointersMade']:.1f}")

    return SequenceSimResult(
        player_name=resolved_name,
        scenario=scenario,
        game_predictions=game_predictions,
        game_uncertainty=game_uncertainty,
    )
