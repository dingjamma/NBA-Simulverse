"""
Ensemble inference for NBA-Simulverse.

Loads N trained models and runs each once per query.
The spread across model predictions is calibrated uncertainty —
not dropout noise, not heuristic variance, but actual disagreement
between independently trained models.

This is the right way to answer "how confident is the simulation?"
"""
import numpy as np
import torch
from pathlib import Path
from typing import Any

from nba_gpt.config import MODEL_CONFIG, TRAIN_CONFIG, TARGET_STATS
from nba_gpt.model.transformer import NBAGPTModel


ENSEMBLE_DIR = TRAIN_CONFIG.checkpoint_dir / "ensemble"


def _load_member(checkpoint_path: Path, device: torch.device) -> NBAGPTModel:
    model = NBAGPTModel(MODEL_CONFIG).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def discover_ensemble(ensemble_dir: Path | None = None) -> list[Path]:
    """Find all seed_N/best.pt checkpoints under ensemble_dir."""
    ensemble_dir = ensemble_dir or ENSEMBLE_DIR
    paths = sorted(ensemble_dir.glob("seed_*/best.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No ensemble checkpoints found in {ensemble_dir}. "
            f"Run scripts/09_train_ensemble.py first."
        )
    return paths


class EnsemblePredictor:
    """
    Runs inference across all ensemble members and returns
    mean, std, and per-member predictions.

    Usage:
        predictor = EnsemblePredictor()
        result = predictor.predict(player_id, era_id, input_seq, next_game_ctx)
        # result["mean"]   shape: (n_targets,)
        # result["std"]    shape: (n_targets,)
        # result["all"]    shape: (n_members, n_targets)
    """

    def __init__(
        self,
        ensemble_dir: Path | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paths = discover_ensemble(ensemble_dir)
        print(f"Loading {len(paths)} ensemble members from {ensemble_dir or ENSEMBLE_DIR}...")
        self.models = [_load_member(p, self.device) for p in paths]
        self.n_members = len(self.models)
        print(f"  Loaded {self.n_members} models.")

    @torch.no_grad()
    def predict(
        self,
        player_id: torch.Tensor,      # (batch,)
        era_id: torch.Tensor,         # (batch,)
        input_seq: torch.Tensor,      # (batch, seq_len, n_features)
        next_game_ctx: torch.Tensor,  # (batch, n_context)
    ) -> dict[str, np.ndarray]:
        """Run all ensemble members, return mean/std/all predictions."""
        all_preds = []
        for model in self.models:
            pred = model(
                player_id.to(self.device),
                era_id.to(self.device),
                input_seq.to(self.device),
                next_game_ctx.to(self.device),
            ).float().cpu().numpy()
            all_preds.append(pred)

        all_preds = np.stack(all_preds)  # (n_members, batch, n_targets)
        return {
            "mean": all_preds.mean(axis=0),   # (batch, n_targets)
            "std":  all_preds.std(axis=0),    # (batch, n_targets)
            "all":  all_preds,                 # (n_members, batch, n_targets)
        }

    @torch.no_grad()
    def predict_distribution(
        self,
        player_id: torch.Tensor,
        era_id: torch.Tensor,
        input_seq: torch.Tensor,
        next_game_ctx: torch.Tensor,
        window_mean: np.ndarray,       # (n_targets,) rolling mean for delta reconstruction
    ) -> dict[str, Any]:
        """
        Full distribution in raw stat space.
        Returns per-stat mean, std, percentiles across the ensemble.
        """
        preds = self.predict(player_id, era_id, input_seq, next_game_ctx)

        # Each member predicts a delta — reconstruct actual stats
        # preds["all"]: (n_members, 1, n_targets)
        member_actuals = preds["all"][:, 0, :] + window_mean  # (n_members, n_targets)
        member_actuals = np.clip(member_actuals, 0, None)

        result: dict[str, Any] = {"mean": {}, "std": {}, "p10": {}, "p25": {}, "p75": {}, "p90": {}}
        for i, stat in enumerate(TARGET_STATS):
            col = member_actuals[:, i]
            result["mean"][stat] = float(col.mean())
            result["std"][stat]  = float(col.std())
            result["p10"][stat]  = float(np.percentile(col, 10))
            result["p25"][stat]  = float(np.percentile(col, 25))
            result["p75"][stat]  = float(np.percentile(col, 75))
            result["p90"][stat]  = float(np.percentile(col, 90))

        return result
