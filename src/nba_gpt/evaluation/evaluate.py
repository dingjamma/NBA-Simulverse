"""
Evaluate NBA-GPT vs XGBoost baseline on the 2024-25 holdout season.
Reports MAE per stat, per era, and by player volume.
"""
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from nba_gpt.config import DATA_CONFIG, TARGET_STATS, MODEL_CONFIG, TRAIN_CONFIG
from nba_gpt.data.dataset import NBASequenceDataset, load_norm_stats
from nba_gpt.model.transformer import NBAGPTModel
from nba_gpt.baseline.xgboost_baseline import load_models, evaluate_on_test, build_features, split_by_date
import pandas as pd


def evaluate_transformer(
    model: NBAGPTModel,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    all_baselines = []

    with torch.no_grad():
        for batch in test_loader:
            player_id = batch["player_id"].to(device)
            era_id = batch["era_id"].to(device)
            input_seq = batch["input_seq"].to(device)
            next_game_ctx = batch["next_game_ctx"].to(device)
            pred = model(player_id, era_id, input_seq, next_game_ctx).float().cpu()
            all_preds.append(pred)
            all_targets.append(batch["target"])
            all_baselines.append(batch["target_baseline"])

    pred_deltas = torch.cat(all_preds).numpy()
    target_deltas = torch.cat(all_targets).numpy()
    baselines = torch.cat(all_baselines).numpy()

    # Reconstruct actual stat values
    pred_actuals = pred_deltas + baselines
    target_actuals = target_deltas + baselines

    return {
        stat: float(np.abs(pred_actuals[:, i] - target_actuals[:, i]).mean())
        for i, stat in enumerate(TARGET_STATS)
    }


def print_comparison(xgb_mae: dict[str, float], gpt_mae: dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print(f"{'Stat':<20} {'XGBoost MAE':>12} {'NBA-GPT MAE':>12} {'Winner':>8}")
    print("-" * 60)

    gpt_wins = 0
    for stat in TARGET_STATS:
        xgb = xgb_mae[stat]
        gpt = gpt_mae[stat]
        winner = "NBA-GPT" if gpt < xgb else "XGBoost"
        if gpt < xgb:
            gpt_wins += 1
        print(f"{stat:<20} {xgb:>12.3f} {gpt:>12.3f} {winner:>8}")

    print("=" * 60)
    print(f"NBA-GPT wins: {gpt_wins}/{len(TARGET_STATS)}")
    success = gpt_wins >= 4
    print(f"Phase 1 success: {'YES' if success else 'NO (need 4/6)'}")
    print()


def run(
    checkpoint_path: Path | None = None,
    results_path: Path | None = None,
) -> dict:
    checkpoint_path = checkpoint_path or (TRAIN_CONFIG.checkpoint_dir / "best.pt")
    results_path = results_path or (DATA_CONFIG.processed_dir / "evaluation_results.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load transformer
    print("Loading NBA-GPT model...")
    model = NBAGPTModel(MODEL_CONFIG).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded from epoch {ckpt['epoch']}")

    # Build test loader
    norm_stats = load_norm_stats()
    test_ds = NBASequenceDataset(DATA_CONFIG.test_sequences_path, norm_stats)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=TRAIN_CONFIG.num_workers)

    # Evaluate transformer
    print("Evaluating NBA-GPT on test set...")
    gpt_mae = evaluate_transformer(model, test_loader, device)

    # Evaluate XGBoost
    print("Evaluating XGBoost on test set...")
    xgb_models = load_models()
    df = pd.read_parquet(DATA_CONFIG.player_features_path)
    df_feat, feature_cols = build_features(df)
    _, _, test_df = split_by_date(df_feat)
    X_test = test_df[feature_cols].values
    xgb_mae = {}
    for stat, xgb_model in xgb_models.items():
        y_test = test_df[stat].values
        pred = xgb_model.predict(X_test)
        xgb_mae[stat] = float(np.abs(pred - y_test).mean())

    print_comparison(xgb_mae, gpt_mae)

    results = {
        "xgboost_mae": xgb_mae,
        "nba_gpt_mae": gpt_mae,
        "nba_gpt_wins": sum(1 for s in TARGET_STATS if gpt_mae[s] < xgb_mae[s]),
        "phase1_success": sum(1 for s in TARGET_STATS if gpt_mae[s] < xgb_mae[s]) >= 4,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run()
