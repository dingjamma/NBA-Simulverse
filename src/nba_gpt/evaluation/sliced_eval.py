"""
Sliced evaluation — where the transformer's edge over XGBoost actually shows.

Instead of aggregate MAE, we slice the test set by conditions where
sequence modeling should matter:
  - Back-to-back games (rest_days == 1)
  - Players on hot streaks (last 5 avg > season avg by 15%+)
  - Cold streaks (last 5 avg < season avg by 15%+)
  - Road games vs home games
  - Tough defensive matchups (opp_pts_allowed bottom quartile)
  - High-pace vs low-pace games

If the transformer wins specifically on these slices, it's learning physics.
If it wins uniformly, it might just be memorizing better averages.
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from nba_gpt.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, TARGET_STATS, INPUT_FEATURES
from nba_gpt.data.dataset import NBASequenceDataset, load_norm_stats
from nba_gpt.model.transformer import NBAGPTModel
from nba_gpt.baseline.xgboost_baseline import load_models, build_features, split_by_date


# ── Slice definitions ────────────────────────────────────────────────────────

def _feat_idx(name: str) -> int:
    return INPUT_FEATURES.index(name)


def define_slices(
    inputs: np.ndarray,
    norm_stats: dict,
) -> dict[str, np.ndarray]:
    """
    Build boolean masks over test sequences.
    `inputs` shape: (N, seq_len, n_features) — normalized.
    We un-normalize relevant features before applying thresholds so
    slices are in interpretable raw space (days, points, etc.).
    """
    last = inputs[:, -1, :]  # (N, n_features)

    def unnorm(feat: str, arr: np.ndarray) -> np.ndarray:
        m = norm_stats[feat]["mean"]
        s = norm_stats[feat]["std"]
        return arr * s + m

    rest_raw = unnorm("rest_days", last[:, _feat_idx("rest_days")])
    home_raw = unnorm("home", last[:, _feat_idx("home")])
    opp_def_raw = unnorm("opp_pts_allowed_roll10", last[:, _feat_idx("opp_pts_allowed_roll10")])
    pace_raw = unnorm("game_pace", last[:, _feat_idx("game_pace")])

    # Hot/cold streak: compare roll5_pts (last 5 avg) vs window mean (last 20 avg)
    roll5_pts_raw = unnorm("roll5_points", last[:, _feat_idx("roll5_points")])
    pts_raw = unnorm("points", inputs[:, :, _feat_idx("points")])
    window_pts_mean = pts_raw.mean(axis=1)
    streak_ratio = np.where(window_pts_mean > 1, roll5_pts_raw / window_pts_mean, 1.0)

    opp_def_q25 = np.percentile(opp_def_raw, 25)
    opp_def_q75 = np.percentile(opp_def_raw, 75)
    pace_q25 = np.percentile(pace_raw, 25)
    pace_q75 = np.percentile(pace_raw, 75)

    return {
        "all":             np.ones(len(inputs), dtype=bool),
        # Rest-based slices (raw days)
        "back_to_back":    rest_raw <= 1,
        "normal_rest":     (rest_raw >= 2) & (rest_raw <= 3),
        "long_rest":       rest_raw >= 5,
        # Venue
        "home":            home_raw > 0.5,
        "away":            home_raw <= 0.5,
        # Streak (ratio of last-5 avg to last-20 avg)
        "hot_streak":      streak_ratio >= 1.15,
        "cold_streak":     streak_ratio <= 0.85,
        # Matchup context
        "tough_defense":   opp_def_raw <= opp_def_q25,
        "weak_defense":    opp_def_raw >= opp_def_q75,
        "high_pace":       pace_raw >= pace_q75,
        "low_pace":        pace_raw <= pace_q25,
    }


# ── Inference ────────────────────────────────────────────────────────────────

def _collect_predictions(
    model: NBAGPTModel,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pred_actuals, target_actuals, inputs_raw, target_baselines)."""
    model.eval()
    all_preds, all_targets, all_inputs, all_baselines = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            pid = batch["player_id"].to(device)
            eid = batch["era_id"].to(device)
            seq = batch["input_seq"].to(device)
            ctx = batch["next_game_ctx"].to(device)
            pred_delta = model(pid, eid, seq, ctx).float().cpu().numpy()
            baseline = batch["target_baseline"].numpy()
            all_preds.append(pred_delta + baseline)
            all_targets.append(batch["target"].numpy() + baseline)
            all_inputs.append(batch["input_seq"].numpy())
            all_baselines.append(baseline)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_inputs),
        np.concatenate(all_baselines),
    )


# ── XGBoost reference ────────────────────────────────────────────────────────

def _xgb_predictions(df_feat: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xgb_preds, targets) aligned to test rows."""
    xgb_models = load_models()
    _, _, test_df = split_by_date(df_feat)
    X = test_df[feature_cols].values
    preds = np.stack([xgb_models[s].predict(X) for s in TARGET_STATS], axis=1)
    targets = test_df[TARGET_STATS].values
    return preds, targets


# ── Main ─────────────────────────────────────────────────────────────────────

def run(
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    checkpoint_path = checkpoint_path or (TRAIN_CONFIG.checkpoint_dir / "best.pt")
    output_path = output_path or (DATA_CONFIG.processed_dir / "sliced_eval.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load transformer
    print("Loading NBA-GPT...")
    model = NBAGPTModel(MODEL_CONFIG).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    norm_stats = load_norm_stats()
    test_ds = NBASequenceDataset(DATA_CONFIG.test_sequences_path, norm_stats)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=TRAIN_CONFIG.num_workers)

    print("Collecting transformer predictions...")
    gpt_preds, gpt_targets, inputs_raw, _ = _collect_predictions(model, test_loader, device)

    slices = define_slices(inputs_raw, norm_stats)

    print("Collecting XGBoost predictions...")
    df = pd.read_parquet(DATA_CONFIG.player_features_path)
    df_feat, feature_cols = build_features(df)
    xgb_preds, xgb_targets = _xgb_predictions(df_feat, feature_cols)

    # XGBoost test set is aligned by date split but may differ in size from
    # transformer test set (XGBoost doesn't require seq_len games of history).
    # Use the smaller size and align from the end (most recent games).
    n_gpt = len(gpt_preds)
    n_xgb = len(xgb_preds)
    if n_xgb != n_gpt:
        print(f"  Note: GPT test={n_gpt:,}, XGB test={n_xgb:,} — aligning to GPT size")
        if n_xgb > n_gpt:
            xgb_preds = xgb_preds[-n_gpt:]
            xgb_targets = xgb_targets[-n_gpt:]
        else:
            gpt_preds = gpt_preds[-n_xgb:]
            gpt_targets = gpt_targets[-n_xgb:]
            inputs_raw = inputs_raw[-n_xgb:]
            slices = define_slices(inputs_raw, norm_stats)

    # Compute MAE per slice per stat
    results = {}
    print("\n" + "=" * 80)
    print(f"{'Slice':<18} {'N':>6}  " + "  ".join(f"{s[:5]:>10}" for s in TARGET_STATS))
    print("-" * 80)

    for slice_name, mask in slices.items():
        n = mask.sum()
        if n < 10:
            continue

        gpt_mae = {
            stat: float(np.abs(gpt_preds[mask, i] - gpt_targets[mask, i]).mean())
            for i, stat in enumerate(TARGET_STATS)
        }
        xgb_mae = {
            stat: float(np.abs(xgb_preds[mask, i] - xgb_targets[mask, i]).mean())
            for i, stat in enumerate(TARGET_STATS)
        }
        gpt_wins = sum(1 for s in TARGET_STATS if gpt_mae[s] < xgb_mae[s])

        results[slice_name] = {
            "n": int(n),
            "gpt_mae": gpt_mae,
            "xgb_mae": xgb_mae,
            "gpt_wins": gpt_wins,
        }

        # Print row: GPT MAE with * where GPT wins
        row = f"{slice_name:<18} {n:>6}  "
        row += "  ".join(
            f"{'*' if gpt_mae[s] < xgb_mae[s] else ' '}{gpt_mae[s]:>9.3f}"
            for s in TARGET_STATS
        )
        print(row)

    print("=" * 80)
    print("* = NBA-GPT wins that slice\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

    return results


if __name__ == "__main__":
    run()
