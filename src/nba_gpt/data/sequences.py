"""
Build sliding window sequences from the feature parquet.
Each sequence: 20 input games -> 1 target game.
Splits by target game date into train/val/test.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from nba_gpt.config import DATA_CONFIG, INPUT_FEATURES, TARGET_STATS, CONTEXT_FEATURES
# DATA_CONFIG.min_target_year filters sequences to post-2002 target games


def build_sequences(df: pd.DataFrame, seq_len: int = 20) -> dict[str, np.ndarray]:
    """
    Process all players and build sliding window sequences.
    Returns dict with arrays: player_ids, era_ids, inputs, targets.
    """
    player_ids_list = []
    era_ids_list = []
    inputs_list = []
    targets_list = []
    target_baselines_list = []
    next_game_ctx_list = []
    game_dates_list = []

    input_cols = INPUT_FEATURES
    target_cols = TARGET_STATS
    context_cols = CONTEXT_FEATURES

    n_players = df["personId"].nunique()
    processed = 0

    for pid, group in df.groupby("personId"):
        group = group.sort_values("gameDateTimeEst").reset_index(drop=True)
        n_games = len(group)

        if n_games < DATA_CONFIG.min_player_games:
            continue

        player_id_enc = int(group["player_id_encoded"].iloc[0])

        input_vals = group[input_cols].values.astype(np.float32)    # (n_games, n_features)
        target_vals = group[target_cols].values.astype(np.float32)  # (n_games, n_targets)
        context_vals = group[context_cols].values.astype(np.float32)  # (n_games, n_context)
        era_vals = group["era_id"].values.astype(np.int64)
        dates = group["gameDateTimeEst"].values  # numpy datetime64

        # Sliding windows: game i is target, games i-seq_len .. i-1 are input
        min_year = DATA_CONFIG.min_target_year
        for i in range(seq_len, n_games):
            target_year = pd.Timestamp(dates[i]).year
            if target_year < min_year:
                continue  # skip pre-modern-era target games
            window_targets = target_vals[i - seq_len : i]      # (seq_len, n_targets)
            window_mean = window_targets.mean(axis=0)           # (n_targets,)
            delta = target_vals[i] - window_mean                # (n_targets,) deviation
            inputs_list.append(input_vals[i - seq_len : i])    # (seq_len, n_features)
            targets_list.append(delta)                          # predict delta
            target_baselines_list.append(window_mean)           # save baseline for reconstruction
            next_game_ctx_list.append(context_vals[i])         # (n_context,) target game conditions
            player_ids_list.append(player_id_enc)
            era_ids_list.append(int(era_vals[i]))               # era of target game
            game_dates_list.append(dates[i])

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{n_players} players...")

    n = len(inputs_list)
    print(f"  Total sequences: {n:,}")

    n_input_feat = len(INPUT_FEATURES)
    n_target_feat = len(TARGET_STATS)

    n_context_feat = len(CONTEXT_FEATURES)

    if n == 0:
        seq_len = DATA_CONFIG.sequence_length
        return {
            "player_ids": np.array([], dtype=np.int64),
            "era_ids": np.array([], dtype=np.int64),
            "inputs": np.empty((0, seq_len, n_input_feat), dtype=np.float32),
            "targets": np.empty((0, n_target_feat), dtype=np.float32),
            "target_baselines": np.empty((0, n_target_feat), dtype=np.float32),
            "next_game_ctx": np.empty((0, n_context_feat), dtype=np.float32),
            "game_dates": np.array([], dtype="datetime64[ns]"),
        }

    return {
        "player_ids": np.array(player_ids_list, dtype=np.int64),
        "era_ids": np.array(era_ids_list, dtype=np.int64),
        "inputs": np.stack(inputs_list).astype(np.float32),                      # (N, seq_len, n_features)
        "targets": np.stack(targets_list).astype(np.float32),                    # (N, n_targets) deltas
        "target_baselines": np.stack(target_baselines_list).astype(np.float32),  # (N, n_targets) window means
        "next_game_ctx": np.stack(next_game_ctx_list).astype(np.float32),        # (N, n_context)
        "game_dates": np.array(game_dates_list),                                  # (N,) datetime64
    }


def split_sequences(
    data: dict[str, np.ndarray],
    val_start: str,
    test_start: str,
) -> tuple[dict, dict, dict]:
    """Split sequences by target game date."""
    dates = pd.to_datetime(data["game_dates"])
    val_ts = pd.Timestamp(val_start)
    test_ts = pd.Timestamp(test_start)

    train_mask = dates < val_ts
    val_mask = (dates >= val_ts) & (dates < test_ts)
    test_mask = dates >= test_ts

    def subset(mask: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "player_ids": data["player_ids"][mask],
            "era_ids": data["era_ids"][mask],
            "inputs": data["inputs"][mask],
            "targets": data["targets"][mask],
            "target_baselines": data["target_baselines"][mask],
            "next_game_ctx": data["next_game_ctx"][mask],
        }

    return subset(train_mask), subset(val_mask), subset(test_mask)


def save_split(data: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {len(data['player_ids']):,} sequences to {path} ({size_mb:.1f} MB)")


def run(
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[dict, dict, dict]:
    input_path = input_path or DATA_CONFIG.player_features_path
    output_dir = output_dir or DATA_CONFIG.processed_dir

    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df):,} rows, {df['personId'].nunique():,} players")

    print(f"Building sequences (seq_len={DATA_CONFIG.sequence_length})...")
    all_data = build_sequences(df, seq_len=DATA_CONFIG.sequence_length)
    print(f"  Built {len(all_data['player_ids']):,} total sequences")

    print("Splitting into train / val / test...")
    train, val, test = split_sequences(
        all_data,
        val_start=DATA_CONFIG.val_season_start,
        test_start=DATA_CONFIG.test_season_start,
    )

    print(f"  Train: {len(train['player_ids']):,}")
    print(f"  Val:   {len(val['player_ids']):,}")
    print(f"  Test:  {len(test['player_ids']):,}")

    save_split(train, DATA_CONFIG.train_sequences_path)
    save_split(val, DATA_CONFIG.val_sequences_path)
    save_split(test, DATA_CONFIG.test_sequences_path)

    return train, val, test


if __name__ == "__main__":
    run()
