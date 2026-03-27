"""
PyTorch Dataset and DataLoader factory for NBA-GPT sequences.
Applies z-score normalization to inputs; targets are deltas (actual - window_mean).
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Any

from nba_gpt.config import DATA_CONFIG, INPUT_FEATURES, TARGET_STATS, CONTEXT_FEATURES


class NBASequenceDataset(Dataset):
    def __init__(self, npz_path: Path, norm_stats: dict[str, Any]):
        data = np.load(npz_path)
        self.player_ids = torch.from_numpy(data["player_ids"]).long()
        self.era_ids = torch.from_numpy(data["era_ids"]).long()
        self.inputs = torch.from_numpy(data["inputs"]).float()            # (N, seq_len, n_feat)
        self.targets = torch.from_numpy(data["targets"]).float()          # (N, n_targets) deltas
        self.target_baselines = torch.from_numpy(data["target_baselines"]).float()  # (N, n_targets)
        self.next_game_ctx = torch.from_numpy(data["next_game_ctx"]).float()  # (N, n_context)

        # Normalize sequence inputs
        means = torch.tensor([norm_stats[f]["mean"] for f in INPUT_FEATURES], dtype=torch.float32)
        stds = torch.tensor([norm_stats[f]["std"] for f in INPUT_FEATURES], dtype=torch.float32)
        self.inputs = (self.inputs - means.unsqueeze(0).unsqueeze(0)) / stds.unsqueeze(0).unsqueeze(0)

        # Normalize next-game context features
        ctx_means = torch.tensor([norm_stats[f]["mean"] for f in CONTEXT_FEATURES], dtype=torch.float32)
        ctx_stds = torch.tensor([norm_stats[f]["std"] for f in CONTEXT_FEATURES], dtype=torch.float32)
        self.next_game_ctx = (self.next_game_ctx - ctx_means) / ctx_stds

    def __len__(self) -> int:
        return len(self.player_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "player_id": self.player_ids[idx],
            "era_id": self.era_ids[idx],
            "input_seq": self.inputs[idx],                   # (seq_len, n_features)
            "target": self.targets[idx],                     # (n_targets,) delta
            "target_baseline": self.target_baselines[idx],   # (n_targets,) window mean
            "next_game_ctx": self.next_game_ctx[idx],        # (n_context,) next-game conditions
        }


def load_norm_stats(path: Path | None = None) -> dict[str, Any]:
    path = path or DATA_CONFIG.norm_stats_path
    with open(path) as f:
        return json.load(f)


def create_dataloaders(
    norm_stats: dict[str, Any] | None = None,
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_path = train_path or DATA_CONFIG.train_sequences_path
    val_path = val_path or DATA_CONFIG.val_sequences_path
    test_path = test_path or DATA_CONFIG.test_sequences_path

    if norm_stats is None:
        norm_stats = load_norm_stats()

    train_ds = NBASequenceDataset(train_path, norm_stats)
    val_ds = NBASequenceDataset(val_path, norm_stats)
    test_ds = NBASequenceDataset(test_path, norm_stats)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader
