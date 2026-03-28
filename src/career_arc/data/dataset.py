"""
PyTorch Dataset and DataLoader factory for career arc sequences.
Applies z-score normalization to stat and context features.
"""
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Any

from career_arc.config import (
    PROCESSED_DIR,
    CAREER_STAT_FEATURES, CAREER_CONTEXT_FEATURES,
    CAREER_MODEL_CONFIG,
)

CAREER_NORM_STATS_PATH = PROCESSED_DIR / "career_norm_stats.json"
CAREER_SEQUENCES_PATH = PROCESSED_DIR / "career_sequences.npz"


def load_career_norm_stats(path: Path | None = None) -> dict[str, Any]:
    path = path or CAREER_NORM_STATS_PATH
    with open(path) as f:
        return json.load(f)


class CareerDataset(Dataset):
    def __init__(self, npz_path: Path, norm_stats: dict[str, Any]):
        data = np.load(npz_path)

        # Raw tensors
        self.stat_seqs = torch.from_numpy(data["stat_seqs"]).float()            # (N, seq_len, 10)
        self.ctx_seqs = torch.from_numpy(data["ctx_seqs"]).float()              # (N, seq_len, 4)
        self.team_ids = torch.from_numpy(data["team_ids"]).long()                # (N, seq_len)
        self.era_ids = torch.from_numpy(data["era_ids"]).long()                  # (N, seq_len)
        self.targets = torch.from_numpy(data["targets"]).float()                 # (N, 10)
        self.breakout_labels = torch.from_numpy(data["breakout_labels"]).float() # (N,)
        self.decline_labels = torch.from_numpy(data["decline_labels"]).float()   # (N,)
        self.injury_labels = torch.from_numpy(data["injury_labels"]).float()     # (N,)
        self.player_ids = torch.from_numpy(data["player_ids"]).long()            # (N,)

        # Normalize stat sequences: (N, seq_len, 10)
        stat_means = torch.tensor(
            [norm_stats[f]["mean"] for f in CAREER_STAT_FEATURES], dtype=torch.float32
        )
        stat_stds = torch.tensor(
            [norm_stats[f]["std"] for f in CAREER_STAT_FEATURES], dtype=torch.float32
        )
        self.stat_seqs = (
            (self.stat_seqs - stat_means.unsqueeze(0).unsqueeze(0))
            / stat_stds.unsqueeze(0).unsqueeze(0)
        )

        # Normalize context sequences: (N, seq_len, 4)
        ctx_means = torch.tensor(
            [norm_stats[f]["mean"] for f in CAREER_CONTEXT_FEATURES], dtype=torch.float32
        )
        ctx_stds = torch.tensor(
            [norm_stats[f]["std"] for f in CAREER_CONTEXT_FEATURES], dtype=torch.float32
        )
        self.ctx_seqs = (
            (self.ctx_seqs - ctx_means.unsqueeze(0).unsqueeze(0))
            / ctx_stds.unsqueeze(0).unsqueeze(0)
        )

        # Keep unscaled targets for MAE reporting; normalized targets for loss
        self.targets_raw = self.targets.clone()
        self.targets = (self.targets - stat_means) / stat_stds

    def __len__(self) -> int:
        return len(self.player_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "stat_seq": self.stat_seqs[idx],              # (seq_len, 10)
            "ctx_seq": self.ctx_seqs[idx],                # (seq_len, 4)
            "team_ids": self.team_ids[idx],               # (seq_len,)
            "era_ids": self.era_ids[idx],                 # (seq_len,)
            "target": self.targets[idx],                  # (10,) normalized
            "target_raw": self.targets_raw[idx],          # (10,) raw stat values
            "breakout_label": self.breakout_labels[idx],  # scalar
            "decline_label": self.decline_labels[idx],    # scalar
            "injury_label": self.injury_labels[idx],      # scalar
            "player_id": self.player_ids[idx],            # scalar
        }


def create_career_dataloaders(
    norm_stats: dict[str, Any] | None = None,
    sequences_path: Path | None = None,
    batch_size: int = CAREER_MODEL_CONFIG.seq_len,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the full career sequences dataset and split into train/val/test.

    Splits are done by random stratification rather than time-based splits,
    since career sequences already span the full historical range.
    """
    sequences_path = sequences_path or CAREER_SEQUENCES_PATH
    if norm_stats is None:
        norm_stats = load_career_norm_stats()

    full_ds = CareerDataset(sequences_path, norm_stats)
    n = len(full_ds)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train : n_train + n_val].tolist()
    test_idx = indices[n_train + n_val :].tolist()

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    pin = torch.cuda.is_available()
    persist = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=persist,
    )

    return train_loader, val_loader, test_loader
