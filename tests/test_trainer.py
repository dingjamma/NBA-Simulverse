"""
Test training loop mechanics on a tiny synthetic dataset.
"""
import tempfile
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from nba_gpt.config import MODEL_CONFIG, TRAIN_CONFIG, INPUT_FEATURES, TARGET_STATS, N_TARGETS
from nba_gpt.model.transformer import NBAGPTModel
from nba_gpt.training.trainer import Trainer
from nba_gpt.training.scheduler import cosine_warmup_scheduler
import torch.optim as optim


def _tiny_loader(n_samples: int = 64, batch_size: int = 16) -> DataLoader:
    cfg = MODEL_CONFIG
    player_ids = torch.randint(0, 50, (n_samples,))
    era_ids = torch.randint(0, cfg.n_eras, (n_samples,))
    inputs = torch.randn(n_samples, cfg.sequence_length, cfg.n_input_features)
    targets = torch.rand(n_samples, N_TARGETS) * 30  # stats in 0-30 range

    ds = TensorDataset(player_ids, era_ids, inputs, targets)

    def collate(batch):
        pids, eids, inps, tgts = zip(*batch)
        return {
            "player_id": torch.stack(pids),
            "era_id": torch.stack(eids),
            "input_seq": torch.stack(inps),
            "target": torch.stack(tgts),
        }

    return DataLoader(ds, batch_size=batch_size, collate_fn=collate, shuffle=True)


def test_trainer_one_step_no_crash():
    model = NBAGPTModel(MODEL_CONFIG)
    loader = _tiny_loader()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_override = TRAIN_CONFIG.__class__(
            **{**TRAIN_CONFIG.__dict__, "checkpoint_dir": Path(tmpdir), "epochs": 1}
        )
        trainer = Trainer(model, loader, loader, cfg=cfg_override)
        history = trainer.train()
    assert len(history["train_loss"]) == 1


def test_trainer_loss_decreases():
    """Model should be able to overfit a tiny dataset."""
    model = NBAGPTModel(MODEL_CONFIG)
    loader = _tiny_loader(n_samples=32, batch_size=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_override = TRAIN_CONFIG.__class__(
            **{**TRAIN_CONFIG.__dict__,
               "checkpoint_dir": Path(tmpdir),
               "epochs": 10,
               "patience": 10}
        )
        trainer = Trainer(model, loader, loader, cfg=cfg_override)
        history = trainer.train()

    first_loss = history["train_loss"][0]
    last_loss = history["train_loss"][-1]
    assert last_loss < first_loss, f"Loss did not decrease: {first_loss:.4f} -> {last_loss:.4f}"


def test_checkpoint_round_trip():
    model = NBAGPTModel(MODEL_CONFIG)
    loader = _tiny_loader()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cfg_override = TRAIN_CONFIG.__class__(
            **{**TRAIN_CONFIG.__dict__, "checkpoint_dir": tmpdir, "epochs": 2}
        )
        trainer = Trainer(model, loader, loader, cfg=cfg_override)
        trainer.train()

        # Load checkpoint into new model
        model2 = NBAGPTModel(MODEL_CONFIG)
        trainer2 = Trainer(model2, loader, loader, cfg=cfg_override)
        trainer2.load_checkpoint(tmpdir / "latest.pt")

        # State dicts should match
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), model2.state_dict().items()
        ):
            assert torch.allclose(v1.float(), v2.float()), f"Mismatch at {k1}"


def test_cosine_warmup_scheduler():
    optimizer = optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=10, total_steps=100)

    # During warmup: LR should increase
    lrs = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    assert lrs[-1] > lrs[0]

    # After warmup: LR should decrease
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    lr_mid = scheduler.get_last_lr()[0]

    for _ in range(40):
        optimizer.step()
        scheduler.step()
    lr_end = scheduler.get_last_lr()[0]
    assert lr_end < lr_mid
