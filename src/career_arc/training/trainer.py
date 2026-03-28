"""
Training loop for CareerArcModel.
Mixed loss: MSE on season-stat regression + BCE on 3 auxiliary classification heads.
Mixed precision (fp16), checkpointing, early stopping.
"""
import json
import math
import time
import warnings
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from career_arc.config import (
    CareerTrainConfig, CAREER_TRAIN_CONFIG,
    CAREER_STAT_FEATURES,
)
from career_arc.model import CareerArcModel


def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _no_decay_params(model: nn.Module) -> tuple[list, list]:
    """Split params into weight-decay and no-weight-decay groups."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return decay_params, no_decay_params


class CareerTrainer:
    def __init__(
        self,
        model: CareerArcModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: CareerTrainConfig = CAREER_TRAIN_CONFIG,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")
        self.model.to(self.device)

        # Optimizer: separate weight decay for bias / norm params
        decay_p, no_decay_p = _no_decay_params(model)
        self.optimizer = torch.optim.AdamW([
            {"params": decay_p, "weight_decay": cfg.weight_decay},
            {"params": no_decay_p, "weight_decay": 0.0},
        ], lr=cfg.lr)

        steps_per_epoch = len(train_loader)
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.scheduler = _cosine_warmup_scheduler(
                self.optimizer, warmup_steps, total_steps
            )

        self.scaler = GradScaler("cuda")
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _forward_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stat_seq = batch["stat_seq"].to(self.device)
        ctx_seq = batch["ctx_seq"].to(self.device)
        team_ids = batch["team_ids"].to(self.device)
        era_ids = batch["era_ids"].to(self.device)
        return self.model(stat_seq, ctx_seq, team_ids, era_ids)

    def _compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        stat_pred: torch.Tensor,
        breakout_prob: torch.Tensor,
        decline_prob: torch.Tensor,
        injury_prob: torch.Tensor,
    ) -> torch.Tensor:
        target = batch["target"].to(self.device)
        breakout_label = batch["breakout_label"].to(self.device)
        decline_label = batch["decline_label"].to(self.device)
        injury_label = batch["injury_label"].to(self.device)

        reg_loss = self.mse_loss(stat_pred.float(), target.float())

        # BCELoss requires float32 and must be computed outside any autocast context.
        # We cast here to float32 unconditionally; the caller is responsible for
        # ensuring this method is invoked outside torch.amp.autocast.
        aux_loss = (
            self.bce_loss(breakout_prob.float(), breakout_label.float())
            + self.bce_loss(decline_prob.float(), decline_label.float())
            + self.bce_loss(injury_prob.float(), injury_label.float())
        ) / 3.0

        return self.cfg.regression_weight * reg_loss + self.cfg.aux_weight * aux_loss

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass under autocast for reduced memory / faster matmuls.
            # Loss is computed outside autocast: BCELoss requires float32.
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                stat_pred, breakout_prob, decline_prob, injury_prob = self._forward_batch(batch)

            loss = self._compute_loss(batch, stat_pred, breakout_prob, decline_prob, injury_prob)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / n_batches

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                stat_pred, breakout_prob, decline_prob, injury_prob = self._forward_batch(batch)

            # BCE must be outside autocast; same pattern as train loop.
            loss = self._compute_loss(batch, stat_pred, breakout_prob, decline_prob, injury_prob)
            total_loss += loss.item()

        return total_loss / len(self.val_loader), {}

    def _save_checkpoint(self, is_best: bool) -> None:
        ckpt = {
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        latest_path = self.cfg.checkpoint_dir / "latest.pt"
        torch.save(ckpt, latest_path)
        if is_best:
            best_path = self.cfg.checkpoint_dir / "best.pt"
            torch.save(ckpt, best_path)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.epoch = ckpt["epoch"]
        print(f"Resumed from epoch {self.epoch}")

    def train(self) -> dict:
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)

        history: dict[str, list] = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epoch + 1, self.cfg.epochs + 1):
            self.epoch = epoch
            t0 = time.time()

            train_loss = self._train_epoch()
            val_loss, _mae = self._val_epoch()

            elapsed = time.time() - t0
            lr = self.scheduler.get_last_lr()[0]

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self._save_checkpoint(is_best)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            best_mark = " *" if is_best else ""
            print(
                f"Epoch {epoch:3d}/{self.cfg.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f}{best_mark} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )

            if self.patience_counter >= self.cfg.patience:
                print(f"Early stopping after {epoch} epochs (patience={self.cfg.patience})")
                break

        # Save training history
        history_path = self.cfg.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        return history
