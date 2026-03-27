"""
Training loop for NBA-GPT.
Mixed precision (fp16), checkpointing, early stopping.
Loss computed on normalized targets (equal weight per stat).
MAE reported in raw stat space.
"""
import json
import time
import warnings
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from nba_gpt.config import TrainConfig, TRAIN_CONFIG, TARGET_STATS, DATA_CONFIG
from nba_gpt.data.dataset import load_norm_stats
from nba_gpt.model.transformer import NBAGPTModel
from nba_gpt.training.scheduler import cosine_warmup_scheduler


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


class Trainer:
    def __init__(
        self,
        model: NBAGPTModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig = TRAIN_CONFIG,
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.scheduler = cosine_warmup_scheduler(
                self.optimizer, cfg.warmup_steps, total_steps
            )

        self.scaler = GradScaler("cuda")
        self.loss_fn = nn.MSELoss()

        # Delta targets are zero-centered; only need per-stat std for equal-weight loss.
        # Std of delta ≈ std of raw stat (mean shift cancels out).
        norm_stats = load_norm_stats()
        target_stds = torch.tensor(
            [norm_stats[s]["std"] for s in TARGET_STATS], dtype=torch.float32
        ).to(self.device)
        self.target_stds = target_stds

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        player_id = batch["player_id"].to(self.device)
        era_id = batch["era_id"].to(self.device)
        input_seq = batch["input_seq"].to(self.device)
        next_game_ctx = batch["next_game_ctx"].to(self.device)
        return self.model(player_id, era_id, input_seq, next_game_ctx)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch in self.train_loader:
            target = batch["target"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                pred = self._forward_batch(batch)
                # Scale by stat std for equal-weight loss (deltas are already zero-centered)
                pred_scaled = pred / self.target_stds
                target_scaled = target / self.target_stds
                loss = self.loss_fn(pred_scaled, target_scaled)

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
        all_preds = []
        all_targets = []

        all_baselines = []
        for batch in self.val_loader:
            target = batch["target"].to(self.device)
            baseline = batch["target_baseline"].to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                pred = self._forward_batch(batch)
                pred_scaled = pred / self.target_stds
                target_scaled = target / self.target_stds
                loss = self.loss_fn(pred_scaled, target_scaled)
            total_loss += loss.item()
            all_preds.append(pred.float().cpu())
            all_targets.append(target.float().cpu())
            all_baselines.append(baseline.float().cpu())

        preds = torch.cat(all_preds)       # predicted deltas
        targets = torch.cat(all_targets)   # actual deltas
        baselines = torch.cat(all_baselines)  # window means

        # MAE in original stat space: reconstruct actual = delta + baseline
        pred_actuals = preds + baselines
        target_actuals = targets + baselines
        mae_per_stat = {
            stat: float((pred_actuals[:, i] - target_actuals[:, i]).abs().mean())
            for i, stat in enumerate(TARGET_STATS)
        }

        return total_loss / len(self.val_loader), mae_per_stat

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

        history = {"train_loss": [], "val_loss": [], "val_mae": []}

        for epoch in range(self.epoch + 1, self.cfg.epochs + 1):
            self.epoch = epoch
            t0 = time.time()

            train_loss = self._train_epoch()
            val_loss, mae = self._val_epoch()

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
            history["val_mae"].append(mae)

            mae_str = " | ".join(f"{s[:3]}={v:.2f}" for s, v in mae.items())
            best_mark = " *" if is_best else ""
            print(
                f"Epoch {epoch:3d}/{self.cfg.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f}{best_mark} | "
                f"{mae_str} | "
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
