"""Step 4: Train NBA-GPT transformer model."""
import argparse
import dataclasses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.config import MODEL_CONFIG, TRAIN_CONFIG
from nba_gpt.data.dataset import create_dataloaders, load_norm_stats
from nba_gpt.model.transformer import NBAGPTModel
from nba_gpt.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train NBA-GPT")
    p.add_argument("--epochs", type=int, default=TRAIN_CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=TRAIN_CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=TRAIN_CONFIG.lr)
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading data...")
    norm_stats = load_norm_stats()
    train_loader, val_loader, _ = create_dataloaders(
        norm_stats=norm_stats,
        batch_size=args.batch_size,
    )
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")

    print("\nBuilding model...")
    model = NBAGPTModel(MODEL_CONFIG)
    print(f"  Parameters: {model.count_parameters():,}")

    # Override train config with CLI args
    base = {f.name: getattr(TRAIN_CONFIG, f.name) for f in dataclasses.fields(TRAIN_CONFIG)}
    cfg = TRAIN_CONFIG.__class__(**{**base, "epochs": args.epochs, "lr": args.lr})
    trainer = Trainer(model, train_loader, val_loader, cfg=cfg)

    if args.resume:
        latest = TRAIN_CONFIG.checkpoint_dir / "latest.pt"
        if latest.exists():
            trainer.load_checkpoint(latest)

    print("\nStarting training...")
    history = trainer.train()
    print("\nTraining complete. Best val loss:", min(history["val_loss"]))
