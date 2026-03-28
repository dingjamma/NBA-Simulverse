"""
Step 11: Train the CareerArcModel on player-season sequences.

Usage:
    python scripts/11_train_career.py
    python scripts/11_train_career.py --epochs 50 --batch-size 128
    python scripts/11_train_career.py --resume
"""
import argparse
import dataclasses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from career_arc.config import CAREER_MODEL_CONFIG, CAREER_TRAIN_CONFIG
from career_arc.data.dataset import create_career_dataloaders, load_career_norm_stats
from career_arc.model import CareerArcModel
from career_arc.training.trainer import CareerTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train CareerArcModel")
    p.add_argument("--epochs", type=int, default=CAREER_TRAIN_CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=CAREER_TRAIN_CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=CAREER_TRAIN_CONFIG.lr)
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sequences_path = CAREER_TRAIN_CONFIG.checkpoint_dir.parent.parent / "data" / "processed" / "career_sequences.npz"
    # Canonical path
    from career_arc.config import PROCESSED_DIR
    sequences_path = PROCESSED_DIR / "career_sequences.npz"

    if not sequences_path.exists():
        print(f"ERROR: {sequences_path} not found. Run scripts/10_build_career_sequences.py first.")
        sys.exit(1)

    print("Loading data...")
    norm_stats = load_career_norm_stats()
    train_loader, val_loader, _ = create_career_dataloaders(
        norm_stats=norm_stats,
        sequences_path=sequences_path,
        batch_size=args.batch_size,
        num_workers=CAREER_TRAIN_CONFIG.num_workers,
    )
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")

    print("\nBuilding model...")
    model = CareerArcModel(CAREER_MODEL_CONFIG)
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  d_model={CAREER_MODEL_CONFIG.d_model}, "
          f"n_layers={CAREER_MODEL_CONFIG.n_layers}, "
          f"n_heads={CAREER_MODEL_CONFIG.n_heads}, "
          f"seq_len={CAREER_MODEL_CONFIG.seq_len}")

    # Override train config with CLI args
    base = {f.name: getattr(CAREER_TRAIN_CONFIG, f.name) for f in dataclasses.fields(CAREER_TRAIN_CONFIG)}
    cfg = CAREER_TRAIN_CONFIG.__class__(**{**base, "epochs": args.epochs, "lr": args.lr})

    trainer = CareerTrainer(model, train_loader, val_loader, cfg=cfg)

    if args.resume:
        latest = cfg.checkpoint_dir / "latest.pt"
        if latest.exists():
            trainer.load_checkpoint(latest)
        else:
            print("  No latest checkpoint found, starting from scratch.")

    print("\nStarting training...")
    history = trainer.train()
    print(f"\nTraining complete.")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")
    print(f"  Checkpoint:    {cfg.checkpoint_dir / 'best.pt'}")
