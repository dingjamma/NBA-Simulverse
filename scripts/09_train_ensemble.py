"""
Step 9: Train an ensemble of NBA-GPT models.

Trains N models with different random seeds. Each model is identical
in architecture but initialized differently — the spread across their
predictions is real uncertainty, not MC dropout noise.

Checkpoints saved to: checkpoints/ensemble/seed_{N}/best.pt

Usage:
  python scripts/09_train_ensemble.py --n-models 5
  python scripts/09_train_ensemble.py --n-models 5 --seeds 10 20 30 40 50
"""
import argparse
import dataclasses
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.config import MODEL_CONFIG, TRAIN_CONFIG, TrainConfig
from nba_gpt.data.dataset import create_dataloaders, load_norm_stats
from nba_gpt.model.transformer import NBAGPTModel
from nba_gpt.training.trainer import Trainer


ENSEMBLE_DIR = TRAIN_CONFIG.checkpoint_dir / "ensemble"


def train_one(seed: int, norm_stats: dict, train_loader, val_loader) -> float:
    ckpt_dir = ENSEMBLE_DIR / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base = {f.name: getattr(TRAIN_CONFIG, f.name) for f in dataclasses.fields(TRAIN_CONFIG)}
    cfg = TrainConfig(**{**base, "seed": seed, "checkpoint_dir": ckpt_dir})

    model = NBAGPTModel(MODEL_CONFIG)
    trainer = Trainer(model, train_loader, val_loader, cfg=cfg)
    history = trainer.train()
    best = min(history["val_loss"])
    print(f"\n[seed={seed}] Best val loss: {best:.4f} → {ckpt_dir}/best.pt\n")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seed list (overrides --n-models)")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(42, 42 + args.n_models))
    print(f"Training ensemble of {len(seeds)} models: seeds={seeds}")
    print(f"Checkpoints → {ENSEMBLE_DIR}/seed_N/best.pt\n")

    norm_stats = load_norm_stats()
    train_loader, val_loader, _ = create_dataloaders(norm_stats=norm_stats,
                                                      batch_size=TRAIN_CONFIG.batch_size)

    results = {}
    for seed in seeds:
        print(f"{'='*60}")
        print(f"Training model seed={seed}")
        print(f"{'='*60}")
        best_loss = train_one(seed, norm_stats, train_loader, val_loader)
        results[seed] = best_loss

    print("\n" + "="*50)
    print("Ensemble training complete.")
    print(f"{'Seed':>6}  {'Best Val Loss':>14}")
    print("-"*25)
    for seed, loss in results.items():
        print(f"{seed:>6}  {loss:>14.4f}")
    print(f"{'Mean':>6}  {sum(results.values())/len(results):>14.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
