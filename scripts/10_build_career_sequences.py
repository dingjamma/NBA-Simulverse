"""
Step 10: Aggregate game-level NBA data into player-season sequences for career arc modeling.

Usage:
    python scripts/10_build_career_sequences.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from career_arc.data.build_career_sequences import build_career_sequences


if __name__ == "__main__":
    print("Building career arc sequences from raw game data...")
    summary = build_career_sequences()

    if summary.get("n_sequences", 0) == 0:
        print("\nERROR: No sequences were generated. Check that data/raw/ contains:")
        print("  - PlayerStatistics.csv")
        print("  - Players.csv")
        sys.exit(1)

    print(f"\nDone.")
    print(f"  Sequences:       {summary['n_sequences']:,}")
    print(f"  Players:         {summary['n_players']:,}")
    print(f"  Sequences file:  {summary['sequences_path']}")
    print(f"  Norm stats file: {summary['norm_stats_path']}")
