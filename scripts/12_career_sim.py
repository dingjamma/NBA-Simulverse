"""
Step 12: Simulate a player's career trajectory using the trained CareerArcModel.

Examples:
    python scripts/12_career_sim.py --player "LeBron James" --seasons 5
    python scripts/12_career_sim.py --player "Curry" --seasons 3
    python scripts/12_career_sim.py --player "Luka" --seasons 5 --team-id 10
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from career_arc.simulation.career_sim import simulate_career


def parse_args():
    p = argparse.ArgumentParser(description="Career arc trajectory simulation")
    p.add_argument("--player", required=True, help="Player name (partial match ok)")
    p.add_argument("--seasons", type=int, default=5, help="Number of seasons to project")
    p.add_argument(
        "--team-id", type=int, default=None,
        help="Override team_id for all projected seasons (int, 0=unknown)"
    )
    p.add_argument(
        "--age-offset", type=float, default=None,
        help="Apply a constant age offset to projected seasons (e.g. +1 for injury year)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    scenario = {}
    if args.team_id is not None:
        scenario["team_id"] = args.team_id
    if args.age_offset is not None:
        scenario["age_offset"] = args.age_offset

    print(f"Simulating career for '{args.player}' ({args.seasons} seasons)...")

    try:
        result = simulate_career(
            player_name=args.player,
            n_seasons=args.seasons,
            scenario=scenario if scenario else None,
        )
        print(result.summary())
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
