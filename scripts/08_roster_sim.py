"""
Step 8: Roster simulation — what happens when a player goes down?

Examples:

  # Warriors without Steph (home game)
  python scripts/08_roster_sim.py --team "Warriors" --absent "Curry" --home

  # Celtics without Tatum on the road
  python scripts/08_roster_sim.py --team "Celtics" --absent "Tatum" --away

"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.simulation.roster import RosterScenario, simulate_roster


def main():
    parser = argparse.ArgumentParser(description="NBA-Simulverse roster simulation")
    parser.add_argument("--team", required=True, help="Team name (partial match ok)")
    parser.add_argument("--absent", required=True, help="Absent player name")
    parser.add_argument("--games", type=int, default=1, help="Number of games missed")
    parser.add_argument("--home", action="store_true", default=None)
    parser.add_argument("--away", action="store_true", default=None)
    parser.add_argument("--rest", type=float, default=None, help="Days rest")
    parser.add_argument("--samples", type=int, default=200, help="MC samples per player")
    args = parser.parse_args()

    home = None
    if args.home:
        home = True
    elif args.away:
        home = False

    scenario = RosterScenario(
        absent_player=args.absent,
        n_games=args.games,
        home=home,
        rest_days=args.rest,
    )

    result = simulate_roster(args.team, scenario, n_samples=args.samples)
    print(result.summary())


if __name__ == "__main__":
    main()
