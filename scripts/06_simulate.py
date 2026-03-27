"""
Step 6: Run a counterfactual simulation.

Examples:

  # Baseline prediction for Luka
  python scripts/06_simulate.py --player "Luka"

  # What if Luka plays only 28 minutes (load management)?
  python scripts/06_simulate.py --player "Luka" --minutes 28

  # What if Steph plays 40 minutes on the road against a tough defense?
  python scripts/06_simulate.py --player "Curry" --minutes 40 --away --opp-defense 98

  # Compare: 32 min vs 40 min
  python scripts/06_simulate.py --player "Luka" --compare --minutes-a 32 --minutes-b 40

"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.simulation.engine import ScenarioOverride, simulate, compare_scenarios


def main():
    parser = argparse.ArgumentParser(description="NBA-Simulverse counterfactual engine")
    parser.add_argument("--player", required=True, help="Player name (partial match ok)")
    parser.add_argument("--samples", type=int, default=500, help="MC dropout samples")

    # Scenario controls
    parser.add_argument("--minutes", type=float, default=None, help="Projected minutes")
    parser.add_argument("--home", action="store_true", default=None, help="Home game")
    parser.add_argument("--away", action="store_true", default=None, help="Away game")
    parser.add_argument("--rest", type=float, default=None, help="Days rest")
    parser.add_argument("--opp-defense", type=float, default=None,
                        help="Opponent pts allowed rolling avg (lower = tougher defense)")
    parser.add_argument("--pace", type=float, default=None, help="Game pace (total FGA)")

    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare two scenarios (varies --minutes)")
    parser.add_argument("--minutes-a", type=float, default=32)
    parser.add_argument("--minutes-b", type=float, default=40)

    args = parser.parse_args()

    # Resolve home/away
    home = None
    if args.home:
        home = True
    elif args.away:
        home = False

    if args.compare:
        scenario_a = ScenarioOverride(
            minutes=args.minutes_a,
            home=home,
            rest_days=args.rest,
            opp_pts_allowed=args.opp_defense,
            game_pace=args.pace,
        )
        scenario_b = ScenarioOverride(
            minutes=args.minutes_b,
            home=home,
            rest_days=args.rest,
            opp_pts_allowed=args.opp_defense,
            game_pace=args.pace,
        )
        compare_scenarios(
            args.player,
            scenario_a,
            scenario_b,
            labels=(f"{args.minutes_a} min", f"{args.minutes_b} min"),
            n_samples=args.samples,
        )
    else:
        override = ScenarioOverride(
            minutes=args.minutes,
            home=home,
            rest_days=args.rest,
            opp_pts_allowed=args.opp_defense,
            game_pace=args.pace,
        )
        result = simulate(args.player, override, n_samples=args.samples)
        print(result.summary())


if __name__ == "__main__":
    main()
