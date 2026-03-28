"""
Roster-level simulation — Phase 3.

Simulates an entire team's box score under counterfactual conditions.
Key use case: "What happens to [Team] if [Player X] misses N games?"

Approach:
1. Identify the absent player's average minutes/usage contribution.
2. Redistribute those minutes proportionally to remaining players.
3. Run individual player simulations with updated minute projections.
4. Aggregate to team box score.

This captures the first-order cascade effect: when Steph is out,
Klay absorbs more usage → his efficiency context changes → the model
sees different conditions and produces different predicted distributions.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

from nba_gpt.config import DATA_CONFIG, TARGET_STATS
from nba_gpt.simulation.engine import ScenarioOverride, simulate, SimulationResult


@dataclass
class RosterScenario:
    """
    Describes a counterfactual roster scenario.

    absent_player:   Name of the player who is out.
    n_games:         How many games they miss (used for multi-game projection).
    home:            True/False/None for home/away/keep-as-is.
    rest_days:       Override rest days for all active players.
    """
    absent_player: str
    n_games: int = 1
    home: bool | None = None
    rest_days: float | None = None


@dataclass
class RosterSimResult:
    team_name: str
    scenario: RosterScenario
    player_results: dict[str, SimulationResult] = field(default_factory=dict)
    absent_baseline: dict[str, float] = field(default_factory=dict)  # absent player's avg

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"Team simulation: {self.team_name}",
            f"Absent: {self.scenario.absent_player} ({self.scenario.n_games} games)",
            f"{'='*70}",
            f"{'Player':<22} " + "  ".join(f"{s[:5]:>7}" for s in TARGET_STATS),
            f"{'-'*70}",
        ]

        team_totals = {s: 0.0 for s in TARGET_STATS}
        for name, res in self.player_results.items():
            row = f"{name[:22]:<22} "
            row += "  ".join(f"{res.mean[s]:>7.1f}" for s in TARGET_STATS)
            lines.append(row)
            for s in TARGET_STATS:
                team_totals[s] += res.mean[s]

        lines.append("-" * 70)
        totals_row = f"{'TEAM TOTAL':<22} "
        totals_row += "  ".join(f"{team_totals[s]:>7.1f}" for s in TARGET_STATS)
        lines.append(totals_row)

        if self.absent_baseline:
            lines.append("")
            lines.append(f"  (Without {self.scenario.absent_player}'s avg: "
                         + ", ".join(f"{s[:3]}={self.absent_baseline[s]:.1f}"
                                     for s in TARGET_STATS) + ")")

        lines.append("=" * 70)
        return "\n".join(lines)


def _get_team_roster(
    team_name: str,
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    min_games: int = 20,
    recent_games: int = 10,
) -> list[dict]:
    """
    Return active roster members for a team, sorted by recent minutes played.
    Each entry: {name, personId, avg_minutes, player_df_rows}
    """
    # Find team rows in raw data
    team_mask = (
        raw_df["playerteamName"].str.contains(team_name, case=False, na=False) |
        raw_df["playerteamCity"].str.contains(team_name, case=False, na=False)
    )
    team_raw = raw_df[team_mask]

    if team_raw.empty:
        raise ValueError(f"No team found matching '{team_name}'")

    # Get player IDs who played recently for this team
    team_raw = team_raw.sort_values("gameDateTimeEst")
    recent = team_raw.tail(recent_games * 15)  # approx 15-man roster × recent games
    active_ids = recent["personId"].value_counts()
    active_ids = active_ids[active_ids >= 3].index.tolist()  # played at least 3 recent games

    # Build roster entries
    roster = []
    for pid in active_ids:
        pid_rows = team_raw[team_raw["personId"] == pid]
        if len(pid_rows) < min_games:
            continue
        if "firstName" in pid_rows.columns:
            fname = str(pid_rows["firstName"].iloc[-1] or "").strip()
            lname = str(pid_rows["lastName"].iloc[-1] or "").strip()
            name = f"{fname} {lname}".strip() or str(pid)
        else:
            name = str(pid)
        avg_min = pd.to_numeric(pid_rows["numMinutes"], errors="coerce").dropna().tail(10).mean()
        if avg_min < 5:  # skip deep bench (<5 min/game)
            continue
        roster.append({"name": name, "personId": int(pid), "avg_minutes": float(avg_min)})

    return sorted(roster, key=lambda x: x["avg_minutes"], reverse=True)


def simulate_roster(
    team_name: str,
    scenario: RosterScenario,
    n_samples: int = 200,
    checkpoint_path: Path | None = None,
) -> RosterSimResult:
    """
    Simulate a team's box score with one player absent.

    The absent player's minutes are redistributed proportionally
    to remaining players (usage redistribution model).
    """
    features_path = DATA_CONFIG.player_features_path
    df = pd.read_parquet(features_path)

    # Load raw data for roster identification
    raw_path = DATA_CONFIG.raw_dir / "PlayerStatistics.csv"
    raw_df = pd.read_csv(raw_path, low_memory=False)
    raw_df["gameDateTimeEst"] = pd.to_datetime(raw_df["gameDateTimeEst"], errors="coerce")
    raw_df["personId"] = pd.to_numeric(raw_df["personId"], errors="coerce")
    raw_df = raw_df.dropna(subset=["personId", "gameDateTimeEst"])
    raw_df["personId"] = raw_df["personId"].astype(int)

    print(f"Building roster for {team_name}...")
    roster = _get_team_roster(team_name, df, raw_df)

    if not roster:
        raise ValueError(f"Could not build roster for '{team_name}'")

    print(f"  Found {len(roster)} active players")

    # Find absent player
    absent_lower = scenario.absent_player.lower()
    absent_entry = next(
        (p for p in roster if absent_lower in p["name"].lower()), None
    )

    if absent_entry is None:
        raise ValueError(
            f"'{scenario.absent_player}' not found on {team_name} roster. "
            f"Roster: {[p['name'] for p in roster]}"
        )

    absent_minutes = absent_entry["avg_minutes"]
    active_roster = [p for p in roster if p["personId"] != absent_entry["personId"]]

    # Redistribute minutes proportionally
    total_active_minutes = sum(p["avg_minutes"] for p in active_roster)
    for player in active_roster:
        share = player["avg_minutes"] / total_active_minutes
        player["sim_minutes"] = player["avg_minutes"] + absent_minutes * share

    print(f"  Absent: {absent_entry['name']} ({absent_minutes:.1f} min/g)")
    print(f"  Redistributing {absent_minutes:.1f} min across {len(active_roster)} players")

    # Get absent player's baseline stats (for comparison)
    absent_result = None
    try:
        absent_result = simulate(
            absent_entry["name"],
            ScenarioOverride(home=scenario.home, rest_days=scenario.rest_days),
            n_samples=n_samples,
            checkpoint_path=checkpoint_path,
        )
    except Exception:
        pass

    # Simulate each active player with redistributed minutes
    player_results = {}
    for player in active_roster:
        print(f"  Simulating {player['name']} ({player['sim_minutes']:.1f} min)...")
        override = ScenarioOverride(
            minutes=player["sim_minutes"],
            home=scenario.home,
            rest_days=scenario.rest_days,
        )
        try:
            result = simulate(
                player["name"],
                override,
                n_samples=n_samples,
                checkpoint_path=checkpoint_path,
            )
            player_results[player["name"]] = result
        except Exception as e:
            print(f"    Skipped ({e})")
            continue

    result = RosterSimResult(
        team_name=team_name,
        scenario=scenario,
        player_results=player_results,
        absent_baseline=absent_result.mean if absent_result else {},
    )

    return result
