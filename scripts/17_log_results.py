"""
Mark yesterday's picks as HIT or MISS using ESPN box scores.

Run the morning after games are played:
  python scripts/17_log_results.py           # grades yesterday's picks
  python scripts/17_log_results.py --date 2026-03-29
"""
import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

LOGS_DIR  = Path(__file__).parent.parent / "logs"
PICKS_CSV = LOGS_DIR / "picks.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_HDR  = {"User-Agent": "Mozilla/5.0"}

STAT_MAP = {
    "Points":          "points",
    "Rebounds":        "rebounds",
    "Assists":         "assists",
    "3-PT Made":       "threePointFieldGoalsMade",
    "3-Pointers Made": "threePointFieldGoalsMade",
}


def get_box_scores(date_str: str) -> dict[str, dict]:
    """
    Returns {player_full_name_lower: {stat_key: value}} for a game date YYYYMMDD.
    """
    r = requests.get(
        f"{ESPN_BASE}/scoreboard",
        params={"dates": date_str},
        headers=ESPN_HDR,
        timeout=15,
    )
    r.raise_for_status()
    events = r.json().get("events", [])

    player_stats: dict[str, dict] = {}

    for ev in events:
        event_id = ev["id"]
        # Fetch box score
        try:
            br = requests.get(
                f"{ESPN_BASE}/summary",
                params={"event": event_id},
                headers=ESPN_HDR,
                timeout=15,
            )
            br.raise_for_status()
            summary = br.json()
        except Exception:
            continue

        for team_box in summary.get("boxscore", {}).get("players", []):
            for stat_group in team_box.get("statistics", []):
                keys = stat_group.get("keys", [])
                for athlete in stat_group.get("athletes", []):
                    name = athlete.get("athlete", {}).get("displayName", "").lower()
                    stats_raw = athlete.get("stats", [])
                    if not name or not stats_raw:
                        continue
                    stat_dict = dict(zip(keys, stats_raw))
                    player_stats[name] = stat_dict

    return player_stats


def grade_picks(date_str: str) -> None:
    """date_str: YYYY-MM-DD"""
    if not PICKS_CSV.exists():
        print("No picks log found. Run 16_daily_picks.py first.")
        return

    espn_date = date_str.replace("-", "")

    print(f"\nFetching box scores for {date_str}...")
    box = get_box_scores(espn_date)
    print(f"  Found {len(box)} players in box scores")

    rows = []
    with open(PICKS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    updated = 0
    for row in rows:
        if row["date"] != date_str or row["result"]:
            continue  # skip already graded or wrong date

        player_lower = row["player"].lower()
        stat         = row["stat"]
        line         = float(row["line"])
        direction    = row["direction"]

        # fuzzy name match
        match = next(
            (k for k in box if all(part in k for part in player_lower.split()[:2])),
            None,
        )
        if not match:
            print(f"  {row['player']}: not found in box score")
            row["result"] = "NO_DATA"
            continue

        espn_key = STAT_MAP.get(stat)
        actual_str = box[match].get(espn_key, "")
        try:
            actual = float(actual_str)
        except (ValueError, TypeError):
            row["result"] = "NO_DATA"
            continue

        hit = (direction == "OVER" and actual > line) or (direction == "UNDER" and actual < line)
        row["result"] = "HIT" if hit else "MISS"
        row["actual"] = actual
        updated += 1

        flag = "HIT" if hit else "MISS"
        print(f"  {row['player']:<24} {stat:<12} {direction} {line} | actual={actual} -> {flag}")

    # Write back
    with open(PICKS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated {updated} picks in {PICKS_CSV}")
    print_record()


def print_record() -> None:
    if not PICKS_CSV.exists():
        return
    with open(PICKS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    graded = [r for r in rows if r["result"] in ("HIT", "MISS")]
    if not graded:
        print("No graded picks yet.")
        return

    hits   = sum(1 for r in graded if r["result"] == "HIT")
    total  = len(graded)
    print(f"\n{'='*40}")
    print(f"RECORD: {hits}/{total} ({100*hits/total:.0f}%)")

    # by stat
    from collections import defaultdict
    by_stat: dict[str, list] = defaultdict(list)
    for r in graded:
        by_stat[r["stat"]].append(r["result"] == "HIT")
    for stat, results in sorted(by_stat.items()):
        h = sum(results); t = len(results)
        print(f"  {stat:<14} {h}/{t} ({100*h/t:.0f}%)")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None,
                        help="Game date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--record", action="store_true", help="Just print record")
    args = parser.parse_args()

    if args.record:
        print_record()
        return

    if args.date:
        date_str = args.date
    else:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date_str  = yesterday.strftime("%Y-%m-%d")

    grade_picks(date_str)


if __name__ == "__main__":
    main()
