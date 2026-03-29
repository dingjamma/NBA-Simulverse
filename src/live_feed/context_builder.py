"""
Build rich natural-language context blocks for Sports GPT.

For each player in today's picks, fetches:
  - Injury status
  - Recent ESPN news (up to 3 headlines)
  - Last 5 game log summary

This context is injected into the Qwen3.5 prompt so it can reason about
injuries, role changes, matchups, and recent form — not just raw model numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd

from live_feed.fetcher import fetch_player_live, find_espn_athlete_id
from live_feed.news import fetch_player_injury, fetch_player_news
from nba_gpt.config import TARGET_STATS


# Map picks.csv stat names -> DataFrame column names
_STAT_COL = {
    "Points":    "points",
    "Rebounds":  "reboundsTotal",
    "Assists":   "assists",
    "3-PT Made": "threePointersMade",
}


def _last5_lines(df: pd.DataFrame, stat_col: str) -> str:
    """Return a compact table of last-5 games for a given stat column."""
    cols = ["gameDateTimeEst", "opp_abbr", stat_col, "minutes"]
    available = [c for c in cols if c in df.columns]
    tail = df[available].tail(5).copy()

    lines = []
    for _, row in tail.iterrows():
        date_s = str(row.get("gameDateTimeEst", ""))[:10]
        opp    = row.get("opp_abbr", "???")
        val    = row.get(stat_col, "?")
        mins   = row.get("minutes", "?")
        val_s  = f"{float(val):.0f}" if val != "?" else "?"
        mins_s = f"{float(mins):.0f}m" if mins != "?" else "?"
        lines.append(f"  {date_s} vs {opp:<4} {val_s:>4} ({mins_s})")
    return "\n".join(lines) if lines else "  (no recent games)"


def build_player_context(name: str, stat: str) -> str:
    """
    Return a multi-line context block for one player+stat.

    Example output:
        [Josh Giddey — Assists]
        Injury: Active
        News: "Giddey takes on bigger playmaking role" (2026-03-25)
        Last 5 games (assists):
          2026-03-24 vs MEM    4
          2026-03-22 vs HOU    6
          ...
    """
    lines = [f"[{name} — {stat}]"]

    # 1. Injury
    try:
        espn_id, _ = find_espn_athlete_id(name)
        inj = fetch_player_injury(espn_id)
        status_str = inj["status"]
        if inj["detail"]:
            status_str += f": {inj['detail']}"
        lines.append(f"Injury: {status_str}")
    except Exception:
        espn_id = None
        lines.append("Injury: Unknown")

    # 2. News
    if espn_id:
        try:
            news = fetch_player_news(espn_id, limit=3)
            if news:
                lines.append("Recent news:")
                for item in news:
                    lines.append(f'  - "{item["headline"]}" ({item["published"]})')
                    if item["summary"]:
                        # truncate long summaries
                        summary = item["summary"][:120] + ("..." if len(item["summary"]) > 120 else "")
                        lines.append(f"    {summary}")
            else:
                lines.append("Recent news: none found")
        except Exception:
            lines.append("Recent news: fetch error")

    # 3. Last 5 game log
    stat_col = _STAT_COL.get(stat)
    if stat_col:
        try:
            df = fetch_player_live(name, n_games=10)
            if not df.empty:
                lines.append(f"Last 5 games ({stat}):")
                lines.append(_last5_lines(df, stat_col))
            else:
                lines.append("Game log: unavailable")
        except Exception as e:
            lines.append(f"Game log: fetch error ({e})")

    return "\n".join(lines)


def build_picks_news_context(picks: list[dict]) -> str:
    """
    Build a combined context block for all picks — used by Sports GPT.

    Only fetches each unique player once (even if they have multiple stats).
    """
    if not picks:
        return ""

    seen_players: set[str] = set()
    blocks: list[str] = []

    for pick in picks:
        name = pick["player"]
        stat = pick["stat"]

        # Avoid duplicate fetches per player
        player_stat_key = f"{name}|{stat}"
        if player_stat_key in seen_players:
            continue
        seen_players.add(player_stat_key)

        try:
            block = build_player_context(name, stat)
            blocks.append(block)
        except Exception as e:
            blocks.append(f"[{name} — {stat}]\n  Context fetch error: {e}")

    if not blocks:
        return ""

    header = "=== PLAYER CONTEXT (injuries, news, recent form) ===\n"
    return header + "\n\n".join(blocks)
