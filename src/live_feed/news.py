"""
ESPN news and injury scraper for NBA players.

Cached daily to disk so Sports GPT runs fast on repeated calls.
"""
import json
from datetime import date
from pathlib import Path

import requests

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_HEADERS   = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT   = 15

_CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
_today     = date.today().isoformat()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}_{_today}.json"


def _load_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_cache(key: str, data: dict) -> None:
    _cache_path(key).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_player_injury(athlete_id: str | int) -> dict:
    """
    Return injury info for a player.

    Returns dict with keys:
        status   : "Active" | "Day-To-Day" | "Out" | "Questionable" | ...
        detail   : free-text injury description (may be empty)
    """
    key  = f"injury_{athlete_id}"
    hit  = _load_cache(key)
    if hit:
        return hit

    try:
        data   = _get(f"{_ESPN_BASE}/athletes/{athlete_id}")
        ath    = data.get("athlete", {})
        status = ath.get("status", {})
        result = {
            "status": status.get("type", {}).get("description", "Active"),
            "detail": status.get("shortComment", ""),
        }
    except Exception:
        result = {"status": "Unknown", "detail": ""}

    _save_cache(key, result)
    return result


def fetch_player_news(athlete_id: str | int, limit: int = 3) -> list[dict]:
    """
    Return recent ESPN news items for a player.

    Each item:
        headline : str
        published: str  (ISO date)
        summary  : str  (may be empty)
    """
    key = f"news_{athlete_id}"
    hit = _load_cache(key)
    if hit:
        return hit.get("items", [])

    try:
        data  = _get(f"{_ESPN_BASE}/athletes/{athlete_id}/news", {"limit": limit})
        items = []
        for art in data.get("articles", [])[:limit]:
            items.append({
                "headline":  art.get("headline", ""),
                "published": art.get("published", "")[:10],
                "summary":   art.get("description", ""),
            })
    except Exception:
        items = []

    _save_cache(key, {"items": items})
    return items
