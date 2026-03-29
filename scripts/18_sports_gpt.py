"""
Sports GPT — Ask Qwen3.5 questions about today's NBA-GPT picks.

Usage:
  python scripts/18_sports_gpt.py
  python scripts/18_sports_gpt.py "Should I parlay Giddey and Embiid tonight?"
  python scripts/18_sports_gpt.py --date 2026-03-28
"""
import argparse
import csv
import sys
from datetime import date, timedelta, timezone, datetime
from pathlib import Path

import requests

LOGS_DIR  = Path(__file__).parent.parent / "logs"
PICKS_CSV = LOGS_DIR / "picks.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen3.5:35b"

SYSTEM_PROMPT = """You are a sharp NBA prop betting analyst working with an ML model called NBA-GPT.

NBA-GPT is a custom Transformer trained on sequential NBA game logs (20-game windows).
It predicts next-game stats for each player. The "edge" is model_mean minus the prop line —
a large negative edge means the model projects well UNDER the line, large positive means OVER.

The model is strong on:
- Players with consistent usage patterns
- Identifying inflated lines (edge > 7 is very high confidence)
- Home/away splits, rest days, opponent defense by position

The model is weaker on:
- Hot shooting nights (single-game variance)
- Players with very few games this season (< 15 games = less reliable)
- Injury/lineup changes not reflected in recent game logs

Your job: reason about the picks, flag any concerns, and give clear actionable advice.
Be concise. Think like a sharp bettor, not a TV analyst."""


def load_picks(date_str: str) -> list[dict]:
    if not PICKS_CSV.exists():
        return []
    with open(PICKS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["date"] == date_str]


def format_picks_context(picks: list[dict], date_str: str) -> str:
    if not picks:
        return f"No picks found for {date_str}."

    graded   = [p for p in picks if p["result"] in ("HIT", "MISS")]
    pending  = [p for p in picks if not p["result"]]

    lines = [f"NBA-GPT Picks — {date_str}\n"]

    if pending:
        lines.append("PENDING PICKS:")
        for p in pending:
            edge = float(p["edge"])
            sign = "+" if edge > 0 else ""
            lines.append(
                f"  {p['player']:<24} {p['stat']:<12} {p['direction']} {p['line']:>5} "
                f"| model={float(p['model_mean']):.1f} "
                f"| p25={float(p['model_p25']):.1f} p75={float(p['model_p75']):.1f} "
                f"| edge={sign}{edge:.1f}"
            )

    if graded:
        lines.append("\nGRADED PICKS:")
        hits  = sum(1 for p in graded if p["result"] == "HIT")
        total = len(graded)
        lines.append(f"  Record: {hits}/{total} ({100*hits//total}%)")
        for p in graded:
            flag = "✓" if p["result"] == "HIT" else "✗"
            lines.append(
                f"  {flag} {p['player']:<24} {p['stat']:<12} {p['direction']} {p['line']:>5} "
                f"| actual={p['actual']} | model={float(p['model_mean']):.1f}"
            )

    # Running record across all time
    with open(PICKS_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))
    all_graded = [r for r in all_rows if r["result"] in ("HIT", "MISS")]
    if all_graded:
        all_hits = sum(1 for r in all_graded if r["result"] == "HIT")
        lines.append(f"\nAll-time record: {all_hits}/{len(all_graded)} ({100*all_hits//len(all_graded)}%)")

    return "\n".join(lines)


def ask_qwen(question: str, context: str, think: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{context}\n\n---\nUser question: {question}"},
    ]
    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model":    MODEL,
                "messages": messages,
                "stream":   False,
                "think":    think,
                "options":  {"temperature": 0.7, "num_predict": 1024},
            },
            timeout=180,
        )
        r.raise_for_status()
        msg      = r.json().get("message", {})
        content  = msg.get("content", "")
        thinking = msg.get("thinking", "")
        # Qwen3.5 puts full answer in thinking when think=True, content when think=False
        return content if content else (thinking if thinking else "No response from model.")
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama not running. Start it with: ollama serve"
    except Exception as e:
        return f"ERROR: {e}"


def interactive_mode(date_str: str) -> None:
    picks   = load_picks(date_str)
    context = format_picks_context(picks, date_str)

    print(f"\n{'='*60}")
    print(f"Sports GPT — {date_str}")
    print(f"Model: {MODEL}")
    print(f"{'='*60}")
    print(context)
    print(f"\n{'='*60}")
    print("Ask anything about today's picks. Type 'quit' to exit.")
    print(f"{'='*60}\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        print(f"\nQwen3.5: ", end="", flush=True)
        response = ask_qwen(question, context)
        print(response)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None,
                        help="Question to ask (omit for interactive mode)")
    parser.add_argument("--date", default=None,
                        help="Date YYYY-MM-DD (default: today ET)")
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    else:
        ET       = timezone(timedelta(hours=-4))
        date_str = datetime.now(ET).strftime("%Y-%m-%d")

    picks   = load_picks(date_str)
    context = format_picks_context(picks, date_str)

    if args.question:
        print(f"\nQwen3.5: ", end="", flush=True)
        print(ask_qwen(args.question, context))
    else:
        interactive_mode(date_str)


if __name__ == "__main__":
    main()
