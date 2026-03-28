"""
Step 15: Live next-game prediction using real-time nba.com data.

Fetches a player's recent games, builds features on the fly, runs
the NBA-GPT ensemble, and prints a stat distribution for their next game.

Usage:
  python scripts/15_live_predict.py "Luka Doncic"
  python scripts/15_live_predict.py "Steph Curry" --minutes 36 --home
  python scripts/15_live_predict.py "Nikola Jokic" --rest 2 --opp-defense 108.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from live_feed.fetcher import fetch_player_live
from nba_gpt.config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, TARGET_STATS
from nba_gpt.data.dataset import load_norm_stats
from nba_gpt.simulation.engine import (
    ScenarioOverride, SimulationResult,
    _build_input_sequence, _build_context_tensor,
    _enable_dropout, _load_model,
)

import json
import numpy as np
import torch


def predict_live(
    player_name: str,
    season_year: int = 2024,
    override: ScenarioOverride | None = None,
) -> SimulationResult:
    override = override or ScenarioOverride()

    print(f"\nFetching live data for '{player_name}'...")
    player_df = fetch_player_live(player_name, season_year=season_year, n_games=25)
    print(f"  Got {len(player_df)} games")

    if len(player_df) < MODEL_CONFIG.sequence_length:
        raise ValueError(
            f"Only {len(player_df)} games fetched, need {MODEL_CONFIG.sequence_length}. "
            "Try a different season or player."
        )

    norm_stats = load_norm_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_seq, player_id_enc, era_id = _build_input_sequence(
        player_df, norm_stats, seq_len=MODEL_CONFIG.sequence_length
    )
    next_game_ctx = _build_context_tensor(player_df, override, norm_stats)

    last_games = player_df.tail(MODEL_CONFIG.sequence_length)
    window_mean = last_games[TARGET_STATS].values.mean(axis=0).astype(np.float32)

    player_id_t   = torch.tensor([player_id_enc], dtype=torch.long).to(device)
    era_id_t      = torch.tensor([era_id],        dtype=torch.long).to(device)
    input_seq_t   = input_seq.to(device)
    ctx_t         = next_game_ctx.to(device)

    # Ensemble if available, else MC dropout
    from nba_gpt.simulation.ensemble import EnsemblePredictor, ENSEMBLE_DIR
    use_ensemble = ENSEMBLE_DIR.exists() and any(ENSEMBLE_DIR.glob("seed_*/best.pt"))

    if use_ensemble:
        predictor = EnsemblePredictor(device=device)
        member_actuals = np.stack([
            m(player_id_t, era_id_t, input_seq_t, ctx_t).float().detach().cpu().numpy()[0] + window_mean
            for m in predictor.models
        ])
        samples = np.clip(member_actuals, 0, None)
    else:
        model = _load_model(TRAIN_CONFIG.checkpoint_dir / "best.pt", device)
        model.eval()
        _enable_dropout(model)
        samples_list = []
        with torch.no_grad():
            for _ in range(500):
                pred = model(player_id_t, era_id_t, input_seq_t, ctx_t).float().cpu().numpy()
                samples_list.append(pred[0] + window_mean)
        samples = np.clip(np.stack(samples_list), 0, None)

    resolved_name = player_df["personId"].iloc[0]  # fallback
    result = SimulationResult(
        player_name=player_name,
        scenario=override,
        n_samples=len(samples),
        samples=samples,
    )
    for i, stat in enumerate(TARGET_STATS):
        col = samples[:, i]
        result.mean[stat]  = float(col.mean())
        result.std[stat]   = float(col.std())
        result.p10[stat]   = float(np.percentile(col, 10))
        result.p25[stat]   = float(np.percentile(col, 25))
        result.p75[stat]   = float(np.percentile(col, 75))
        result.p90[stat]   = float(np.percentile(col, 90))

    return result


def main():
    parser = argparse.ArgumentParser(description="Live NBA-GPT next-game prediction")
    parser.add_argument("player", help="Player name (partial match ok)")
    parser.add_argument("--season", type=int, default=2024,
                        help="Season start year (default: 2024 = 2024-25)")
    parser.add_argument("--minutes", type=float, default=None, help="Projected minutes")
    parser.add_argument("--home",   action="store_true", help="Home game")
    parser.add_argument("--away",   action="store_true", help="Away game")
    parser.add_argument("--rest",   type=float, default=None, help="Rest days")
    parser.add_argument("--opp-defense", type=float, default=None,
                        help="Opponent pts allowed per game (lower = tougher defense)")
    args = parser.parse_args()

    home = None
    if args.home:
        home = True
    elif args.away:
        home = False

    override = ScenarioOverride(
        minutes=args.minutes,
        home=home,
        rest_days=args.rest,
        opp_pts_allowed=args.opp_defense,
    )

    result = predict_live(args.player, season_year=args.season, override=override)
    print(result.summary())


if __name__ == "__main__":
    main()
