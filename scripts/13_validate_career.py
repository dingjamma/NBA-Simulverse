"""
Step 13: Backtest the CareerArcModel against known player trajectories.

For each target player:
  1. Find all their seasons in the dataset
  2. Hold out the last N_HOLDOUT seasons
  3. Feed the preceding seq_len seasons as input
  4. Predict held-out seasons autoregressively
  5. Measure MAE vs actual and check if breakout/decline labels were called

Usage:
  python scripts/13_validate_career.py
  python scripts/13_validate_career.py --holdout 3 --players "Luka Doncic" "Chris Paul"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from career_arc.config import (
    CAREER_STAT_FEATURES, CAREER_CONTEXT_FEATURES,
    CAREER_MODEL_CONFIG, CAREER_TRAIN_CONFIG,
    RAW_DIR, PROCESSED_DIR, CAREER_ERA_BOUNDARIES,
)
from career_arc.model import CareerArcModel
from career_arc.data.build_career_sequences import (
    _compute_per_season, _build_team_id_map, _compute_labels,
)

N_HOLDOUT = 3
SEQ_LEN = CAREER_MODEL_CONFIG.seq_len  # 5

CHECKPOINT_PATH = CAREER_TRAIN_CONFIG.checkpoint_dir / "best.pt"
NORM_STATS_PATH = PROCESSED_DIR / "career_norm_stats.json"
SEQUENCES_PATH  = PROCESSED_DIR / "career_sequences.npz"

DEFAULT_PLAYERS = [
    "Luka Doncic",
    "Chris Paul",
    "Derrick Rose",
    "Nikola Jokic",
    "LeBron James",
    "Stephen Curry",
    "Kevin Durant",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_era(year: int) -> int:
    era_id = 0
    for i, (start_year, _) in enumerate(CAREER_ERA_BOUNDARIES):
        if year >= start_year:
            era_id = i
    return era_id


def _load_model(device: torch.device) -> CareerArcModel:
    model = CareerArcModel(CAREER_MODEL_CONFIG).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_norm_stats() -> dict:
    with open(NORM_STATS_PATH) as f:
        return json.load(f)


def _normalize(arr: np.ndarray, norm_stats: dict, features: list[str]) -> np.ndarray:
    means = np.array([norm_stats[f]["mean"] for f in features], dtype=np.float32)
    stds  = np.array([norm_stats[f]["std"]  for f in features], dtype=np.float32)
    return (arr - means) / stds


def _denormalize(arr: np.ndarray, norm_stats: dict) -> np.ndarray:
    means = np.array([norm_stats[f]["mean"] for f in CAREER_STAT_FEATURES], dtype=np.float32)
    stds  = np.array([norm_stats[f]["std"]  for f in CAREER_STAT_FEATURES], dtype=np.float32)
    return arr * stds + means


def _get_player_seasons(player_name: str, games_df: pd.DataFrame,
                        players_df: pd.DataFrame, team_id_map: dict) -> tuple[pd.DataFrame, int]:
    """Return season-level DataFrame for a player + their personId."""
    players_df["fullName"] = (
        players_df["firstName"].fillna("") + " " + players_df["lastName"].fillna("")
    ).str.strip()
    matches = players_df[
        players_df["fullName"].str.contains(player_name, case=False, na=False)
    ].drop_duplicates("personId")

    if matches.empty:
        raise ValueError(f"Player '{player_name}' not found.")

    if len(matches) > 1:
        # Pick the one with the most game rows
        counts = {
            int(row["personId"]): int((games_df["personId"] == int(row["personId"])).sum())
            for _, row in matches.iterrows()
        }
        best_id = max(counts, key=counts.get)
        matches = matches[matches["personId"] == best_id]

    pid = int(matches.iloc[0]["personId"])
    player_games = games_df[games_df["personId"] == pid]
    if player_games.empty:
        raise ValueError(f"No game data found for '{player_name}' (id={pid})")

    season_df = _compute_per_season(player_games, team_id_map, players_df)
    return season_df, pid


def _predict_autoregressive(
    model: CareerArcModel,
    stat_window: np.ndarray,   # (seq_len, 10) normalized
    ctx_window: np.ndarray,    # (seq_len, 4)  normalized
    team_window: np.ndarray,   # (seq_len,)
    era_window: np.ndarray,    # (seq_len,)
    n_steps: int,
    norm_stats: dict,
    raw_ctx_last: np.ndarray,  # (4,) last raw context for extrapolation
    device: torch.device,
) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    """
    Autoregressively predict n_steps seasons.

    Returns:
        preds_raw:    (n_steps, 10) denormalized stat predictions
        breakout_probs, decline_probs, injury_probs: lists of length n_steps
    """
    ctx_means = np.array([norm_stats[f]["mean"] for f in CAREER_CONTEXT_FEATURES], dtype=np.float32)
    ctx_stds  = np.array([norm_stats[f]["std"]  for f in CAREER_CONTEXT_FEATURES], dtype=np.float32)

    working_stat  = stat_window.copy()
    working_ctx   = ctx_window.copy()
    working_teams = team_window.copy()
    working_eras  = era_window.copy()

    preds_raw      = []
    breakout_probs = []
    decline_probs  = []
    injury_probs   = []

    with torch.no_grad():
        for step in range(1, n_steps + 1):
            s_t = torch.tensor(working_stat,  dtype=torch.float32).unsqueeze(0).to(device)
            c_t = torch.tensor(working_ctx,   dtype=torch.float32).unsqueeze(0).to(device)
            tm_t = torch.tensor(
                np.clip(working_teams, 0, CAREER_MODEL_CONFIG.n_teams - 1),
                dtype=torch.long,
            ).unsqueeze(0).to(device)
            e_t  = torch.tensor(
                np.clip(working_eras, 0, CAREER_MODEL_CONFIG.n_eras - 1),
                dtype=torch.long,
            ).unsqueeze(0).to(device)
            stat_pred, bp, dp, ip = model(s_t, c_t, tm_t, e_t)

            pred_norm = stat_pred[0].float().detach().cpu().numpy()
            pred_raw  = np.clip(_denormalize(pred_norm, norm_stats), 0.0, None)

            preds_raw.append(pred_raw)
            breakout_probs.append(float(bp[0].cpu()))
            decline_probs.append(float(dp[0].cpu()))
            injury_probs.append(float(ip[0].cpu()))

            # Slide window — extrapolate context
            # CAREER_CONTEXT_FEATURES = [age, years_in_league, draft_round, draft_pick]
            ctx_raw_next = raw_ctx_last.copy()
            ctx_raw_next[0] += float(step)   # age
            ctx_raw_next[1] += float(step)   # years_in_league
            # draft_round and draft_pick stay constant
            norm_ctx_next = ((ctx_raw_next - ctx_means) / ctx_stds).reshape(1, -1)

            working_stat  = np.concatenate([working_stat[1:],  pred_norm.reshape(1, -1)], axis=0)
            working_ctx   = np.concatenate([working_ctx[1:],   norm_ctx_next], axis=0)
            working_teams = np.concatenate([working_teams[1:], working_teams[-1:]], axis=0)
            working_eras  = np.concatenate([working_eras[1:],  working_eras[-1:]], axis=0)

    return np.stack(preds_raw), breakout_probs, decline_probs, injury_probs


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def validate_player(
    player_name: str,
    games_df: pd.DataFrame,
    players_df: pd.DataFrame,
    team_id_map: pd.DataFrame,
    model: CareerArcModel,
    norm_stats: dict,
    device: torch.device,
    n_holdout: int = N_HOLDOUT,
) -> dict | None:
    try:
        season_df, pid = _get_player_seasons(player_name, games_df, players_df, team_id_map)
    except ValueError as e:
        print(f"  SKIP: {e}")
        return None

    n_seasons = len(season_df)
    required = SEQ_LEN + n_holdout
    if n_seasons < required:
        print(f"  SKIP: only {n_seasons} seasons, need {required}")
        return None

    # Split: use seasons up to [-n_holdout] as known history, last n_holdout as ground truth
    known_df  = season_df.iloc[:n_seasons - n_holdout]
    holdout_df = season_df.iloc[n_seasons - n_holdout:]

    # Take the last SEQ_LEN seasons from known history as the input window
    input_df = known_df.iloc[-SEQ_LEN:]
    if len(input_df) < SEQ_LEN:
        print(f"  SKIP: not enough history (got {len(input_df)}, need {SEQ_LEN})")
        return None

    stat_window_raw = input_df[CAREER_STAT_FEATURES].values.astype(np.float32)
    ctx_window_raw  = input_df[CAREER_CONTEXT_FEATURES].values.astype(np.float32)
    team_window     = input_df["team_id"].values.astype(np.int64)
    era_window      = input_df["era_id"].values.astype(np.int64)

    stat_window_norm = _normalize(stat_window_raw, norm_stats, CAREER_STAT_FEATURES)
    ctx_window_norm  = _normalize(ctx_window_raw,  norm_stats, CAREER_CONTEXT_FEATURES)

    raw_ctx_last = ctx_window_raw[-1].copy()

    # Predict n_holdout seasons autoregressively
    preds_raw, bp_list, dp_list, ip_list = _predict_autoregressive(
        model, stat_window_norm, ctx_window_norm, team_window, era_window,
        n_holdout, norm_stats, raw_ctx_last, device,
    )

    # Ground truth
    actuals_raw = holdout_df[CAREER_STAT_FEATURES].values.astype(np.float32)

    # MAE per stat, per season
    mae_per_season = np.abs(preds_raw - actuals_raw)  # (n_holdout, 10)
    mae_per_stat   = mae_per_season.mean(axis=0)       # (10,)
    overall_mae    = float(mae_per_stat.mean())

    # Classification: breakout / decline / injury
    # Ground truth labels for each held-out season
    gt_breakout = []
    gt_decline  = []
    gt_injury   = []
    for i in range(n_seasons - n_holdout, n_seasons):
        bo, dec, inj = _compute_labels(season_df, i)
        gt_breakout.append(bo)
        gt_decline.append(dec)
        gt_injury.append(inj)

    threshold = 0.5
    pred_breakout = [int(p >= threshold) for p in bp_list]
    pred_decline  = [int(p >= threshold) for p in dp_list]
    pred_injury   = [int(p >= threshold) for p in ip_list]

    bo_acc  = float(sum(p == g for p, g in zip(pred_breakout, gt_breakout)) / n_holdout)
    dec_acc = float(sum(p == g for p, g in zip(pred_decline,  gt_decline))  / n_holdout)
    inj_acc = float(sum(p == g for p, g in zip(pred_injury,   gt_injury))   / n_holdout)

    # Build per-season detail rows
    seasons_detail = []
    for i in range(n_holdout):
        actual_yr = int(holdout_df.iloc[i]["season_year"])
        row = {
            "season_year": actual_yr,
        }
        for j, feat in enumerate(CAREER_STAT_FEATURES):
            row[f"pred_{feat}"]   = round(float(preds_raw[i, j]), 2)
            row[f"actual_{feat}"] = round(float(actuals_raw[i, j]), 2)
            row[f"err_{feat}"]    = round(float(mae_per_season[i, j]), 2)
        row["pred_breakout"]  = round(bp_list[i], 3)
        row["pred_decline"]   = round(dp_list[i], 3)
        row["pred_injury"]    = round(ip_list[i], 3)
        row["gt_breakout"]    = gt_breakout[i]
        row["gt_decline"]     = gt_decline[i]
        row["gt_injury"]      = gt_injury[i]
        seasons_detail.append(row)

    return {
        "player": player_name,
        "pid": pid,
        "n_seasons_total": n_seasons,
        "n_holdout": n_holdout,
        "overall_mae": round(overall_mae, 4),
        "mae_per_stat": {f: round(float(mae_per_stat[j]), 4) for j, f in enumerate(CAREER_STAT_FEATURES)},
        "breakout_accuracy": round(bo_acc, 3),
        "decline_accuracy":  round(dec_acc, 3),
        "injury_accuracy":   round(inj_acc, 3),
        "seasons_detail":    seasons_detail,
    }


def print_player_report(result: dict) -> None:
    p = result["player"]
    print(f"\n{'='*90}")
    print(f"  {p}  |  {result['n_seasons_total']} total seasons  |  {result['n_holdout']} held out")
    print(f"  Overall MAE: {result['overall_mae']:.4f}")
    print(f"  Breakout acc: {result['breakout_accuracy']:.0%}  |  "
          f"Decline acc: {result['decline_accuracy']:.0%}  |  "
          f"Injury acc:  {result['injury_accuracy']:.0%}")
    print(f"{'='*90}")

    # Per-stat MAE summary
    stat_short = {"pts_per_game": "PTS", "reb_per_game": "REB", "ast_per_game": "AST",
                  "stl_per_game": "STL", "blk_per_game": "BLK", "fg_pct": "FG%",
                  "fg3_pct": "3P%", "ft_pct": "FT%", "minutes_per_game": "MIN",
                  "games_played": "GP"}
    mae_line = "  MAE: " + "  ".join(
        f"{stat_short[f]}={result['mae_per_stat'][f]:.2f}"
        for f in CAREER_STAT_FEATURES
    )
    print(mae_line)

    # Per-season detail
    print(f"\n  {'Season':>6}  {'Stat':>10}  {'Pred':>7}  {'Actual':>7}  {'Err':>6}  "
          f"{'BO?':>4}  {'Dec?':>4}  {'Inj?':>4}")
    print(f"  {'-'*75}")
    for row in result["seasons_detail"]:
        yr = row["season_year"]
        pts_pred   = row["pred_pts_per_game"]
        pts_actual = row["actual_pts_per_game"]
        pts_err    = row["err_pts_per_game"]
        ast_pred   = row["pred_ast_per_game"]
        ast_actual = row["actual_ast_per_game"]
        reb_pred   = row["pred_reb_per_game"]
        reb_actual = row["actual_reb_per_game"]

        bo_flag  = "YES" if row["gt_breakout"] else "no"
        dec_flag = "YES" if row["gt_decline"]  else "no"
        inj_flag = "YES" if row["gt_injury"]   else "no"

        print(f"  {yr:>6}  {'PTS':>10}  {pts_pred:>7.1f}  {pts_actual:>7.1f}  "
              f"{pts_err:>6.2f}  {bo_flag:>4}  {dec_flag:>4}  {inj_flag:>4}")
        print(f"  {'':>6}  {'REB':>10}  {reb_pred:>7.1f}  {reb_actual:>7.1f}  "
              f"{abs(reb_pred - reb_actual):>6.2f}")
        print(f"  {'':>6}  {'AST':>10}  {ast_pred:>7.1f}  {ast_actual:>7.1f}  "
              f"{abs(ast_pred - ast_actual):>6.2f}")
        # Classification probs
        print(f"  {'':>6}  {'probs':>10}  breakout={row['pred_breakout']:.2f}  "
              f"decline={row['pred_decline']:.2f}  injury={row['pred_injury']:.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Backtest career arc model")
    parser.add_argument("--players", nargs="+", default=DEFAULT_PLAYERS)
    parser.add_argument("--holdout", type=int, default=N_HOLDOUT,
                        help="Number of seasons to hold out (default 3)")
    parser.add_argument("--out", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: No checkpoint at {CHECKPOINT_PATH}. Run scripts/11_train_career.py first.")
        sys.exit(1)

    print("Loading data...")
    games_df = pd.read_csv(RAW_DIR / "PlayerStatistics.csv", low_memory=False)
    games_df = games_df[games_df["gameType"] == "Regular Season"].copy()
    games_df["gameDateTimeEst"] = pd.to_datetime(games_df["gameDateTimeEst"], errors="coerce")
    games_df = games_df.dropna(subset=["gameDateTimeEst"])
    games_df["season_year"] = games_df["gameDateTimeEst"].apply(
        lambda ts: ts.year if ts.month >= 7 else ts.year - 1
    )
    games_df["personId"] = pd.to_numeric(games_df["personId"], errors="coerce")
    games_df = games_df.dropna(subset=["personId"])
    games_df["personId"] = games_df["personId"].astype(int)

    numeric_cols = ["points","reboundsTotal","assists","steals","blocks",
                    "fieldGoalsAttempted","fieldGoalsMade","threePointersAttempted",
                    "threePointersMade","freeThrowsAttempted","freeThrowsMade","numMinutes"]
    for col in numeric_cols:
        if col in games_df.columns:
            games_df[col] = pd.to_numeric(games_df[col], errors="coerce").fillna(0.0)

    players_df = pd.read_csv(RAW_DIR / "Players.csv", low_memory=False)
    players_df["personId"] = pd.to_numeric(players_df["personId"], errors="coerce")
    players_df = players_df.dropna(subset=["personId"])
    players_df["personId"] = players_df["personId"].astype(int)

    team_id_map = _build_team_id_map(games_df)
    norm_stats  = _load_norm_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = _load_model(device)
    print(f"  Device: {device}\n")

    all_results = []
    for player_name in args.players:
        print(f"Validating: {player_name}")
        result = validate_player(
            player_name, games_df, players_df, team_id_map,
            model, norm_stats, device, n_holdout=args.holdout,
        )
        if result:
            print_player_report(result)
            all_results.append(result)

    # Aggregate summary
    if all_results:
        print(f"\n{'='*90}")
        print("SUMMARY")
        print(f"{'='*90}")
        print(f"{'Player':<22}  {'MAE':>6}  {'BO%':>5}  {'Dec%':>5}  {'Inj%':>5}  "
              f"{'PTS MAE':>7}  {'REB MAE':>7}  {'AST MAE':>7}")
        print(f"{'-'*70}")
        for r in all_results:
            print(
                f"{r['player']:<22}  {r['overall_mae']:>6.4f}  "
                f"{r['breakout_accuracy']:>5.0%}  {r['decline_accuracy']:>5.0%}  "
                f"{r['injury_accuracy']:>5.0%}  "
                f"{r['mae_per_stat']['pts_per_game']:>7.2f}  "
                f"{r['mae_per_stat']['reb_per_game']:>7.2f}  "
                f"{r['mae_per_stat']['ast_per_game']:>7.2f}"
            )
        avg_mae = sum(r["overall_mae"] for r in all_results) / len(all_results)
        avg_bo  = sum(r["breakout_accuracy"] for r in all_results) / len(all_results)
        avg_dec = sum(r["decline_accuracy"]  for r in all_results) / len(all_results)
        avg_inj = sum(r["injury_accuracy"]   for r in all_results) / len(all_results)
        print(f"{'-'*70}")
        print(f"{'MEAN':<22}  {avg_mae:>6.4f}  {avg_bo:>5.0%}  {avg_dec:>5.0%}  {avg_inj:>5.0%}")
        print(f"{'='*90}")

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
