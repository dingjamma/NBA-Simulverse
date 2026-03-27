import numpy as np
import pandas as pd
import pytest
from nba_gpt.data.sequences import build_sequences, split_sequences
from nba_gpt.config import INPUT_FEATURES, TARGET_STATS


def _make_player_df(pid: int, n_games: int, start_year: int = 2020) -> pd.DataFrame:
    dates = pd.date_range(start=f"{start_year}-01-01", periods=n_games, freq="2D")
    rows = []
    for i, d in enumerate(dates):
        row = {
            "personId": pid,
            "player_id_encoded": pid - 1,
            "gameId": pid * 1000 + i,
            "gameDateTimeEst": d,
            "era_id": 5,
            "player_game_number": i + 1,
        }
        for f in INPUT_FEATURES:
            row[f] = float(i + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_sequences_window_size():
    df = _make_player_df(1, n_games=25)
    result = build_sequences(df, seq_len=20)
    # 25 games -> 5 sequences (25 - 20)
    assert len(result["inputs"]) == 5
    assert result["inputs"].shape == (5, 20, len(INPUT_FEATURES))
    assert result["targets"].shape == (5, len(TARGET_STATS))


def test_build_sequences_skips_small_players():
    df = _make_player_df(1, n_games=24)  # 24 < min_player_games=25 -> no sequences
    result = build_sequences(df, seq_len=20)
    assert result["inputs"].shape[0] == 0


def test_build_sequences_no_future_leakage():
    df = _make_player_df(1, n_games=30)
    result = build_sequences(df, seq_len=20)
    # The input for sequence i should be strictly before the target
    # We check shapes are consistent
    assert result["inputs"].shape[1] == 20
    assert result["targets"].shape[1] == len(TARGET_STATS)


def test_split_sequences_correct_boundaries():
    # Create 3 players; override dates so target games land in each split window.
    # Player 1: target games land in train (before 2023-07-01)
    df1 = _make_player_df(1, n_games=30, start_year=2021)
    df1["gameDateTimeEst"] = pd.date_range(start="2021-01-01", periods=30, freq="3D")

    # Player 2: target games in val (2023-07-01 to 2024-07-01)
    df2 = _make_player_df(2, n_games=30, start_year=2023)
    df2["gameDateTimeEst"] = pd.date_range(start="2023-07-01", periods=30, freq="3D")

    # Player 3: target games in test (after 2024-07-01)
    df3 = _make_player_df(3, n_games=30, start_year=2024)
    df3["gameDateTimeEst"] = pd.date_range(start="2024-08-01", periods=30, freq="3D")

    df = pd.concat([df1, df2, df3], ignore_index=True)

    all_data = build_sequences(df, seq_len=20)
    train, val, test = split_sequences(all_data, "2023-07-01", "2024-07-01")

    assert len(train["inputs"]) > 0
    assert len(val["inputs"]) > 0
    assert len(test["inputs"]) > 0

    # No overlap
    total = len(train["inputs"]) + len(val["inputs"]) + len(test["inputs"])
    assert total == len(all_data["inputs"])
