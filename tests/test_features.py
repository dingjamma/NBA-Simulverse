import pandas as pd
import pytest
from nba_gpt.data.features import assign_era, compute_rest_days, encode_player_ids


def test_assign_era_boundaries():
    assert assign_era(1950) == 0  # Pre-Shot Clock
    assert assign_era(1954) == 0  # still Pre-Shot Clock (boundary is 1955)
    assert assign_era(1955) == 1  # Shot Clock Era
    assert assign_era(1976) == 1
    assert assign_era(1977) == 2  # Showtime
    assert assign_era(1993) == 2
    assert assign_era(1994) == 3  # Jordan Era
    assert assign_era(2002) == 3
    assert assign_era(2003) == 4  # Post-Jordan
    assert assign_era(2012) == 4
    assert assign_era(2013) == 5  # Three-Point Revolution
    assert assign_era(2025) == 5


def test_compute_rest_days_first_game():
    df = pd.DataFrame({
        "personId": [1],
        "gameDateTimeEst": pd.to_datetime(["2024-01-01"]),
    })
    result = compute_rest_days(df)
    assert result["rest_days"].iloc[0] == 7.0  # neutral value for first game


def test_compute_rest_days_two_games():
    df = pd.DataFrame({
        "personId": [1, 1],
        "gameDateTimeEst": pd.to_datetime(["2024-01-01", "2024-01-03"]),
    })
    result = compute_rest_days(df)
    assert result["rest_days"].iloc[0] == 7.0   # first game
    assert result["rest_days"].iloc[1] == 2.0   # 2 days later


def test_compute_rest_days_back_to_back():
    df = pd.DataFrame({
        "personId": [1, 1],
        "gameDateTimeEst": pd.to_datetime(["2024-01-01", "2024-01-02"]),
    })
    result = compute_rest_days(df)
    assert result["rest_days"].iloc[1] == 1.0


def test_compute_rest_days_capped_at_30():
    df = pd.DataFrame({
        "personId": [1, 1],
        "gameDateTimeEst": pd.to_datetime(["2024-01-01", "2024-04-01"]),
    })
    result = compute_rest_days(df)
    assert result["rest_days"].iloc[1] == 30.0


def test_compute_rest_days_separate_players():
    df = pd.DataFrame({
        "personId": [1, 2, 1],
        "gameDateTimeEst": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })
    result = compute_rest_days(df)
    # Player 2's game should have neutral value (first game)
    p2 = result[result["personId"] == 2]["rest_days"].values
    assert p2[0] == 7.0


def test_encode_player_ids_contiguous():
    df = pd.DataFrame({"personId": [100, 200, 100, 300]})
    result, id_map = encode_player_ids(df)
    assert set(result["player_id_encoded"].unique()) == {0, 1, 2}
    assert len(id_map) == 3
