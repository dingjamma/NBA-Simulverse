import io
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from nba_gpt.data.preprocess import load_raw, clean, validate


def _make_raw_csv() -> pd.DataFrame:
    rows = [
        # Regular Season, played
        dict(firstName="A", lastName="X", personId=1, gameId=101,
             gameDateTimeEst="2024-01-10 19:00:00", gameType="Regular Season",
             numMinutes=30.0, points=20, assists=5, blocks=1, steals=2,
             fieldGoalsAttempted=15, fieldGoalsMade=8, fieldGoalsPercentage=0.533,
             threePointersAttempted=5, threePointersMade=2, threePointersPercentage=0.4,
             freeThrowsAttempted=4, freeThrowsMade=4, freeThrowsPercentage=1,
             reboundsDefensive=4, reboundsOffensive=1, reboundsTotal=5,
             foulsPersonal=2, turnovers=1, plusMinusPoints=10, home=1,
             win=1, playerteamCity="LA", playerteamName="Lakers",
             opponentteamCity="BOS", opponentteamName="Celtics",
             gameLabel=None, gameSubLabel=None, seriesGameNumber=None),
        # Preseason - should be filtered out
        dict(firstName="B", lastName="Y", personId=2, gameId=102,
             gameDateTimeEst="2023-10-01 19:00:00", gameType="Preseason",
             numMinutes=20.0, points=10, assists=2, blocks=0, steals=1,
             fieldGoalsAttempted=8, fieldGoalsMade=4, fieldGoalsPercentage=0.5,
             threePointersAttempted=2, threePointersMade=1, threePointersPercentage=0.5,
             freeThrowsAttempted=2, freeThrowsMade=2, freeThrowsPercentage=1,
             reboundsDefensive=3, reboundsOffensive=0, reboundsTotal=3,
             foulsPersonal=1, turnovers=1, plusMinusPoints=5, home=0,
             win=0, playerteamCity="BOS", playerteamName="Celtics",
             opponentteamCity="LA", opponentteamName="Lakers",
             gameLabel=None, gameSubLabel=None, seriesGameNumber=None),
        # DNP - should be filtered out
        dict(firstName="A", lastName="X", personId=1, gameId=103,
             gameDateTimeEst="2024-01-12 19:00:00", gameType="Regular Season",
             numMinutes=0.0, points=0, assists=0, blocks=0, steals=0,
             fieldGoalsAttempted=0, fieldGoalsMade=0, fieldGoalsPercentage=0,
             threePointersAttempted=0, threePointersMade=0, threePointersPercentage=0,
             freeThrowsAttempted=0, freeThrowsMade=0, freeThrowsPercentage=0,
             reboundsDefensive=0, reboundsOffensive=0, reboundsTotal=0,
             foulsPersonal=0, turnovers=0, plusMinusPoints=0, home=1,
             win=1, playerteamCity="LA", playerteamName="Lakers",
             opponentteamCity="BOS", opponentteamName="Celtics",
             gameLabel=None, gameSubLabel=None, seriesGameNumber=None),
    ]
    return pd.DataFrame(rows)


def test_clean_filters_preseason():
    df = _make_raw_csv()
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    result = clean(df)
    assert all(result["gameType"] == "Regular Season")


def test_clean_filters_dnp():
    df = _make_raw_csv()
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    result = clean(df)
    assert all(result["numMinutes"] >= 1.0)


def test_clean_assigns_game_number():
    df = _make_raw_csv()
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    result = clean(df)
    # Player 1 has one valid game -> game_number == 1
    p1_rows = result[result["personId"] == 1]
    assert list(p1_rows["player_game_number"]) == [1]


def test_validate_passes_clean_data():
    df = _make_raw_csv()
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    cleaned = clean(df)
    validate(cleaned)  # Should not raise


def test_validate_fails_on_duplicates():
    df = _make_raw_csv()
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    cleaned = clean(df)
    # Artificially create a duplicate
    dup = pd.concat([cleaned, cleaned.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate"):
        validate(dup)
