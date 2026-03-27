"""
XGBoost baseline: rolling 20-game averages -> predict next game stats.
One regressor per target stat. Used as benchmark for NBA-GPT.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

from nba_gpt.config import DATA_CONFIG, INPUT_FEATURES, TARGET_STATS


ROLLING_WINDOW = 20
_EXTRA_FEATURES = ["rest_days", "home", "era_id", "player_game_number"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game row, create a flat feature vector from the previous
    ROLLING_WINDOW games (rolling mean) + extra contextual features.
    """
    groups = []
    for pid, group in df.groupby("personId"):
        group = group.sort_values("gameDateTimeEst").copy()

        # Rolling mean of input features (shifted to exclude current game)
        for feat in INPUT_FEATURES:
            group[f"roll_{feat}"] = (
                group[feat].rolling(ROLLING_WINDOW, min_periods=5).mean().shift(1)
            )

        groups.append(group)

    out = pd.concat(groups, ignore_index=True)

    roll_cols = [f"roll_{f}" for f in INPUT_FEATURES]
    feature_cols = roll_cols + _EXTRA_FEATURES
    # Drop rows where rolling features aren't ready
    out = out.dropna(subset=roll_cols)
    return out, feature_cols


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_ts = pd.Timestamp(DATA_CONFIG.val_season_start)
    test_ts = pd.Timestamp(DATA_CONFIG.test_season_start)
    train = df[df["gameDateTimeEst"] < val_ts]
    val = df[(df["gameDateTimeEst"] >= val_ts) & (df["gameDateTimeEst"] < test_ts)]
    test = df[df["gameDateTimeEst"] >= test_ts]
    return train, val, test


def train_models(
    features_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, XGBRegressor]:
    features_path = features_path or DATA_CONFIG.player_features_path
    output_dir = output_dir or (DATA_CONFIG.processed_dir / "xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {features_path}...")
    df = pd.read_parquet(features_path)

    print("Building rolling features...")
    df_feat, feature_cols = build_features(df)

    print("Splitting by date...")
    train_df, val_df, test_df = split_by_date(df_feat)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values

    models: dict[str, XGBRegressor] = {}
    mae_results: dict[str, float] = {}

    for stat in TARGET_STATS:
        print(f"Training XGBoost for {stat}...")
        y_train = train_df[stat].values
        y_val = val_df[stat].values

        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            eval_metric="mae",
            device="cuda",
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_pred = model.predict(X_val)
        val_mae = float(np.abs(val_pred - y_val).mean())
        mae_results[stat] = val_mae
        print(f"  Val MAE: {val_mae:.3f}")

        model_path = output_dir / f"xgb_{stat}.joblib"
        joblib.dump(model, model_path)
        models[stat] = model

    # Save val MAE results
    with open(output_dir / "val_mae.json", "w") as f:
        json.dump(mae_results, f, indent=2)

    return models, feature_cols


def evaluate_on_test(
    models: dict[str, XGBRegressor],
    feature_cols: list[str],
    features_path: Path | None = None,
) -> dict[str, float]:
    features_path = features_path or DATA_CONFIG.player_features_path

    df = pd.read_parquet(features_path)
    df_feat, _ = build_features(df)
    _, _, test_df = split_by_date(df_feat)

    X_test = test_df[feature_cols].values
    mae_results: dict[str, float] = {}

    for stat, model in models.items():
        y_test = test_df[stat].values
        pred = model.predict(X_test)
        mae = float(np.abs(pred - y_test).mean())
        mae_results[stat] = mae

    return mae_results


def load_models(output_dir: Path | None = None) -> dict[str, XGBRegressor]:
    output_dir = output_dir or (DATA_CONFIG.processed_dir / "xgboost")
    return {
        stat: joblib.load(output_dir / f"xgb_{stat}.joblib")
        for stat in TARGET_STATS
    }
