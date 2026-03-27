"""Step 3: Train XGBoost baseline models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.baseline.xgboost_baseline import train_models

if __name__ == "__main__":
    models, feature_cols = train_models()
    print("\nXGBoost baseline training complete.")
    print(f"Models saved to data/processed/xgboost/")
