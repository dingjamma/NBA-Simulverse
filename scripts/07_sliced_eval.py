"""Step 7: Sliced evaluation — where does the transformer actually beat XGBoost?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.evaluation.sliced_eval import run

if __name__ == "__main__":
    run()
