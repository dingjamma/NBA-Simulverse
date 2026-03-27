"""Step 5: Evaluate NBA-GPT vs XGBoost on holdout test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.evaluation.evaluate import run

if __name__ == "__main__":
    results = run()
    if results["phase1_success"]:
        print("Phase 1 complete. NBA-GPT beats XGBoost on 4+ of 6 stats.")
    else:
        print("NBA-GPT did not beat XGBoost on 4/6 stats. See evaluation_results.json.")
