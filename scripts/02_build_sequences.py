"""Step 2: Feature engineering + build sliding window sequences."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.data.features import run as run_features
from nba_gpt.data.sequences import run as run_sequences

if __name__ == "__main__":
    run_features()
    run_sequences()
