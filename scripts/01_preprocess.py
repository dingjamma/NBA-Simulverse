"""Step 1: Preprocess raw CSV -> clean parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_gpt.data.preprocess import run

if __name__ == "__main__":
    run()
