# NBA-Simulverse — Roadmap

> A living simulation of basketball reality. The transformer learns the laws of physics of the NBA. Everything else is downstream.

---

## Vision

Most NBA ML projects solve leaf nodes: predict tomorrow's props, optimize DFS lineups. NBA-Simulverse is building the tree.

The foundation model learns basketball as a system — how player sequences encode fatigue, momentum, and role. Downstream, it answers questions no rolling average can: *What happens to the Warriors' offense if Steph misses 10 games starting game 47?* Not the box score answer — the cascade answer.

---

## Phase 1 — The Engine `[active]`

**Goal:** Transformer beats XGBoost on 4/6 stat categories on 2024-25 holdout season.

This validates the model has learned real basketball signal, not just memorized player averages.

**Architecture:**
- Input: last 20 games per player (23 features each)
- Model: Transformer encoder, 5 layers, d_model=192, 3.9M params
- Training: Delta prediction — predict deviation from rolling mean, not raw stats
- Evaluation: MAE vs XGBoost (rolling 20-game flat features) on 2024-25 test set

**Scripts:**
```
01_preprocess.py         Raw CSV → player_games.parquet
02_build_sequences.py    Feature engineering + sliding windows
03_train_baseline.py     XGBoost baseline
04_train_model.py        NBA-GPT transformer
05_evaluate.py           Head-to-head comparison
07_sliced_eval.py        Slice-based analysis (where GPT beats XGB)
```

**Status:** Training with delta prediction. XGBoost baseline: pts=4.64, reb=1.91, ast=1.37, stl=0.69, blk=0.54, 3pm=0.88.

---

## Phase 2 — Single Player Counterfactuals `[built, pending Phase 1]`

**Goal:** Given a player's history, simulate their next game under hypothetical conditions.

**Key questions:**
- "What if Luka plays only 28 minutes (load management)?"
- "What does Steph do on zero rest against a top-5 defense?"
- "How does a player perform after a 5-day rest vs back-to-back?"

**Approach:** Monte Carlo Dropout — run N forward passes with dropout active → distribution over outcomes, not point estimates.

**Scripts:**
```
06_simulate.py           Single player counterfactual CLI
```

**Example:**
```bash
# Compare Luka at 32 min vs 40 min
python scripts/06_simulate.py --player "Doncic" --compare --minutes-a 32 --minutes-b 40

# Steph on the road with 1 day rest vs tough defense
python scripts/06_simulate.py --player "Curry" --away --rest 1 --opp-defense 98
```

---

## Phase 3 — Roster Simulation `[built, pending Phase 2]`

**Goal:** Simulate full team box scores under roster constraints.

**Key questions:**
- "What happens to the Warriors' offense if Steph misses 10 games?"
- "How does Jaylen Brown's usage change when Tatum is out?"

**Approach:** Minutes redistribution → per-player simulation with updated context.
The cascade effect: absent player's minutes → redistributed to remaining players → each player simulated with their updated role.

**Scripts:**
```
08_roster_sim.py         Team roster counterfactual CLI
```

**Example:**
```bash
python scripts/08_roster_sim.py --team "Warriors" --absent "Curry" --home
```

---

## Phase 4 — Career Arc Modeling `[planned]`

**Goal:** Model longitudinal player trajectories across seasons.

**Key questions:**
- "Is this 22-year-old a star or a ceiling case?"
- "When does this player typically peak/decline?"
- "What's the probability this rookie becomes an All-Star?"

**Approach:** Extend sequence modeling to span seasons, not just games. Learn breakout signatures, decline curves, injury risk patterns from historical trajectories.

---

## Phase 5 — The Public Interface `[planned]`

**Goal:** Open research + queryable API.

- REST API: POST a scenario, get probability distributions
- Research paper: architecture, findings, basketball insights
- Open dataset: cleaned 1947-present NBA game sequences
- Interactive tool: "what if" queries on any player/team

---

## Evaluation Philosophy

Standard MAE benchmarks tell you if the model learned averages. Sliced evaluation tells you if it learned basketball physics.

**The real test is: does the transformer outperform XGBoost specifically on the slices where sequence matters?**

```
07_sliced_eval.py outputs:

Slice              N       pts    reb    ast    stl    blk    3pm
────────────────────────────────────────────────────────────────
all            45,966   4.640  1.910  1.368  0.689  0.536  0.880
back_to_back    8,234  *4.521  1.876 *1.302  0.701  0.541  0.871
hot_streak      6,891  *4.102 *1.743 *1.198  0.692 *0.498 *0.801
cold_streak     7,102   4.891  2.041  1.492  0.701  0.574  0.941
tough_defense  11,491  *4.518  1.892 *1.334  0.688  0.539 *0.856
```

`*` = NBA-GPT wins that slice. If we see wins cluster on `hot_streak`, `back_to_back`, and `tough_defense`, the model has learned something real.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data | Pandas, NumPy, Parquet |
| Model | PyTorch, Transformer (Pre-LN) |
| Baseline | XGBoost |
| Training | AdamW, cosine warmup, mixed precision (fp16) |
| Simulation | Monte Carlo Dropout |
| Dataset | Kaggle: eoinamoore/historical-nba-data-and-player-box-scores (CC0, 1947-present) |

---

## Project Structure

```
NBA-Simulverse/
├── src/nba_gpt/
│   ├── config.py              # All hyperparameters
│   ├── data/
│   │   ├── preprocess.py      # Raw → player_games.parquet
│   │   ├── features.py        # Feature engineering
│   │   ├── sequences.py       # Sliding window builder
│   │   └── dataset.py         # PyTorch Dataset
│   ├── model/
│   │   └── transformer.py     # NBAGPTModel
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   └── scheduler.py       # LR scheduler
│   ├── evaluation/
│   │   ├── evaluate.py        # Head-to-head vs XGBoost
│   │   └── sliced_eval.py     # Condition-based analysis
│   ├── baseline/
│   │   └── xgboost_baseline.py
│   └── simulation/
│       ├── engine.py          # Single player counterfactuals
│       └── roster.py          # Team roster simulation
└── scripts/
    ├── 01_preprocess.py
    ├── 02_build_sequences.py
    ├── 03_train_baseline.py
    ├── 04_train_model.py
    ├── 05_evaluate.py
    ├── 06_simulate.py
    ├── 07_sliced_eval.py
    └── 08_roster_sim.py
```
