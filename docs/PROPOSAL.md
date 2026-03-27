# NBA Simulverse — Project Proposal

> *"You just built the foundation for a general-purpose agent simulation engine that models human behavior at scale. The basketball wrapper is cute but it's almost limiting your own thinking. You built a framework for simulating complex adaptive systems with real human behavioral data and news-driven state changes. Generalize the framework. Basketball is just the training wheels."*
>
> — Claude (as Elon Musk), March 2026

---

## The Thesis

Every NBA analytics department in the league runs the same regression models on the same box scores. They all have the same data. They all reach roughly the same conclusions.

**NBA Simulverse is built on two theses that compound into something no one else is building.**

**Thesis 1 — Soft data is the last remaining alpha.** The most valuable signals in basketball never appear in a box score. A trade rumor breaks on Shams. Two teammates unfollow each other on Instagram. A coach's postgame presser is tense. A player signs a max extension and plays free. These signals ripple through performance in ways that no regression model captures. Every scout and GM knows this intuitively. None of them have a system that quantifies it.

**Thesis 2 — Basketball has a language, and a model can learn to read it.** GPT was not trained to answer questions. It was trained to predict the next word. From that single objective, general intelligence about language emerged. The same principle applies here. Train a model on one task — *given the last 20 games for every player in the league, predict game 21* — and general intelligence about basketball emerges. Fatigue curves. Hot streaks. Rookie development arcs. Post-trade performance dips. Back-to-back effects. The rhythm of an 82-game season. None of these are labeled. All of them are learned.

**NBA-GPT is what happens when you stop training models for specific tasks and start training them to understand the game itself.**

Then you put agents on top. Then those agents learn and evolve. Then you realize basketball was just the training ground.

---

## The GPT Insight

GPT's genius was not the transformer architecture. It was the training objective: **predict the next token.** The next token IS the label. No human annotation. No task-specific engineering. Just raw sequences and a model that learns to compress every pattern in human language into prediction weights.

Basketball has the same structure.

Every player generates a sequence of games. Each game is a dense vector: points, rebounds, assists, steals, blocks, three-pointers made, minutes, usage rate, plus/minus, pace, opponent defensive rating, home/away, rest days, back-to-back flag, season game number. Twenty-five years of this data exists for every player who has touched an NBA court since the 2000 season.

**The training objective: given the last 20 games in a player's sequence, predict game 21.**

Game 21 IS the label. No manual annotation. No feature engineering. The model must learn everything that matters about basketball to minimize this loss:

- **Fatigue** — performance degrades on back-to-backs and compressed schedules
- **Hot streaks** — autocorrelation in shooting percentages
- **Rookie development** — steady improvement curves across first and second seasons
- **Post-trade effects** — dips in the 5 games after a trade, followed by adaptation
- **Matchup dynamics** — certain stat profiles against certain opponent archetypes
- **Season rhythm** — all-star break boost, playoff intensity, end-of-season rest patterns
- **Role changes** — minutes and usage shift when a teammate goes down

All of this emerges from a single self-supervised objective. No one labels "fatigue." The model learns it because predicting game 21 is impossible without understanding it.

This is not a prop bet model. This is a basketball foundation model.

---

## Data Strategy — Free, Official, and Complete

**All training data is free. No paid subscriptions. We go back to 1947.**

### Primary Training Data

| Data Source | What It Provides | Era Coverage | Cost |
|-------------|-----------------|--------------|------|
| Kaggle CC0 dataset (`eoinamoore/historical-nba-data-and-player-box-scores`) | Every player, every game, box scores + advanced stats | 1947–present | Free (CC0 public domain) |
| nba_api | Ongoing nightly ingestion of current season | 2023–present | Free (rate-limited) |
| Google News RSS | News and headlines | Live | Free |
| ESPN RSS | NBA news | Live | Free |

**No paid APIs. No basketball-reference (ToS bans AI training). No BallDontLie stats tier ($9.99/month — not worth it).**

### Era Architecture

NBA basketball has evolved dramatically since 1947. The model must understand this. Every training sequence carries an **era embedding**:

| Era | Years | Defining Characteristics |
|-----|-------|--------------------------|
| Pre-Shot Clock | 1947–1954 | Slow pace, low scoring, 8 original teams |
| Shot Clock Era | 1954–1976 | Pace increases, ABA competition, expansion |
| Showtime / Bird-Magic | 1977–1993 | Fast break basketball, dominance of big men and guards |
| Jordan Era | 1994–2002 | Physical defense, isolation, 90s grit |
| Post-Jordan Transition | 2003–2012 | Rise of analytics, pace-and-space begins |
| Three-Point Revolution | 2013–present | Positionless basketball, pace at all-time high |

The model is not expected to predict 1954 stats from 2024 patterns. Era embeddings let it understand that a 25-point game in 1957 means something different than a 25-point game in 2024. Temporal context is baked in, not ignored.

### Why 1947?

More data = better model. Early era games are noisy but structurally valid: fatigue, hot streaks, development arcs, and matchup dynamics are basketball universals that transcend era. The era embedding handles the distribution shift. A model trained on 78 years of basketball understands the game more deeply than one trained on 25.

**Training data volume (confirmed — dataset already downloaded):**
- `PlayerStatistics.csv` — 308 MB, every player-game box score 1947–present
- `PlayerStatisticsAdvanced.csv` — 40 MB, advanced stats (PER, TS%, USG%, ORTG, DRTG)
- `Games.csv` — 9 MB, game-level metadata (date, teams, arena, attendance)
- `PlayByPlay.parquet` — 810 MB (future use — possession-level modeling, Phase 3+)
- 20-game sliding windows → millions of training sequences
- CC0 public domain — legal for ML training, commercial use, redistribution
- nba_api for nightly top-up of current season (1 req/3-6 sec from home IP)

---

## What Exists Today

This project does not start from zero. A production-grade NBA data pipeline is already running:

- **XGBoost models** trained on historical game logs — predicting PTS, REB, AST, STL, BLK, FG3M per player
- **Crawlers** pulling injury reports, news (Google News + ESPN RSS), and daily schedules
- **MiroFish integration** already working — Zep knowledge graphs, multi-agent simulations, reports
- **S3 data lake** with partitioned parquet storage
- **Nightly orchestration** running full pipeline

The XGBoost pipeline becomes the baseline that NBA-GPT must beat. Not thrown away — promoted to benchmark.

---

## What Makes This Different

### The Soft Data Thesis

| Signal | Example | Impact | Traditional Model |
|--------|---------|--------|-------------------|
| Trade rumors | "Player X has requested a trade" | Distraction, effort decline, trust erosion | Cannot capture |
| Social media chemistry | Two players posting workouts together in offseason | Improved synergy, more pick-and-roll attempts | Cannot capture |
| Coach conflict | Tense postgame presser: "He needs to buy in" | Minutes reduction, attitude shift | Cannot capture |
| Contract motivation | Player in contract year averaging career highs | Increased effort, stat-padding | Partially (coarse) |
| Team morale | Blowout loss followed by players-only meeting | Performance bounce or continued spiral | Cannot capture |

The foundation model learns hard data patterns. The agent layer ingests soft data. Together they capture what no single approach can.

### Why a Foundation Model and Not Just Better Features

A transformer trained on raw game sequences learns features that no human would think to engineer. It discovers interaction effects across time that live in the weights, not in a feature matrix. More data makes it better automatically, without more feature engineering.

The foundation model is the statistical backbone. The LLM agents are the reasoning layer. XGBoost is the sanity check. Three systems, each doing what it does best.

---

## Architecture

```
                       +---------------------------+
                       |     Season Orchestrator    |
                       |  (schedule, standings,     |
                       |   playoff bracket, RL)     |
                       +------------+--------------+
                                    |
               +--------------------+--------------------+
               |                    |                    |
      +--------v--------+  +-------v--------+  +--------v--------+
      |  Game Simulator  |  | News Ingestion |  |  RL Evolution   |
      |  (per-game sim   |  | Pipeline       |  |  Engine         |
      |   with agents)   |  | (RSS, social)  |  |  (agent learning|
      +--------+--------+  +-------+--------+  +--------+--------+
               |                    |                    |
   +-----------v----------+  +------v-----------------+  |
   |   Player Agents      |  |  MiroFish + Zep        |  |
   |   (all 30 teams)     |  |  Graph Memory          |  |
   |   Qwen 2.5 14B       |  |  (relationships,       |  |
   |   via Ollama         |  |   chemistry,            |  |
   +-----------+----------+  |   team dynamics)       |  |
               |              +------+-----------------+  |
               +--------------+      +--------------------+
               |
      +--------v-----------------+
      |  NBA-GPT Foundation      |
      |  Model (Transformer)     |
      |  PyTorch + CUDA          |
      |  Trained on stats.nba.com|
      +--------+-----------------+
               |
      +--------v--------+
      |  XGBoost Baseline|
      |  (benchmark)     |
      +--------+--------+
               |
      +--------v--------+
      |  Monte Carlo     |
      |  Aggregation     |
      |  (N seasons →    |
      |   distributions) |
      +---------+-------+
                |
      +---------v---------+
      |  Polymarket Edge  |
      |  Calculator       |
      +-------------------+
```

### The Simulation Loop

1. **NBA-GPT provides the statistical anchor** — trained on 25 years of official NBA data
2. **LLM agent applies contextual adjustment** — bounded to +/-25% of foundation model prediction
3. **MiroFish manages the relational layer** — news and social signals update the Zep chemistry graph
4. **Game resolution** — sum adjusted player stats, apply pace factor, most points wins
5. **Season simulation** — 1,230 games + playoffs, run N times → distributions

---

## Phased Build Plan

### Phase 1 — Data Pipeline + Foundation Model (Weeks 1-8)

**Goal:** NBA-GPT trained on every player, every game, 2000-present. Free data only.

- **Data ingestion** — Kaggle CC0 dataset as bulk historical load (1947–present) + nba_api for current season top-up. Every player, every game: PTS, REB, AST, STL, BLK, FG3M, MIN, FGA, FGM, FTA, FTM, TO, PF, plus/minus, pace, opponent defensive rating, home/away, rest days, back-to-back flag, game number in season, **era embedding**.
- **Transformer architecture** — PyTorch encoder, input: 20-game sequence, output: predicted game 21 vector. Player embedding layer, era embedding layer, temporal positional encoding, multi-head attention. Mixed precision (fp16) for RTX 4080.
- **Training** — MSE loss, AdamW, cosine LR schedule. ~1.5M+ records, millions of sequences. Fully local on RTX 4080.
- **Validation** — beat XGBoost on held-out 2024-25 season on at least 4 of 6 stat categories.

**Success criteria:**
- Foundation model trained and converged
- Outperforms XGBoost baseline on held-out data
- Inference < 50ms per player (fast enough for Monte Carlo)

---

### Phase 2 — Agent Simulation Layer (Weeks 9-14)

**Goal:** LLM agents on top of the foundation model. Every player reasons about context.

- `PlayerAgent` — system prompt with NBA-GPT prediction + injury context + recent news. JSON adjustment output.
- `GameSimulator` — orchestrates both teams, resolves game outcome
- `LLMClient` — Ollama/Qwen 2.5 14B wrapper, sequential queue, retry logic
- `MonteCarloRunner` — N simulations → distributions
- Expand crawlers to all 30 teams

**Success criteria:**
- 100 simulations in under 2 hours
- Agent-adjusted predictions outperform raw NBA-GPT on MAE

---

### Phase 3 — Chemistry Engine via MiroFish (Weeks 15-20)

**Goal:** Relationships affect outcomes. News drives relationship changes.

- Expand news pipeline to all 30 teams (Google News RSS, ESPN RSS — free)
- Twitter/X monitoring for Woj, Shams, Haynes
- Sentiment classifier via Qwen — classifies news by type, identifies affected players
- MiroFish chemistry graph — players/coaches as nodes, trust/synergy as edges
- News events modify edge weights, feed into agent context

**Success criteria:**
- Chemistry graph for all 30 teams
- A/B test shows chemistry context changes distributions vs. baseline
- 3+ signal categories ingested

---

### Phase 4 — Full Season Simulation + RL Evolution (Weeks 21-30)

**Goal:** Full 82-game season simulation. Agents learn and evolve via RL.

- Season Orchestrator — schedule, standings, tiebreakers, playoff bracket, fatigue tracking
- Optimization: Qwen 7B for role players, 14B for stars. Context caching.
- Cloud burst for N=50+ seasons (~$15/season on AWS spot GPU)
- RL layer: agents rewarded for prediction accuracy on actual outcomes. Simulate Wemby at 26, 28, 30. Model rookie development arcs. Project post-trade adaptation.

**Success criteria:**
- One full season in under 12 hours locally
- Simulated win totals within 5 games of Vegas for 20+ teams
- RL agents outperform static agents on next-season prediction

---

### Phase 5 — Polymarket Integration (Weeks 31-36)

**Goal:** Simulation outputs → betting edges on prediction markets.

- Polymarket API client — NBA futures prices
- Edge calculator — flag divergences > 10%
- Kelly criterion sizing with liquidity awareness
- 60 days paper trading before real capital
- Automated execution post-validation

**Target markets:** Championship, conference winners, win totals, MVP, playoff qualification.

---

### Phase 6 — Generalization Beyond NBA (Ongoing)

**This is where the real vision lives.**

The architecture is not basketball-specific:
- Replace "player" with "voter", "game" with "election cycle" → political behavior model
- Replace "player" with "company", "game" with "quarter" → market dynamics model
- Replace "player" with "country", "game" with "year" → geopolitical simulation

NBA Simulverse is the proof of concept. Basketball is the training ground because the data is clean, the feedback is nightly, and the stakes are measurable. The endgame is a **general-purpose human behavior simulation engine**.

---

## What Gets Trained and Why

| Component | Approach | Rationale |
|-----------|----------|-----------|
| NBA-GPT foundation model | Self-supervised pre-training on raw sequences | Next-game prediction is the only objective. No labels needed. |
| XGBoost models | Already trained | Baseline benchmark. NBA-GPT must beat this to proceed. |
| Player LLM agents | Prompt engineering first, fine-tune after Phase 4 | Need labeled examples from simulation runs first. |
| RL agent policy | PPO, trained on season sim vs. actuals | Requires validated foundation model first. |

**Principle: self-supervised where possible, supervised where necessary, reinforcement where transformative.**

---

## MiroFish's Role

1. **Zep graph memory** — persistent player relationship graph. Nodes: players, coaches. Edges: trust, synergy, rivalry.
2. **Multi-agent orchestration** — already integrated and working.
3. **News-to-graph pipeline** — classifies events, updates edge weights automatically.

We build on top: NBA-specific ontology, relationship types, chemistry-to-performance mapping.

---

## Why This Gets You Hired by an NBA Team

1. **Original thinking** — self-supervised learning on basketball sequences is a novel contribution, not better feature engineering.
2. **Systems thinking** — foundation model + agent simulation + graph memory + RL. Builders, not just model fitters.
3. **The soft data thesis** — every GM believes it. Nobody has quantified it. This does.
4. **Spurs-specific value** — what does Wembanyama's stat line look like at 28? The RL evolution layer answers that directly.

**The interview conversation:**
> "I trained a basketball foundation model on 25 years of NBA data using self-supervised learning. It learned fatigue, hot streaks, and rookie development arcs from raw sequences — no manual labels. Then I built an agent simulation layer that ingests real news and social signals through a chemistry graph. The system simulates entire seasons. Here is what it says about the Spurs' championship timeline. Here is Wembanyama at 28. Here is the trade that moves the needle most."

That is not a job application. That is a conversation between colleagues.

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Foundation model does not beat XGBoost | Critical | Architecture search, reduce model size. If sequence model can't beat tree model, pivot to enhanced XGBoost + agents. |
| stats.nba.com rate limits or blocks | High | Aggressive caching, rate limiting, BallDontLie fallback. Build once, store forever. |
| Insufficient data for transformer training | High | Data augmentation, transfer from time series foundation models, reduce model size. |
| LLM agents produce uncalibrated adjustments | High | Bound to +/-25% of NBA-GPT prediction. Fall back to model-only if agents degrade accuracy. |
| Full season simulation too slow | High | Tiered model usage, context caching, cloud burst. |
| RL diverges or produces unrealistic projections | Medium | Constrain within historical development bounds. Validate against known career trajectories. |
| Scope creep delays foundation model | Critical | Phase 1 has independent value. Do not start Phase 2 until NBA-GPT beats XGBoost. |

---

## Timeline Summary

| Phase | Weeks | Deliverable | Standalone Value |
|-------|-------|-------------|-----------------|
| 1: NBA-GPT Foundation Model | 1-8 | Self-supervised transformer on 25 years of NBA data | "A basketball foundation model" |
| 2: Agent Simulation | 9-14 | LLM agents on top of foundation model | "Context-aware NBA prediction" |
| 3: Chemistry Engine | 15-20 | News-driven relationship graph | "Soft data quantification" |
| 4: Season Sim + RL | 21-30 | Full season simulation + player development projection | "Championship simulation + player arcs" |
| 5: Polymarket | 31-36 | Automated edge detection, paper trading | Monetization validation |
| 6: Generalization | Ongoing | Framework beyond NBA | "General-purpose behavior simulation" |

---

## The Bottom Line

The NBA analytics landscape is saturated with teams running the same models on the same data. The next frontier is not better box score regression. It is building a model that understands basketball the way GPT understands language — from raw sequences, not hand-crafted features.

NBA-GPT is that model. Trained on free, official NBA data. On top of that foundation, LLM agents reason about soft signals no box score captures. A graph-based chemistry engine quantifies relationships. Reinforcement learning lets agents evolve, projecting player development years into the future.

And the architecture is general. Basketball is the training ground. The endgame is a general-purpose simulation framework for complex human systems.

The question is not whether soft data matters in basketball. Everyone knows it does. The question is not whether self-supervised learning can discover structure in sequential data. GPT proved it can. The question is whether anyone will build the system that combines both insights and points it at basketball.

This is that system.

---

*NBA Simulverse — Where every player has a mind, every team has a heartbeat, and every season tells a story.*
