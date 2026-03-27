# NBA Simulverse — Project Proposal

## The Vision

Every NBA analytics department in the league runs the same regression models on the same box scores. They all have the same data. They all reach roughly the same conclusions.

**NBA Simulverse is built on a different thesis: the most valuable signals in basketball are the ones that never appear in a box score.**

A trade rumor breaks on Shams. Two teammates unfollow each other on Instagram. A coach's postgame presser is tense. A player signs a max extension and plays free. These signals — locker room chemistry, social media sentiment, news-driven morale shifts — ripple through performance in ways that no regression model captures. Every scout and GM knows this intuitively. None of them have a system that quantifies it.

NBA Simulverse is a full-season simulation engine where every player is an autonomous LLM agent with memory, personality, and relationships. These agents do not just crunch stats. They react to news. They have chemistry with teammates that evolves over the season. They get rattled by trade rumors and energized by contract extensions. They play 82 games, a playoff bracket, and crown a champion — hundreds of times — producing probability distributions grounded in both hard stats and soft context that no existing model touches.

This is not a dashboard. It is not a prop bet calculator. It is a living, breathing NBA universe that runs in parallel to the real one — and it captures the human element that decides championships.

---

## What Exists Today

This project does not start from zero. A production-grade NBA data pipeline (`NBA-Player-Prop`) is already running nightly:

- **XGBoost models** trained on historical game logs with rolling averages, rest days, back-to-back flags, usage proxies, opponent-specific splits, and season averages — predicting PTS, REB, AST, STL, BLK, FG3M per player
- **Crawlers** pulling injury reports, prop lines (DraftKings/FanDuel/Pinnacle), news (Google News + ESPN RSS), and daily schedules
- **MiroFish integration** already working — uploading seed files, building Zep knowledge graphs, running multi-agent simulations, generating reports
- **S3 data lake** with partitioned parquet storage
- **Nightly orchestration** running full pipeline from crawl to prediction to report

The foundation is laid. Simulverse scales it from single-game prop predictions to a full-season simulation engine powered by agent intelligence.

---

## What Makes This Different

### The Soft Data Thesis

Every public model in sports analytics operates on the same hard data: box scores, play-by-play logs, tracking data. The edge has been arbitraged away. The remaining alpha lives in **soft data** — information that is abundant, public, but structurally impossible for traditional models to ingest:

| Signal | Example | Impact | Traditional Model |
|--------|---------|--------|-------------------|
| Trade rumors | "Player X has requested a trade" | Distraction, effort decline, trust erosion | Cannot capture |
| Social media chemistry | Two players posting workouts together in offseason | Improved synergy, more pick-and-roll attempts | Cannot capture |
| Coach conflict | Tense postgame presser: "He needs to buy in" | Minutes reduction, attitude shift | Cannot capture |
| Contract motivation | Player in contract year averaging career highs | Increased effort, stat-padding | Partially (coarse) |
| Team morale | Blowout loss followed by players-only meeting | Performance bounce or continued spiral | Cannot capture |

NBA Simulverse is the first system that turns these signals into simulated outcomes at scale.

### Why LLM Agents and Not Just Sentiment Scores

A naive approach: scrape news, run sentiment analysis, add a "morale score" feature to XGBoost. That misses the point. Soft signals are **relational and contextual**. A trade rumor about Player A affects the teammate who gets more shots, the young player auditioning for the departing role, and the opponent's defensive scheme. These are second and third-order effects that require reasoning, not regression.

An LLM agent for Victor Wembanyama knows:
- His stat baseline and recent form
- His relationships with teammates (chemistry graph via Zep memory)
- His matchup history against tonight's opponent
- Current news context (hot streak, back-to-back, trade deadline chaos)

When the agent "plays" a simulated game, it reasons about all of this simultaneously — the way a scout does, but at simulation scale.

---

## Architecture

```
                    +---------------------------+
                    |     Season Orchestrator    |
                    |  (schedule, standings,     |
                    |   playoff bracket logic)   |
                    +------------+--------------+
                                 |
               +-----------------+-----------------+
               |                                   |
      +--------v--------+               +----------v--------+
      |  Game Simulator  |               |  News Ingestion   |
      |  (per-game sim   |               |  Pipeline         |
      |   with agents)   |               |  (RSS, social,    |
      +--------+--------+               |   sentiment)      |
               |                         +----------+--------+
               |                                    |
    +----------v-----------+           +------------v-----------+
    |   Player Agents      |           |  MiroFish + Zep        |
    |   (all 30 teams)     |           |  Graph Memory          |
    |                      |           |  (relationships,       |
    |   Qwen 2.5 14B       |           |   chemistry,           |
    |   via Ollama         |           |   team dynamics)       |
    +----------+-----------+           +------------+-----------+
               |                                    |
               +------------------------------------+
               |
      +--------v--------+
      |  XGBoost Anchor  |
      |  (stat baseline  |
      |   per player)    |
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

1. **XGBoost provides the statistical anchor.** For every player in every game, the trained models produce a baseline prediction. This prevents LLM hallucination of numbers.

2. **The LLM agent applies contextual adjustment.** Each player agent receives its XGBoost baseline plus current context. It outputs an adjustment factor and reasoning. Adjustments are bounded to +/-25% of baseline to prevent runaway hallucination.

3. **MiroFish manages the relational layer.** Zep's graph memory stores player-to-player and player-to-team relationships. When a trade rumor hits, MiroFish propagates the effect: the rumored player's morale drops, the replacement's confidence rises, team cohesion shifts.

4. **Game resolution is simple arithmetic.** Sum adjusted player stats per team. Apply pace factor. Most points wins. Quarter-level granularity — enough to capture blowouts, foul trouble, and rest without burning VRAM.

5. **Season simulation repeats 1,230 times** (82 games × 30 teams / 2) plus playoffs. Run N full seasons → probability distributions.

---

## Phased Build Plan

### Phase 1 — Single-Game Agent Simulation (Weeks 1-4)

**Goal:** One game, simulated 100 times, with player agents producing calibrated stat distributions.

- `PlayerAgent` class — system prompt with stats, XGBoost baseline, injury context, recent news. Outputs JSON adjustment to baseline.
- `GameSimulator` — orchestrates 10 players per team, resolves game outcome
- `LLMClient` — Ollama wrapper with sequential queue, retry logic, structured output parsing
- `MonteCarloRunner` — N simulations → distribution (mean, std, percentiles)
- Expand crawlers from Spurs-only to all 30 teams

**Hardware reality:** 20 agents × 3 sec/inference × 100 sims = ~100 min per game. Feasible nightly.

**Success criteria:**
- 100 simulations complete in under 2 hours on local hardware
- Backtested against 20 historical games — agent-adjusted predictions outperform raw XGBoost on MAE

---

### Phase 2 — Chemistry Engine via MiroFish (Weeks 5-8)

**Goal:** Player relationships affect simulation outcomes. News drives relationship changes.

- Expand news pipeline to all 30 teams. Add Twitter/X monitoring for Woj, Shams, Haynes. Add Instagram activity tracking.
- Sentiment classifier — Qwen classifies news by type (trade rumor, injury, chemistry signal, coach conflict, contract news) and affected players
- MiroFish chemistry graph — players and coaches as nodes, relationship strength/trust/synergy as edges. News events modify edge weights.
- Chemistry-to-performance mapping — graph state feeds into each agent's context window
- Daily news digest — runs every 6 hours, classifies events, updates graph

**Why MiroFish:** It already handles multi-agent simulation with Zep graph memory. The relationship propagation logic is exactly what Zep is designed for. Building from scratch would take months.

**Success criteria:**
- Chemistry graph populated for all 30 teams
- A/B test shows chemistry context produces different distributions vs. no-chemistry baseline
- 3+ signal categories ingested: trade rumors, injury uncertainty, social media interactions

---

### Phase 3 — Full Season Simulation (Weeks 9-14)

**Goal:** Simulate an entire 82-game NBA season including standings, tiebreakers, and playoffs.

- **Season Orchestrator** — full schedule management, standings, tiebreakers, playoff bracket, cumulative fatigue tracking
- **Batch inference optimization:**
  - Qwen 7B for role players, 14B for stars
  - Context caching for similar game situations
  - Target: one full season in 8-12 hours (overnight batch)
- **Cloud burst** — for N=50+ seasons, AWS spot GPU instances (~$15/season). 50 seasons ≈ $750 for a full championship probability analysis.
- **Mid-season event injection** — handles real trades, injuries, coaching changes at the correct point in the simulated schedule

**Success criteria:**
- One full season completes in under 12 hours locally
- 10 season simulations produce stable playoff probability distributions (SE < 5% for top teams)
- Simulated win totals within 5 games of Vegas over/unders for 20+ teams

---

### Phase 4 — Polymarket Integration (Weeks 15-18)

**Goal:** Turn simulation outputs into betting edges on prediction markets.

- **Polymarket API client** — fetch NBA futures prices (championship, conference, MVP, win totals, playoff qualification)
- **Edge calculator** — compare sim probabilities vs. Polymarket implied probabilities. Flag divergences >10%.
- **Kelly criterion sizing** — fractional Kelly, factor in market liquidity
- **Paper trading** — log all recommended bets for 60 days before real capital. Track ROI, Sharpe, max drawdown.
- **Automated execution** (post-validation) — limit orders only, daily loss limits, circuit breaker at 15% drawdown

**Target markets:**
- NBA Championship winner (largest liquidity)
- Conference and division winners
- Individual team win totals
- MVP race
- Playoff qualification (yes/no per team)

**Success criteria:**
- 60 days paper trading with logged results
- Positive simulated ROI after Polymarket fees (2%)
- 3+ identified edges per month where simulation diverges from market by >10%

---

### Phase 5 — Continuous Learning (Ongoing)

- Backtesting harness — after each real game, compare simulated outcomes to actual. Track calibration curves.
- Agent prompt evolution — data-driven refinement of what contextual factors actually improve predictions
- Chemistry model validation — rigorous A/B testing: does chemistry context improve calibration?
- Fine-tuning investigation — after 500+ labeled agent response/outcome pairs, explore fine-tuning Qwen on NBA-specific reasoning

---

## What Gets Trained and Why

| Component | Approach | Rationale |
|-----------|----------|-----------|
| XGBoost stat models | Train on historical game logs (done) | Proven, fast, interpretable. Prevents LLM hallucination. |
| Player LLM agents | Prompt engineering first, fine-tune in Phase 5 | Need hundreds of labeled examples before fine-tuning is justified |
| News sentiment classifier | Few-shot prompting → fine-tune when data accumulates | Same logic |
| Chemistry-to-performance | Heuristic rules → learned weights after 50+ observed events | Start simple, learn real coefficients from data |

**Principle: do not fine-tune until you have the data to justify it.**

---

## MiroFish's Role

MiroFish is not a nice-to-have — it is the infrastructure that makes the chemistry engine possible without building a custom graph database from scratch.

**What MiroFish provides:**
1. **Zep graph memory** — persistent, queryable graph of player relationships. Nodes are players and coaches. Edges carry trust, synergy, and relationship history. Survives across simulation runs.
2. **Multi-agent orchestration** — handles agent lifecycle, message passing, and simulation execution. Already integrated and working.
3. **News-to-graph pipeline** — classifies news events, identifies affected entities, updates edge weights automatically.

**What we build on top:**
- NBA-specific ontology (teams, players, coaches, agents, front offices)
- NBA-specific relationship types (teammate, rival, mentor/mentee, coach-player)
- Chemistry-to-performance mapping that converts graph state into agent prompt context

---

## Why This Gets You Hired by an NBA Team

Every applicant to an NBA analytics role submits a regression model and some shot chart visualizations. Table stakes.

**NBA Simulverse demonstrates something different:**

1. **Systems thinking.** This is an integrated system — data pipelines, multi-agent simulation, graph-based relationship modeling, automated decision-making. NBA front offices need builders.

2. **The soft data thesis is the thesis every GM already believes.** Every GM talks about culture, fit, and locker room chemistry. They make trades based on it. No analytics department has a systematic way to quantify it. Showing up with a working prototype changes that conversation.

3. **Applied AI.** The NBA is actively exploring LLMs and AI agents. A candidate who has built a multi-agent simulation with graph-based memory, grounded in real statistical models, is exactly what forward-thinking organizations are looking for.

4. **Spurs-specific value.** The existing pipeline is already Spurs-focused. The Spurs, under RC Buford and Brian Wright, have historically been analytically progressive. A system that quantifies how Wembanyama's development, teammate chemistry, and coaching dynamics interact is directly relevant to their most important strategic question.

**The conversation in the interview:**

> "I built a system that simulates entire NBA seasons using LLM agents that react to real news and social signals, with a graph-based chemistry engine that captures locker room dynamics. Here is what it says about the Spurs' championship timeline. Here is where the simulation disagrees with Vegas. Here is the specific trade that improves the Spurs' title odds by 8%."

That is not a job application. That is a conversation between colleagues.

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| LLM agents produce uncalibrated adjustments | High | Bound to +/-25% of XGBoost baseline. Fall back to XGBoost-only if agents degrade accuracy. |
| Full season simulation too slow on local hardware | High | Tiered model (7B for role players, 14B for stars). Context caching. Cloud burst for Monte Carlo. |
| Chemistry engine adds noise instead of signal | Medium | Rigorous A/B testing from Phase 2. Remove if no improvement after 100 games. |
| Polymarket markets have insufficient liquidity | High | Check order book depth. Use limit orders. Pivot to analysis/content if liquidity too thin. |
| Scope creep delays core simulation | High | Each phase has independent value. Do not start Phase N+1 until Phase N is validated. |

---

## Timeline Summary

| Phase | Weeks | Deliverable | Standalone Value |
|-------|-------|-------------|-----------------|
| 1: Single-Game Agent Sim | 1-4 | 100-sim game predictions with agent reasoning | "LLM agents that reason about NBA matchups" |
| 2: Chemistry Engine | 5-8 | News-driven relationship graph affecting outcomes | "Soft data quantification system" |
| 3: Full Season Sim | 9-14 | Complete season simulation with playoff bracket | "Full NBA Monte Carlo with agent intelligence" |
| 4: Polymarket Integration | 15-18 | Automated edge detection and paper trading | Monetization validation |
| 5: Continuous Learning | Ongoing | Calibration, fine-tuning, production hardening | Production system |

---

## The Bottom Line

The NBA analytics landscape is saturated with teams running the same models on the same data. The next frontier is not better box score regression — it is systematic ingestion of the soft signals that every basketball person knows matter but no one has figured out how to quantify at scale.

NBA Simulverse does this. It is technically ambitious but architecturally grounded, built on an existing production pipeline, and designed to produce something no other model produces: championship probability distributions that account for locker room chemistry, trade deadline chaos, and the human dynamics that actually decide who raises the Larry O'Brien trophy.

The question is not whether soft data matters in basketball. Everyone knows it does. The question is whether anyone can build a system that captures it.

This is that system.

---

*NBA Simulverse — Where every player has a mind, every team has a heartbeat, and every season tells a story.*
