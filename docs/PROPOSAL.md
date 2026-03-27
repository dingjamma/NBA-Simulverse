# NBA Player Agent Simulation System — Proposal

## North Star
Each NBA player is an LLM agent with real stats. They "play" against each other in simulation. Run 100-200 games → probability distributions → compare vs Polymarket lines → bet when you have edge.

## Architecture
**XGBoost (statistical baseline) + LLM agents (contextual reasoning) = calibrated probabilities**

The LLM layer is your edge over the books — it understands "Wemby on a back-to-back against a top defense after a road trip." Pure stats can't reason about that.

---

## 4 Phases

### Phase 0 — Data Foundation (Weeks 1-2)
- Expand crawlers from Wemby-only → all players in tonight's games
- Add team pace + defensive rating features
- Build a bet tracking SQLite database (start recording every prediction from day 1)

### Phase 1 — MVP Simulation (Weeks 3-5)
- `PlayerAgent` class — each player gets a system prompt with their stats, matchup, injury context, anchored to XGBoost predictions
- `GameOrchestrator` — runs 4 quarters, tracks fouls/blowouts/bench decisions
- `LLM Client` — wraps Ollama, sequential queue (4080 can only do 1 inference at a time)
- Run 100 sims → distributions → edge calculator (P_over_simulated vs P_over_implied)

**Reality check on speed:** 40 LLM calls per sim × 3 sec each × 100 sims = ~2 hours. Feasible. Not 1000 sims though — that's 40 hours. Start with 100.

### Phase 2 — Calibration + Polymarket (Weeks 6-8)
- Backtesting against historical games
- Calibration layer (your distributions will be wrong at first, fix systematically)
- Polymarket API client — paper trade for 30 days before real money
- Risk management — Kelly sizing, daily loss limits

### Phase 3 — Go Live (Weeks 9-12)
- Lighter model experiments (Qwen 7B for agents, 14B for orchestrator)
- Live execution only after 30 days paper trading with positive ROI

---

## Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Simulation granularity | Quarter-level (not possession) | 40 LLM calls vs 200+ per sim |
| LLM | Qwen 2.5 14B local | Already running, free |
| Stats backbone | Keep XGBoost | LLMs hallucinate numbers without an anchor |
| MiroFish | Keep for now, retire by Phase 2 | Agent simulation replaces it |

---

## New Directory Structure

```
NBA-Player-Prop/
├── crawlers/
│   ├── player_logs.py         (new — Phase 0)
│   └── team_stats.py          (new — Phase 0)
├── model/
│   ├── features.py            (modify — Phase 0)
│   ├── train.py               (modify — Phase 0)
│   └── predict.py             (modify — Phase 0)
├── simulation/                (new — Phase 1)
│   ├── agents/
│   │   ├── player_agent.py
│   │   └── prompts.py
│   ├── orchestrator.py
│   ├── runner.py
│   ├── llm_client.py
│   ├── edge.py
│   ├── calibrator.py          (Phase 2)
│   └── models.py
├── execution/                 (new — Phase 2)
│   ├── polymarket.py
│   └── risk.py
├── tracking/                  (new — Phase 0)
│   ├── tracker.py
│   └── schema.py
├── backtest/                  (new — Phase 2)
│   ├── runner.py
│   └── metrics.py
└── docs/
    └── PROPOSAL_agent_simulation.md
```

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| LLM produces uncalibrated distributions | High | Calibration layer + always ensemble with XGBoost |
| 100 sims insufficient for stable distributions | Medium | Bootstrap confidence intervals, add cloud GPU if needed |
| Polymarket NBA markets have low liquidity | High | Check depth before placing, use limit orders |
| Simulation takes too long for nightly window | Medium | Start pipeline at 8 PM, run sims right after lineups announced |
| Edge disappears when lines move | Medium | Fetch lines at 6 PM, 8 PM, 10 PM — only bet stable edges |

---

## Success Criteria

### Phase 0
- [ ] XGBoost predictions for all starters in tonight's games
- [ ] Team pace and defensive rating features added
- [ ] Bet tracking database recording every prediction

### Phase 1
- [ ] 100 simulations complete in under 3 hours on local hardware
- [ ] Stat distributions with mean, std, and percentiles per player
- [ ] Edge calculator flags bets with >5% edge
- [ ] Nightly pipeline runs end-to-end without manual intervention

### Phase 2
- [ ] Backtesting over 50+ historical games shows simulation adds value over XGBoost alone
- [ ] Calibration brings Brier score below 0.25
- [ ] 30 days paper trading on Polymarket with logged results
- [ ] Positive simulated ROI after vig

### Phase 3
- [ ] Live execution on Polymarket
- [ ] 30-day rolling ROI of 5%+ after vig
- [ ] Fully autonomous nightly pipeline through bet execution

---

## What NOT to Build
- Custom LLM fine-tuning (overkill for now)
- Real-time in-game betting (too complex)
- Multi-sport expansion (nail NBA first)
- Custom frontend (Streamlit is fine)
- Distributed simulation infrastructure (one machine, sequential)
