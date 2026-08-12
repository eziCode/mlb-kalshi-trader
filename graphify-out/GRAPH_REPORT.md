# Graph Report - .  (2026-08-12)

## Corpus Check
- 154 files · ~93,387 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 968 nodes · 2282 edges · 43 communities (40 shown, 3 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 114 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Mispricing Research
- Live Execution Risk
- Scoring and Latency
- Competing Risks Runtime
- Shared Data Assembly
- MLB Feed Runtime
- Kalshi Feed Runtime
- Trade Tape Backtests
- Game Market Snapshots
- Strategy Runner Config
- Trade Entry Logic
- MLB Event Alignment
- Paper Portfolio Models
- Kalshi Trade Download
- Kalshi Market Discovery
- MLB Feed Download
- Settlement Policy Tests
- Portfolio Settlement
- Strategy Concepts
- Hybrid Strategy Model
- Pipeline Integration Tests
- Slate Coordination
- Worker Process Control
- Portfolio Reporting
- Event Feature Engineering
- Model Tuning Runtime
- Slate Date Scheduling
- Portfolio Data Paths
- Combined Live Runner
- Combined Paper Runner
- Causal Entry Replay
- Live Feature Construction
- Robust Policy Tuning
- Data Setup Pipeline
- Statcast Download
- Container Entrypoint
- Holdout Diagnostics
- Raw Data Collection
- Worker Log Relay
- Pitch State Features
- Shared Live Infrastructure
- Settlement Strategy Package
- Project Package Root

## God Nodes (most connected - your core abstractions)
1. `TradeTapeConfig` - 71 edges
2. `TradeTapeStrategyTests` - 69 edges
3. `simulate_trade_tape()` - 53 edges
4. `main()` - 38 edges
5. `MispricingConfig` - 36 edges
6. `PipelineTests` - 35 edges
7. `LiveExecutor` - 32 edges
8. `LiveRiskLedger` - 31 edges
9. `SharedPaperPortfolio` - 29 edges
10. `simulate_paired_both()` - 28 edges

## Surprising Connections (you probably didn't know these)
- `State-Adjusted Market Target` --semantically_similar_to--> `Anchored State Target`  [INFERRED] [semantically similar]
  hit_reversion_strategy/README.md → settlement_value_strategy/README.md
- `Causal Trade-Tape Fill Proxy` --semantically_similar_to--> `Causal Decision Timeline`  [INFERRED] [semantically similar]
  hit_reversion_strategy/README.md → settlement_value_strategy/README.md
- `MarketSnapshot` --uses--> `LiveExecutor`  [INFERRED]
  hit_reversion_strategy/scripts/paper_trade.py → live_trading/execution.py
- `GameSnapshot` --uses--> `LiveExecutor`  [INFERRED]
  hit_reversion_strategy/scripts/paper_trade.py → live_trading/execution.py
- `Position` --uses--> `LiveExecutor`  [INFERRED]
  hit_reversion_strategy/scripts/paper_trade.py → live_trading/execution.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Shared Causal Market Data Flow** — data_processing_scripts_readme_pitch_state_feature_builder, data_processing_scripts_readme_shared_data_builder, data_readme_shared_normalized_inputs, settlement_value_strategy_readme_causal_decision_timeline, hit_reversion_strategy_readme_causal_fill_proxy [INFERRED 0.95]
- **Anchored State Repricing Framework** — data_readme_exact_kalshi_executions, hit_reversion_strategy_readme_state_adjusted_market_target, settlement_value_strategy_readme_anchored_state_target [INFERRED 0.95]
- **Strategy Safety and Validation** — readme_guarded_real_money_execution, settlement_value_strategy_readme_hold_to_settlement_policy, settlement_value_strategy_results_readme_forward_validation_requirement [INFERRED 0.85]

## Communities (43 total, 3 thin omitted)

### Community 0 - "Mispricing Research"
Cohesion: 0.05
Nodes (79): range, deployed_config(), main(), metrics(), Leak-free chronological backtest of the deployed latency policy., main(), Evaluate the frozen settlement-value strategy on development holdout., apply_early_exits() (+71 more)

### Community 1 - "Live Execution Risk"
Cohesion: 0.06
Nodes (18): contracts_for_budget(), KalshiAccountClient, LiveExecutor, LiveFill, LiveRiskLedger, main(), Connection, datetime (+10 more)

### Community 2 - "Scoring and Latency"
Cohesion: 0.08
Nodes (48): main(), Diagnose where causal hit-reversion opportunities leave the entry funnel., apply_live_paired_execution_prices(), apply_publication_latency(), load_latency_profile(), main(), parse_args(), publication_delay_seconds() (+40 more)

### Community 3 - "Competing Risks Runtime"
Cohesion: 0.07
Nodes (41): candidate_value_features(), canonical_kalshi_code(), ensure_decision_log_schema(), event_scheduled_time(), event_within_entry_window(), fetch_market_snapshot(), fetch_mlb_payload(), fetch_model_pregame_prior() (+33 more)

### Community 4 - "Shared Data Assembly"
Cohesion: 0.08
Nodes (38): CatBoostClassifier, build_shared(), canonical_team(), feed_rows(), load_downloaded_trades(), load_feed(), load_pitch_states(), map_games_to_markets() (+30 more)

### Community 5 - "MLB Feed Runtime"
Cohesion: 0.06
Nodes (11): Event, Exception, FeedState, GameFeed, Handler, BaseHTTPRequestHandler, Session, One adaptive MLB live-feed poller shared by all paper workers. (+3 more)

### Community 6 - "Kalshi Feed Runtime"
Cohesion: 0.08
Nodes (18): _add_trade(), _bootstrap(), FeedMarket, FeedState, Handler, _headers(), _number(), _process() (+10 more)

### Community 7 - "Trade Tape Backtests"
Cohesion: 0.16
Nodes (8): TradeTapeStrategyTests, event_target(), DataFrame, Apply a symmetric, bounded model state move to the market anchor., simulate_trade_tape(), TradeTapeConfig, TradeTapeResult, Portable exact-timestamp Kalshi MLB trade-tape strategy.

### Community 8 - "Game Market Snapshots"
Cohesion: 0.09
Nodes (30): _completed_play_score_for_pitch(), consecutive_pitch(), _event_scheduled_time(), execution_within_window(), fetch_game_snapshot(), fetch_market_snapshot(), fetch_mlb_payload(), fetch_pregame_anchor() (+22 more)

### Community 9 - "Strategy Runner Config"
Cohesion: 0.11
Nodes (28): add_runner_flags(), build_state_updates(), extract_cached_pitch_times(), load_game_map(), main(), parse_utc(), DataFrame, Timestamp (+20 more)

### Community 10 - "Trade Entry Logic"
Cohesion: 0.09
Nodes (25): Candidate, compatible_taker(), _direct_model_accepts(), _dynamic_target(), _material_state(), _ns_to_timestamp(), PendingEntry, position_exit_target() (+17 more)

### Community 11 - "MLB Event Alignment"
Cohesion: 0.10
Nodes (18): completed_play_pitch_token(), event_inputs_aligned(), fetch_game_snapshot(), GameSnapshot, latest_completed_pitch_token(), latest_resolved_play(), play_has_atomic_runner_state(), Return a stable identity for the latest pitch with an end timestamp. (+10 more)

### Community 12 - "Paper Portfolio Models"
Cohesion: 0.13
Nodes (9): PortfolioMetrics, Position, Connection, Find the backtest-style trade after target reversion that can exit., SQLite-backed cash and positions shared by every game process., Credit a partial sale and return the persisted remainder., replay_position_exit(), SharedPaperPortfolio (+1 more)

### Community 13 - "Kalshi Trade Download"
Cohesion: 0.20
Nodes (25): api_get(), discover_events(), fetch_all_pages(), fetch_event_markets(), fetch_market_trades(), get_historical_cutoffs(), main(), make_session() (+17 more)

### Community 14 - "Kalshi Market Discovery"
Cohesion: 0.20
Nodes (25): api_get(), discover_events(), fetch_all_pages(), fetch_event_markets(), fetch_market_trades(), get_historical_cutoffs(), main(), make_session() (+17 more)

### Community 15 - "MLB Feed Download"
Cohesion: 0.13
Nodes (23): api_get(), cache_path_for(), discover_game_pks(), extract_pitches_from_feed(), fetch_live_feed(), load_cached_feed(), main(), make_session() (+15 more)

### Community 16 - "Settlement Policy Tests"
Cohesion: 0.17
Nodes (14): conflicting_positions(), MarketSnapshot, PaperPosition, Paired home YES and away YES positions settle oppositely., Find positions whose executable bid crossed the configured stop., Apply the backtest's post-signal compatible-trade fill contract., replay_fill_from_observed_trades(), stop_loss_positions() (+6 more)

### Community 17 - "Portfolio Settlement"
Cohesion: 0.13
Nodes (10): Connection, SQLite-backed cash and settlement positions shared by game workers., Settle positions whose workers disappeared before observing Final., Durably import an exchange fill exactly once after a crash/restart., Remove one sold position and credit its net exchange proceeds., Credit a partial sale and return the contracts still open., Finalize a game once, including the live recovery ledger., reconcile_final_positions() (+2 more)

### Community 18 - "Strategy Concepts"
Cohesion: 0.12
Nodes (22): Avoid One-Minute Candles, Canonical Data Pipeline, Exact Kalshi Executions, Shared Normalized Inputs, Causal Trade-Tape Fill Proxy, Delayed Market Reaction, Event-Reaction Reversion, State-Adjusted Market Target (+14 more)

### Community 19 - "Hybrid Strategy Model"
Cohesion: 0.16
Nodes (16): add_event_targets(), anchored_event_target(), _expit(), hybrid_signal(), HybridConfig, HybridPosition, HybridResult, _logit() (+8 more)

### Community 20 - "Pipeline Integration Tests"
Cohesion: 0.17
Nodes (4): discover_daily_games(), DiscoveredGame, _market_team_code(), PipelineTests

### Community 21 - "Slate Coordination"
Cohesion: 0.18
Nodes (12): clock_time_delta(), current_slate_date(), discover_daily_games(), DiscoveredGame, match_games_to_home_markets(), date, timedelta, Compare Eastern clock times while ignoring a postponed ticker's date. (+4 more)

### Community 22 - "Worker Process Control"
Cohesion: 0.20
Nodes (7): ForkedWorker, Linux fork workers that share imported runtime pages copy-on-write., Fork an imported coordinator and capture the child's combined output., spawn_forked_worker(), worker_lifecycle_line(), ForkedWorkerTests, skipUnless

### Community 23 - "Portfolio Reporting"
Cohesion: 0.23
Nodes (10): format_live_portfolio_summary(), Path, Authoritative live-account reporting with per-strategy shadow-book detail., Read a strategy book without treating its duplicated seed cash as capital., Format one account total; strategy books contribute detail, not capital., read_strategy_book(), StrategyBookSnapshot, FakeClient (+2 more)

### Community 24 - "Event Feature Engineering"
Cohesion: 0.26
Nodes (12): add_completed_event_availability(), build_game_state(), load_mlb_timestamps(), load_statcast(), main(), merge_pitch_timestamps(), DataFrame, data/processed/scripts/build_event_state_features.py… (+4 more)

### Community 25 - "Model Tuning Runtime"
Cohesion: 0.27
Nodes (10): BaseContext, _evaluate_configuration(), _initialize_worker(), main(), parse_args(), print_progress(), _process_context(), DataFrame (+2 more)

### Community 26 - "Slate Date Scheduling"
Cohesion: 0.25
Nodes (11): _clock_time_delta(), current_slate_date(), _daily_kalshi_events(), _event_game_date(), date, timedelta, Return active events that can settle games on this slate. Kalshi tickers retain…, Compare Eastern clock times while ignoring a postponed ticker's date. (+3 more)

### Community 27 - "Portfolio Data Paths"
Cohesion: 0.38
Nodes (6): hit_reversion_path(), date, Path, Slate-scoped strategy ledger paths for continuous live trading., settlement_value_path(), PortfolioPathTest

### Community 28 - "Combined Live Runner"
Cohesion: 0.36
Nodes (8): main(), date, Popen, Run both real-money strategies behind one Kalshi and one MLB feed., run(), settlement_enabled(), stop(), wait_ready()

### Community 29 - "Combined Paper Runner"
Cohesion: 0.33
Nodes (7): main(), date, Popen, Run both paper strategies behind one shared Kalshi WebSocket., run(), _stop(), _wait_for_feed()

### Community 30 - "Causal Entry Replay"
Cohesion: 0.39
Nodes (5): EventCandidate, Replay the backtest's confirmation and strictly later entry fill., replay_candidate_entry(), position_contracts(), taker_fee()

### Community 31 - "Live Feature Construction"
Cohesion: 0.29
Nodes (7): build_live_decision_row(), flow_features(), _logit(), DataFrame, Training-identical two-second flow strictly before the signal trade., Construct one causal model row, or return None until inputs qualify., state_delta()

### Community 32 - "Robust Policy Tuning"
Cohesion: 0.43
Nodes (6): assert_leak_free_inputs(), candidate_config(), evaluate(), main(), DataFrame, Robustness-first, pre-holdout tuning for the hit-reversion policy.

### Community 33 - "Data Setup Pipeline"
Cohesion: 0.47
Nodes (5): main(), parse_args(), Namespace, Download and process every input required by both trading strategies., run_step()

### Community 34 - "Statcast Download"
Cohesion: 0.60
Nodes (4): download_season(), main(), month_ranges(), Generate month-sized date ranges from start_date to end_date.

### Community 35 - "Container Entrypoint"
Cohesion: 0.90
Nodes (4): run_mispricing(), run_trade_tape(), docker-entrypoint.sh script, usage()

### Community 36 - "Holdout Diagnostics"
Cohesion: 0.60
Nodes (4): evaluate(), initialize(), main(), Report holdout sensitivity by hit type, side, and edge threshold.

### Community 37 - "Raw Data Collection"
Cohesion: 0.67
Nodes (3): get_json(), main(), Collect public Kalshi trades and MLB live-feed snapshots for explicit IDs.

### Community 38 - "Worker Log Relay"
Cohesion: 0.50
Nodes (3): Persist all worker detail, but surface only executions and settlement., _relay(), should_surface_worker_line()

### Community 39 - "Pitch State Features"
Cohesion: 0.67
Nodes (3): Exclude Unused Context Features, Pitch-State Feature Builder, Shared Data Builder

## Knowledge Gaps
- **6 isolated node(s):** `mlb-kalshi-hit-reversion`, `StrategyConfig`, `Canonical Data Pipeline`, `Shared Data Builder`, `Hit-Reversion Dependencies` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LiveExecutor` connect `Live Execution Risk` to `Competing Risks Runtime`, `Game Market Snapshots`, `MLB Event Alignment`, `Paper Portfolio Models`, `Settlement Policy Tests`, `Portfolio Settlement`, `Pipeline Integration Tests`, `Slate Coordination`, `Slate Date Scheduling`, `Causal Entry Replay`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Why does `TradeTapeConfig` connect `Trade Tape Backtests` to `Robust Policy Tuning`, `Scoring and Latency`, `Competing Risks Runtime`, `Holdout Diagnostics`, `Trade Entry Logic`, `MLB Event Alignment`, `Paper Portfolio Models`, `Slate Coordination`, `Model Tuning Runtime`, `Causal Entry Replay`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `main()` connect `Competing Risks Runtime` to `Live Execution Risk`, `Shared Data Assembly`, `Trade Tape Backtests`, `Trade Entry Logic`, `MLB Event Alignment`, `Paper Portfolio Models`, `Hybrid Strategy Model`, `Causal Entry Replay`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `TradeTapeConfig` (e.g. with `DiscoveredGame` and `EventCandidate`) actually correct?**
  _`TradeTapeConfig` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `TradeTapeStrategyTests` (e.g. with `EventCandidate` and `GameSnapshot`) actually correct?**
  _`TradeTapeStrategyTests` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `main()` (e.g. with `fetch_game_snapshot()` and `fetch_market_snapshot()`) actually correct?**
  _`main()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `mlb-kalshi-hit-reversion`, `StrategyConfig`, `Canonical Data Pipeline` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._