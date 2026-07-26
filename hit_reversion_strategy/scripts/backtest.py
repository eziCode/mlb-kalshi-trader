"""Evaluate the selected hybrid strategy on exact-timestamp Kalshi trades."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trade_tape_strategy.core import (  # noqa: E402
    TradeTapeConfig,
    simulate_trade_tape,
)
from trade_tape_strategy.reversion_value import ReversionValueModel  # noqa: E402


DATA_DIR = REPOSITORY_ROOT / "data/shared"
STATE_UPDATES_PATH = REPOSITORY_ROOT / "data/shared/state_updates.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = MODEL_DIR / "trade_tape_config.json"
REVERSION_MODEL_PATH = MODEL_DIR / "reversion_value.cbm"
REVERSION_METADATA_PATH = MODEL_DIR / "reversion_value.metadata.json"
STUDY_DIR = PROJECT_ROOT / "artifacts"
OUTER_HOLDOUT_START = pd.Timestamp("2026-06-28").date()
TRADE_COLUMNS = [
    "game_pk", "game_date", "home_win", "trade_id", "created_time",
    "yes_price_dollars", "no_price_dollars", "count_fp",
    "taker_outcome_side",
]
AWAY_TRADE_COLUMNS = [
    "game_pk", "game_date", "home_win", "trade_id", "created_time",
    "yes_price_dollars", "count_fp", "taker_outcome_side",
]
STATE_COLUMNS = [
    "game_pk", "game_date", "at_bat_number", "pitch_number",
    "pitch_start_time", "pitch_end_time", "completed_event",
    "completed_event_batting_home", "is_hit", "fair_before", "fair_after",
    "atomic_play_input",
    "inning_after", "inning_topbot_after", "outs_when_up_after",
    "score_diff_after", "runner_on_first_after", "runner_on_second_after",
    "runner_on_third_after",
]

LATENCY_PROFILE_PATH = MODEL_DIR / "event_observation_latency.json"


def load_latency_profile() -> dict:
    profile = json.loads(LATENCY_PROFILE_PATH.read_text())
    quantiles = {
        float(probability): float(delay)
        for probability, delay in profile["quantiles_seconds"].items()
    }
    if not quantiles or min(quantiles) != 0.0 or max(quantiles) != 1.0:
        raise RuntimeError("Latency profile must cover quantiles 0 through 1")
    profile["quantiles_seconds"] = quantiles
    return profile


def publication_delay_seconds(
    row: pd.Series, profile: dict | None = None,
) -> float:
    identity = (
        f"{int(row.game_pk)}:{int(row.at_bat_number)}:"
        f"{int(row.pitch_number)}"
    ).encode()
    uniform = int.from_bytes(
        hashlib.sha256(identity).digest()[:8], "big"
    ) / float(2**64 - 1)
    quantiles = (profile or load_latency_profile())["quantiles_seconds"]
    probabilities = sorted(quantiles)
    delays = [quantiles[probability] for probability in probabilities]
    return float(np.interp(uniform, probabilities, delays))


def apply_publication_latency(updates: pd.DataFrame) -> pd.DataFrame:
    invalid_clocks = (
        pd.to_datetime(updates["pitch_end_time"], utc=True)
        <= pd.to_datetime(updates["pitch_start_time"], utc=True)
    )
    if invalid_clocks.any():
        raise ValueError(
            f"State updates contain {int(invalid_clocks.sum())} non-causal "
            "pitch timestamps (pitch_end_time <= pitch_start_time)"
        )
    event_rows = updates["completed_event"].notna()
    non_atomic = event_rows & ~updates["atomic_play_input"].fillna(False).astype(bool)
    if non_atomic.any():
        raise ValueError(
            f"State updates contain {int(non_atomic.sum())} completed events "
            "without atomic play inputs"
        )
    delayed = updates.copy()
    profile = load_latency_profile()
    # Parquet stores pitch timestamps at microsecond resolution, while the
    # empirical interpolation can produce nanoseconds. Promote the destination
    # before assigning delayed values so pandas does not truncate them.
    delayed["event_available_time"] = delayed["pitch_end_time"].astype(
        "datetime64[ns, UTC]"
    )
    # Apply the measured terminal-play publication distribution to every
    # completed event considered by research configurations. The archived
    # sample was collected on hits, so this is an explicit approximation for
    # walks/outs until event-specific observation samples accumulate.
    hit_mask = delayed["completed_event"].notna()
    delays = delayed.loc[hit_mask].apply(
        publication_delay_seconds, axis=1, profile=profile
    )
    delayed.loc[hit_mask, "event_available_time"] = (
        delayed.loc[hit_mask, "pitch_end_time"]
        + pd.to_timedelta(delays, unit="s")
    )
    return delayed


def apply_live_paired_execution_prices(
    home_trades: pd.DataFrame, away_trades: pd.DataFrame,
) -> pd.DataFrame:
    """Build a causal two-market tape with market-specific fill liquidity."""
    home = home_trades.sort_values(["created_time", "game_pk"]).copy()
    away = away_trades.sort_values(["created_time", "game_pk"]).copy()
    tolerance = pd.Timedelta(seconds=10)

    away_quotes = away[["game_pk", "created_time", "yes_price_dollars"]].rename(
        columns={"created_time": "away_quote_time", "yes_price_dollars": "away_yes_price"}
    ).sort_values(["away_quote_time", "game_pk"])
    home_rows = pd.merge_asof(
        home, away_quotes, by="game_pk", left_on="created_time",
        right_on="away_quote_time", direction="backward", tolerance=tolerance,
    )
    home_rows["no_price_dollars"] = home_rows["away_yes_price"]
    home_rows["yes_count_fp"] = home_rows["count_fp"]
    home_rows["no_count_fp"] = 0.0
    home_rows["yes_taker_outcome_side"] = home_rows["taker_outcome_side"]
    home_rows["no_taker_outcome_side"] = ""
    home_rows["home_market_observation"] = True
    home_rows["trade_id"] = "home:" + home_rows["trade_id"].astype(str)

    home_quotes = home[["game_pk", "created_time", "yes_price_dollars"]].rename(
        columns={"created_time": "home_quote_time", "yes_price_dollars": "home_yes_price"}
    ).sort_values(["home_quote_time", "game_pk"])
    away_rows = pd.merge_asof(
        away, home_quotes, by="game_pk", left_on="created_time",
        right_on="home_quote_time", direction="backward", tolerance=tolerance,
    )
    away_rows = away_rows[away_rows["home_yes_price"].notna()].copy()
    away_rows["no_price_dollars"] = away_rows["yes_price_dollars"]
    away_rows["yes_price_dollars"] = away_rows["home_yes_price"]
    away_rows["yes_count_fp"] = 0.0
    away_rows["no_count_fp"] = away_rows["count_fp"]
    away_rows["yes_taker_outcome_side"] = ""
    # Away YES is the home-economic NO contract; normalize aggressor direction.
    away_rows["no_taker_outcome_side"] = away_rows["taker_outcome_side"].map(
        {"yes": "no", "no": "yes"}
    ).fillna("")
    away_rows["home_market_observation"] = False
    away_rows["trade_id"] = "away:" + away_rows["trade_id"].astype(str)

    paired = pd.concat([home_rows, away_rows], ignore_index=True, sort=False)
    return paired.sort_values(["game_pk", "created_time", "trade_id"])


def main() -> None:
    config = TradeTapeConfig(**json.loads(CONFIG_PATH.read_text()))
    trades = pd.read_parquet(
        DATA_DIR / "home_market_trades.parquet", columns=TRADE_COLUMNS
    )
    away_trades = pd.read_parquet(
        DATA_DIR / "away_market_trades.parquet", columns=AWAY_TRADE_COLUMNS
    )
    updates = pd.read_parquet(STATE_UPDATES_PATH, columns=STATE_COLUMNS)
    trades["game_date"] = pd.to_datetime(trades["game_date"]).dt.date
    away_trades["game_date"] = pd.to_datetime(away_trades["game_date"]).dt.date
    updates["game_date"] = pd.to_datetime(updates["game_date"]).dt.date
    test_trades = trades[trades["game_date"] >= OUTER_HOLDOUT_START].copy()
    test_games = set(test_trades["game_pk"].unique())
    test_trades = apply_live_paired_execution_prices(
        test_trades,
        away_trades[away_trades["game_pk"].isin(test_games)].copy(),
    )
    test_updates = apply_publication_latency(
        updates[updates["game_pk"].isin(test_games)].copy()
    )

    entry_scorer = (
        ReversionValueModel(
            REVERSION_MODEL_PATH, REVERSION_METADATA_PATH,
            CONFIG_PATH, LATENCY_PROFILE_PATH,
        )
        if config.direct_value_model_enabled else None
    )
    result = simulate_trade_tape(
        test_trades, test_updates, config, entry_scorer=entry_scorer
    )
    records = pd.DataFrame(asdict(record) for record in result.records)
    game_pnl = records.groupby("game_pk").pnl.sum()
    segment_results = {
        f"{event_type}:{side}": {
            "trades": int(len(segment)),
            "pnl": float(segment.pnl.sum()),
            "wins": int((segment.pnl > 0).sum()),
        }
        for (event_type, side), segment in records.groupby(
            ["event_type", "side"]
        )
    }
    pnl_without_best_game = float(result.pnl - game_pnl.nlargest(1).sum())
    pnl_without_top_four_games = float(
        result.pnl - game_pnl.nlargest(min(4, len(game_pnl))).sum()
    )
    deployment_enabled = bool(
        config.enabled and result.trades >= 20 and result.pnl > 0
        and result.roi > 0 and pnl_without_best_game > 0
    )
    deployment_config = replace(config, enabled=deployment_enabled)
    summary = {
        "evaluation_status": (
            "reused_research_holdout_not_an_unbiased_forward_estimate"
        ),
        "holdout_start": str(OUTER_HOLDOUT_START),
        "selected_config": asdict(config),
        "deployment_config": asdict(deployment_config),
        "games": len(test_games),
        "trade_tape_rows": len(test_trades),
        "state_updates": len(test_updates),
        "observed_events": result.observed_hits,
        "misaligned_event_updates": result.misaligned_event_updates,
        "eligible_hit_updates": result.eligible_hit_updates,
        "rejected_fair_updates": result.rejected_fair_updates,
        "invalidated_candidates": result.invalidated_candidates,
        "expired_candidates": result.expired_candidates,
        "fresh_hit_anchors": result.fresh_hit_anchors,
        "confirmed_signals": result.confirmed_signals,
        "model_rejected_signals": result.model_rejected_signals,
        "trades": result.trades,
        "yes_trades": result.yes_trades,
        "no_trades": result.no_trades,
        "reversion_exits": result.reversion_exits,
        "momentum_exits": result.momentum_exits,
        "timeout_exits": result.timeout_exits,
        "settlements": result.settlements,
        "fees": result.fees,
        "capital": result.capital,
        "pnl": result.pnl,
        "net_ev_per_scheduled_game": result.pnl / len(test_games),
        "roi": result.roi,
        "pnl_without_best_game": pnl_without_best_game,
        "pnl_without_top_four_games": pnl_without_top_four_games,
        "segment_results": segment_results,
        "time_based_exit": config.maximum_hold_seconds > 0,
        "exit_target_mode": config.exit_target_mode,
        "latch_reversion_exit": config.latch_reversion_exit,
        "state_model": "MLB-only batting-perspective local win expectancy",
        "event_availability": {
            key: value for key, value in load_latency_profile().items()
            if key != "quantiles_seconds"
        },
        "no_execution_contract": "paired_away_yes_independent_trade_liquidity",
    }
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    records.to_csv(STUDY_DIR / "holdout_trades.csv", index=False)

    print("EXACT-TIMESTAMP TRADE-TAPE HYBRID")
    print(f"Live enabled:          {deployment_enabled}")
    print(f"Minimum edge:          {config.minimum_edge:.1%}")
    print(f"Confirmation:          {config.confirmation_seconds:g} seconds")
    print(
        "Maximum hold:         "
        f"{config.maximum_hold_seconds:g} seconds"
        if config.maximum_hold_seconds > 0 else "Maximum hold:         none"
    )
    print(f"Exit target:           {config.exit_target_mode}")
    print(f"Latch target touch:    {config.latch_reversion_exit}")
    print(f"Games:                 {len(test_games):,}")
    print(f"Observed trades:       {len(test_trades):,}")
    print(f"Observed events:       {result.observed_hits:,}")
    print("Event availability:    empirical live publication latency")
    print("NO execution:          paired away-team YES trade")
    print(f"Misaligned hit state:  {result.misaligned_event_updates:,}")
    print(f"Eligible fair moves:   {result.eligible_hit_updates:,}")
    print(f"Rejected fair moves:   {result.rejected_fair_updates:,}")
    print(f"Invalidated signals:   {result.invalidated_candidates:,}")
    print(f"Expired signals:       {result.expired_candidates:,}")
    print(f"Fresh hit anchors:     {result.fresh_hit_anchors:,}")
    print(f"Confirmed signals:     {result.confirmed_signals:,}")
    print(f"Model-rejected signals:{result.model_rejected_signals:>9,}")
    print(f"Filled trades:         {result.trades:,}")
    print(f"YES / NO:              {result.yes_trades:,} / {result.no_trades:,}")
    print(f"Reversion exits:       {result.reversion_exits:,}")
    print(f"Momentum-delayed exits:{result.momentum_exits:>9,}")
    print(f"Settlements:           {result.settlements:,}")
    print(f"Fees:                  ${result.fees:,.2f}")
    print(f"Capital:               ${result.capital:,.2f}")
    print(f"Net PnL:               ${result.pnl:,.2f}")
    print(
        "Net EV / game:         "
        f"${result.pnl / len(test_games):,.4f}"
    )
    print(f"ROI:                   {result.roi:.2%}")


if __name__ == "__main__":
    main()
