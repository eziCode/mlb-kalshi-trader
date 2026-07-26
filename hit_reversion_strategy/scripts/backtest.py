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

# Empirical completed-hit publication latency measured from 184 archived MLB
# live-feed observations on 2026-07-25. Historical rows do not contain an
# observation timestamp, so stable event identities are mapped through this
# CDF. This is deterministic and keeps pitch_end_time as the signal-age clock.
PUBLICATION_DELAY_QUANTILES = {
    0.00: 2.523033,
    0.05: 4.150827,
    0.10: 5.662205,
    0.20: 7.876134,
    0.30: 9.631647,
    0.40: 10.908609,
    0.50: 12.933865,
    0.60: 14.024072,
    0.70: 15.513805,
    0.75: 19.111486,
    0.80: 23.557750,
    0.90: 41.331502,
    0.95: 55.757119,
    0.975: 80.516275,
    0.99: 126.110700,
    1.00: 266.310212,
}


def publication_delay_seconds(row: pd.Series) -> float:
    identity = (
        f"{int(row.game_pk)}:{int(row.at_bat_number)}:"
        f"{int(row.pitch_number)}"
    ).encode()
    uniform = int.from_bytes(
        hashlib.sha256(identity).digest()[:8], "big"
    ) / float(2**64 - 1)
    probabilities = list(PUBLICATION_DELAY_QUANTILES)
    delays = list(PUBLICATION_DELAY_QUANTILES.values())
    return float(np.interp(uniform, probabilities, delays))


def apply_publication_latency(updates: pd.DataFrame) -> pd.DataFrame:
    delayed = updates.copy()
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
        publication_delay_seconds, axis=1
    )
    delayed.loc[hit_mask, "event_available_time"] = (
        delayed.loc[hit_mask, "pitch_end_time"]
        + pd.to_timedelta(delays, unit="s")
    )
    return delayed


def apply_live_paired_execution_prices(
    home_trades: pd.DataFrame, away_trades: pd.DataFrame,
) -> pd.DataFrame:
    """Use the causal away-YES trade that the live NO route can execute.

    The old replay used NO on the home ticker even though production buys YES
    on the paired away ticker. Keep only an away price observed in the prior
    ten seconds, matching the live quote-freshness requirement.
    """
    left = home_trades.sort_values(["created_time", "game_pk"]).copy()
    right = away_trades[[
        "game_pk", "created_time", "yes_price_dollars",
    ]].rename(columns={
        "created_time": "away_created_time",
        "yes_price_dollars": "away_yes_price",
    }).sort_values(["away_created_time", "game_pk"])
    paired = pd.merge_asof(
        left, right, by="game_pk", left_on="created_time",
        right_on="away_created_time", direction="backward",
        tolerance=pd.Timedelta(seconds=10),
    )
    paired["no_price_dollars"] = paired["away_yes_price"]
    return paired.sort_values(["game_pk", "created_time", "trade_id"])


def main() -> None:
    config = TradeTapeConfig(**json.loads(CONFIG_PATH.read_text()))
    trades = pd.read_parquet(DATA_DIR / "home_market_trades.parquet")
    away_trades = pd.read_parquet(DATA_DIR / "away_market_trades.parquet")
    updates = pd.read_parquet(STATE_UPDATES_PATH)
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
        ReversionValueModel(REVERSION_MODEL_PATH, REVERSION_METADATA_PATH)
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
            "mode": "empirical_live_publication_latency",
            "sample_date": "2026-07-25",
            "sample_hits": 184,
            "latency_seconds_median": 12.933865,
            "latency_seconds_p75": 19.111486,
            "latency_seconds_p95": 55.757119,
        },
        "no_execution_contract": "paired_away_yes",
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
