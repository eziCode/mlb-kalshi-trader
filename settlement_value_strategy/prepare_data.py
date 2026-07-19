"""Build model decisions and the causal replay tape from normalized raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from settlement_value_strategy.strategy import MispricingConfig, build_mispricing_dataset


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DATA = ROOT.parent / "data/settlement_value"
SHARED_DATA = ROOT.parent / "data/shared"

TRADE_COLUMNS = {
    "game_pk", "game_date", "trade_id", "created_time",
    "yes_price_dollars", "no_price_dollars", "count_fp",
    "taker_outcome_side",
}
STATE_COLUMNS = {
    "game_pk", "game_date", "home_win", "at_bat_number", "pitch_number",
    "pitch_end_time", "fair_before", "fair_after", "inning_after",
    "inning_topbot_after", "outs_when_up_after", "score_diff_after",
    "balls_after", "strikes_after", "runner_on_first_after",
    "runner_on_second_after", "runner_on_third_after",
}


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def compact_execution_tape(
    decisions: pd.DataFrame, trades: pd.DataFrame, maximum_delay: float = 5.0,
) -> pd.DataFrame:
    """Keep only strictly-later trades inside some candidate fill window."""
    chunks = []
    delay_ns = int(maximum_delay * 1e9)
    tapes = {
        int(game_pk): tape.sort_values(["created_time", "trade_id"])
        for game_pk, tape in trades.groupby("game_pk", sort=False)
    }
    for game_pk, rows in decisions.groupby("game_pk", sort=False):
        tape = tapes.get(int(game_pk))
        if tape is None or tape.empty:
            continue
        times = pd.to_datetime(tape.created_time, utc=True).array.as_unit("ns").asi8
        keep = np.zeros(len(tape), dtype=bool)
        for row in rows.itertuples(index=False):
            start = pd.Timestamp(row.signal_time).value
            end = start + delay_ns
            if pd.notna(row.next_update_time):
                end = min(end, pd.Timestamp(row.next_update_time).value)
            left = int(np.searchsorted(times, start, side="right"))
            right = int(np.searchsorted(times, end, side="left"))
            keep[left:right] = True
        if keep.any():
            chunks.append(tape.iloc[np.flatnonzero(keep)])
    if not chunks:
        return trades.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True).drop_duplicates("trade_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trades", type=Path,
        default=SHARED_DATA / "home_market_trades.parquet",
    )
    parser.add_argument(
        "--states", type=Path,
        default=DATA / "state_updates.parquet",
    )
    parser.add_argument(
        "--away-trades", type=Path,
        default=SHARED_DATA / "away_market_trades.parquet",
    )
    parser.add_argument("--output", type=Path, default=DATA)
    args = parser.parse_args()
    trades = pd.read_parquet(args.trades)
    away_trades = pd.read_parquet(args.away_trades)
    states = pd.read_parquet(args.states)
    require_columns(trades, TRADE_COLUMNS, "trade tape")
    require_columns(away_trades, TRADE_COLUMNS, "away trade tape")
    require_columns(states, STATE_COLUMNS, "state updates")
    trades["created_time"] = pd.to_datetime(trades.created_time, utc=True)
    away_trades["created_time"] = pd.to_datetime(
        away_trades.created_time, utc=True
    )
    states["pitch_end_time"] = pd.to_datetime(states.pitch_end_time, utc=True)
    decisions = build_mispricing_dataset(trades, states, MispricingConfig())
    compact = compact_execution_tape(decisions, trades)
    compact_away = compact_execution_tape(decisions, away_trades)
    args.output.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(args.output / "decision_rows.parquet", index=False)
    compact.to_parquet(args.output / "execution_trades.parquet", index=False)
    compact_away.to_parquet(
        args.output / "away_execution_trades.parquet", index=False
    )
    print(
        f"wrote {len(decisions):,} decisions and {len(compact):,} execution "
        f"home trades and {len(compact_away):,} away YES trades to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
