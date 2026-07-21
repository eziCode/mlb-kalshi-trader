"""Report holdout sensitivity by hit type, side, and edge threshold."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trade_tape_strategy.core import TradeTapeConfig, simulate_trade_tape

TRADES = None
UPDATES = None


def initialize(trades, updates):
    global TRADES, UPDATES
    TRADES, UPDATES = trades, updates


def evaluate(parameters):
    event_type, side, edge = parameters
    config = TradeTapeConfig(
        minimum_edge=edge,
        confirmation_seconds=2.0,
        allowed_event_types=(event_type,),
        minimum_reversion_move=0.01,
        side_filter=side,
        position_sizing="fixed_payout",
    )
    result = simulate_trade_tape(TRADES, UPDATES, config)
    return event_type, side, edge, result.trades, result.pnl, result.roi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-holdout", action="store_true")
    parser.add_argument(
        "--state-updates",
        type=Path,
        default=REPOSITORY_ROOT / "data/shared/state_updates.parquet",
        help="State-update parquet to evaluate (defaults to the hit model).",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data = REPOSITORY_ROOT / "data/shared"
    trades = pd.read_parquet(data / "home_market_trades.parquet")
    updates = pd.read_parquet(args.state_updates)
    trades["game_date"] = pd.to_datetime(trades.game_date).dt.date
    start = pd.Timestamp("2026-06-28").date()
    if args.pre_holdout:
        trades = trades[trades.game_date < start]
        dates = sorted(trades.game_date.unique())
        trades = trades[trades.game_date >= dates[int(len(dates) * .75)]]
    else:
        trades = trades[trades.game_date >= start]
    updates = updates[updates.game_pk.isin(set(trades.game_pk))]
    parameters = [
        (event, side, edge)
        for event in ("single", "double", "triple")
        for side in ("yes", "no")
        for edge in (0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize, initargs=(trades, updates)
    ) as executor:
        for row in executor.map(evaluate, parameters):
            event, side, edge, count, pnl, roi = row
            print(
                f"{event:6} {side:3} edge={edge:5.1%} trades={count:3} "
                f"pnl=${pnl:7.2f} roi={roi:7.2%}", flush=True,
            )


if __name__ == "__main__":
    main()
