"""Causal next-observation backtest using the shared live feature contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.strategy import (  # noqa: E402
    CONFIG,
    RAW_STATE_FEATURES,
    add_reaction_features,
    fee_aware_signal_side,
    reaction_feature_frame,
    predict_home_probability,
    signal_side,
    taker_fee,
    validate_market_prices,
)


TEST_DATA = PROJECT_ROOT / "data/processed/test/test_dataset.parquet"
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"


@dataclass
class Result:
    trades: int = 0
    early_exits: int = 0
    settlements: int = 0
    fees: float = 0.0
    capital: float = 0.0
    pnl: float = 0.0

    @property
    def roi(self) -> float:
        return self.pnl / self.capital if self.capital else 0.0


@dataclass
class Position:
    side: str
    contracts: float
    entry_price: float
    entry_fee: float


ENTRY_STATE_FIELDS = tuple(
    feature for feature in RAW_STATE_FEATURES if feature != "pregame_prob"
)


@dataclass(frozen=True)
class PendingEntry:
    side: str
    signal_probability: float
    state_key: tuple


def entry_state_key(row) -> tuple:
    """Identify the complete live state that authorized an entry signal."""
    return tuple(getattr(row, field) for field in ENTRY_STATE_FIELDS)


def add_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    state_model = CatBoostClassifier()
    state_model.load_model(MODEL_DIR / "local_win_expectancy.cbm")
    reaction_model = CatBoostClassifier()
    reaction_model.load_model(MODEL_DIR / "reaction_model.cbm")
    fair = predict_home_probability(state_model, frame)
    result = add_reaction_features(frame, fair)
    baseline = np.log(result["fair_prob"] / (1 - result["fair_prob"]))
    pool = Pool(reaction_feature_frame(result), baseline=baseline)
    result["final_prob"] = reaction_model.predict_proba(pool)[:, 1]
    return result


def simulate(
    frame: pd.DataFrame,
    exit_hysteresis: float = CONFIG.exit_hysteresis,
    hold_to_settlement: bool = False,
) -> Result:
    if exit_hysteresis < 0:
        raise ValueError("exit_hysteresis cannot be negative")
    missing_state = set(ENTRY_STATE_FIELDS) - set(frame.columns)
    if missing_state:
        raise ValueError(
            "Missing fields required to bind entries to game state: "
            f"{sorted(missing_state)}"
        )
    result = Result()
    for _, game in frame.groupby("game_pk", sort=False):
        game = game.sort_values("decision_time")
        position: Position | None = None
        pending_entry: PendingEntry | None = None
        pending_exit = False

        for row in game.itertuples():
            bid = float(row.yes_bid_close)
            ask = float(row.yes_ask_close)

            # Decisions made on the prior observation can execute only now.
            if pending_exit and position is not None:
                exit_price = bid if position.side == "yes" else 1.0 - ask
                exit_fee = taker_fee(position.contracts, exit_price)
                proceeds = position.contracts * exit_price - exit_fee
                result.pnl += (
                    proceeds
                    - position.contracts * position.entry_price
                    - position.entry_fee
                )
                result.fees += exit_fee
                result.early_exits += 1
                position = None
                pending_exit = False

            if pending_entry is not None and position is None:
                state_unchanged = entry_state_key(row) == pending_entry.state_key
                if state_unchanged:
                    side = pending_entry.side
                    price = ask if side == "yes" else 1.0 - bid
                    confirmed_side, _ = fee_aware_signal_side(
                        pending_entry.signal_probability,
                        bid,
                        ask,
                        CONFIG.edge_threshold,
                    )
                    if confirmed_side == side and 0 < price < 1:
                        contracts = CONFIG.bet_size / price
                        entry_fee = taker_fee(contracts, price)
                        position = Position(side, contracts, price, entry_fee)
                        result.trades += 1
                        result.capital += CONFIG.bet_size + entry_fee
                        result.fees += entry_fee
                pending_entry = None

            if position is not None and not hold_to_settlement:
                if (
                    position.side == "yes"
                    and row.final_prob < bid - exit_hysteresis
                ) or (
                    position.side == "no"
                    and row.final_prob > ask + exit_hysteresis
                ):
                    pending_exit = True
            elif pending_entry is None:
                side, _ = signal_side(row.final_prob, bid, ask)
                if side is not None:
                    pending_entry = PendingEntry(
                        side=side,
                        signal_probability=float(row.final_prob),
                        state_key=entry_state_key(row),
                    )

        if position is not None:
            home_win = int(game.iloc[-1]["home_win"])
            won = (
                position.side == "yes" and home_win == 1
            ) or (
                position.side == "no" and home_win == 0
            )
            payout = position.contracts if won else 0.0
            result.pnl += (
                payout
                - position.contracts * position.entry_price
                - position.entry_fee
            )
            result.settlements += 1
    return result


def main() -> None:
    frame = pd.read_parquet(TEST_DATA)
    validate_market_prices(frame)
    frame = add_predictions(frame)
    result = simulate(frame)
    settlement_result = simulate(frame, hold_to_settlement=True)
    print("CAUSAL MARKET-OBSERVATION BACKTEST (EXIT HYSTERESIS)")
    print(f"Decision rows:       {len(frame):,}")
    print(f"Games:               {frame['game_pk'].nunique():,}")
    print(f"Trades:              {result.trades:,}")
    print(f"Early exits:         {result.early_exits:,}")
    print(f"Settlements:         {result.settlements:,}")
    print(f"Fees:                ${result.fees:,.2f}")
    print(f"Capital:             ${result.capital:,.2f}")
    print(f"Net PnL:             ${result.pnl:,.2f}")
    print(f"ROI:                 {result.roi:.2%}")
    print(f"Exit hysteresis:     {CONFIG.exit_hysteresis:.1%}")
    print("\nHOLD-TO-SETTLEMENT COUNTERFACTUAL")
    print(f"Trades:              {settlement_result.trades:,}")
    print(f"Fees:                ${settlement_result.fees:,.2f}")
    print(f"Capital:             ${settlement_result.capital:,.2f}")
    print(f"Net PnL:             ${settlement_result.pnl:,.2f}")
    print(f"ROI:                 {settlement_result.roi:.2%}")


if __name__ == "__main__":
    main()
