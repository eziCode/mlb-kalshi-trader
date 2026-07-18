"""Empirical post-hit market reaction and profitable-reversion modeling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlb_kalshi.strategy import CONFIG, taker_fee


REACTION_NUMERIC_FEATURES = (
    "pre_batting_price",
    "inning",
    "batting_score_diff",
    "outs_after",
    "runner_on_first_after",
    "runner_on_second_after",
    "runner_on_third_after",
    "pre_trade_count_5s",
    "pre_volume_5s",
    "pre_flow_imbalance_5s",
    "pre_volatility_5s",
    "pitch_duration_seconds",
)
REACTION_CATEGORICAL_FEATURES = ("event_type",)
REACTION_FEATURES = REACTION_NUMERIC_FEATURES + REACTION_CATEGORICAL_FEATURES

REVERSION_NUMERIC_FEATURES = REACTION_NUMERIC_FEATURES + (
    "predicted_batting_logit_move",
    "actual_batting_logit_move",
    "excess_logit_move",
    "probability_residual",
    "entry_price",
    "target_reversion_pnl",
    "breakeven_reversion_probability",
    "event_to_entry_seconds",
    "post_trade_count_5s",
    "post_volume_5s",
    "post_flow_imbalance_5s",
    "post_volatility_5s",
)
REVERSION_CATEGORICAL_FEATURES = ("event_type", "entry_side")
REVERSION_FEATURES = REVERSION_NUMERIC_FEATURES + REVERSION_CATEGORICAL_FEATURES


def logit(value):
    clipped = np.clip(np.asarray(value, dtype=float), 0.01, 0.99)
    return np.log(clipped / (1.0 - clipped))


def expit(value):
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-value))


def expected_home_probability(
    pre_batting_price,
    predicted_batting_logit_move,
    batting_home,
):
    """Convert a batting-oriented reaction prediction back to HOME YES."""
    expected_batting = expit(
        logit(pre_batting_price) + predicted_batting_logit_move
    )
    return np.where(
        np.asarray(batting_home, dtype=bool),
        expected_batting,
        1.0 - expected_batting,
    )


def reaction_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REACTION_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reaction features: {sorted(missing)}")
    result = frame.loc[:, REACTION_FEATURES].copy()
    for column in REACTION_NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    for column in REACTION_CATEGORICAL_FEATURES:
        result[column] = result[column].fillna("unknown").astype(str)
    return result


def reversion_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REVERSION_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reversion features: {sorted(missing)}")
    result = frame.loc[:, REVERSION_FEATURES].copy()
    for column in REVERSION_NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    for column in REVERSION_CATEGORICAL_FEATURES:
        result[column] = result[column].fillna("unknown").astype(str)
    return result


@dataclass(frozen=True)
class EmpiricalStrategyConfig:
    enabled: bool = False
    minimum_probability_residual: float = 0.05
    minimum_reversion_probability: float = 0.70
    minimum_reversion_probability_margin: float = 0.02
    reaction_window_seconds: float = 5.0


@dataclass
class EmpiricalResult:
    trades: int = 0
    reversion_exits: int = 0
    settlements: int = 0
    fees: float = 0.0
    capital: float = 0.0
    pnl: float = 0.0
    accepted_ids: list[int] = field(default_factory=list)

    @property
    def roi(self) -> float:
        return self.pnl / self.capital if self.capital else 0.0


def build_reversion_candidates(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    reaction_model,
    minimum_entry_residual: float = 0.01,
) -> pd.DataFrame:
    """Create causally fillable post-reaction entries and reversion labels.

    The reaction is observed for five seconds before an entry is considered.
    Entry and exit fills require a strictly later compatible aggressor-side
    trade with sufficient reported size.  A positive label requires the
    market to cross the empirical expected price before game settlement and
    the round trip to be profitable after fees.
    """
    required_events = {
        "event_id", "game_pk", "decision_time", "valid_until",
        "game_end_time", "batting_home", "pre_batting_price", "home_win",
    } | set(REACTION_FEATURES)
    required_trades = {
        "game_pk", "trade_id", "created_time", "yes_price_dollars",
        "no_price_dollars", "count_fp", "taker_outcome_side",
    }
    if missing := required_events - set(events.columns):
        raise ValueError(f"Reaction events are missing columns: {sorted(missing)}")
    if missing := required_trades - set(trades.columns):
        raise ValueError(f"Trade tape is missing columns: {sorted(missing)}")

    frame = events.copy()
    frame["predicted_batting_logit_move"] = reaction_model.predict(
        reaction_feature_frame(frame)
    )
    frame["expected_home_price"] = expected_home_probability(
        frame["pre_batting_price"],
        frame["predicted_batting_logit_move"],
        frame["batting_home"],
    )
    trades_by_game = {}
    for game_pk, game in trades.groupby("game_pk", sort=False):
        game = game.sort_values(["created_time", "trade_id"])
        trades_by_game[int(game_pk)] = (
            pd.to_datetime(game["created_time"], utc=True)
            .array.as_unit("ns").asi8,
            game["yes_price_dollars"].to_numpy(dtype=float),
            game["no_price_dollars"].to_numpy(dtype=float),
            game["count_fp"].to_numpy(dtype=float),
            game["taker_outcome_side"].astype(str).to_numpy(),
        )
    rows: list[dict] = []
    for event in frame.sort_values(["game_pk", "decision_time"]).itertuples():
        game_arrays = trades_by_game.get(int(event.game_pk))
        if game_arrays is None:
            continue
        times, yes_prices, no_prices, sizes, taker_sides = game_arrays
        decision_ns = pd.Timestamp(event.decision_time).value
        valid_ns = pd.Timestamp(event.valid_until).value
        entry_start = int(np.searchsorted(times, decision_ns, side="right"))
        entry_stop = int(np.searchsorted(times, valid_ns, side="right"))
        if entry_start >= entry_stop:
            continue

        expected_home = float(event.expected_home_price)
        entry_index = None
        entry_side = None
        entry_price = None
        contracts = None
        for index in range(entry_start, entry_stop):
            home_residual = float(yes_prices[index] - expected_home)
            side = "no" if home_residual >= minimum_entry_residual else (
                "yes" if home_residual <= -minimum_entry_residual else None
            )
            if side is None or taker_sides[index] != side:
                continue
            price = float(
                yes_prices[index] if side == "yes" else no_prices[index]
            )
            if price <= 0:
                continue
            required_contracts = CONFIG.bet_size / price
            if sizes[index] < required_contracts:
                continue
            entry_index = index
            entry_side = side
            entry_price = price
            contracts = required_contracts
            break
        if entry_index is None:
            continue

        assert entry_side is not None and entry_price is not None
        assert contracts is not None
        entry_fee = taker_fee(contracts, entry_price)
        target_exit_price = (
            expected_home if entry_side == "yes" else 1.0 - expected_home
        )
        target_exit_fee = taker_fee(contracts, target_exit_price)
        target_reversion_pnl = (
            contracts * target_exit_price - target_exit_fee
            - contracts * entry_price - entry_fee
        )
        maximum_settlement_loss = contracts * entry_price + entry_fee
        breakeven_reversion_probability = (
            maximum_settlement_loss
            / (maximum_settlement_loss + target_reversion_pnl)
            if target_reversion_pnl > 0 else 1.0
        )
        entry_home_price = float(yes_prices[entry_index])
        entry_batting_price = (
            entry_home_price
            if bool(event.batting_home)
            else 1.0 - entry_home_price
        )
        actual_logit_move = float(
            logit(entry_batting_price) - logit(event.pre_batting_price)
        )

        exit_index = None
        exit_price = np.nan
        exit_fee = np.nan
        exit_start = entry_index + 1
        exit_stop = int(np.searchsorted(
            times, pd.Timestamp(event.game_end_time).value, side="right"
        ))
        exit_taker_side = "no" if entry_side == "yes" else "yes"
        for index in range(exit_start, exit_stop):
            if taker_sides[index] != exit_taker_side or sizes[index] < contracts:
                continue
            crossed_expected = (
                entry_side == "yes" and yes_prices[index] >= expected_home
            ) or (
                entry_side == "no" and yes_prices[index] <= expected_home
            )
            if not crossed_expected:
                continue
            price = float(
                yes_prices[index] if entry_side == "yes" else no_prices[index]
            )
            fee = taker_fee(contracts, price)
            pnl = (
                contracts * price - fee
                - contracts * entry_price - entry_fee
            )
            if pnl > 0:
                exit_index = index
                exit_price = price
                exit_fee = fee
                break

        row = event._asdict()
        row.update({
            "candidate_id": len(rows),
            "entry_time": pd.Timestamp(times[entry_index], tz="UTC"),
            "entry_side": entry_side,
            "entry_price": entry_price,
            "entry_home_price": entry_home_price,
            "contracts": contracts,
            "entry_fee": entry_fee,
            "target_exit_price": target_exit_price,
            "target_exit_fee": target_exit_fee,
            "target_reversion_pnl": target_reversion_pnl,
            "maximum_settlement_loss": maximum_settlement_loss,
            "breakeven_reversion_probability": (
                breakeven_reversion_probability
            ),
            "event_to_entry_seconds": (
                times[entry_index] - pd.Timestamp(event.event_end_time).value
            ) / 1e9,
            "actual_batting_logit_move": actual_logit_move,
            "excess_logit_move": (
                actual_logit_move - float(event.predicted_batting_logit_move)
            ),
            "probability_residual": abs(entry_home_price - expected_home),
            "profitable_reversion": int(exit_index is not None),
            "reversion_exit_time": (
                pd.Timestamp(times[exit_index], tz="UTC")
                if exit_index is not None else pd.NaT
            ),
            "reversion_exit_price": exit_price,
            "reversion_exit_fee": exit_fee,
        })
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values(
        ["game_date", "entry_time", "game_pk"]
    ).reset_index(drop=True)
    result["candidate_id"] = np.arange(len(result), dtype=int)
    return result


def evaluate_candidates(
    candidates: pd.DataFrame,
    config: EmpiricalStrategyConfig,
) -> EmpiricalResult:
    result = EmpiricalResult()
    eligible = candidates[
        (candidates["probability_residual"] >= config.minimum_probability_residual)
        & (
            candidates["predicted_reversion_probability"]
            >= config.minimum_reversion_probability
        )
        & (
            candidates["predicted_reversion_probability"]
            >= candidates["breakeven_reversion_probability"]
            + config.minimum_reversion_probability_margin
        )
    ].sort_values(["game_pk", "entry_time", "candidate_id"])

    for _, game in eligible.groupby("game_pk", sort=False):
        occupied_until: pd.Timestamp | None = None
        settled = False
        for row in game.itertuples():
            if settled:
                break
            entry_time = pd.Timestamp(row.entry_time)
            if occupied_until is not None and entry_time <= occupied_until:
                continue
            contracts = float(row.contracts)
            entry_price = float(row.entry_price)
            entry_fee = float(row.entry_fee)
            result.trades += 1
            result.capital += contracts * entry_price + entry_fee
            result.fees += entry_fee
            result.accepted_ids.append(int(row.candidate_id))
            if bool(row.profitable_reversion):
                exit_price = float(row.reversion_exit_price)
                exit_fee = float(row.reversion_exit_fee)
                pnl = (
                    contracts * exit_price - exit_fee
                    - contracts * entry_price - entry_fee
                )
                result.reversion_exits += 1
                result.fees += exit_fee
                occupied_until = pd.Timestamp(row.reversion_exit_time)
            else:
                won = (
                    row.entry_side == "yes" and int(row.home_win) == 1
                ) or (
                    row.entry_side == "no" and int(row.home_win) == 0
                )
                pnl = (
                    (contracts if won else 0.0)
                    - contracts * entry_price - entry_fee
                )
                result.settlements += 1
                settled = True
            result.pnl += pnl
    return result
