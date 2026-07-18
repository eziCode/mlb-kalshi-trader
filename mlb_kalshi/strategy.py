"""One feature and execution contract for every strategy surface."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


STATE_FEATURES = (
    "pregame_prob",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "score_diff",
    "balls",
    "strikes",
    "runner_on_first",
    "runner_on_second",
    "runner_on_third",
)

REACTION_FEATURES = (
    "market_error",
    "kalshi_price",
    "pregame_prob",
    "spread",
    "inning",
)

TAKER_FEE_RATE = 0.07


@dataclass(frozen=True)
class StrategyConfig:
    edge_threshold: float = 0.15
    exit_hysteresis: float = 0.00
    bet_size: float = 10.0
    maximum_quote_age_seconds: float = 2.0
    maximum_feed_age_seconds: float = 15.0


CONFIG = StrategyConfig()


def _numeric_frame(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    result = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(f"Model features contain nulls: {bad}")
    return result.astype(float)


def state_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = _numeric_frame(df, STATE_FEATURES)
    checks = {
        "pregame_prob": frame["pregame_prob"].between(0.01, 0.99),
        "inning": frame["inning"].between(1, 30),
        "inning_topbot": frame["inning_topbot"].isin([0, 1]),
        "outs_when_up": frame["outs_when_up"].between(0, 2),
        "balls": frame["balls"].between(0, 4),
        "strikes": frame["strikes"].between(0, 3),
    }
    invalid = [name for name, valid in checks.items() if not valid.all()]
    if invalid:
        raise ValueError(f"State features outside expected ranges: {invalid}")
    return frame


def add_reaction_features(df: pd.DataFrame, fair_probability) -> pd.DataFrame:
    result = df.copy()
    fair = np.clip(np.asarray(fair_probability, dtype=float), 1e-4, 1 - 1e-4)
    result["fair_prob"] = fair
    result["market_error"] = result["kalshi_price"].astype(float) - fair
    return result


def reaction_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = _numeric_frame(df, REACTION_FEATURES)
    if not frame["kalshi_price"].between(0.01, 0.99).all():
        raise ValueError("kalshi_price must be between 0.01 and 0.99")
    if not frame["pregame_prob"].between(0.01, 0.99).all():
        raise ValueError("pregame_prob must be between 0.01 and 0.99")
    if not frame["spread"].between(0.0, 0.50).all():
        raise ValueError("spread must be between 0 and 0.50")
    return frame


def validate_market_prices(df: pd.DataFrame) -> None:
    required = {"yes_bid_close", "yes_ask_close", "kalshi_price", "spread"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing market prices: {sorted(missing)}")
    valid = (
        df["yes_bid_close"].between(0.01, 0.99)
        & df["yes_ask_close"].between(0.01, 0.99)
        & (df["yes_ask_close"] > df["yes_bid_close"])
    )
    if not valid.all():
        raise ValueError(f"{(~valid).sum()} invalid market price rows")
    midpoint = (df["yes_bid_close"] + df["yes_ask_close"]) / 2
    spread = df["yes_ask_close"] - df["yes_bid_close"]
    if not np.allclose(df["kalshi_price"], midpoint):
        raise ValueError("kalshi_price is not the actual bid/ask midpoint")
    if not np.allclose(df["spread"], spread):
        raise ValueError("spread does not match actual bid/ask")


def taker_fee(contracts: float, price: float) -> float:
    if contracts <= 0:
        return 0.0
    raw = TAKER_FEE_RATE * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0 - 1e-12) / 100.0


def estimated_round_trip_fee_per_contract(price: float) -> float:
    """Conservative fee reserve for an entry and an early taker exit."""
    if not 0 < price < 1:
        return math.inf
    contracts = CONFIG.bet_size / price
    return 2.0 * taker_fee(contracts, price) / contracts


def fee_aware_signal_side(
    final_probability: float,
    bid: float,
    ask: float,
    buffer: float,
) -> tuple[str | None, float]:
    """Select a side only after spread and estimated round-trip fees.

    The returned edge is net of the executable half-spread and fee reserve.
    Starting from midpoint edge makes those costs visible; equivalently, this
    requires executable-price edge to exceed fees plus ``buffer``.
    """
    midpoint = (float(bid) + float(ask)) / 2.0
    half_spread = (float(ask) - float(bid)) / 2.0
    yes_price = float(ask)
    no_price = 1.0 - float(bid)
    yes_net_edge = (
        float(final_probability) - midpoint - half_spread
        - estimated_round_trip_fee_per_contract(yes_price)
    )
    no_net_edge = (
        midpoint - float(final_probability) - half_spread
        - estimated_round_trip_fee_per_contract(no_price)
    )
    if yes_net_edge >= buffer and yes_net_edge >= no_net_edge:
        return "yes", yes_net_edge
    if no_net_edge >= buffer:
        return "no", no_net_edge
    return None, max(yes_net_edge, no_net_edge)


def signal_side(final_probability: float, bid: float, ask: float) -> tuple[str | None, float]:
    return fee_aware_signal_side(
        final_probability, bid, ask, CONFIG.edge_threshold
    )
