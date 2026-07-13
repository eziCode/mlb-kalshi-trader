"""Paper trader using the exact raw feature contract used in training."""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier, Pool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.strategy import (  # noqa: E402
    CONFIG,
    add_reaction_features,
    reaction_feature_frame,
    signal_side,
    state_feature_frame,
    taker_fee,
)


GAME_PK = int(os.getenv("MLB_GAME_PK", "824491"))
MARKET_TICKER = os.getenv(
    "KALSHI_MARKET_TICKER", "KXMLBGAME-26JUL121340CHCCIN-CIN"
)
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
MLB_API = "https://statsapi.mlb.com/api"


@dataclass
class MarketSnapshot:
    received_at: datetime
    bid: float
    ask: float

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class GameSnapshot:
    received_at: datetime
    status: str
    state: dict
    home_score: int
    away_score: int


@dataclass
class Position:
    side: str
    contracts: float
    entry_price: float
    entry_fee: float


def fetch_market_snapshot() -> MarketSnapshot:
    response = requests.get(
        f"{KALSHI_API}/markets/{MARKET_TICKER}/orderbook", timeout=5
    )
    response.raise_for_status()
    book = response.json().get("orderbook_fp") or {}
    yes_bids = book.get("yes_dollars") or []
    no_bids = book.get("no_dollars") or []
    if not yes_bids or not no_bids:
        raise RuntimeError("Kalshi order book is not two-sided")
    bid = float(yes_bids[-1][0])
    ask = 1.0 - float(no_bids[-1][0])
    if not 0 < bid < ask < 1:
        raise RuntimeError(f"Invalid Kalshi order book: bid={bid}, ask={ask}")
    return MarketSnapshot(datetime.now(timezone.utc), bid, ask)


def fetch_game_snapshot() -> GameSnapshot:
    response = requests.get(
        f"{MLB_API}/v1.1/game/{GAME_PK}/feed/live", timeout=5
    )
    response.raise_for_status()
    payload = response.json()
    live = payload.get("liveData") or {}
    linescore = live.get("linescore") or {}
    status = payload.get("gameData", {}).get("status", {}).get(
        "abstractGameState", "Preview"
    )
    inning_state = str(linescore.get("inningState") or "")
    if inning_state.lower().startswith("top"):
        topbot = 0
    elif inning_state.lower().startswith("bottom"):
        topbot = 1
    elif status == "Final":
        topbot = 0
    else:
        raise RuntimeError(f"No active inning half: {inning_state!r}")
    teams = linescore.get("teams") or {}
    home_score = int(teams.get("home", {}).get("runs") or 0)
    away_score = int(teams.get("away", {}).get("runs") or 0)
    offense = linescore.get("offense") or {}
    state = {
        "inning": int(linescore.get("currentInning") or 1),
        "inning_topbot": topbot,
        "outs_when_up": int(linescore.get("outs") or 0),
        "score_diff": home_score - away_score,
        "balls": int(linescore.get("balls") or 0),
        "strikes": int(linescore.get("strikes") or 0),
        "runner_on_first": int("first" in offense),
        "runner_on_second": int("second" in offense),
        "runner_on_third": int("third" in offense),
    }
    return GameSnapshot(
        datetime.now(timezone.utc), status, state, home_score, away_score
    )


def first_pitch_time(payload: dict) -> datetime | None:
    values = []
    for play in payload.get("liveData", {}).get("plays", {}).get("allPlays", []):
        for event in play.get("playEvents") or []:
            if event.get("isPitch") and event.get("startTime"):
                values.append(pd.to_datetime(event["startTime"], utc=True).to_pydatetime())
    return min(values) if values else None


def fetch_pregame_anchor() -> float:
    feed = requests.get(
        f"{MLB_API}/v1.1/game/{GAME_PK}/feed/live", timeout=5
    ).json()
    first_pitch = first_pitch_time(feed)
    if first_pitch is None:
        snapshot = fetch_market_snapshot()
        return snapshot.midpoint
    params = {
        "start_ts": int(first_pitch.timestamp()) - 14_400,
        "end_ts": int(first_pitch.timestamp()),
        "period_interval": 1,
    }
    response = requests.get(
        f"https://api.elections.kalshi.com/trade-api/v2/series/"
        f"KXMLBGAME/markets/{MARKET_TICKER}/candlesticks",
        params=params,
        timeout=5,
    )
    response.raise_for_status()
    candles = response.json().get("candlesticks") or []
    if not candles:
        raise RuntimeError("No pre-first-pitch Kalshi anchor; refusing to trade")
    candle = candles[-1]
    bid = float(candle.get("yes_bid", {}).get("close_dollars"))
    ask = float(candle.get("yes_ask", {}).get("close_dollars"))
    if not 0 < bid < ask < 1:
        raise RuntimeError("Invalid pregame anchor quote")
    return (bid + ask) / 2.0


async def main() -> None:
    state_model = CatBoostClassifier()
    state_model.load_model(MODEL_DIR / "local_win_expectancy.cbm")
    reaction_model = CatBoostClassifier()
    reaction_model.load_model(MODEL_DIR / "reaction_model.cbm")
    pregame_prob = await asyncio.to_thread(fetch_pregame_anchor)
    print(f"Pregame Kalshi anchor: {pregame_prob:.1%}")

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"paper_trade_log_{MARKET_TICKER}_{int(time.time())}.csv"
    with log_path.open("w", newline="") as handle:
        csv.writer(handle).writerow([
            "decision_time", "market_received_at", "state_received_at",
            "bid", "ask", "inning", "outs", "score_diff", "fair_prob",
            "final_prob", "edge", "action", "cash",
        ])

    cash = 1000.0
    position: Position | None = None
    while True:
        try:
            market, game = await asyncio.gather(
                asyncio.to_thread(fetch_market_snapshot),
                asyncio.to_thread(fetch_game_snapshot),
            )
        except Exception as error:
            print(f"Snapshot rejected: {error}")
            await asyncio.sleep(POLL_SECONDS)
            continue
        now = datetime.now(timezone.utc)
        if game.status == "Final":
            if position is not None:
                won = (
                    position.side == "yes" and game.home_score > game.away_score
                ) or (
                    position.side == "no" and game.home_score < game.away_score
                )
                if won:
                    cash += position.contracts
            print(f"Final paper cash: ${cash:.2f}")
            break
        if game.status != "Live":
            await asyncio.sleep(POLL_SECONDS)
            continue
        if (now - market.received_at).total_seconds() > CONFIG.maximum_quote_age_seconds:
            await asyncio.sleep(POLL_SECONDS)
            continue
        if (now - game.received_at).total_seconds() > CONFIG.maximum_feed_age_seconds:
            await asyncio.sleep(POLL_SECONDS)
            continue

        values = {**game.state, "pregame_prob": pregame_prob}
        state_row = pd.DataFrame([values])
        fair_prob = state_model.predict_proba(state_feature_frame(state_row))[0, 1]
        reaction_row = add_reaction_features(pd.DataFrame([{
            "kalshi_price": market.midpoint,
            "pregame_prob": pregame_prob,
            "spread": market.spread,
            "inning": game.state["inning"],
        }]), [fair_prob])
        baseline = [np.log(fair_prob / (1 - fair_prob))]
        final_prob = reaction_model.predict_proba(Pool(
            reaction_feature_frame(reaction_row), baseline=baseline
        ))[0, 1]
        side, edge = signal_side(final_prob, market.bid, market.ask)
        action = "HOLD"

        if position is not None:
            should_exit = (
                position.side == "yes" and final_prob < market.bid
            ) or (
                position.side == "no" and final_prob > market.ask
            )
            if should_exit:
                price = market.bid if position.side == "yes" else 1.0 - market.ask
                fee = taker_fee(position.contracts, price)
                cash += position.contracts * price - fee
                action = f"CLOSE_{position.side.upper()}"
                position = None
        elif side is not None:
            price = market.ask if side == "yes" else 1.0 - market.bid
            contracts = CONFIG.bet_size / price
            fee = taker_fee(contracts, price)
            cash -= CONFIG.bet_size + fee
            position = Position(side, contracts, price, fee)
            action = f"OPEN_{side.upper()}"

        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                now.isoformat(), market.received_at.isoformat(),
                game.received_at.isoformat(), market.bid, market.ask,
                game.state["inning"], game.state["outs_when_up"],
                game.state["score_diff"], fair_prob, final_prob, edge,
                action, cash,
            ])
        print(
            f"{now.time()} {market.bid:.2f}/{market.ask:.2f} "
            f"fair={fair_prob:.1%} model={final_prob:.1%} "
            f"edge={edge:+.1%} {action}"
        )
        await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Paper trader stopped")
