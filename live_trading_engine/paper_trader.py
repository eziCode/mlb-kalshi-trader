"""Paper trader for the event-conditioned hybrid residual strategy."""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
import requests
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_kalshi.hybrid import (  # noqa: E402
    HIT_EVENTS,
    anchored_event_target,
)
from mlb_kalshi.trade_tape import TradeTapeConfig  # noqa: E402
from mlb_kalshi.strategy import (  # noqa: E402
    CONFIG,
    fee_aware_signal_side,
    state_feature_frame,
    taker_fee,
)


GAME_PK_TEXT = os.getenv("MLB_GAME_PK")
MARKET_TICKER = os.getenv("KALSHI_MARKET_TICKER")
GAME_PK = int(GAME_PK_TEXT) if GAME_PK_TEXT else None
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
HYBRID_CONFIG_PATH = MODEL_DIR / "trade_tape_config.json"
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
    completed_event_id: int | None
    completed_event: str | None
    completed_event_batting_home: bool | None
    latest_completed_pitch_token: tuple | None


@dataclass
class Position:
    side: str
    contracts: float
    entry_price: float
    entry_fee: float
    entry_time: datetime
    anchor_target: float
    anchor_fair: float
    event_id: int


@dataclass
class EventCandidate:
    side: str
    target: float
    event_id: int
    event_type: str
    observed_at: datetime
    event_time: datetime
    pre_market: float
    pre_fair: float
    post_fair: float
    material_state: tuple
    pitch_token: tuple | None


def material_state(state: dict) -> tuple:
    return (
        int(state["score_diff"]),
        int(state["outs_when_up"]),
        int(state["runner_on_first"]),
        int(state["runner_on_second"]),
        int(state["runner_on_third"]),
    )


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


def latest_completed_pitch_token(payload: dict) -> tuple | None:
    """Return a stable identity for the latest pitch with an end timestamp."""
    latest: tuple | None = None
    for play in payload.get("liveData", {}).get("plays", {}).get("allPlays", []):
        at_bat = play.get("atBatIndex")
        for event in play.get("playEvents") or []:
            if not event.get("isPitch") or not event.get("endTime"):
                continue
            token = (
                int(at_bat) if at_bat is not None else -1,
                int(event.get("pitchNumber") or event.get("index") or 0),
                str(event["endTime"]),
            )
            latest = token
    return latest


def pitch_token_time(token: tuple | None) -> datetime | None:
    if token is None:
        return None
    return pd.to_datetime(token[2], utc=True).to_pydatetime()


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
    elif status != "Live":
        topbot = 0
    else:
        raise RuntimeError(f"No active inning half: {inning_state!r}")
    teams = linescore.get("teams") or {}
    home_score = int(teams.get("home", {}).get("runs") or 0)
    away_score = int(teams.get("away", {}).get("runs") or 0)
    offense = linescore.get("offense") or {}
    completed_plays = [
        play for play in (live.get("plays", {}).get("allPlays") or [])
        if play.get("about", {}).get("isComplete")
    ]
    latest_play = completed_plays[-1] if completed_plays else None
    completed_event_id = (
        int(latest_play.get("atBatIndex"))
        if latest_play is not None and latest_play.get("atBatIndex") is not None
        else None
    )
    completed_event = (
        str(latest_play.get("result", {}).get("eventType") or "").lower()
        if latest_play is not None
        else None
    )
    completed_event_batting_home = (
        not bool(latest_play.get("about", {}).get("isTopInning"))
        if latest_play is not None
        else None
    )
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
        datetime.now(timezone.utc), status, state, home_score, away_score,
        completed_event_id, completed_event, completed_event_batting_home,
        latest_completed_pitch_token(payload),
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
    if GAME_PK is None or not MARKET_TICKER:
        raise RuntimeError(
            "Set MLB_GAME_PK and KALSHI_MARKET_TICKER explicitly; "
            "there are no safe default games or markets."
        )
    state_model = CatBoostClassifier()
    state_model.load_model(MODEL_DIR / "local_win_expectancy.cbm")
    hybrid_config = TradeTapeConfig(**json.loads(HYBRID_CONFIG_PATH.read_text()))
    allow_unvalidated = os.getenv("ALLOW_UNVALIDATED_HYBRID") == "1"
    if not hybrid_config.enabled and not allow_unvalidated:
        raise RuntimeError(
            "Hybrid policy has not passed a fresh forward test and is disabled. "
            "Set ALLOW_UNVALIDATED_HYBRID=1 only to collect paper observations."
        )
    pregame_prob = await asyncio.to_thread(fetch_pregame_anchor)
    print(f"Pregame Kalshi anchor: {pregame_prob:.1%}")
    print(
        f"Hybrid threshold={hybrid_config.minimum_edge:.1%}, "
        f"confirmation={hybrid_config.confirmation_seconds:g} seconds, "
        f"entry expiry={hybrid_config.maximum_event_to_entry_seconds:g} seconds, "
        "target-reversion exit"
    )

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"paper_trade_log_{MARKET_TICKER}_{int(time.time())}.csv"
    with log_path.open("w", newline="") as handle:
        csv.writer(handle).writerow([
            "decision_time", "market_received_at", "state_received_at",
            "bid", "ask", "inning", "outs", "score_diff", "fair_prob",
            "completed_event_id", "completed_event", "target", "excess_move",
            "edge", "continuation_value", "exit_advantage", "action", "cash",
        ])

    cash = 1000.0
    position: Position | None = None
    candidate: EventCandidate | None = None
    previous_market: float | None = None
    previous_fair: float | None = None
    previous_event_id: int | None = None
    exit_watch_started: datetime | None = None
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
        action = "HOLD"
        edge = float("nan")
        target = float("nan")
        continuation_value = float("nan")
        exit_advantage = float("nan")

        new_event = (
            previous_event_id is not None
            and game.completed_event_id is not None
            and game.completed_event_id == previous_event_id + 1
        )

        if position is not None:
            target = float(anchored_event_target(
                position.anchor_target, position.anchor_fair, fair_prob
            ))
            should_exit = (
                position.side == "yes" and market.bid >= target
            ) or (
                position.side == "no" and market.ask <= target
            )
            if should_exit and exit_watch_started is None:
                exit_watch_started = now
            elif not should_exit:
                exit_watch_started = None
            confirmed_exit = (
                should_exit
                and exit_watch_started is not None
                and (now - exit_watch_started).total_seconds()
                >= hybrid_config.confirmation_seconds
            )
            if confirmed_exit:
                price = market.bid if position.side == "yes" else 1.0 - market.ask
                fee = taker_fee(position.contracts, price)
                cash += position.contracts * price - fee
                reason = "TARGET_REVERSION"
                action = f"CLOSE_{position.side.upper()}_{reason}"
                position = None
                candidate = None
                exit_watch_started = None

        if position is None:
            if candidate is not None:
                candidate_age = (now - candidate.event_time).total_seconds()
                pitch_changed = (
                    hybrid_config.invalidate_on_next_pitch
                    and game.latest_completed_pitch_token != candidate.pitch_token
                )
                if (
                    candidate_age
                    > hybrid_config.maximum_event_to_entry_seconds
                    or pitch_changed
                ):
                    action = "EXPIRE_EVENT_CANDIDATE"
                    candidate = None

            if (
                candidate is not None
                and (
                    game.completed_event_id != candidate.event_id
                    or material_state(game.state) != candidate.material_state
                )
            ):
                candidate = None

            if (
                new_event
                and game.completed_event in HIT_EVENTS
                and previous_market is not None
                and previous_fair is not None
                and pitch_token_time(game.latest_completed_pitch_token) is not None
            ):
                signed_fair_move = (
                    float(fair_prob) - previous_fair
                ) * (
                    1.0 if game.completed_event_batting_home else -1.0
                )
                if signed_fair_move >= hybrid_config.minimum_fair_move:
                    target = float(anchored_event_target(
                        previous_market, previous_fair, fair_prob
                    ))
                    side, edge = fee_aware_signal_side(
                        target, market.bid, market.ask,
                        hybrid_config.minimum_edge,
                    )
                    candidate = (
                        EventCandidate(
                            side=side,
                            target=target,
                            event_id=int(game.completed_event_id),
                            event_type=str(game.completed_event),
                            observed_at=now,
                            event_time=pitch_token_time(
                                game.latest_completed_pitch_token
                            ),
                            pre_market=previous_market,
                            pre_fair=previous_fair,
                            post_fair=fair_prob,
                            material_state=material_state(game.state),
                            pitch_token=game.latest_completed_pitch_token,
                        )
                        if side is not None
                        else None
                    )
                    if candidate is not None:
                        action = (
                            f"WATCH_{side.upper()}_"
                            f"{game.completed_event.upper()}"
                        )

            if candidate is not None and position is None:
                target = float(anchored_event_target(
                    candidate.target, candidate.post_fair, fair_prob
                ))
                confirmed_side, edge = fee_aware_signal_side(
                    target, market.bid, market.ask,
                    hybrid_config.minimum_edge,
                )
                confirmation_age = (
                    now - candidate.observed_at
                ).total_seconds()
                if (
                    confirmation_age >= hybrid_config.confirmation_seconds
                    and confirmed_side == candidate.side
                    and game.completed_event_id == candidate.event_id
                ):
                    price = (
                        market.ask if candidate.side == "yes"
                        else 1.0 - market.bid
                    )
                    contracts = CONFIG.bet_size / price
                    fee = taker_fee(contracts, price)
                    cash -= CONFIG.bet_size + fee
                    position = Position(
                        side=candidate.side,
                        contracts=contracts,
                        entry_price=price,
                        entry_fee=fee,
                        entry_time=now,
                        anchor_target=candidate.target,
                        anchor_fair=candidate.post_fair,
                        event_id=candidate.event_id,
                    )
                    action = f"OPEN_{candidate.side.upper()}_{candidate.event_type.upper()}"
                    candidate = None
                    exit_watch_started = None

        excess_move = (
            market.midpoint - target if pd.notna(target) else float("nan")
        )

        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                now.isoformat(), market.received_at.isoformat(),
                game.received_at.isoformat(), market.bid, market.ask,
                game.state["inning"], game.state["outs_when_up"],
                game.state["score_diff"], fair_prob,
                game.completed_event_id, game.completed_event, target,
                excess_move, edge, continuation_value, exit_advantage,
                action, cash,
            ])
        print(
            f"{now.time()} {market.bid:.2f}/{market.ask:.2f} "
            f"fair={fair_prob:.1%} target={target:.1%} "
            f"edge={edge:+.1%} {action}"
        )
        previous_market = market.midpoint
        previous_fair = float(fair_prob)
        previous_event_id = game.completed_event_id
        await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Paper trader stopped")
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
