"""Live paper trader for the calibrated event-agnostic mispricing strategy.

This module never submits an order.  It polls the public MLB live feed, the
public Kalshi trade feed, and the public Kalshi order book.  Model features
are created only after a completed pitch and only from trades observable at
the configured signal delay.  Paper positions are held to settlement, which
matches the backtest contract.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from settlement_value_strategy.build_normalized_raw import state_model_frame
from settlement_value_strategy.predict import MispricingPredictor
from settlement_value_strategy.strategy import (
    MISPRICING_FEATURES, anchored_event_target, taker_fee,
)
from shared_kalshi_feed import get_market as get_shared_market


GAME_PK_TEXT = os.getenv("MLB_GAME_PK")
MARKET_TICKER = os.getenv("KALSHI_MARKET_TICKER")
AWAY_MARKET_TICKER = os.getenv("KALSHI_AWAY_MARKET_TICKER")
GAME_PK = int(GAME_PK_TEXT) if GAME_PK_TEXT else None
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))
LOG_DIR = Path(os.getenv("PAPER_LOG_DIR", str(ROOT / "results/live")))
KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
MLB_API = "https://statsapi.mlb.com/api"
MLB_PRIOR_PATH = ROOT / "model/mlb_pregame_prior.json"
SLATE_TIMEZONE = ZoneInfo(os.getenv("SLATE_TIMEZONE", "America/Chicago"))

MLB_TEAM_CODES = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY", 158: "MIL",
}
KALSHI_TEAM_CODES = {
    "AZ": "ARI", "ARI": "ARI", "CWS": "CHW", "CHW": "CHW",
    "KC": "KC", "KCR": "KC", "OAK": "ATH", "ATH": "ATH",
    "SD": "SD", "SDP": "SD", "SF": "SF", "SFG": "SF",
    "TB": "TB", "TBR": "TB", "WAS": "WSH", "WSH": "WSH",
}


@dataclass(frozen=True)
class MarketSnapshot:
    received_at: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float


@dataclass(frozen=True)
class GameSnapshot:
    received_at: datetime
    status: str
    state: dict
    home_score: int
    away_score: int
    pitch_token: tuple[int, int, str] | None


@dataclass(frozen=True)
class PaperPosition:
    side: str
    contracts: float
    entry_price: float
    entry_fee: float
    entry_time: datetime
    settlement_probability: float
    trigger_pitch: str


@dataclass(frozen=True)
class PortfolioMetrics:
    cash: float
    equity: float
    pnl: float
    open_positions: int


@dataclass(frozen=True)
class DiscoveredGame:
    game_pk: int
    scheduled_time: datetime
    away_code: str
    home_code: str
    market_ticker: str
    away_market_ticker: str


class SharedPaperPortfolio:
    """SQLite-backed cash and settlement positions shared by game workers."""

    def __init__(self, path: Path, starting_cash: float = 1000.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS portfolio ("
                "id INTEGER PRIMARY KEY CHECK (id=1), "
                "starting_cash REAL NOT NULL, cash REAL NOT NULL)"
            )
            columns = connection.execute(
                "PRAGMA table_info(positions)"
            ).fetchall()
            if any(row[1] == "game_pk" and row[5] == 1 for row in columns):
                connection.execute(
                    "ALTER TABLE positions RENAME TO positions_single_position"
                )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS positions (
                    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_pk INTEGER NOT NULL,
                    market_ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    contracts REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_fee REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    settlement_probability REAL NOT NULL,
                    trigger_pitch TEXT NOT NULL,
                    mark_price REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(game_pk, trigger_pitch)
                )"""
            )
            if columns and any(
                row[1] == "game_pk" and row[5] == 1 for row in columns
            ):
                connection.execute(
                    """INSERT OR IGNORE INTO positions (
                        game_pk,market_ticker,side,contracts,entry_price,
                        entry_fee,entry_time,settlement_probability,
                        trigger_pitch,mark_price,updated_at
                    ) SELECT game_pk,market_ticker,side,contracts,entry_price,
                        entry_fee,entry_time,settlement_probability,
                        trigger_pitch,mark_price,updated_at
                    FROM positions_single_position"""
                )
                connection.execute("DROP TABLE positions_single_position")
            connection.execute(
                "INSERT OR IGNORE INTO portfolio VALUES (1, ?, ?)",
                (starting_cash, starting_cash),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def open_position(
        self, game_pk: int, ticker: str, position: PaperPosition,
        minimum_seconds_between_entries: float = 0.0,
    ) -> bool:
        cost = position.contracts * position.entry_price + position.entry_fee
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            cash = float(connection.execute(
                "SELECT cash FROM portfolio WHERE id=1"
            ).fetchone()[0])
            duplicate = connection.execute(
                "SELECT 1 FROM positions WHERE game_pk=? AND trigger_pitch=?",
                (game_pk, position.trigger_pitch),
            ).fetchone()
            latest = connection.execute(
                "SELECT entry_time FROM positions WHERE game_pk=? "
                "ORDER BY entry_time DESC LIMIT 1", (game_pk,),
            ).fetchone()
            cooling_down = bool(
                latest
                and (
                    position.entry_time
                    - pd.to_datetime(latest[0], utc=True).to_pydatetime()
                ).total_seconds() < minimum_seconds_between_entries
            )
            if duplicate or cooling_down or cash + 1e-9 < cost:
                connection.rollback()
                return False
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "UPDATE portfolio SET cash=cash-? WHERE id=1", (cost,)
            )
            connection.execute(
                """INSERT INTO positions (
                    game_pk,market_ticker,side,contracts,entry_price,entry_fee,
                    entry_time,settlement_probability,trigger_pitch,mark_price,
                    updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_pk, ticker, position.side, position.contracts,
                    position.entry_price, position.entry_fee,
                    position.entry_time.isoformat(),
                    position.settlement_probability, position.trigger_pitch,
                    position.entry_price, now,
                ),
            )
        return True

    def load_positions(self, game_pk: int) -> list[PaperPosition]:
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "SELECT side,contracts,entry_price,entry_fee,entry_time,"
                "settlement_probability,trigger_pitch FROM positions "
                "WHERE game_pk=? ORDER BY entry_time", (game_pk,),
            ).fetchall()
        return [
            PaperPosition(
                side=str(row[0]), contracts=float(row[1]),
                entry_price=float(row[2]), entry_fee=float(row[3]),
                entry_time=pd.to_datetime(row[4], utc=True).to_pydatetime(),
                settlement_probability=float(row[5]),
                trigger_pitch=str(row[6]),
            )
            for row in rows
        ]

    def open_game_pks(self) -> list[int]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT DISTINCT game_pk FROM positions ORDER BY game_pk"
            ).fetchall()
        return [int(row[0]) for row in rows]

    def update_marks(
        self, game_pk: int, yes_bid: float, yes_ask: float,
        away_yes_bid: float | None = None,
    ) -> None:
        for attempt in range(6):
            try:
                with closing(self._connect()) as connection, connection:
                    connection.execute(
                        """UPDATE positions SET mark_price=CASE
                            WHEN side='yes' THEN ?
                            WHEN side='away_yes' THEN ?
                            ELSE 1.0-? END,
                            updated_at=? WHERE game_pk=?""",
                        (
                            yes_bid,
                            away_yes_bid
                            if away_yes_bid is not None else 1.0-yes_ask,
                            yes_ask,
                            datetime.now(timezone.utc).isoformat(), game_pk,
                        ),
                    )
                return
            except sqlite3.OperationalError as error:
                transient = (
                    "locked" in str(error).lower()
                    or "locking protocol" in str(error).lower()
                )
                if not transient or attempt == 5:
                    raise
                time.sleep(0.05 * (2 ** attempt))

    def settle(self, game_pk: int, proceeds: float) -> bool:
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM positions WHERE game_pk=?", (game_pk,)
            ).fetchone()
            if not exists:
                connection.rollback()
                return False
            connection.execute(
                "UPDATE portfolio SET cash=cash+? WHERE id=1", (proceeds,)
            )
            connection.execute("DELETE FROM positions WHERE game_pk=?", (game_pk,))
        return True

    def metrics(self) -> PortfolioMetrics:
        with closing(self._connect()) as connection, connection:
            starting, cash = connection.execute(
                "SELECT starting_cash,cash FROM portfolio WHERE id=1"
            ).fetchone()
            rows = connection.execute(
                "SELECT contracts,mark_price FROM positions"
            ).fetchall()
        liquidation = sum(
            float(contracts) * float(price)
            - taker_fee(float(contracts), float(price))
            for contracts, price in rows
        )
        equity = float(cash) + liquidation
        return PortfolioMetrics(
            float(cash), equity, equity - float(starting), len(rows)
        )


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1 - 1e-4)
    return math.log(value / (1 - value))


def state_delta(before: dict, after: dict) -> dict:
    names = (
        "inning", "outs_when_up", "score_diff", "balls", "strikes",
        "runner_on_first", "runner_on_second", "runner_on_third",
    )
    return {
        ("delta_outs" if name == "outs_when_up" else f"delta_{name}"):
        float(after[name]) - float(before[name])
        for name in names
    }


def consecutive_pitch(
    previous: tuple[int, int, str] | None,
    current: tuple[int, int, str] | None,
) -> bool:
    """Reject polling gaps rather than attributing several pitches to one row."""
    if previous is None or current is None:
        return False
    previous_at_bat, previous_pitch, _ = previous
    current_at_bat, current_pitch, _ = current
    return (
        current_at_bat == previous_at_bat
        and current_pitch == previous_pitch + 1
    ) or (
        current_at_bat == previous_at_bat + 1 and current_pitch == 1
    )


def flow_features(trades: pd.DataFrame, signal_index: int) -> dict:
    """Training-identical two-second flow strictly before the signal trade."""
    signal_time = pd.Timestamp(trades.iloc[signal_index].created_time)
    prior = trades.iloc[:signal_index]
    prior = prior[
        pd.to_datetime(prior.created_time, utc=True)
        >= signal_time - pd.Timedelta(seconds=2)
    ]
    sizes = prior.count_fp.astype(float).to_numpy()
    prices = prior.yes_price_dollars.astype(float).to_numpy()
    volume = float(sizes.sum())
    signs = np.where(prior.taker_outcome_side.astype(str) == "yes", 1.0, -1.0)
    return {
        "pre_trade_count_2s": float(len(prior)),
        "pre_volume_2s": volume,
        "pre_flow_imbalance_2s": (
            float((sizes * signs).sum() / volume) if volume else 0.0
        ),
        "pre_price_volatility_2s": float(np.std(prices)) if len(prior) else 0.0,
    }


def build_live_decision_row(
    *, game_pk: int, before: dict, after: dict, fair_before: float,
    fair_after: float, pitch_token: tuple[int, int, str],
    trades: pd.DataFrame, config,
) -> dict | None:
    """Construct one causal model row, or return None until inputs qualify."""
    if trades.empty:
        return None
    tape = trades.sort_values(["created_time", "trade_id"]).reset_index(drop=True)
    times = pd.to_datetime(tape.created_time, utc=True)
    event_time = pd.Timestamp(pitch_token[2])
    cutoff = event_time - pd.Timedelta(seconds=config.anchor_buffer_seconds)
    anchor_positions = np.flatnonzero((times < cutoff).to_numpy())
    signal_positions = np.flatnonzero(
        (times >= event_time + pd.Timedelta(
            seconds=config.observation_delay_seconds
        )).to_numpy()
    )
    if not len(anchor_positions) or not len(signal_positions):
        return None
    anchor_index = int(anchor_positions[-1])
    signal_index = int(signal_positions[0])
    anchor_age = (cutoff - times.iloc[anchor_index]).total_seconds()
    observation_delay = (times.iloc[signal_index] - event_time).total_seconds()
    if (
        anchor_age > config.maximum_anchor_age_seconds
        or observation_delay > 10.0
    ):
        return None
    anchor_market = float(tape.iloc[anchor_index].yes_price_dollars)
    market = float(tape.iloc[signal_index].yes_price_dollars)
    target = float(anchored_event_target(anchor_market, fair_before, fair_after))
    row = {
        "game_pk": int(game_pk),
        "trigger_at_bat": int(pitch_token[0]) + 1,
        "trigger_pitch": int(pitch_token[1]),
        "trigger_time": event_time.isoformat(),
        "signal_time": times.iloc[signal_index].isoformat(),
        "market_home_price": market,
        "market_logit": _logit(market),
        "local_fair_after": float(fair_after),
        "local_fair_before": float(fair_before),
        "fair_logit_move": _logit(fair_after) - _logit(fair_before),
        "market_logit_move": _logit(market) - _logit(anchor_market),
        "anchored_state_target": target,
        "market_target_residual": market - target,
        "anchor_age_seconds": anchor_age,
        "observation_delay_seconds": observation_delay,
        **{f"{name}_after": float(value) for name, value in after.items()},
        **state_delta(before, after),
        **flow_features(tape, signal_index),
    }
    return row


def execution_within_window(
    signal_time: object, execution_time: datetime, maximum_delay: float,
) -> bool:
    age = (
        execution_time
        - pd.Timestamp(signal_time).to_pydatetime()
    ).total_seconds()
    # Backtest fills use searchsorted(..., side="right"), so equality is not
    # executable: the observed quote must be strictly later than the signal.
    return 0.0 < age < maximum_delay


def replay_fill_from_observed_trades(
    trades: pd.DataFrame, signal_time: object, execution_probability: float,
    positions: list[PaperPosition], execution_side: str, config,
) -> dict | None:
    """Apply the backtest's post-signal compatible-trade fill contract."""
    if trades.empty:
        return None
    signal = pd.Timestamp(signal_time)
    deadline = signal + pd.Timedelta(seconds=config.maximum_fill_delay_seconds)
    earliest = signal
    if positions:
        latest = max(pd.Timestamp(position.entry_time) for position in positions)
        earliest = max(
            earliest,
            latest + pd.Timedelta(
                seconds=config.minimum_seconds_between_entries
            ),
        )
    comparable = [
        position for position in positions if position.side == execution_side
    ]
    best_probability = max(
        (position.settlement_probability for position in comparable),
        default=float("-inf"),
    )
    best_return = max((
        (
            position.contracts
            * (position.settlement_probability - position.entry_price)
            - position.entry_fee
        ) / (
            position.contracts * position.entry_price + position.entry_fee
        )
        for position in comparable
    ), default=float("-inf"))
    tape = trades.sort_values(["created_time", "trade_id"])
    times = pd.to_datetime(tape.created_time, utc=True)
    eligible = tape[
        times.gt(signal) & times.ge(earliest) & times.lt(deadline)
        & tape.taker_outcome_side.astype(str).eq("yes")
    ]
    for trade in eligible.itertuples(index=False):
        price = float(trade.yes_price_dollars)
        contracts = config.bet_size / price
        if float(trade.count_fp) < contracts:
            continue
        fee = taker_fee(contracts, price)
        edge = execution_probability - price
        expected = contracts * edge - fee
        expected_return = expected / (config.bet_size + fee)
        if (
            expected < config.minimum_expected_pnl
            or edge < config.minimum_probability_edge
        ):
            continue
        if config.conditional_stacking and not (
            execution_probability > best_probability
            and expected_return > best_return
        ):
            continue
        return {
            "time": pd.Timestamp(trade.created_time).to_pydatetime(),
            "price": price, "contracts": contracts, "fee": fee,
            "edge": edge, "expected_pnl": expected,
        }
    return None


def fetch_market_snapshot(ticker: str | None = None) -> MarketSnapshot:
    ticker = ticker or MARKET_TICKER
    if os.getenv("KALSHI_FEED_URL"):
        payload = get_shared_market(str(ticker))
        snapshot = payload["snapshot"]
        return MarketSnapshot(
            pd.to_datetime(snapshot["received_at"], utc=True).to_pydatetime(),
            float(snapshot["bid"]), float(snapshot["ask"]),
            float(snapshot["bid_size"]), float(snapshot["ask_size"]),
        )
    response = requests.get(
        f"{KALSHI_API}/markets/{ticker}/orderbook", timeout=5
    )
    response.raise_for_status()
    book = response.json().get("orderbook_fp") or {}
    yes = book.get("yes_dollars") or []
    no = book.get("no_dollars") or []
    if not yes or not no:
        raise RuntimeError("Kalshi order book is not two-sided")
    bid, ask = float(yes[-1][0]), 1.0 - float(no[-1][0])
    bid_size, ask_size = float(yes[-1][1]), float(no[-1][1])
    if not 0 < bid < ask < 1:
        raise RuntimeError(f"Invalid Kalshi order book: {bid}/{ask}")
    return MarketSnapshot(
        datetime.now(timezone.utc), bid, ask, bid_size, ask_size
    )


def fetch_recent_trades(ticker: str | None = None) -> pd.DataFrame:
    ticker = ticker or MARKET_TICKER
    if os.getenv("KALSHI_FEED_URL"):
        rows = get_shared_market(str(ticker)).get("trades") or []
    else:
        response = requests.get(
            f"{KALSHI_API}/markets/trades",
            params={"ticker": ticker, "limit": 1000}, timeout=5,
        )
        response.raise_for_status()
        rows = response.json().get("trades") or []
    frame = pd.DataFrame(rows)
    required = {
        "trade_id", "created_time", "yes_price_dollars", "count_fp",
        "taker_outcome_side",
    }
    if frame.empty or (missing := required - set(frame.columns)):
        if frame.empty:
            return pd.DataFrame(columns=sorted(required))
        raise RuntimeError(f"Kalshi trade response missing {sorted(missing)}")
    frame["created_time"] = pd.to_datetime(frame.created_time, utc=True)
    for name in ("yes_price_dollars", "count_fp"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame.sort_values(["created_time", "trade_id"]).drop_duplicates("trade_id")


def _latest_pitch(payload: dict) -> tuple[int, int, str] | None:
    latest = None
    for play in payload.get("liveData", {}).get("plays", {}).get("allPlays", []):
        at_bat = int(play.get("atBatIndex", -1))
        for event in play.get("playEvents") or []:
            if event.get("isPitch") and event.get("endTime"):
                latest = (
                    at_bat, int(event.get("pitchNumber") or event.get("index") or 0),
                    str(event["endTime"]),
                )
    return latest


def fetch_game_snapshot() -> GameSnapshot:
    response = requests.get(f"{MLB_API}/v1.1/game/{GAME_PK}/feed/live", timeout=5)
    response.raise_for_status()
    payload = response.json()
    linescore = payload.get("liveData", {}).get("linescore") or {}
    status = payload.get("gameData", {}).get("status", {}).get(
        "abstractGameState", "Preview"
    )
    inning_state = str(linescore.get("inningState") or "").lower()
    if inning_state.startswith("top"):
        topbot = 0
    elif inning_state.startswith("bottom"):
        topbot = 1
    elif status != "Live":
        topbot = 0
    else:
        raise RuntimeError(f"No active inning half: {inning_state!r}")
    teams = linescore.get("teams") or {}
    home = int(teams.get("home", {}).get("runs") or 0)
    away = int(teams.get("away", {}).get("runs") or 0)
    offense = linescore.get("offense") or {}
    state = {
        "inning": int(linescore.get("currentInning") or 1),
        "inning_topbot": topbot,
        "outs_when_up": int(linescore.get("outs") or 0),
        "score_diff": home - away,
        "balls": int(linescore.get("balls") or 0),
        "strikes": int(linescore.get("strikes") or 0),
        "runner_on_first": int("first" in offense),
        "runner_on_second": int("second" in offense),
        "runner_on_third": int("third" in offense),
    }
    return GameSnapshot(
        datetime.now(timezone.utc), status, state, home, away,
        _latest_pitch(payload),
    )


def home_fair_probability(model, pregame_prob: float, state: dict) -> float:
    frame = pd.DataFrame([{**state, "pregame_prob": pregame_prob}])
    batting = float(model.predict_proba(
        state_model_frame(frame), thread_count=1,
    )[0, 1])
    return batting if int(state["inning_topbot"]) == 1 else 1.0 - batting


def pregame_probability_from_rating_state(
    state: dict, home_code: str, away_code: str,
) -> float:
    ratings = state["ratings"]
    aliases = {
        "ARI": "AZ", "CHW": "CWS", "OAK": "ATH",
        "KCR": "KC", "SDP": "SD", "SFG": "SF",
        "TBR": "TB", "WAS": "WSH",
    }

    def rating(code: str) -> float:
        code = str(code).upper()
        key = code if code in ratings else aliases.get(code, code)
        return float(ratings.get(key, state["initial_rating"]))

    home_rating = rating(home_code)
    away_rating = rating(away_code)
    return 1.0 / (
        1.0 + 10.0 ** (-(
            home_rating + float(state["home_advantage"]) - away_rating
        ) / 400.0)
    )


def fetch_pregame_anchor() -> float:
    state = json.loads(MLB_PRIOR_PATH.read_text())
    home = MARKET_TICKER.rsplit("-", 1)[-1]
    away = AWAY_MARKET_TICKER.rsplit("-", 1)[-1]
    return pregame_probability_from_rating_state(state, home, away)


async def wait_for_pregame_anchor() -> float:
    """Wait through MLB's transient Live-without-pitch startup state."""
    while True:
        try:
            return await asyncio.to_thread(fetch_pregame_anchor)
        except RuntimeError as error:
            if "no authoritative first-pitch time" not in str(error).lower():
                raise
            print(
                "Pregame anchor pending: MLB marks the game Live but has not "
                "published a pitch start time; retrying",
                flush=True,
            )
            await asyncio.sleep(max(POLL_SECONDS, 1.0))


def _market_team_code(market: dict) -> str | None:
    ticker = str(market.get("ticker") or "")
    code = ticker.rsplit("-", 1)[-1].upper() if "-" in ticker else ""
    return KALSHI_TEAM_CODES.get(code, code) or None


def _event_game_date(event: dict) -> date | None:
    match = re.match(
        r"^KXMLBGAME-(\d{2})([A-Z]{3})(\d{2})",
        str(event.get("event_ticker") or ""),
    )
    if not match:
        return None
    year, month_name, day = match.groups()
    months = {
        name: number for number, name in enumerate(
            ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
             "SEP", "OCT", "NOV", "DEC"), 1
        )
    }
    return date(2000 + int(year), months[month_name], int(day))


def _daily_kalshi_events(game_date: date) -> list[dict]:
    params = {
        "series_ticker": "KXMLBGAME", "limit": 200,
        "with_nested_markets": "true",
        "min_close_ts": int((
            datetime.combine(game_date, datetime.min.time(), tzinfo=timezone.utc)
            - timedelta(days=2)
        ).timestamp()),
    }
    events = []
    while True:
        response = requests.get(f"{KALSHI_API}/events", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        events.extend(payload.get("events") or [])
        cursor = payload.get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor
    return [event for event in events if _event_game_date(event) == game_date]


def discover_daily_games(game_date: date) -> tuple[list[DiscoveredGame], list[str]]:
    response = requests.get(
        f"{MLB_API}/v1/schedule",
        params={"sportId": 1, "date": game_date.isoformat()}, timeout=30,
    )
    response.raise_for_status()
    games = [
        game for day in response.json().get("dates") or []
        for game in day.get("games") or []
        if str(game.get("status", {}).get("detailedState") or "").lower()
        not in {"postponed", "cancelled", "canceled"}
    ]
    events = _daily_kalshi_events(game_date)
    event_groups = {}
    for event in events:
        markets = event.get("markets") or []
        by_team = {
            code: market for market in markets
            if (code := _market_team_code(market)) is not None
        }
        if len(by_team) == 2:
            event_groups.setdefault(frozenset(by_team), []).append(
                (str(event.get("event_ticker") or ""), by_team)
            )
    for values in event_groups.values():
        values.sort(key=lambda value: value[0])
    game_groups = {}
    warnings = []
    for game in games:
        teams = game.get("teams") or {}
        away_id = teams.get("away", {}).get("team", {}).get("id")
        home_id = teams.get("home", {}).get("team", {}).get("id")
        away = MLB_TEAM_CODES.get(int(away_id)) if away_id else None
        home = MLB_TEAM_CODES.get(int(home_id)) if home_id else None
        if not away or not home:
            warnings.append(f"Game {game.get('gamePk')} has unknown team IDs")
            continue
        scheduled = pd.to_datetime(game["gameDate"], utc=True).to_pydatetime()
        game_groups.setdefault(frozenset({away, home}), []).append(
            (scheduled, game, away, home)
        )
    matched = []
    for matchup, scheduled_games in game_groups.items():
        scheduled_games.sort(key=lambda value: value[0])
        markets = event_groups.get(matchup, [])
        if len(scheduled_games) != len(markets):
            warnings.append(
                f"{sorted(matchup)}: {len(scheduled_games)} games and "
                f"{len(markets)} Kalshi events; skipped"
            )
            continue
        for (scheduled, game, away, home), (_, by_team) in zip(
            scheduled_games, markets
        ):
            if game.get("status", {}).get("abstractGameState") == "Final":
                continue
            matched.append(DiscoveredGame(
                int(game["gamePk"]), scheduled, away, home,
                str(by_team[home]["ticker"]),
                str(by_team[away]["ticker"]),
            ))
    return sorted(matched, key=lambda game: game.scheduled_time), warnings


async def run_worker() -> None:
    if GAME_PK is None or not MARKET_TICKER:
        raise RuntimeError("Set MLB_GAME_PK and KALSHI_MARKET_TICKER")
    predictor = MispricingPredictor()
    if (
        predictor.config.execution_contract in {"away_yes", "paired_both"}
        and not AWAY_MARKET_TICKER
    ):
        raise RuntimeError("Set KALSHI_AWAY_MARKET_TICKER for away YES routing")
    if not predictor.config.enabled and os.getenv(
        "ALLOW_UNVALIDATED_MISPRICING"
    ) != "1":
        raise RuntimeError(
            "Mispricing policy is disabled; set "
            "ALLOW_UNVALIDATED_MISPRICING=1 for paper observation only"
        )
    state_model = CatBoostClassifier()
    state_model.load_model(ROOT / "model/local_win_expectancy.cbm")
    portfolio_path = Path(os.getenv(
        "PAPER_PORTFOLIO_DB",
        str(LOG_DIR / f"settlement_value_portfolio_{date.today()}_game_{GAME_PK}.sqlite3"),
    ))
    portfolio = SharedPaperPortfolio(
        portfolio_path, float(os.getenv("PAPER_STARTING_CASH", "1000"))
    )
    pregame_prob = await wait_for_pregame_anchor()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"settlement_value_decisions_v2_{MARKET_TICKER}.csv"
    new_log = not log_path.exists() or log_path.stat().st_size == 0
    if new_log:
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                "decision_time", "pitch", "signal_time", "execution_age_seconds",
                "home_bid", "home_ask", "home_ask_size", "away_bid",
                "away_ask", "away_ask_size", "settlement_probability",
                "model_side", "model_expected_pnl", "model_edge", "fill_price",
                "fill_expected_pnl", "fill_edge", "action", "cash", "equity",
                "portfolio_pnl", *MISPRICING_FEATURES,
            ])
    positions = portfolio.load_positions(int(GAME_PK))
    if positions:
        print(f"Recovered {len(positions)} open position(s) for game {GAME_PK}")
    previous_game = None
    handled_tokens: set[tuple[int, int, str]] = set()
    print(f"Pregame Kalshi anchor: {pregame_prob:.1%}; log={log_path}")
    while True:
        try:
            # Check the authoritative game state before reading Kalshi.  Kalshi
            # can remove the live order book as soon as a game ends; if that
            # request is grouped with the feed request, its error masks the
            # MLB "Final" response and this process polls the dead market
            # forever.
            game = await asyncio.to_thread(fetch_game_snapshot)
        except Exception as error:
            print(f"Game snapshot rejected: {error}", flush=True)
            await asyncio.sleep(POLL_SECONDS)
            continue
        if game.status == "Final":
            if positions:
                total_payout = 0.0
                for position in positions:
                    won = (
                        position.side == "yes"
                        and game.home_score > game.away_score
                    ) or (
                        position.side == "no"
                        and game.home_score < game.away_score
                    ) or (
                        position.side == "away_yes"
                        and game.home_score < game.away_score
                    )
                    payout = position.contracts if won else 0.0
                    total_payout += payout
                    print(
                        f"TRADE SETTLE {position.side.upper()} "
                        f"result={'WIN' if won else 'LOSS'} "
                        f"contracts={position.contracts:.4f} "
                        f"payout={payout:.4f} reason=GAME_FINAL "
                        f"game_pk={GAME_PK} ticker="
                        f"{AWAY_MARKET_TICKER if position.side == 'away_yes' else MARKET_TICKER}",
                        flush=True,
                    )
                portfolio.settle(int(GAME_PK), total_payout)
            metrics = portfolio.metrics()
            print(f"Shared portfolio: equity=${metrics.equity:.2f} PnL=${metrics.pnl:+.2f}")
            return
        if game.status != "Live":
            previous_game = game
            await asyncio.sleep(POLL_SECONDS)
            continue
        try:
            market, trades, away_market, away_trades = await asyncio.gather(
                asyncio.to_thread(fetch_market_snapshot, MARKET_TICKER),
                asyncio.to_thread(fetch_recent_trades),
                asyncio.to_thread(fetch_market_snapshot, AWAY_MARKET_TICKER),
                asyncio.to_thread(fetch_recent_trades, AWAY_MARKET_TICKER),
            )
        except Exception as error:
            print(f"Market snapshot rejected: {error}", flush=True)
            await asyncio.sleep(POLL_SECONDS)
            continue
        if positions:
            portfolio.update_marks(
                int(GAME_PK), market.bid, market.ask, away_market.bid
            )
        if previous_game is None or previous_game.status != "Live":
            print(
                f"TRADER READY game_pk={GAME_PK} ticker={MARKET_TICKER} "
                f"status=LIVE inning={game.state['inning']} "
                f"home_bid={market.bid:.4f} home_ask={market.ask:.4f} "
                f"away_bid={away_market.bid:.4f} "
                f"away_ask={away_market.ask:.4f}",
                flush=True,
            )
            previous_game = game
            if game.pitch_token:
                handled_tokens.add(game.pitch_token)
            print("INITIALIZE_LIVE_BASELINE; visible pitches will not be traded")
            await asyncio.sleep(POLL_SECONDS)
            continue
        token = game.pitch_token
        if token is None or token == previous_game.pitch_token or token in handled_tokens:
            previous_game = game
            await asyncio.sleep(POLL_SECONDS)
            continue
        if not consecutive_pitch(previous_game.pitch_token, token):
            handled_tokens.add(token)
            previous_game = game
            print(
                f"SKIP_PITCH_GAP reset baseline at {token}; causal before-state "
                "is unavailable",
                flush=True,
            )
            await asyncio.sleep(POLL_SECONDS)
            continue
        fair_before = home_fair_probability(
            state_model, pregame_prob, previous_game.state
        )
        fair_after = home_fair_probability(state_model, pregame_prob, game.state)
        row = build_live_decision_row(
            game_pk=int(GAME_PK), before=previous_game.state, after=game.state,
            fair_before=fair_before, fair_after=fair_after, pitch_token=token,
            trades=trades, config=predictor.config,
        )
        fill_ev = fill_edge = price = None
        if row is None:
            # Do not mark handled until the post-delay trade arrives or expires.
            if (
                datetime.now(timezone.utc)
                - pd.Timestamp(token[2]).to_pydatetime()
            ).total_seconds() <= 10:
                await asyncio.sleep(POLL_SECONDS)
                continue
            action, decision = "SKIP_INCOMPLETE_CAUSAL_INPUTS", None
        else:
            decision = predictor.decision(row)
            action = "NO_SIGNAL"
            decision_time = datetime.now(timezone.utc)
            signal_time = pd.Timestamp(row["signal_time"]).to_pydatetime()
            execution_age = (decision_time - signal_time).total_seconds()
            if decision["eligible"] and (
                predictor.config.execution_contract != "away_yes"
                or decision["side"] == "no"
            ):
                route_away_yes = (
                    predictor.config.execution_contract == "away_yes"
                    or (
                        predictor.config.execution_contract == "paired_both"
                        and decision["side"] == "no"
                    )
                )
                side = str(decision["side"])
                execution_side = "away_yes" if route_away_yes else side
                probability = float(decision["settlement_probability"])
                execution_probability = (
                    1.0 - probability if route_away_yes else probability
                )
                execution_tape = away_trades if route_away_yes else trades
                fill = replay_fill_from_observed_trades(
                    execution_tape, row["signal_time"], execution_probability,
                    positions, execution_side, predictor.config,
                )
                deadline = (
                    pd.Timestamp(row["signal_time"])
                    + pd.Timedelta(
                        seconds=predictor.config.maximum_fill_delay_seconds
                    )
                )
                if fill is None and pd.Timestamp.now(tz="UTC") < deadline:
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                if fill is None:
                    action = "SKIP_NO_COMPATIBLE_POST_SIGNAL_FILL"
                else:
                    price = fill["price"]
                    contracts = fill["contracts"]
                    fee = fill["fee"]
                    fill_edge = fill["edge"]
                    fill_ev = fill["expected_pnl"]
                    execution_time = fill["time"]
                    proposed = PaperPosition(
                        execution_side, contracts, price, fee,
                        execution_time, execution_probability,
                        str(token),
                    )
                    if (
                        predictor.config.maximum_positions_per_game > 0
                        and len(positions)
                        >= predictor.config.maximum_positions_per_game
                    ):
                        action = "SKIP_MAXIMUM_GAME_POSITIONS"
                        handled_tokens.add(token)
                        previous_game = game
                        await asyncio.sleep(POLL_SECONDS)
                        continue
                    if portfolio.open_position(
                        int(GAME_PK),
                        AWAY_MARKET_TICKER if route_away_yes else MARKET_TICKER,
                        proposed,
                        predictor.config.minimum_seconds_between_entries,
                    ):
                        positions.append(proposed)
                        action = f"OPEN_{execution_side.upper()}"
                        print(
                            f"TRADE BUY {'AWAY YES' if route_away_yes else side.upper()} "
                            f"contracts={contracts:.4f} "
                            f"price={price:.4f} fee={fee:.4f} "
                            f"signal_age={execution_age:.3f}s "
                            f"game_pk={GAME_PK} ticker="
                            f"{AWAY_MARKET_TICKER if route_away_yes else MARKET_TICKER}",
                            flush=True,
                        )
                    else:
                        action = "SKIP_CASH_COOLDOWN_OR_DUPLICATE_SIGNAL"
        handled_tokens.add(token)
        metrics = portfolio.metrics()
        values = decision or {}
        decision_time = datetime.now(timezone.utc)
        signal_time_value = row.get("signal_time") if row else None
        execution_age_value = (
            (decision_time - pd.Timestamp(signal_time_value).to_pydatetime())
            .total_seconds()
            if signal_time_value else None
        )
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                decision_time.isoformat(), str(token), signal_time_value,
                execution_age_value, market.bid, market.ask, market.ask_size,
                away_market.bid, away_market.ask, away_market.ask_size,
                values.get("settlement_probability"), values.get("side"),
                values.get("expected_pnl"), values.get("probability_edge"),
                price, fill_ev, fill_edge, action, metrics.cash, metrics.equity,
                metrics.pnl,
                *[(row or {}).get(name) for name in MISPRICING_FEATURES],
            ])
        print(
            f"{token} {market.bid:.2f}/{market.ask:.2f} "
            f"fair={values.get('settlement_probability', float('nan')):.1%} "
            f"{action} equity=${metrics.equity:.2f} pnl=${metrics.pnl:+.2f}",
            flush=True,
        )
        previous_game = game
        await asyncio.sleep(POLL_SECONDS)


MAIN_LOG_ACTIONS = ("TRADER READY", "TRADE ", "Shared portfolio")


def should_surface_worker_line(line: str) -> bool:
    return any(marker in line for marker in MAIN_LOG_ACTIONS)


def _relay(stream, handle, label: str) -> None:
    """Persist all worker detail, but surface only executions and settlement."""
    for line in stream:
        handle.write(line)
        handle.flush()
        if should_surface_worker_line(line):
            print(f"[{label}] {line.rstrip()}", flush=True)


def reconcile_final_positions(portfolio: SharedPaperPortfolio) -> int:
    """Settle positions whose workers disappeared before observing Final."""
    settled = 0
    for game_pk in portfolio.open_game_pks():
        try:
            response = requests.get(
                f"{MLB_API}/v1.1/game/{game_pk}/feed/live", timeout=15
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            print(
                f"WARNING: could not reconcile game {game_pk}: {error}",
                flush=True,
            )
            continue
        status = payload.get("gameData", {}).get("status", {}).get(
            "abstractGameState"
        )
        if status != "Final":
            continue
        teams = payload.get("liveData", {}).get("linescore", {}).get(
            "teams", {}
        )
        home_score = int(teams.get("home", {}).get("runs") or 0)
        away_score = int(teams.get("away", {}).get("runs") or 0)
        positions = portfolio.load_positions(game_pk)
        payout = sum(
            position.contracts
            for position in positions
            if (
                position.side == "yes" and home_score > away_score
            ) or (
                position.side in {"no", "away_yes"}
                and away_score > home_score
            )
        )
        if portfolio.settle(game_pk, payout):
            settled += len(positions)
            print(
                f"Reconciled final game {game_pk}: {away_score}-{home_score} "
                f"positions={len(positions)} payout=${payout:.2f}",
                flush=True,
            )
    return settled


def run_all_games(game_date: date) -> int:
    if os.getenv("ALLOW_UNVALIDATED_MISPRICING") != "1":
        raise RuntimeError("Set ALLOW_UNVALIDATED_MISPRICING=1 for paper mode")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    database = Path(os.getenv(
        "PAPER_PORTFOLIO_DB",
        str(LOG_DIR / f"settlement_value_portfolio_{game_date}.sqlite3"),
    ))
    portfolio = SharedPaperPortfolio(
        database, float(os.getenv("PAPER_STARTING_CASH", "1000"))
    )
    reconcile_final_positions(portfolio)
    games, warnings = discover_daily_games(game_date)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if not games:
        metrics = portfolio.metrics()
        print(
            f"No active matched games for {game_date}; "
            f"equity=${metrics.equity:.2f} PnL=${metrics.pnl:+.2f}",
            flush=True,
        )
        return 0
    print(f"Games for {game_date} ({len(games)}):", flush=True)
    for game in games:
        print(
            f"  {game.scheduled_time.isoformat()} "
            f"{game.away_code}@{game.home_code} game_pk={game.game_pk} "
            f"ticker={game.market_ticker}",
            flush=True,
        )
    children = {}
    restart_counts = {game.game_pk: 0 for game in games}
    maximum_restarts = int(os.getenv("PAPER_WORKER_MAX_RESTARTS", "10"))
    restart_delay = float(os.getenv("PAPER_WORKER_RESTART_DELAY", "2"))

    def start_worker(game: DiscoveredGame, restarted: bool = False):
        path = LOG_DIR / f"settlement_value_console_{game.market_ticker}.log"
        handle = path.open("a")
        env = os.environ.copy()
        env.update({
            "MLB_GAME_PK": str(game.game_pk),
            "KALSHI_MARKET_TICKER": game.market_ticker,
            "KALSHI_AWAY_MARKET_TICKER": game.away_market_ticker,
            "PAPER_PORTFOLIO_DB": str(database),
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        })
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve())],
            cwd=REPOSITORY_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        thread = threading.Thread(
            target=_relay, args=(process.stdout, handle,
                f"{game.away_code}@{game.home_code} {game.market_ticker}"),
            daemon=True,
        )
        thread.start()
        children[game.game_pk] = (game, process, handle, thread)
        action = "restarted" if restarted else "started"
        print(
            f"Trader {action}: {game.away_code}@{game.home_code} "
            f"game_pk={game.game_pk} home_ticker={game.market_ticker} "
            f"away_ticker={game.away_market_ticker} "
            f"game_log={path}",
            flush=True,
        )

    try:
        for game in games:
            start_worker(game)
        opening = portfolio.metrics()
        print(
            f"Running {len(children)} isolated paper traders with shared "
            f"cash=${opening.cash:.2f}."
        )
        return_code = 0
        while children:
            for game_pk, child in list(children.items()):
                game, process, handle, thread = child
                code = process.poll()
                if code is None:
                    continue
                thread.join(timeout=5)
                handle.close()
                del children[game_pk]
                if code == 0:
                    continue
                restart_counts[game_pk] += 1
                attempt = restart_counts[game_pk]
                print(
                    f"Game {game_pk} exited with status {code}; "
                    f"restart {attempt}/{maximum_restarts}",
                    flush=True,
                )
                if attempt <= maximum_restarts:
                    time.sleep(restart_delay)
                    start_worker(game, restarted=True)
                else:
                    return_code = code
            if children:
                time.sleep(1)
        final = portfolio.metrics()
        print(
            f"Shared portfolio: cash=${final.cash:.2f} "
            f"equity=${final.equity:.2f} PnL=${final.pnl:+.2f} "
            f"open_positions={final.open_positions}"
        )
        return return_code
    except KeyboardInterrupt:
        for _, process, _, _ in children.values():
            if process.poll() is None:
                process.terminate()
        return 130
    finally:
        for _, process, handle, thread in children.values():
            if process.poll() is None:
                process.terminate()
            thread.join(timeout=5)
            handle.close()


def current_slate_date() -> date:
    return datetime.now(SLATE_TIMEZONE).date()


def run_continuous_slates(start_date: date | None = None) -> int:
    """Run today's slate and automatically advance to the next matched one."""
    cursor = max(start_date or current_slate_date(), current_slate_date())
    poll_seconds = float(os.getenv("SLATE_DISCOVERY_POLL_SECONDS", "300"))
    lookahead_days = int(os.getenv("SLATE_LOOKAHEAD_DAYS", "7"))
    while True:
        selected = None
        for offset in range(lookahead_days + 1):
            candidate = cursor + timedelta(days=offset)
            try:
                games, _ = discover_daily_games(candidate)
            except Exception as error:
                print(
                    f"Slate discovery failed for {candidate}: {error}",
                    flush=True,
                )
                continue
            if games:
                selected = candidate
                break
        if selected is None:
            cursor = max(cursor, current_slate_date())
            print(
                f"No matched slate from {cursor} through "
                f"{cursor + timedelta(days=lookahead_days)}; "
                f"retrying in {poll_seconds:g}s",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        print(f"Selected next settlement-value slate: {selected}", flush=True)
        code = run_all_games(selected)
        if code:
            print(
                f"Slate {selected} exited with status {code}; retrying",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        cursor = selected + timedelta(days=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-games", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--portfolio-status", action="store_true")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    parser.add_argument("--continuous", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.discover_only:
        discovered, discovery_warnings = discover_daily_games(
            args.date or current_slate_date()
        )
        for warning in discovery_warnings:
            print(f"WARNING: {warning}")
        for game in discovered:
            print(
                f"{game.scheduled_time.isoformat()} {game.away_code}@{game.home_code} "
                f"game_pk={game.game_pk} home_ticker={game.market_ticker} "
                f"away_ticker={game.away_market_ticker}"
            )
    elif args.portfolio_status:
        path = Path(os.getenv(
            "PAPER_PORTFOLIO_DB", str(LOG_DIR / "settlement_value_portfolio.sqlite3")
        ))
        metrics = SharedPaperPortfolio(path).metrics()
        print(
            f"cash=${metrics.cash:.2f} equity=${metrics.equity:.2f} "
            f"pnl=${metrics.pnl:+.2f} open_positions={metrics.open_positions}"
        )
    elif args.continuous:
        raise SystemExit(run_continuous_slates(args.date))
    elif args.all_games:
        raise SystemExit(run_all_games(args.date or current_slate_date()))
    else:
        asyncio.run(run_worker())
