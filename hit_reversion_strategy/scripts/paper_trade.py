"""Paper trader for the event-conditioned hybrid residual strategy."""

from __future__ import annotations

import asyncio
import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from trade_tape_strategy.hybrid import anchored_event_target  # noqa: E402
from trade_tape_strategy.core import (  # noqa: E402
    TradeTapeConfig, position_contracts, segmented_trade_signal,
    segment_value,
)
from trade_tape_strategy.strategy import (  # noqa: E402
    CONFIG,
    estimated_round_trip_fee_per_contract,
    taker_fee,
)
from shared_kalshi_feed import get_market as get_shared_market  # noqa: E402
from shared_mlb_feed import get_game as get_shared_game  # noqa: E402
from settlement_value_strategy.live_execution import (  # noqa: E402
    LiveExecutor, REAL_MONEY_ACK,
)


GAME_PK_TEXT = os.getenv("MLB_GAME_PK")
MARKET_TICKER = os.getenv("KALSHI_MARKET_TICKER")
AWAY_MARKET_TICKER = os.getenv("KALSHI_AWAY_MARKET_TICKER")
GAME_PK = int(GAME_PK_TEXT) if GAME_PK_TEXT else None
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))
LOG_DIR = Path(os.getenv(
    "PAPER_LOG_DIR",
    str(Path(__file__).resolve().parent / "logs"),
))
MODEL_DIR = PROJECT_ROOT / "models"
HYBRID_CONFIG_PATH = MODEL_DIR / "trade_tape_config.json"
STATE_MODEL_PATH = PROJECT_ROOT / "models/local_win_expectancy.cbm"
MLB_PRIOR_PATH = (
    PROJECT_ROOT / "models/mlb_pregame_prior.json"
)
KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
MLB_API = "https://statsapi.mlb.com/api"
SLATE_TIMEZONE = ZoneInfo(os.getenv("SLATE_TIMEZONE", "America/Chicago"))
KALSHI_EVENT_TIMEZONE = ZoneInfo("America/New_York")
MAX_EVENT_TIME_DELTA = timedelta(minutes=90)
LIVE_MODE = os.getenv("LIVE_TRADING_ENABLED") == REAL_MONEY_ACK
LIVE_ORDER_BUDGET = 2.0

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
    market_ticker: str = ""
    entry_client_order_id: str = ""


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


@dataclass(frozen=True)
class PortfolioMetrics:
    cash: float
    equity: float
    pnl: float
    open_positions: int


class SharedPaperPortfolio:
    """SQLite-backed cash and positions shared by every game process."""

    def __init__(self, path: Path, starting_cash: float = 1000.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    starting_cash REAL NOT NULL,
                    cash REAL NOT NULL
                )"""
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
                    anchor_target REAL NOT NULL,
                    anchor_fair REAL NOT NULL,
                    event_id INTEGER NOT NULL,
                    mark_price REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    entry_client_order_id TEXT NOT NULL DEFAULT '',
                    UNIQUE(game_pk, event_id)
                )"""
            )
            if columns and any(
                row[1] == "game_pk" and row[5] == 1 for row in columns
            ):
                connection.execute(
                    """INSERT OR IGNORE INTO positions (
                        game_pk,market_ticker,side,contracts,entry_price,
                        entry_fee,entry_time,anchor_target,anchor_fair,event_id,
                        mark_price,updated_at
                    ) SELECT game_pk,market_ticker,side,contracts,entry_price,
                        entry_fee,entry_time,anchor_target,anchor_fair,event_id,
                        mark_price,updated_at FROM positions_single_position"""
                )
                connection.execute("DROP TABLE positions_single_position")
            existing_columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(positions)"
                ).fetchall()
            }
            if "entry_client_order_id" not in existing_columns:
                connection.execute(
                    "ALTER TABLE positions ADD COLUMN "
                    "entry_client_order_id TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "INSERT OR IGNORE INTO portfolio(id, starting_cash, cash) "
                "VALUES (1, ?, ?)",
                (starting_cash, starting_cash),
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS entry_history (
                    game_pk INTEGER NOT NULL,
                    event_id INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    PRIMARY KEY (game_pk, event_id)
                )"""
            )
            connection.execute(
                "INSERT OR IGNORE INTO entry_history(game_pk,event_id,entry_time) "
                "SELECT game_pk,event_id,entry_time FROM positions"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def open_position(
        self, game_pk: int, ticker: str, position: Position,
        minimum_seconds_between_entries: float = 0.0,
    ) -> bool:
        cost = position.contracts * position.entry_price + position.entry_fee
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cash = float(connection.execute(
                "SELECT cash FROM portfolio WHERE id = 1"
            ).fetchone()[0])
            duplicate = connection.execute(
                "SELECT 1 FROM entry_history WHERE game_pk=? AND event_id=?",
                (game_pk, position.event_id),
            ).fetchone()
            latest = connection.execute(
                "SELECT entry_time FROM entry_history WHERE game_pk=? "
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
                "UPDATE portfolio SET cash = cash - ? WHERE id = 1", (cost,)
            )
            connection.execute(
                """INSERT INTO positions (
                    game_pk,market_ticker,side,contracts,entry_price,entry_fee,
                    entry_time,anchor_target,anchor_fair,event_id,mark_price,
                    updated_at,entry_client_order_id
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    game_pk, ticker, position.side, position.contracts,
                    position.entry_price, position.entry_fee,
                    position.entry_time.isoformat(), position.anchor_target,
                    position.anchor_fair, position.event_id,
                    position.entry_price, now, position.entry_client_order_id,
                ),
            )
            connection.execute(
                "INSERT INTO entry_history(game_pk,event_id,entry_time) "
                "VALUES (?,?,?)",
                (game_pk, position.event_id, position.entry_time.isoformat()),
            )
        return True

    def update_marks(self, game_pk: int, yes_bid: float, yes_ask: float) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE positions SET mark_price=CASE
                    WHEN side='yes' THEN ? ELSE 1.0-? END,
                    updated_at=? WHERE game_pk=?""",
                (
                    yes_bid, yes_ask, datetime.now(timezone.utc).isoformat(),
                    game_pk,
                ),
            )

    def close_position(
        self, game_pk: int, event_id: int, proceeds: float,
    ) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM positions WHERE game_pk=? AND event_id=?",
                (game_pk, event_id),
            ).fetchone()
            if not exists:
                connection.rollback()
                return False
            connection.execute(
                "UPDATE portfolio SET cash = cash + ? WHERE id = 1",
                (proceeds,),
            )
            connection.execute(
                "DELETE FROM positions WHERE game_pk=? AND event_id=?",
                (game_pk, event_id),
            )
        return True

    def load_positions(self, game_pk: int) -> list[Position]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT side, contracts, entry_price, entry_fee, entry_time,
                          anchor_target, anchor_fair, event_id, market_ticker,
                          entry_client_order_id
                   FROM positions WHERE game_pk = ? ORDER BY entry_time""",
                (game_pk,),
            ).fetchall()
        return [
            Position(
                side=str(row[0]), contracts=float(row[1]),
                entry_price=float(row[2]), entry_fee=float(row[3]),
                entry_time=pd.to_datetime(row[4], utc=True).to_pydatetime(),
                anchor_target=float(row[5]), anchor_fair=float(row[6]),
                event_id=int(row[7]), market_ticker=str(row[8]),
                entry_client_order_id=str(row[9]),
            )
            for row in rows
        ]

    def metrics(self) -> PortfolioMetrics:
        with self._connect() as connection:
            starting_cash, cash = connection.execute(
                "SELECT starting_cash, cash FROM portfolio WHERE id = 1"
            ).fetchone()
            rows = connection.execute(
                "SELECT contracts, mark_price FROM positions"
            ).fetchall()
        liquidation = sum(
            float(contracts) * float(price)
            - taker_fee(float(contracts), float(price))
            for contracts, price in rows
        )
        equity = float(cash) + liquidation
        return PortfolioMetrics(
            cash=float(cash), equity=equity,
            pnl=equity - float(starting_cash), open_positions=len(rows),
        )


@dataclass(frozen=True)
class DiscoveredGame:
    game_pk: int
    scheduled_time: datetime
    away_code: str
    home_code: str
    market_ticker: str
    away_market_ticker: str


MAIN_LOG_ACTIONS = ("TRADER READY", "TRADE ", "Shared portfolio:")


def should_surface_worker_line(line: str) -> bool:
    """Keep quotes/model diagnostics per-game; surface executions globally."""
    return any(marker in line for marker in MAIN_LOG_ACTIONS)


def relay_worker_output(
    stream, handle, game_label: str, ticker: str,
) -> None:
    """Tee a worker stream to its file and selected lines to the main log."""
    try:
        for line in stream:
            handle.write(line)
            handle.flush()
            if should_surface_worker_line(line):
                print(f"[{game_label} {ticker}] {line.rstrip()}", flush=True)
    finally:
        stream.close()


def canonical_kalshi_code(value: str) -> str:
    code = str(value).strip().upper()
    return KALSHI_TEAM_CODES.get(code, code)


def market_team_code(market: dict) -> str | None:
    ticker = str(market.get("ticker") or "")
    if "-" not in ticker:
        return None
    return canonical_kalshi_code(ticker.rsplit("-", 1)[-1])


def event_scheduled_time(event: dict) -> datetime | None:
    """Parse the Eastern start time embedded in an MLB event ticker."""
    match = re.match(
        r"^KXMLBGAME-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})",
        str(event.get("event_ticker") or ""),
    )
    if not match:
        return None
    year, month_name, day, hour, minute = match.groups()
    months = {
        name: number for number, name in enumerate(
            ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
             "SEP", "OCT", "NOV", "DEC"), 1
        )
    }
    try:
        local_time = datetime(
            2000 + int(year), months[month_name], int(day),
            int(hour), int(minute), tzinfo=KALSHI_EVENT_TIMEZONE,
        )
    except (KeyError, ValueError):
        return None
    return local_time.astimezone(timezone.utc)


def clock_time_delta(scheduled: datetime, event_time: datetime) -> timedelta:
    """Compare Eastern clock times while ignoring a postponed ticker's date."""
    scheduled_local = scheduled.astimezone(KALSHI_EVENT_TIMEZONE)
    event_local = event_time.astimezone(KALSHI_EVENT_TIMEZONE)
    scheduled_minutes = scheduled_local.hour * 60 + scheduled_local.minute
    event_minutes = event_local.hour * 60 + event_local.minute
    return timedelta(minutes=abs(scheduled_minutes - event_minutes))


def match_games_to_home_markets(
    games: list[dict],
    events: list[dict],
) -> tuple[list[DiscoveredGame], list[str]]:
    """Match same-day games and Kalshi events, including doubleheaders."""
    event_groups: dict[
        frozenset[str], list[tuple[str, dict[str, dict], datetime | None]]
    ] = {}
    for event in events:
        markets = [
            market for market in (event.get("markets") or [])
            if str(market.get("status") or "").lower()
            not in {"closed", "settled", "finalized"}
        ]
        by_team = {
            code: market
            for market in markets
            if (code := market_team_code(market)) is not None
        }
        if len(by_team) != 2:
            continue
        matchup = frozenset(by_team)
        event_groups.setdefault(matchup, []).append((
            str(event.get("event_ticker") or ""), by_team,
            event_scheduled_time(event),
        ))
    for group in event_groups.values():
        group.sort(key=lambda item: item[0])

    game_groups: dict[frozenset[str], list[tuple[datetime, dict]]] = {}
    warnings: list[str] = []
    for game in games:
        teams = game.get("teams") or {}
        away_id = teams.get("away", {}).get("team", {}).get("id")
        home_id = teams.get("home", {}).get("team", {}).get("id")
        away = MLB_TEAM_CODES.get(int(away_id)) if away_id is not None else None
        home = MLB_TEAM_CODES.get(int(home_id)) if home_id is not None else None
        if away is None or home is None:
            warnings.append(f"Game {game.get('gamePk')} has unknown MLB team IDs")
            continue
        scheduled = pd.to_datetime(game.get("gameDate"), utc=True).to_pydatetime()
        game_groups.setdefault(frozenset({away, home}), []).append((
            scheduled, {"row": game, "away": away, "home": home},
        ))
    for group in game_groups.values():
        group.sort(key=lambda item: item[0])

    matched: list[DiscoveredGame] = []
    for matchup, scheduled_games in game_groups.items():
        market_events = event_groups.get(matchup, [])
        if market_events and all(value[2] is not None for value in market_events):
            candidates = sorted(
                (
                    clock_time_delta(scheduled, event_time),
                    game_index, market_index
                )
                for game_index, (scheduled, _) in enumerate(scheduled_games)
                for market_index, (_, _, event_time) in enumerate(market_events)
                if (
                    event_time.astimezone(KALSHI_EVENT_TIMEZONE).date()
                    == scheduled.astimezone(KALSHI_EVENT_TIMEZONE).date()
                    and clock_time_delta(scheduled, event_time)
                    <= MAX_EVENT_TIME_DELTA
                )
            )
            used_games: set[int] = set()
            used_markets: set[int] = set()
            pair_indexes = []
            for delta, game_index, market_index in candidates:
                if game_index in used_games or market_index in used_markets:
                    continue
                used_games.add(game_index)
                used_markets.add(market_index)
                pair_indexes.append((game_index, market_index, delta))
            remaining_games = [
                index for index in range(len(scheduled_games))
                if index not in used_games
            ]
            remaining_markets = [
                index for index in range(len(market_events))
                if index not in used_markets
            ]
            if remaining_games and len(remaining_games) == len(remaining_markets):
                for game_index, market_index in zip(
                    remaining_games, remaining_markets
                ):
                    pair_indexes.append((
                        game_index, market_index,
                        clock_time_delta(
                            scheduled_games[game_index][0],
                            market_events[market_index][2],
                        ),
                    ))
            pairs = [
                (scheduled_games[game_index], market_events[market_index])
                for game_index, market_index, _ in sorted(pair_indexes)
            ]
        elif len(market_events) == len(scheduled_games):
            pairs = list(zip(scheduled_games, market_events))
        else:
            pairs = []
        if len(pairs) != len(scheduled_games) or len(pairs) != len(market_events):
            warnings.append(
                f"{sorted(matchup)}: time-matched {len(pairs)} of "
                f"{len(scheduled_games)} MLB games to "
                f"{len(market_events)} Kalshi events"
            )
        for (scheduled, info), (_, markets, _) in pairs:
            if info["row"].get("status", {}).get(
                "abstractGameState"
            ) == "Final":
                continue
            home_market = markets.get(info["home"])
            if home_market is None:
                warnings.append(
                    f"Game {info['row'].get('gamePk')} has no home-team market"
                )
                continue
            matched.append(DiscoveredGame(
                game_pk=int(info["row"]["gamePk"]),
                scheduled_time=scheduled,
                away_code=info["away"],
                home_code=info["home"],
                market_ticker=str(home_market["ticker"]),
                away_market_ticker=str(markets[info["away"]]["ticker"]),
            ))
    return sorted(matched, key=lambda game: game.scheduled_time), warnings


def discover_daily_games(game_date: date) -> tuple[list[DiscoveredGame], list[str]]:
    schedule = requests.get(
        f"{MLB_API}/v1/schedule",
        params={"sportId": 1, "date": game_date.isoformat()},
        timeout=30,
    )
    schedule.raise_for_status()
    games = [
        game
        for day in schedule.json().get("dates") or []
        for game in day.get("games") or []
        if str(game.get("status", {}).get("detailedState") or "").lower()
        not in {"postponed", "cancelled", "canceled"}
    ]

    from download_market_data import (
        discover_events,
        fetch_event_markets,
    )

    # Active postponed markets retain their original ticker date. Search the
    # two-day postponement window, then let team and start-time matching select
    # the correct games on today's MLB slate.
    events = discover_events(
        game_date - timedelta(days=2), game_date, verbose=False,
    )
    hydrated = []
    for event in events:
        hydrated.append({
            **event,
            "markets": fetch_event_markets(event, verbose=False),
        })
    return match_games_to_home_markets(games, hydrated)


def run_daily_coordinator(game_date: date) -> int:
    if os.getenv("ALLOW_UNVALIDATED_HYBRID") != "1":
        raise RuntimeError(
            "Set ALLOW_UNVALIDATED_HYBRID=1 to run multi-game paper mode."
        )
    games, warnings = discover_daily_games(game_date)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if not games:
        print(f"No active matched MLB/Kalshi games for {game_date}")
        return 0
    print(f"Games for {game_date} ({len(games)}):", flush=True)
    for game in games:
        print(
            f"  {game.scheduled_time.isoformat()} "
            f"{game.away_code}@{game.home_code} game_pk={game.game_pk} "
            f"ticker={game.market_ticker}",
            flush=True,
        )

    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = Path(os.getenv(
        "PAPER_PORTFOLIO_DB",
        str(log_dir / f"hit_reversion_portfolio_{game_date.isoformat()}.sqlite3"),
    ))
    portfolio = SharedPaperPortfolio(
        portfolio_path,
        float(os.getenv("PAPER_STARTING_CASH", "1000")),
    )
    children: list[
        tuple[DiscoveredGame, subprocess.Popen, object, threading.Thread]
    ] = []
    try:
        for game in games:
            console_path = log_dir / f"hit_reversion_console_{game.market_ticker}.log"
            handle = console_path.open("a")
            env = os.environ.copy()
            env["MLB_GAME_PK"] = str(game.game_pk)
            env["KALSHI_MARKET_TICKER"] = game.market_ticker
            env["KALSHI_AWAY_MARKET_TICKER"] = game.away_market_ticker
            env["PAPER_PORTFOLIO_DB"] = str(portfolio_path)
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve())],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("Failed to capture paper-trader output")
            relay = threading.Thread(
                target=relay_worker_output,
                args=(
                    process.stdout, handle,
                    f"{game.away_code}@{game.home_code}", game.market_ticker,
                ),
                daemon=True,
            )
            relay.start()
            children.append((game, process, handle, relay))
            print(
                f"Trader started: {game.away_code}@{game.home_code} "
                f"game_pk={game.game_pk} ticker={game.market_ticker} "
                f"game_log={console_path}",
                flush=True,
            )
        opening = portfolio.metrics()
        print(
            f"Running {len(children)} isolated "
            f"{'LIVE' if LIVE_MODE else 'paper'} traders with shared "
            f"cash=${opening.cash:.2f}."
        )
        return_code = 0
        for game, process, _, relay in children:
            code = process.wait()
            relay.join(timeout=5)
            if code:
                return_code = code
                print(f"Game {game.game_pk} exited with status {code}")
        final = portfolio.metrics()
        print(
            f"Shared portfolio: cash=${final.cash:.2f} "
            f"equity=${final.equity:.2f} PnL=${final.pnl:+.2f} "
            f"open_positions={final.open_positions}"
        )
        return return_code
    except KeyboardInterrupt:
        print("Stopping all paper traders...")
        for _, process, _, _ in children:
            if process.poll() is None:
                process.terminate()
        for _, process, _, relay in children:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            relay.join(timeout=5)
        return 130
    finally:
        for _, _, handle, _ in children:
            handle.close()


def current_slate_date() -> date:
    return datetime.now(SLATE_TIMEZONE).date()


def run_continuous_coordinator(start_date: date | None = None) -> int:
    """Run today's slate, then wait for and advance to each next slate."""
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
        print(f"Selected next trade-tape slate: {selected}", flush=True)
        code = run_daily_coordinator(selected)
        if code:
            print(
                f"Slate {selected} exited with status {code}; retrying",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        cursor = selected + timedelta(days=1)


def material_state(state: dict) -> tuple:
    return (
        int(state["score_diff"]),
        int(state["outs_when_up"]),
        int(state["runner_on_first"]),
        int(state["runner_on_second"]),
        int(state["runner_on_third"]),
    )


def is_next_completed_event(
    previous_event_id: int | None,
    current_event_id: int | None,
) -> bool:
    if current_event_id is None:
        return False
    if previous_event_id is None:
        return current_event_id == 0
    return current_event_id == previous_event_id + 1


def fetch_market_snapshot(ticker: str | None = None) -> MarketSnapshot:
    ticker = str(ticker or MARKET_TICKER)
    if os.getenv("KALSHI_FEED_URL"):
        payload = get_shared_market(ticker)
        snapshot = payload["snapshot"]
        return MarketSnapshot(
            pd.to_datetime(snapshot["received_at"], utc=True).to_pydatetime(),
            float(snapshot["bid"]), float(snapshot["ask"]),
        )
    response = requests.get(
        f"{KALSHI_API}/markets/{ticker}/orderbook", timeout=5
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
                str(event.get("startTime") or event["endTime"]),
            )
            latest = token
    return latest


def pitch_token_time(token: tuple | None) -> datetime | None:
    if token is None:
        return None
    return pd.to_datetime(token[2], utc=True).to_pydatetime()


def pitch_token_start_time(token: tuple | None) -> datetime | None:
    if token is None:
        return None
    value = token[3] if len(token) > 3 else token[2]
    return pd.to_datetime(value, utc=True).to_pydatetime()


def fetch_recent_trades() -> pd.DataFrame:
    if os.getenv("KALSHI_FEED_URL"):
        rows = get_shared_market(str(MARKET_TICKER)).get("trades") or []
    else:
        response = requests.get(
            f"{KALSHI_API}/markets/trades",
            params={"ticker": MARKET_TICKER, "limit": 1000}, timeout=5,
        )
        response.raise_for_status()
        rows = response.json().get("trades") or []
    frame = pd.DataFrame(rows)
    required = {
        "trade_id", "created_time", "yes_price_dollars", "count_fp",
        "taker_outcome_side",
    }
    if frame.empty:
        return pd.DataFrame(columns=sorted(required))
    if missing := required - set(frame):
        raise RuntimeError(f"Trade response missing {sorted(missing)}")
    frame["created_time"] = pd.to_datetime(frame.created_time, utc=True)
    frame["yes_price_dollars"] = pd.to_numeric(
        frame.yes_price_dollars, errors="raise"
    )
    frame["count_fp"] = pd.to_numeric(frame.count_fp, errors="raise")
    return frame.sort_values(["created_time", "trade_id"]).drop_duplicates(
        "trade_id"
    )


def pre_pitch_trade_anchor(
    trades: pd.DataFrame, pitch_start: datetime, maximum_age_seconds: float,
) -> float | None:
    if trades.empty:
        return None
    times = pd.to_datetime(trades.created_time, utc=True)
    eligible = trades[times.lt(pd.Timestamp(pitch_start))]
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    age = (
        pd.Timestamp(pitch_start) - pd.Timestamp(row.created_time)
    ).total_seconds()
    if age > maximum_age_seconds:
        return None
    return float(row.yes_price_dollars)


def replay_candidate_entry(
    trades: pd.DataFrame, candidate: EventCandidate, current_fair: float,
    positions: list[Position], config: TradeTapeConfig,
) -> dict | None:
    """Replay the backtest's confirmation and strictly later entry fill."""
    if trades.empty:
        return None
    target = float(anchored_event_target(
        candidate.target, candidate.post_fair, current_fair
    ))
    deadline = candidate.event_time + timedelta(
        seconds=config.maximum_event_to_entry_seconds
    )
    observable_start = max(candidate.event_time, candidate.observed_at)
    if observable_start >= deadline:
        return None
    last_entry = max(
        (position.entry_time for position in positions), default=None
    )
    watch_started = None
    pending_created = None
    confirmation_price = None
    tape = trades.sort_values(["created_time", "trade_id"])
    for trade in tape.itertuples(index=False):
        when = pd.Timestamp(trade.created_time).to_pydatetime()
        if when <= observable_start or when > deadline:
            continue
        yes_price = float(trade.yes_price_dollars)
        side, _ = segmented_trade_signal(
            target, yes_price, candidate.event_type, config
        )
        if pending_created is not None:
            if side != candidate.side:
                return None
            if (
                when > pending_created
                and (
                    not config.require_compatible_taker
                    or str(trade.taker_outcome_side) == candidate.side
                )
                and (
                    last_entry is None
                    or (when - last_entry).total_seconds()
                    >= config.minimum_seconds_between_entries
                )
            ):
                reversion_move = (
                    yes_price - confirmation_price
                    if candidate.side == "yes"
                    else confirmation_price - yes_price
                )
                minimum_reversion_move = segment_value(
                    config.minimum_reversion_moves_by_segment,
                    candidate.event_type,
                    candidate.side,
                    config.minimum_reversion_move,
                )
                if reversion_move < minimum_reversion_move:
                    continue
                price = yes_price if side == "yes" else 1.0 - yes_price
                contracts = position_contracts(price, config)
                if float(trade.count_fp) >= contracts:
                    return {
                        "time": when, "side": side, "price": price,
                        "contracts": contracts,
                        "fee": taker_fee(contracts, price),
                    }
            continue
        if side != candidate.side:
            watch_started = None
        elif watch_started is None:
            watch_started = when
        elif (when - watch_started).total_seconds() >= segment_value(
            config.confirmation_seconds_by_segment,
            candidate.event_type,
            candidate.side,
            config.confirmation_seconds,
        ):
            pending_created = when
            confirmation_price = yes_price
    return None


def replay_position_exit(
    trades: pd.DataFrame, position: Position, current_fair: float,
    scanned_after: datetime | None = None,
    pending_time: datetime | None = None,
    config: TradeTapeConfig | None = None,
) -> tuple[dict | None, datetime | None, datetime]:
    """Find the backtest-style trade after target reversion that can exit."""
    config = config or TradeTapeConfig()
    if trades.empty:
        return None, pending_time, scanned_after or position.entry_time
    target = (
        float(position.anchor_target)
        if config.exit_target_mode == "frozen"
        else float(anchored_event_target(
            position.anchor_target, position.anchor_fair, current_fair
        ))
    )
    last_seen = scanned_after or position.entry_time
    exit_taker_side = "no" if position.side == "yes" else "yes"
    for trade in trades.sort_values(["created_time", "trade_id"]).itertuples(
        index=False
    ):
        when = pd.Timestamp(trade.created_time).to_pydatetime()
        if when <= (scanned_after or position.entry_time):
            continue
        last_seen = max(last_seen, when)
        yes_price = float(trade.yes_price_dollars)
        reverted = (
            position.side == "yes" and yes_price >= target
        ) or (
            position.side == "no" and yes_price <= target
        )
        timed_out = bool(
            config.maximum_hold_seconds > 0
            and (when - position.entry_time).total_seconds()
            >= config.maximum_hold_seconds
        )
        if pending_time is None:
            if reverted or timed_out:
                pending_time = when
            continue
        if not reverted and not timed_out and not config.latch_reversion_exit:
            pending_time = None
            continue
        if (
            when > pending_time
            and (
                not config.require_compatible_taker
                or str(trade.taker_outcome_side) == exit_taker_side
            )
            and (reverted or timed_out or config.latch_reversion_exit)
        ):
            price = yes_price if position.side == "yes" else 1.0 - yes_price
            if float(trade.count_fp) >= position.contracts:
                fee = taker_fee(position.contracts, price)
                return {
                    "time": when, "price": price, "fee": fee,
                    "reason": "TIMEOUT" if timed_out else "TARGET_REVERSION",
                }, None, last_seen
    return None, pending_time, last_seen


def fetch_mlb_payload(
    game_pk: int, timeout: float = 5.0,
) -> tuple[dict, datetime]:
    if os.getenv("MLB_FEED_URL"):
        wrapper = get_shared_game(game_pk, timeout=timeout)
        return (
            wrapper["payload"],
            pd.to_datetime(wrapper["received_at"], utc=True).to_pydatetime(),
        )
    response = requests.get(
        f"{MLB_API}/v1.1/game/{game_pk}/feed/live", timeout=timeout
    )
    response.raise_for_status()
    return response.json(), datetime.now(timezone.utc)


def fetch_game_snapshot() -> GameSnapshot:
    payload, received_at = fetch_mlb_payload(int(GAME_PK))
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
        received_at, status, state, home_score, away_score,
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


def state_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the batting-perspective feature contract used in research."""
    batting_home = frame.inning_topbot.astype(int)
    result = pd.DataFrame(index=frame.index)
    result["pregame_batting_prob"] = np.where(
        batting_home.eq(1), frame.pregame_prob, 1.0 - frame.pregame_prob
    )
    result["inning"] = frame.inning
    result["batting_team_is_home"] = batting_home
    result["outs_when_up"] = frame.outs_when_up
    result["batting_score_diff"] = np.where(
        batting_home.eq(1), frame.score_diff, -frame.score_diff
    )
    for name in (
        "balls", "strikes", "runner_on_first", "runner_on_second",
        "runner_on_third",
    ):
        result[name] = frame[name]
    return result.astype(float)


def home_fair_probability(
    model: CatBoostClassifier, pregame_prob: float, state: dict,
) -> float:
    frame = pd.DataFrame([{**state, "pregame_prob": pregame_prob}])
    batting = float(model.predict_proba(
        state_model_frame(frame), thread_count=1,
    )[0, 1])
    return batting if int(state["inning_topbot"]) == 1 else 1.0 - batting


def pregame_probability_from_rating_state(
    rating_state: dict, home_code: str, away_code: str,
) -> float:
    aliases = {
        "ARI": "AZ", "CHW": "CWS", "OAK": "ATH", "KCR": "KC",
        "SDP": "SD", "SFG": "SF", "TBR": "TB", "WAS": "WSH",
    }

    def rating(code: str) -> float:
        code = str(code).upper()
        key = code if code in rating_state["ratings"] else aliases.get(code, code)
        return float(rating_state["ratings"].get(
            key, rating_state["initial_rating"]
        ))

    difference = (
        rating(home_code) + float(rating_state["home_advantage"])
        - rating(away_code)
    )
    return 1.0 / (1.0 + 10.0 ** (-difference / 400.0))


def fetch_model_pregame_prior() -> float:
    payload, _ = fetch_mlb_payload(int(GAME_PK))
    teams = payload.get("gameData", {}).get("teams", {})
    home_id = int(teams.get("home", {}).get("id"))
    away_id = int(teams.get("away", {}).get("id"))
    rating_state = json.loads(MLB_PRIOR_PATH.read_text())
    return pregame_probability_from_rating_state(
        rating_state, MLB_TEAM_CODES[home_id], MLB_TEAM_CODES[away_id]
    )


async def wait_for_model_pregame_prior() -> float:
    delay = 1.0
    while True:
        try:
            return await asyncio.to_thread(fetch_model_pregame_prior)
        except requests.RequestException as error:
            print(
                f"MLB model prior pending: {error}; retrying in {delay:g}s",
                flush=True,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


def fetch_pregame_anchor() -> float:
    feed, _ = fetch_mlb_payload(int(GAME_PK))
    first_pitch = first_pitch_time(feed)
    if first_pitch is None:
        status = feed.get("gameData", {}).get("status", {}).get(
            "abstractGameState", "Preview"
        )
        if status == "Live":
            raise RuntimeError(
                "Game is live but first-pitch time is unavailable; "
                "refusing to substitute an in-game price for the pregame anchor"
            )
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


async def wait_for_pregame_anchor() -> float:
    delay = 1.0
    while True:
        try:
            return await asyncio.to_thread(fetch_pregame_anchor)
        except requests.RequestException as error:
            print(
                f"Pregame anchor request failed: {error}; retrying in "
                f"{delay:g}s",
                flush=True,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


async def main() -> None:
    if GAME_PK is None or not MARKET_TICKER:
        raise RuntimeError(
            "Set MLB_GAME_PK and KALSHI_MARKET_TICKER explicitly; "
            "there are no safe default games or markets."
        )
    state_model = CatBoostClassifier()
    state_model.load_model(STATE_MODEL_PATH)
    hybrid_config = TradeTapeConfig(**json.loads(HYBRID_CONFIG_PATH.read_text()))
    allow_unvalidated = os.getenv("ALLOW_UNVALIDATED_HYBRID") == "1"
    if not hybrid_config.enabled and not allow_unvalidated:
        raise RuntimeError(
            "Hybrid policy has not passed a fresh forward test and is disabled. "
            "Set ALLOW_UNVALIDATED_HYBRID=1 only to collect paper observations."
        )
    portfolio_path = Path(os.getenv(
        "PAPER_PORTFOLIO_DB",
        str(LOG_DIR / (
            f"hit_reversion_portfolio_{datetime.now().astimezone().date()}_"
            f"game_{GAME_PK}.sqlite3"
        )),
    ))
    portfolio = SharedPaperPortfolio(
        portfolio_path,
        float(os.getenv("PAPER_STARTING_CASH", "1000")),
    )
    live_executor = (
        LiveExecutor(Path(os.environ["LIVE_RISK_DB"])) if LIVE_MODE else None
    )
    pregame_prob = await wait_for_pregame_anchor()
    print(f"Causal pre-first-pitch market anchor: {pregame_prob:.1%}")
    print(
        f"Hybrid threshold={hybrid_config.minimum_edge:.1%}, "
        f"confirmation={hybrid_config.confirmation_seconds:g} seconds, "
        f"entry expiry={hybrid_config.maximum_event_to_entry_seconds:g} seconds, "
        f"taker direction={'required' if hybrid_config.require_compatible_taker else 'either'}, "
        "target-reversion exit"
    )

    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"hit_reversion_decisions_{MARKET_TICKER}.csv"
    new_log = not log_path.exists() or log_path.stat().st_size == 0
    if new_log:
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                "decision_time", "market_received_at", "state_received_at",
                "bid", "ask", "inning", "outs", "score_diff", "fair_prob",
                "completed_event_id", "completed_event", "target", "excess_move",
                "edge", "continuation_value", "exit_advantage", "action",
                "portfolio_cash", "portfolio_equity", "portfolio_pnl",
                "portfolio_open_positions",
            ])

    positions = portfolio.load_positions(int(GAME_PK))
    if positions:
        print(
            f"Recovered {len(positions)} open position(s) for game "
            f"{GAME_PK} from {portfolio_path}"
        )
    candidate: EventCandidate | None = None
    previous_market: float | None = None
    previous_fair: float | None = None
    previous_event_id: int | None = None
    exit_pending_trade: dict[int, datetime] = {}
    exit_scanned_through: dict[int, datetime] = {
        position.event_id: position.entry_time for position in positions
    }
    while True:
        try:
            # The order book may disappear immediately after the game.  Read
            # the authoritative MLB state first so a Kalshi 404 cannot hide a
            # "Final" response and leave this worker polling forever.
            game = await asyncio.to_thread(fetch_game_snapshot)
        except Exception as error:
            print(f"Game snapshot rejected: {error}")
            await asyncio.sleep(POLL_SECONDS)
            continue
        now = datetime.now(timezone.utc)
        if game.status == "Final":
            for position in list(positions):
                won = (
                    position.side == "yes" and game.home_score > game.away_score
                ) or (
                    position.side == "no" and game.home_score < game.away_score
                )
                payout = position.contracts if won else 0.0
                portfolio.close_position(
                    int(GAME_PK), position.event_id, payout
                )
                if live_executor is not None and position.entry_client_order_id:
                    live_executor.ledger.close_entry(
                        position.entry_client_order_id
                    )
                print(
                    f"TRADE SETTLE {position.side.upper()} "
                    f"result={'WIN' if won else 'LOSS'} "
                    f"contracts={position.contracts:.4f} payout={payout:.4f} "
                    f"reason=GAME_FINAL game_pk={GAME_PK} "
                    f"ticker={MARKET_TICKER}",
                    flush=True,
                )
            positions.clear()
            final = portfolio.metrics()
            print(
                f"Shared portfolio: cash=${final.cash:.2f} "
                f"equity=${final.equity:.2f} PnL=${final.pnl:+.2f}"
            )
            break
        if game.status != "Live":
            await asyncio.sleep(POLL_SECONDS)
            continue
        try:
            market, recent_trades = await asyncio.gather(
                asyncio.to_thread(fetch_market_snapshot),
                asyncio.to_thread(fetch_recent_trades),
            )
        except Exception as error:
            print(f"Market snapshot rejected: {error}")
            await asyncio.sleep(POLL_SECONDS)
            continue
        if (now - market.received_at).total_seconds() > CONFIG.maximum_quote_age_seconds:
            await asyncio.sleep(POLL_SECONDS)
            continue
        if (now - game.received_at).total_seconds() > CONFIG.maximum_feed_age_seconds:
            await asyncio.sleep(POLL_SECONDS)
            continue

        fair_prob = home_fair_probability(
            state_model, pregame_prob, game.state
        )
        action = "HOLD"
        edge = float("nan")
        target = float("nan")
        continuation_value = float("nan")
        exit_advantage = float("nan")

        initializing_live_baseline = previous_market is None or previous_fair is None
        new_event = (
            not initializing_live_baseline
            and is_next_completed_event(
                previous_event_id, game.completed_event_id
            )
        )
        if initializing_live_baseline:
            action = "INITIALIZE_LIVE_BASELINE"
            print(
                f"TRADER READY game_pk={GAME_PK} ticker={MARKET_TICKER} "
                f"status=LIVE inning={game.state['inning']} "
                f"bid={market.bid:.4f} ask={market.ask:.4f}",
                flush=True,
            )
            print(
                f"Initialized {'midgame' if game.completed_event_id is not None else 'pregame'} "
                f"baseline at inning {game.state['inning']}; "
                "events already visible at startup will not be traded."
            )

        portfolio.update_marks(int(GAME_PK), market.bid, market.ask)
        closed_positions: list[Position] = []
        for position in positions:
            target = float(anchored_event_target(
                position.anchor_target, position.anchor_fair, fair_prob
            ))
            exit_fill, pending_exit, scanned_through = replay_position_exit(
                recent_trades, position, float(fair_prob),
                exit_scanned_through.get(position.event_id),
                exit_pending_trade.get(position.event_id),
                hybrid_config,
            )
            exit_scanned_through[position.event_id] = scanned_through
            if pending_exit is None:
                exit_pending_trade.pop(position.event_id, None)
            else:
                exit_pending_trade[position.event_id] = pending_exit
            if exit_fill is not None:
                price = exit_fill["price"]
                fee = exit_fill["fee"]
                closed_side = position.side
                closed_contracts = position.contracts
                if live_executor is not None:
                    actual_ticker = position.market_ticker or (
                        str(MARKET_TICKER) if position.side == "yes"
                        else str(AWAY_MARKET_TICKER)
                    )
                    executable = await asyncio.to_thread(
                        fetch_market_snapshot, actual_ticker
                    )
                    live_exit = await asyncio.to_thread(
                        live_executor.execute_exit,
                        trigger_key=(
                            f"hr-exit:{GAME_PK}:{position.event_id}:"
                            f"{exit_fill.get('reason')}"
                        ),
                        entry_client_order_id=position.entry_client_order_id,
                        ticker=actual_ticker,
                        contracts=position.contracts,
                        price=executable.bid,
                    )
                    if not live_exit.filled:
                        action = f"LIVE_EXIT_SKIP_{live_exit.reason.upper()}"
                        continue
                    price = live_exit.price
                    fee = live_exit.fee
                    closed_contracts = live_exit.contracts
                portfolio.close_position(
                    int(GAME_PK), position.event_id,
                    position.contracts * price - fee,
                )
                reason = str(exit_fill.get("reason") or "TARGET_REVERSION")
                action = f"CLOSE_{position.side.upper()}_{reason}"
                print(
                    f"TRADE SELL {closed_side.upper()} "
                    f"contracts={closed_contracts:.4f} price={price:.4f} "
                    f"fee={fee:.4f} reason={reason} game_pk={GAME_PK} "
                    f"ticker={MARKET_TICKER}",
                    flush=True,
                )
                closed_positions.append(position)
                exit_pending_trade.pop(position.event_id, None)
                exit_scanned_through.pop(position.event_id, None)
        if closed_positions:
            positions = [
                position for position in positions
                if position not in closed_positions
            ]

        if candidate is not None or new_event:
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
                and game.completed_event in hybrid_config.allowed_event_types
                and previous_market is not None
                and previous_fair is not None
                and pitch_token_time(game.latest_completed_pitch_token) is not None
            ):
                pitch_start = pitch_token_start_time(
                    game.latest_completed_pitch_token
                )
                pre_event_market = (
                    pre_pitch_trade_anchor(
                        recent_trades, pitch_start,
                        hybrid_config.maximum_pre_event_trade_age_seconds,
                    )
                    if pitch_start is not None else None
                )
                signed_fair_move = (
                    float(fair_prob) - previous_fair
                ) * (
                    1.0 if game.completed_event_batting_home else -1.0
                )
                if (
                    signed_fair_move >= hybrid_config.minimum_fair_move
                    and pre_event_market is not None
                ):
                    target = float(anchored_event_target(
                        pre_event_market, previous_fair, fair_prob
                    ))
                    event_type = str(game.completed_event)
                    side_candidates = []
                    for evaluated_side in ("yes", "no"):
                        if (
                            hybrid_config.side_filter != "both"
                            and evaluated_side != hybrid_config.side_filter
                        ):
                            continue
                        threshold = segment_value(
                            hybrid_config.minimum_edges_by_segment,
                            event_type,
                            evaluated_side,
                            hybrid_config.minimum_edge,
                        )
                        executable_price = (
                            float(market.ask) if evaluated_side == "yes"
                            else 1.0 - float(market.bid)
                        )
                        evaluated_edge = (
                            target - float(market.ask)
                            if evaluated_side == "yes"
                            else float(market.bid) - target
                        ) - estimated_round_trip_fee_per_contract(
                            executable_price
                        )
                        if evaluated_edge >= threshold:
                            side_candidates.append((
                                evaluated_edge - threshold,
                                evaluated_edge,
                                evaluated_side,
                            ))
                    if side_candidates:
                        _, edge, side = max(side_candidates)
                    else:
                        side, edge = None, 0.0
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
                            pre_market=pre_event_market,
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

            if candidate is not None:
                target = float(anchored_event_target(
                    candidate.target, candidate.post_fair, fair_prob
                ))
                if hybrid_config.require_post_signal_trade:
                    fill = replay_candidate_entry(
                        recent_trades, candidate, float(fair_prob), positions,
                        hybrid_config,
                    )
                else:
                    immediate_price = (
                        float(market.ask) if candidate.side == "yes"
                        else 1.0 - float(market.bid)
                    )
                    fill = {
                        "time": now, "side": candidate.side,
                        "price": immediate_price,
                        "contracts": position_contracts(immediate_price, hybrid_config),
                        "fee": taker_fee(
                            position_contracts(immediate_price, hybrid_config),
                            immediate_price,
                        ),
                    }
                if fill is not None:
                    price = fill["price"]
                    contracts = fill["contracts"]
                    fee = fill["fee"]
                    actual_ticker = (
                        str(MARKET_TICKER) if candidate.side == "yes"
                        else str(AWAY_MARKET_TICKER)
                    )
                    entry_client_order_id = ""
                    if live_executor is not None:
                        if not actual_ticker or actual_ticker == "None":
                            action = "LIVE_SKIP_MISSING_PAIRED_MARKET"
                            candidate = None
                            continue
                        executable = await asyncio.to_thread(
                            fetch_market_snapshot, actual_ticker
                        )
                        contract_target = (
                            target if candidate.side == "yes" else 1.0 - target
                        )
                        minimum_edge = segment_value(
                            hybrid_config.minimum_edges_by_segment,
                            candidate.event_type, candidate.side,
                            hybrid_config.minimum_edge,
                        )
                        actual_edge = (
                            contract_target - executable.ask
                            - estimated_round_trip_fee_per_contract(
                                executable.ask
                            )
                        )
                        if actual_edge < minimum_edge:
                            action = "LIVE_SKIP_ACTUAL_EDGE_CHECK"
                            candidate = None
                            continue
                        live_fill = await asyncio.to_thread(
                            live_executor.execute,
                            trigger_key=(
                                f"hr-entry:{GAME_PK}:{candidate.event_id}:"
                                f"{candidate.side}"
                            ),
                            game_pk=int(GAME_PK), ticker=actual_ticker,
                            price=executable.ask,
                            settlement_probability=contract_target,
                            original_bet_size=10.0,
                            original_minimum_expected_pnl=minimum_edge * 10.0,
                            minimum_seconds_between_entries=(
                                hybrid_config.minimum_seconds_between_entries
                            ),
                            minimum_probability_edge=minimum_edge,
                            strategy="hit_reversion",
                            signal_time=candidate.observed_at,
                            signal_price=float(executable.ask),
                            edge_at_submission=actual_edge,
                            order_budget=LIVE_ORDER_BUDGET,
                        )
                        if not live_fill.filled:
                            action = f"LIVE_SKIP_{live_fill.reason.upper()}"
                            candidate = None
                            continue
                        price = live_fill.price
                        contracts = live_fill.contracts
                        fee = live_fill.fee
                        entry_client_order_id = live_fill.client_order_id
                    proposed_position = Position(
                        side=candidate.side,
                        contracts=contracts,
                        entry_price=price,
                        entry_fee=fee,
                        entry_time=fill["time"],
                        anchor_target=candidate.target,
                        anchor_fair=candidate.post_fair,
                        event_id=candidate.event_id,
                        market_ticker=actual_ticker,
                        entry_client_order_id=entry_client_order_id,
                    )
                    if portfolio.open_position(
                        int(GAME_PK), actual_ticker, proposed_position,
                        hybrid_config.minimum_seconds_between_entries,
                    ):
                        positions.append(proposed_position)
                        action = (
                            f"OPEN_{candidate.side.upper()}_"
                            f"{candidate.event_type.upper()}"
                        )
                        print(
                            f"TRADE BUY {candidate.side.upper()} "
                            f"contracts={contracts:.4f} price={price:.4f} "
                            f"fee={fee:.4f} reason={candidate.event_type.upper()} "
                            f"game_pk={GAME_PK} ticker={MARKET_TICKER}",
                            flush=True,
                        )
                        candidate = None
                    else:
                        action = "SKIP_CASH_COOLDOWN_OR_DUPLICATE_EVENT"

        excess_move = (
            market.midpoint - target if pd.notna(target) else float("nan")
        )

        metrics = portfolio.metrics()
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                now.isoformat(), market.received_at.isoformat(),
                game.received_at.isoformat(), market.bid, market.ask,
                game.state["inning"], game.state["outs_when_up"],
                game.state["score_diff"], fair_prob,
                game.completed_event_id, game.completed_event, target,
                excess_move, edge, continuation_value, exit_advantage,
                action, metrics.cash, metrics.equity, metrics.pnl,
                metrics.open_positions,
            ])
        print(
            f"{now.time()} {market.bid:.2f}/{market.ask:.2f} "
            f"fair={fair_prob:.1%} target={target:.1%} "
            f"edge={edge:+.1%} {action} "
            f"portfolio=${metrics.equity:.2f} pnl=${metrics.pnl:+.2f}"
        )
        previous_market = market.midpoint
        previous_fair = float(fair_prob)
        previous_event_id = game.completed_event_id
        await asyncio.sleep(POLL_SECONDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Discover and run every matched MLB/Kalshi game for a date.",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Local schedule date for --all-games (YYYY-MM-DD).",
    )
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Print discovered mappings without launching paper traders.",
    )
    parser.add_argument(
        "--portfolio-status",
        action="store_true",
        help="Print the shared paper portfolio and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        if args.portfolio_status:
            path = Path(os.getenv(
                "PAPER_PORTFOLIO_DB",
                str(LOG_DIR / "hit_reversion_portfolio.sqlite3"),
            ))
            portfolio = SharedPaperPortfolio(
                path, float(os.getenv("PAPER_STARTING_CASH", "1000"))
            )
            metrics = portfolio.metrics()
            print(
                f"cash=${metrics.cash:.2f} equity=${metrics.equity:.2f} "
                f"pnl=${metrics.pnl:+.2f} "
                f"open_positions={metrics.open_positions} db={path}"
            )
        elif args.discover_only:
            games, warnings = discover_daily_games(
                args.date or current_slate_date()
            )
            for warning in warnings:
                print(f"WARNING: {warning}")
            for game in games:
                print(
                    f"{game.scheduled_time.isoformat()} "
                    f"{game.away_code}@{game.home_code} "
                    f"game_pk={game.game_pk} ticker={game.market_ticker}"
                )
        elif args.continuous:
            raise SystemExit(run_continuous_coordinator(args.date))
        elif args.all_games:
            raise SystemExit(run_daily_coordinator(
                args.date or current_slate_date()
            ))
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("Paper trader stopped")
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
