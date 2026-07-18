"""Paper trader for the event-conditioned hybrid residual strategy."""

from __future__ import annotations

import asyncio
import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
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
LOG_DIR = Path(os.getenv(
    "PAPER_LOG_DIR",
    str(Path(__file__).resolve().parent / "logs"),
))
MODEL_DIR = PROJECT_ROOT / "models/market_reaction_model"
HYBRID_CONFIG_PATH = MODEL_DIR / "trade_tape_config.json"
KALSHI_API = "https://external-api.kalshi.com/trade-api/v2"
MLB_API = "https://statsapi.mlb.com/api"

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
            connection.execute(
                """CREATE TABLE IF NOT EXISTS positions (
                    game_pk INTEGER PRIMARY KEY,
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
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "INSERT OR IGNORE INTO portfolio(id, starting_cash, cash) "
                "VALUES (1, ?, ?)",
                (starting_cash, starting_cash),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def open_position(self, game_pk: int, ticker: str, position: Position) -> bool:
        cost = position.contracts * position.entry_price + position.entry_fee
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cash = float(connection.execute(
                "SELECT cash FROM portfolio WHERE id = 1"
            ).fetchone()[0])
            exists = connection.execute(
                "SELECT 1 FROM positions WHERE game_pk = ?", (game_pk,)
            ).fetchone()
            if exists or cash + 1e-9 < cost:
                connection.rollback()
                return False
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "UPDATE portfolio SET cash = cash - ? WHERE id = 1", (cost,)
            )
            connection.execute(
                """INSERT INTO positions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    game_pk, ticker, position.side, position.contracts,
                    position.entry_price, position.entry_fee,
                    position.entry_time.isoformat(), position.anchor_target,
                    position.anchor_fair, position.event_id,
                    position.entry_price, now,
                ),
            )
        return True

    def update_mark(self, game_pk: int, mark_price: float) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE positions SET mark_price = ?, updated_at = ? "
                "WHERE game_pk = ?",
                (mark_price, datetime.now(timezone.utc).isoformat(), game_pk),
            )

    def close_position(self, game_pk: int, proceeds: float) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM positions WHERE game_pk = ?", (game_pk,)
            ).fetchone()
            if not exists:
                connection.rollback()
                return False
            connection.execute(
                "UPDATE portfolio SET cash = cash + ? WHERE id = 1",
                (proceeds,),
            )
            connection.execute(
                "DELETE FROM positions WHERE game_pk = ?", (game_pk,)
            )
        return True

    def load_position(self, game_pk: int) -> Position | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT side, contracts, entry_price, entry_fee, entry_time,
                          anchor_target, anchor_fair, event_id
                   FROM positions WHERE game_pk = ?""",
                (game_pk,),
            ).fetchone()
        if row is None:
            return None
        return Position(
            side=str(row[0]), contracts=float(row[1]), entry_price=float(row[2]),
            entry_fee=float(row[3]),
            entry_time=pd.to_datetime(row[4], utc=True).to_pydatetime(),
            anchor_target=float(row[5]), anchor_fair=float(row[6]),
            event_id=int(row[7]),
        )

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


MAIN_LOG_ACTIONS = (
    "WATCH_", "OPEN_", "CLOSE_", "EXPIRE_", "SKIP_",
    "INITIALIZE_", "Recovered open", "Snapshot rejected",
    "Shared portfolio:",
)


def should_surface_worker_line(line: str) -> bool:
    """Keep the coordinator log focused on decisions and operational errors."""
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


def match_games_to_home_markets(
    games: list[dict],
    events: list[dict],
) -> tuple[list[DiscoveredGame], list[str]]:
    """Match same-day games and Kalshi events, including doubleheaders."""
    event_groups: dict[frozenset[str], list[tuple[str, dict[str, dict]]]] = {}
    for event in events:
        markets = event.get("markets") or []
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
        if len(market_events) != len(scheduled_games):
            warnings.append(
                f"{sorted(matchup)}: {len(scheduled_games)} MLB games but "
                f"{len(market_events)} Kalshi events; skipping ambiguous matchup"
            )
            continue
        for (scheduled, info), (_, markets) in zip(
            scheduled_games, market_events
        ):
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
        if game.get("status", {}).get("abstractGameState") != "Final"
        and str(game.get("status", {}).get("detailedState") or "").lower()
        not in {"postponed", "cancelled", "canceled"}
    ]

    from data.raw.scripts.download_live_kalshi_market_logs import (
        discover_events,
        fetch_event_markets,
    )

    events = discover_events(game_date, game_date, verbose=False)
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
        raise RuntimeError(f"No matched MLB/Kalshi games for {game_date}")

    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = Path(os.getenv(
        "PAPER_PORTFOLIO_DB",
        str(log_dir / f"paper_portfolio_{game_date.isoformat()}.sqlite3"),
    ))
    portfolio = SharedPaperPortfolio(
        portfolio_path,
        float(os.getenv("PAPER_STARTING_CASH", "1000")),
    )
    run_id = int(time.time())
    children: list[
        tuple[DiscoveredGame, subprocess.Popen, object, threading.Thread]
    ] = []
    try:
        for game in games:
            console_path = log_dir / (
                f"paper_console_{game.market_ticker}_{run_id}.log"
            )
            handle = console_path.open("w")
            env = os.environ.copy()
            env["MLB_GAME_PK"] = str(game.game_pk)
            env["KALSHI_MARKET_TICKER"] = game.market_ticker
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
                f"Started {game.away_code}@{game.home_code} "
                f"game_pk={game.game_pk} ticker={game.market_ticker} "
                f"console={console_path}"
            )
        opening = portfolio.metrics()
        print(
            f"Running {len(children)} isolated paper traders with shared "
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
    feed_response = requests.get(
        f"{MLB_API}/v1.1/game/{GAME_PK}/feed/live", timeout=5
    )
    feed_response.raise_for_status()
    feed = feed_response.json()
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
    portfolio_path = Path(os.getenv(
        "PAPER_PORTFOLIO_DB",
        str(LOG_DIR / (
            f"paper_portfolio_{datetime.now().astimezone().date()}_"
            f"game_{GAME_PK}.sqlite3"
        )),
    ))
    portfolio = SharedPaperPortfolio(
        portfolio_path,
        float(os.getenv("PAPER_STARTING_CASH", "1000")),
    )
    pregame_prob = await asyncio.to_thread(fetch_pregame_anchor)
    print(f"Pregame Kalshi anchor: {pregame_prob:.1%}")
    print(
        f"Hybrid threshold={hybrid_config.minimum_edge:.1%}, "
        f"confirmation={hybrid_config.confirmation_seconds:g} seconds, "
        f"entry expiry={hybrid_config.maximum_event_to_entry_seconds:g} seconds, "
        "target-reversion exit"
    )

    log_dir = LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"paper_trade_log_{MARKET_TICKER}_{int(time.time())}.csv"
    with log_path.open("w", newline="") as handle:
        csv.writer(handle).writerow([
            "decision_time", "market_received_at", "state_received_at",
            "bid", "ask", "inning", "outs", "score_diff", "fair_prob",
            "completed_event_id", "completed_event", "target", "excess_move",
            "edge", "continuation_value", "exit_advantage", "action",
            "portfolio_cash", "portfolio_equity", "portfolio_pnl",
            "portfolio_open_positions",
        ])

    position = portfolio.load_position(int(GAME_PK))
    if position is not None:
        print(
            f"Recovered open {position.side.upper()} position for game "
            f"{GAME_PK} from {portfolio_path}"
        )
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
                portfolio.close_position(
                    int(GAME_PK), position.contracts if won else 0.0
                )
                position = None
            final = portfolio.metrics()
            print(
                f"Shared portfolio: cash=${final.cash:.2f} "
                f"equity=${final.equity:.2f} PnL=${final.pnl:+.2f}"
            )
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
                f"Initialized {'midgame' if game.completed_event_id is not None else 'pregame'} "
                f"baseline at inning {game.state['inning']}; "
                "events already visible at startup will not be traded."
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
            liquidation_price = (
                market.bid if position.side == "yes" else 1.0 - market.ask
            )
            portfolio.update_mark(int(GAME_PK), liquidation_price)
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
                portfolio.close_position(
                    int(GAME_PK), position.contracts * price - fee
                )
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
                    proposed_position = Position(
                        side=candidate.side,
                        contracts=contracts,
                        entry_price=price,
                        entry_fee=fee,
                        entry_time=now,
                        anchor_target=candidate.target,
                        anchor_fair=candidate.post_fair,
                        event_id=candidate.event_id,
                    )
                    if portfolio.open_position(
                        int(GAME_PK), str(MARKET_TICKER), proposed_position
                    ):
                        position = proposed_position
                        action = (
                            f"OPEN_{candidate.side.upper()}_"
                            f"{candidate.event_type.upper()}"
                        )
                        candidate = None
                        exit_watch_started = None
                    else:
                        action = "SKIP_INSUFFICIENT_SHARED_CASH"

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
        default=datetime.now().astimezone().date(),
        help="Local schedule date for --all-games (YYYY-MM-DD).",
    )
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
                str(LOG_DIR / "paper_portfolio.sqlite3"),
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
            games, warnings = discover_daily_games(args.date)
            for warning in warnings:
                print(f"WARNING: {warning}")
            for game in games:
                print(
                    f"{game.scheduled_time.isoformat()} "
                    f"{game.away_code}@{game.home_code} "
                    f"game_pk={game.game_pk} ticker={game.market_ticker}"
                )
        elif args.all_games:
            raise SystemExit(run_daily_coordinator(args.date))
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("Paper trader stopped")
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
