"""One adaptive MLB live-feed poller shared by all paper workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import threading
import time
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
import websockets

from settlement_value_strategy.play_eligibility import (
    incomplete_ball_in_play_reason,
)


MLB_API = "https://statsapi.mlb.com/api"
MLB_GAMEDAY_WS = os.getenv(
    "MLB_GAMEDAY_WS", "wss://ws.statsapi.mlb.com/api/v1/game/push/subscribe/gameday"
).rstrip("/")
FEED_URL = os.getenv("MLB_FEED_URL", "http://127.0.0.1:8766").rstrip("/")


@dataclass
class GameFeed:
    payload: dict | None = None
    received_at: str | None = None
    status: str = "Unknown"
    failures: int = 0
    last_error: str | None = None
    last_error_kind: str | None = None
    last_status_code: int | None = None
    last_play_stage: tuple | None = None
    last_at_bat_index: int | None = None
    websocket_connected: bool = False
    websocket_last_message_at: str | None = None
    websocket_failures: int = 0
    pending_websocket_notification: dict | None = None
    last_websocket_notification: dict | None = None


class FeedState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.games: dict[int, GameFeed] = {}
        self.workers: dict[int, threading.Thread] = {}
        self.websocket_workers: dict[int, threading.Thread] = {}
        self.websocket_wakeups: dict[int, threading.Event] = {}
        self.stopping = threading.Event()

    def request(self, game_pk: int) -> GameFeed:
        game_pk = int(game_pk)
        with self.lock:
            game = self.games.setdefault(game_pk, GameFeed())
            worker = self.workers.get(game_pk)
            if worker is None or not worker.is_alive():
                wakeup = self.websocket_wakeups.setdefault(
                    game_pk, threading.Event()
                )
                worker = threading.Thread(
                    target=self._poll_game, args=(game_pk,), daemon=True,
                    name=f"mlb-feed-{game_pk}",
                )
                self.workers[game_pk] = worker
                worker.start()
                websocket_worker = threading.Thread(
                    target=self._websocket_game, args=(game_pk, wakeup),
                    daemon=True, name=f"mlb-gameday-ws-{game_pk}",
                )
                self.websocket_workers[game_pk] = websocket_worker
                websocket_worker.start()
            return game

    def response(self, game_pk: int) -> dict:
        self.request(game_pk)
        with self.lock:
            game = self.games[int(game_pk)]
            return {
                "game_pk": int(game_pk), "payload": game.payload,
                "received_at": game.received_at, "status": game.status,
                "failures": game.failures, "last_error": game.last_error,
                "last_error_kind": game.last_error_kind,
                "last_status_code": game.last_status_code,
                "websocket_connected": game.websocket_connected,
                "websocket_last_message_at": game.websocket_last_message_at,
                "websocket_failures": game.websocket_failures,
                "websocket_last_notification": game.last_websocket_notification,
            }

    @staticmethod
    def _interval(status: str) -> float:
        if status == "Live":
            return float(os.getenv("MLB_LIVE_POLL_SECONDS", "1.0"))
        if status == "Final":
            return float(os.getenv("MLB_FINAL_POLL_SECONDS", "300"))
        return float(os.getenv("MLB_PREGAME_POLL_SECONDS", "30"))

    @staticmethod
    def _failure_interval(
        status: str, failures: int, error_kind: str = "connection",
        retry_after: float | None = None,
    ) -> float:
        if error_kind == "rate_limit":
            return min(300.0, max(1.0, retry_after or 30.0))
        if error_kind == "client_error":
            return 60.0
        if status == "Live" and error_kind in {"dns", "connection"}:
            # Long backoffs during a game can miss several pitches after the
            # connection has returned. Keep retries bounded while avoiding a
            # tight loop against an unavailable upstream.
            return min(5.0, 0.5 * 2.0 ** min(max(failures - 1, 0), 4))
        if status == "Live" and error_kind == "timeout":
            return min(10.0, 1.0 * 2.0 ** min(max(failures - 1, 0), 4))
        if error_kind == "server_error":
            return min(30.0, 2.0 ** min(failures, 5))
        return min(60.0, 2.0 ** min(failures, 6))

    @staticmethod
    def _websocket_retry_policy(
        status: str, error: Exception, failures: int,
    ) -> tuple[float, bool, bool]:
        """Return (delay, expected_unavailability, permanently_stop)."""
        if status == "Final":
            return 0.0, True, True
        code = getattr(error, "code", None)
        message = str(error).lower()
        unavailable = (
            code == 4400
            and "not available for subscription" in message
        )
        if unavailable:
            return (
                float(os.getenv("MLB_WS_UNAVAILABLE_RETRY_SECONDS", "300")),
                True, False,
            )
        return min(30.0, 2.0 ** min(failures - 1, 5)), False, False

    @staticmethod
    def _classify_failure(
        error: Exception,
    ) -> tuple[str, int | None, float | None]:
        if isinstance(error, requests.Timeout):
            return "timeout", None, None
        if isinstance(error, requests.HTTPError):
            response = error.response
            code = response.status_code if response is not None else None
            retry_after = None
            if response is not None:
                try:
                    retry_after = float(response.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    pass
            if code == 429:
                return "rate_limit", code, retry_after
            if code is not None and code >= 500:
                return "server_error", code, None
            return "client_error", code, None
        if isinstance(error, requests.ConnectionError):
            message = str(error).lower()
            if "nameresolutionerror" in message or "failed to resolve" in message:
                return "dns", None, None
            return "connection", None, None
        if isinstance(error, (ValueError, json.JSONDecodeError)):
            return "invalid_response", None, None
        return "unknown", None, None

    @staticmethod
    def _new_session() -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=2, max_retries=0)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def _append_record(
        record: dict, observed_at: str, filename_prefix: str,
    ) -> None:
        log_dir = Path(os.getenv("PAPER_LOG_DIR", "live_logs"))
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            date = observed_at[:10] if len(observed_at) >= 10 else "unknown"
            path = log_dir / f"{filename_prefix}_{date}.jsonl"
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, separators=(",", ":")))
                output.write("\n")
        except OSError as error:
            print(f"MLB_STRUCTURED_LOG_ERROR error={error}", flush=True)

    @staticmethod
    def _append_transition_record(record: dict, observed_at: str) -> None:
        FeedState._append_record(
            record, observed_at, "mlb_feed_transitions"
        )

    @staticmethod
    def _append_transport_record(record: dict, observed_at: str) -> None:
        FeedState._append_record(
            record, observed_at, "mlb_feed_transport"
        )

    @staticmethod
    def _log_play_transitions(
        game_pk: int, game: GameFeed, payload: dict, observed_at: str,
        upstream: dict | None = None,
    ) -> None:
        plays = payload.get("liveData", {}).get("plays", {}) or {}
        values = list(plays.get("allPlays") or [])
        if isinstance(plays.get("currentPlay"), dict):
            values.append(plays["currentPlay"])
        indexed = [
            play for play in values
            if isinstance(play, dict) and play.get("atBatIndex") is not None
        ]
        if not indexed:
            return
        play = max(indexed, key=lambda item: int(item["atBatIndex"]))
        at_bat = int(play["atBatIndex"])
        previous_at_bat = game.last_at_bat_index
        progressed = previous_at_bat is not None and at_bat > previous_at_bat
        game.last_at_bat_index = max(game.last_at_bat_index or at_bat, at_bat)
        event_type = str(play.get("result", {}).get("eventType") or "")
        runners = play.get("runners")
        pitch_ends = [
            str(event["endTime"])
            for event in play.get("playEvents") or []
            if event.get("isPitch") and event.get("endTime")
        ]
        pitches = [
            event for event in play.get("playEvents") or []
            if event.get("isPitch")
        ]
        latest_pitch = max(
            pitches,
            key=lambda event: int(
                event.get("pitchNumber") or event.get("index") or 0
            ),
            default=None,
        )
        latest_pitch_number = (
            int(latest_pitch.get("pitchNumber") or latest_pitch.get("index") or 0)
            if latest_pitch else None
        )
        is_in_play = bool(
            latest_pitch
            and (latest_pitch.get("details") or {}).get("isInPlay")
        )
        atomic_rejection = (
            incomplete_ball_in_play_reason(play, latest_pitch_number)
            if latest_pitch_number is not None else None
        )
        stage = (
            at_bat, event_type, bool(isinstance(runners, list) and runners),
            bool(play.get("about", {}).get("isComplete")),
            max(pitch_ends) if pitch_ends else None,
            latest_pitch_number, is_in_play, atomic_rejection,
        )
        if stage != game.last_play_stage:
            previous_play = next((
                item for item in indexed
                if progressed and int(item["atBatIndex"]) == previous_at_bat
            ), None)
            FeedState._append_transition_record({
                "kind": "mlb_play_transition",
                "game_pk": game_pk,
                "observed_at": observed_at,
                "previous_at_bat": previous_at_bat,
                "at_bat": at_bat,
                "at_bat_progressed": progressed,
                "stage": {
                    "event_type": event_type or None,
                    "runners_populated": stage[2],
                    "is_complete": stage[3],
                    "latest_pitch_end": stage[4],
                    "latest_pitch_number": latest_pitch_number,
                    "is_in_play": is_in_play,
                    "atomic_play_eligible": atomic_rejection is None,
                    "atomic_rejection_reason": atomic_rejection,
                },
                "upstream": upstream or {},
                "play": play,
                "previous_play": previous_play,
                "linescore": payload.get("liveData", {}).get("linescore"),
                "game_status": payload.get("gameData", {}).get("status"),
            }, observed_at)
            game.last_play_stage = stage

    def _poll_game(self, game_pk: int) -> None:
        session = self._new_session()
        wakeup = self.websocket_wakeups[game_pk]
        try:
            while not self.stopping.is_set():
                started = time.monotonic()
                request_started_at = datetime.now(timezone.utc).isoformat()
                with self.lock:
                    game = self.games[game_pk]
                    notification = game.pending_websocket_notification
                    game.pending_websocket_notification = None
                trigger = "websocket" if notification else "scheduled_poll"
                try:
                    response = session.get(
                        f"{MLB_API}/v1.1/game/{game_pk}/feed/live",
                        # A WebSocket notification should retrieve a fresh
                        # representation rather than a CDN's cached full feed.
                        params={"_": time.time_ns()},
                        timeout=(3.05, 10),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    status = str(payload.get("gameData", {}).get(
                        "status", {}
                    ).get("abstractGameState") or "Unknown")
                    observed_at = datetime.now(timezone.utc).isoformat()
                    upstream = {
                        "trigger": trigger,
                        "request_started_at": request_started_at,
                        "websocket_notification": notification,
                        "request_elapsed_seconds": response.elapsed.total_seconds(),
                        "date": response.headers.get("Date"),
                        "age": response.headers.get("Age"),
                        "cache_control": response.headers.get("Cache-Control"),
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                    }
                    with self.lock:
                        game = self.games[game_pk]
                        self._log_play_transitions(
                            game_pk, game, payload, observed_at, upstream
                        )
                        game.payload = payload
                        game.received_at = observed_at
                        game.status = status
                        game.failures = 0
                        game.last_error = None
                        game.last_error_kind = None
                        game.last_status_code = None
                    if notification is not None:
                        self._append_transport_record({
                            "kind": "mlb_feed_refresh",
                            "game_pk": game_pk,
                            "trigger": trigger,
                            "request_started_at": request_started_at,
                            "observed_at": observed_at,
                            "websocket_notification": notification,
                            "http": upstream,
                            "game_status": status,
                        }, observed_at)
                    delay = self._interval(status)
                except Exception as error:
                    kind, status_code, retry_after = self._classify_failure(error)
                    with self.lock:
                        game = self.games[game_pk]
                        game.failures += 1
                        game.last_error = str(error)
                        game.last_error_kind = kind
                        game.last_status_code = status_code
                        failures = game.failures
                        status = game.status
                    delay = self._failure_interval(
                        status, failures, kind, retry_after
                    )
                    delay *= random.uniform(0.8, 1.2)
                    print(
                        f"MLB feed {game_pk} failed kind={kind} "
                        f"status={status_code} ({failures}): {error}; "
                        f"retrying in {delay:.1f}s", flush=True,
                    )
                remaining = max(0.05, delay - (time.monotonic() - started))
                wakeup.wait(remaining)
                wakeup.clear()
        finally:
            session.close()

    async def _websocket_game_loop(
        self, game_pk: int, wakeup: threading.Event,
    ) -> None:
        url = f"{MLB_GAMEDAY_WS}/{game_pk}"
        async with websockets.connect(
            url, open_timeout=10, ping_interval=20, ping_timeout=20,
            close_timeout=5,
        ) as websocket:
            connected_at = datetime.now(timezone.utc).isoformat()
            connection_notification = {
                "kind": "websocket_connected", "received_at": connected_at,
            }
            with self.lock:
                game = self.games[game_pk]
                game.websocket_connected = True
                game.websocket_failures = 0
                game.pending_websocket_notification = connection_notification
                game.last_websocket_notification = connection_notification
            self._append_transport_record({
                "kind": "mlb_websocket_connected", "game_pk": game_pk,
                "observed_at": connected_at, "url": url,
            }, connected_at)
            # Fetch immediately after the socket handshake, then after every
            # Gameday push message. The one-second poll remains the fallback.
            wakeup.set()
            async for message in websocket:
                observed_at = datetime.now(timezone.utc).isoformat()
                try:
                    decoded = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    decoded = {"raw_message": str(message)}
                notification = {
                    "kind": "websocket_message",
                    "received_at": observed_at,
                    "mlb_timestamp": decoded.get("timeStamp"),
                    "update_id": decoded.get("updateId"),
                    "logical_events": decoded.get("logicalEvents"),
                    "game_events": decoded.get("gameEvents"),
                    "change_event": decoded.get("changeEvent"),
                    "is_delay": decoded.get("isDelay"),
                    "wait": decoded.get("wait"),
                }
                with self.lock:
                    game = self.games[game_pk]
                    game.websocket_last_message_at = observed_at
                    game.pending_websocket_notification = notification
                    game.last_websocket_notification = notification
                self._append_transport_record({
                    "kind": "mlb_websocket_message", "game_pk": game_pk,
                    "observed_at": observed_at, "notification": notification,
                    "message": decoded,
                }, observed_at)
                wakeup.set()

    def _websocket_game(
        self, game_pk: int, wakeup: threading.Event,
    ) -> None:
        failures = 0
        while not self.stopping.is_set():
            try:
                asyncio.run(self._websocket_game_loop(game_pk, wakeup))
                failures = 0
                with self.lock:
                    status = self.games[game_pk].status
                if status == "Final":
                    return
                # A clean upstream close outside Final is reconnectable, but
                # reconnecting in a tight loop can hammer MLB and flood logs.
                self.stopping.wait(5.0)
            except Exception as error:
                failures += 1
                with self.lock:
                    game = self.games[game_pk]
                    game.websocket_connected = False
                    game.websocket_failures = failures
                    status = game.status
                delay, expected, stop = self._websocket_retry_policy(
                    status, error, failures
                )
                observed_at = datetime.now(timezone.utc).isoformat()
                self._append_transport_record({
                    "kind": (
                        "mlb_websocket_unavailable"
                        if expected else "mlb_websocket_error"
                    ),
                    "game_pk": game_pk, "game_status": status,
                    "observed_at": observed_at, "failure_count": failures,
                    "error_type": type(error).__name__, "error": str(error),
                    "close_code": getattr(error, "code", None),
                    "retry_seconds": delay, "terminal": stop,
                }, observed_at)
                if stop:
                    return
                if not expected:
                    print(
                        f"MLB Gameday WebSocket {game_pk} failed ({failures}): "
                        f"{error}; polling remains active; retrying in "
                        f"{delay:.1f}s", flush=True,
                    )
                self.stopping.wait(delay)
            finally:
                with self.lock:
                    if game_pk in self.games:
                        self.games[game_pk].websocket_connected = False


STATE = FeedState()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._reply(200, {"ok": True, "games": len(STATE.games)})
            return
        if path.startswith("/games/"):
            try:
                game_pk = int(path.removeprefix("/games/"))
            except ValueError:
                self._reply(400, {"error": "invalid game_pk"})
                return
            payload = STATE.response(game_pk)
            self._reply(200 if payload["payload"] is not None else 503, payload)
            return
        self._reply(404, {"error": "not found"})

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def get_game(game_pk: int, timeout: float = 5.0) -> dict:
    response = requests.get(f"{FEED_URL}/games/{int(game_pk)}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def serve() -> None:
    server = ThreadingHTTPServer((
        os.getenv("MLB_FEED_BIND", "127.0.0.1"), 8766
    ), Handler)
    print("Shared MLB feed ready", flush=True)
    try:
        server.serve_forever()
    finally:
        STATE.stopping.set()
        server.server_close()


if __name__ == "__main__":
    serve()
