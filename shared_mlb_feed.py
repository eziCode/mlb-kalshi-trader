"""One adaptive MLB live-feed poller shared by all paper workers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import random
import threading
import time
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter


MLB_API = "https://statsapi.mlb.com/api"
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


class FeedState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.games: dict[int, GameFeed] = {}
        self.workers: dict[int, threading.Thread] = {}
        self.stopping = threading.Event()

    def request(self, game_pk: int) -> GameFeed:
        game_pk = int(game_pk)
        with self.lock:
            game = self.games.setdefault(game_pk, GameFeed())
            worker = self.workers.get(game_pk)
            if worker is None or not worker.is_alive():
                worker = threading.Thread(
                    target=self._poll_game, args=(game_pk,), daemon=True,
                    name=f"mlb-feed-{game_pk}",
                )
                self.workers[game_pk] = worker
                worker.start()
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

    def _poll_game(self, game_pk: int) -> None:
        session = self._new_session()
        try:
            while not self.stopping.is_set():
                started = time.monotonic()
                try:
                    response = session.get(
                        f"{MLB_API}/v1.1/game/{game_pk}/feed/live",
                        timeout=(3.05, 10),
                    )
                    response.raise_for_status()
                    payload = response.json()
                    status = str(payload.get("gameData", {}).get(
                        "status", {}
                    ).get("abstractGameState") or "Unknown")
                    with self.lock:
                        game = self.games[game_pk]
                        game.payload = payload
                        game.received_at = datetime.now(timezone.utc).isoformat()
                        game.status = status
                        game.failures = 0
                        game.last_error = None
                        game.last_error_kind = None
                        game.last_status_code = None
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
                self.stopping.wait(remaining)
        finally:
            session.close()


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
