"""One authenticated Kalshi WebSocket shared by all paper-trading workers."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse

import requests
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


REST_URL = "https://external-api.kalshi.com/trade-api/v2"
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"
FEED_URL = os.getenv("KALSHI_FEED_URL", "http://127.0.0.1:8765").rstrip("/")


@dataclass
class FeedMarket:
    snapshot: dict | None = None
    trades: deque | None = None
    anchor: float | None = None

    def __post_init__(self) -> None:
        if self.trades is None:
            self.trades = deque(maxlen=5000)


class FeedState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.markets: dict[str, FeedMarket] = defaultdict(FeedMarket)
        self.requested: set[str] = set()
        self.subscribed: set[str] = set()
        self.connected = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.wakeup: asyncio.Event | None = None

    def request(self, ticker: str) -> None:
        with self.lock:
            is_new = ticker not in self.requested
            self.requested.add(ticker)
        if is_new and self.loop is not None and self.wakeup is not None:
            self.loop.call_soon_threadsafe(self.wakeup.set)

    def payload(self, ticker: str) -> dict:
        self.request(ticker)
        with self.lock:
            market = self.markets[ticker]
            return {
                "connected": self.connected,
                "ticker": ticker,
                "snapshot": market.snapshot,
                "trades": list(market.trades or ()),
                "pregame_anchor": market.anchor,
            }


STATE = FeedState()


def _number(message: dict, *names: str) -> float | None:
    for name in names:
        value = message.get(name)
        if value is not None:
            return float(value)
    return None


def _timestamp(message: dict) -> str:
    value = message.get("created_time") or message.get("time")
    if value:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    milliseconds = message.get("ts_ms")
    if milliseconds is not None:
        return datetime.fromtimestamp(
            float(milliseconds) / 1000, timezone.utc
        ).isoformat(timespec="microseconds")
    seconds = message.get("ts")
    if seconds is not None:
        return datetime.fromtimestamp(
            float(seconds), timezone.utc
        ).isoformat(timespec="microseconds")
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _set_snapshot(ticker: str, bid: float, ask: float,
                  bid_size: float = 0.0, ask_size: float = 0.0) -> None:
    if not 0 < bid < ask < 1:
        return
    with STATE.lock:
        market = STATE.markets[ticker]
        market.snapshot = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "bid": bid, "ask": ask,
            "bid_size": bid_size, "ask_size": ask_size,
        }
        if market.anchor is None:
            market.anchor = (bid + ask) / 2


def _add_trade(ticker: str, row: dict) -> None:
    normalized = {
        "trade_id": str(row.get("trade_id") or row.get("id") or ""),
        "created_time": _timestamp(row),
        "yes_price_dollars": _number(
            row, "yes_price_dollars", "price_dollars", "yes_price"
        ),
        "count_fp": _number(row, "count_fp", "count", "size_fp", "size"),
        "taker_outcome_side": str(
            row.get("taker_outcome_side") or row.get("taker_side") or ""
        ).lower(),
    }
    if not normalized["trade_id"] or normalized["yes_price_dollars"] is None:
        return
    with STATE.lock:
        tape = STATE.markets[ticker].trades
        if tape is not None and all(
            existing["trade_id"] != normalized["trade_id"] for existing in tape
        ):
            tape.append(normalized)


def _bootstrap(ticker: str) -> None:
    """Use bounded REST calls only when a ticker first joins the stream."""
    try:
        response = requests.get(f"{REST_URL}/markets/{ticker}/orderbook", timeout=10)
        response.raise_for_status()
        book = response.json().get("orderbook_fp") or {}
        yes, no = book.get("yes_dollars") or [], book.get("no_dollars") or []
        if yes and no:
            _set_snapshot(
                ticker, float(yes[-1][0]), 1.0 - float(no[-1][0]),
                float(yes[-1][1]), float(no[-1][1]),
            )
        time.sleep(0.1)
        response = requests.get(
            f"{REST_URL}/markets/trades",
            params={"ticker": ticker, "limit": 1000}, timeout=10,
        )
        response.raise_for_status()
        for row in reversed(response.json().get("trades") or []):
            _add_trade(ticker, row)
    except Exception as error:
        print(f"Feed bootstrap failed for {ticker}: {error}", flush=True)


def _headers() -> dict[str, str]:
    key_id = os.environ["KALSHI_API_KEY_ID"]
    key_path = Path(os.environ["KALSHI_PRIVATE_KEY_PATH"])
    private_key = serialization.load_pem_private_key(key_path.read_bytes(), None)
    timestamp = str(int(time.time() * 1000))
    signature = private_key.sign(
        f"{timestamp}GET{WS_PATH}".encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    import base64
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


async def _subscribe(websocket, tickers: list[str]) -> None:
    if not tickers:
        return
    await websocket.send(json.dumps({
        "id": int(time.time() * 1000), "cmd": "subscribe",
        "params": {
            "channels": ["ticker", "trade"],
            "market_tickers": tickers,
        },
    }))
    for ticker in tickers:
        await asyncio.to_thread(_bootstrap, ticker)
    with STATE.lock:
        STATE.subscribed.update(tickers)
    print(f"Kalshi feed subscribed to {len(tickers)} ticker(s)", flush=True)


def _process(message: dict) -> None:
    kind = str(message.get("type") or "")
    row = message.get("msg") or {}
    if kind == "error":
        print(f"Kalshi WebSocket error: {row}", flush=True)
        return
    ticker = str(row.get("market_ticker") or row.get("ticker") or "")
    if not ticker:
        return
    if kind == "ticker":
        bid = _number(row, "yes_bid_dollars", "yes_bid")
        ask = _number(row, "yes_ask_dollars", "yes_ask")
        if bid is not None and ask is not None:
            _set_snapshot(
                ticker, bid, ask,
                _number(row, "yes_bid_size_fp", "yes_bid_size") or 0.0,
                _number(row, "yes_ask_size_fp", "yes_ask_size") or 0.0,
            )
    elif kind == "trade":
        _add_trade(ticker, row)


async def websocket_loop() -> None:
    STATE.loop = asyncio.get_running_loop()
    STATE.wakeup = asyncio.Event()
    delay = 1.0
    while True:
        try:
            async with websockets.connect(
                WS_URL, additional_headers=_headers(), ping_interval=20,
                ping_timeout=20, max_queue=10000,
            ) as websocket:
                with STATE.lock:
                    STATE.connected = True
                    STATE.subscribed.clear()
                delay = 1.0
                print("Kalshi shared WebSocket connected", flush=True)
                while True:
                    with STATE.lock:
                        pending = sorted(STATE.requested - STATE.subscribed)
                    if pending:
                        await _subscribe(websocket, pending)
                    receive = asyncio.create_task(websocket.recv())
                    wake = asyncio.create_task(STATE.wakeup.wait())
                    done, pending_tasks = await asyncio.wait(
                        {receive, wake}, return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending_tasks:
                        task.cancel()
                    if wake in done:
                        STATE.wakeup.clear()
                    if receive in done:
                        _process(json.loads(receive.result()))
        except Exception as error:
            with STATE.lock:
                STATE.connected = False
            print(f"Kalshi WebSocket disconnected: {error}; retrying", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._reply(
                200 if STATE.connected else 503,
                {"ok": STATE.connected, "connected": STATE.connected},
            )
            return
        if path.startswith("/markets/"):
            ticker = unquote(path.removeprefix("/markets/"))
            payload = STATE.payload(ticker)
            ready = bool(payload["connected"] and payload["snapshot"] is not None)
            self._reply(200 if ready else 503, payload)
            return
        self._reply(404, {"error": "not found"})

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def get_market(ticker: str, timeout: float = 5.0) -> dict:
    response = requests.get(f"{FEED_URL}/markets/{quote(ticker, safe='')}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def serve() -> None:
    missing = [name for name in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH")
               if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing shared-feed credentials: {', '.join(missing)}")
    server = ThreadingHTTPServer((
        os.getenv("KALSHI_FEED_BIND", "127.0.0.1"), 8765
    ), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        asyncio.run(websocket_loop())
    finally:
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["serve"], default="serve", nargs="?")
    parser.parse_args()
    serve()
