"""Record prospective MLB observations and Kalshi books without placing orders.

Example: python -m research.record --game-pk 123 --ticker HOME --ticker AWAY
Uses KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH for read-only WebSocket auth.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import uuid

import requests
import websockets

from shared_kalshi_feed import _headers, WS_URL


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class BookSequence:
    """Reject deltas without a snapshot or across a sequence gap."""
    def __init__(self):
        self.last = {}
        self.books = set()

    def observe(self, message):
        sid, seq = message.get("sid"), message.get("seq")
        kind = message.get("type")
        ticker = (message.get("msg") or {}).get("market_ticker")
        if sid is not None and seq is not None:
            if sid in self.last and seq != self.last[sid] + 1:
                self.books.clear()
                raise ValueError(f"sequence gap on subscription {sid}: {self.last[sid]} -> {seq}")
            self.last[sid] = seq
        if kind == "orderbook_snapshot":
            self.books.add((sid, ticker))
        elif kind == "orderbook_delta" and (sid, ticker) not in self.books:
            raise ValueError("orderbook delta without a snapshot")


async def capture(args, write):
    deadline = time.monotonic() + args.duration_seconds

    async def market_stream():
        while time.monotonic() < deadline:
            connection = str(uuid.uuid4())
            write({"type": "connection_start", "connection_id": connection})
            sequence = BookSequence()
            try:
                async with websockets.connect(WS_URL, additional_headers=_headers(), ping_interval=20,
                                              ping_timeout=20, max_queue=10000) as socket:
                    await socket.send(json.dumps({"id": 1, "cmd": "subscribe", "params": {
                        "channels": ["orderbook_delta", "trade"], "market_tickers": args.ticker}}))
                    while time.monotonic() < deadline:
                        remaining = deadline - time.monotonic()
                        raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
                        message = json.loads(raw)
                        write({"type": "kalshi_message", "connection_id": connection, "message": message})
                        sequence.observe(message)
                        if message.get("type") == "error":
                            raise ValueError("Kalshi rejected the subscription; see recorded error")
            except TimeoutError:
                if time.monotonic() < deadline:
                    write({"type": "connection_gap", "connection_id": connection, "error": "timeout"})
            except (websockets.exceptions.WebSocketException, OSError, ValueError) as error:
                write({"type": "connection_gap", "connection_id": connection, "error": str(error)})
                await asyncio.sleep(min(2., max(0., deadline - time.monotonic())))
            finally:
                write({"type": "connection_end", "connection_id": connection})

    async def baseball(game_pk):
        previous = None
        with requests.Session() as session:
            while time.monotonic() < deadline:
                sent_at, sent_ns = utc_now(), time.monotonic_ns()
                try:
                    response = await asyncio.to_thread(session.get,
                        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=10)
                    received_at, received_ns = utc_now(), time.monotonic_ns()
                    response.raise_for_status()
                    payload = response.json()
                    live = payload.get("liveData") or {}
                    plays = live.get("plays") or {}
                    observation = {"gameData": payload.get("gameData"), "linescore": live.get("linescore"),
                        "currentPlay": plays.get("currentPlay"), "recentPlays": (plays.get("allPlays") or [])[-3:]}
                    identity = hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()
                    write({"type": "mlb_observation", "game_pk": game_pk, "request_sent_at": sent_at,
                        "request_sent_monotonic_ns": sent_ns, "received_at": received_at,
                        "received_monotonic_ns": received_ns, "observation_sha256": identity,
                        "observation": observation if identity != previous else None})
                    previous = identity
                except (requests.RequestException, ValueError) as error:
                    write({"type": "mlb_error", "game_pk": game_pk, "request_sent_at": sent_at, "error": str(error)})
                await asyncio.sleep(min(args.poll_seconds, max(0., deadline - time.monotonic())))

    await asyncio.gather(market_stream(), *(baseball(game_pk) for game_pk in args.game_pk))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-pk", type=int, action="append", required=True)
    parser.add_argument("--ticker", action="append", required=True, help="Include both home and away YES tickers")
    parser.add_argument("--duration-seconds", type=float, default=3600)
    parser.add_argument("--poll-seconds", type=float, default=1.)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/forward_observations"))
    args = parser.parse_args()
    if not 0 < args.duration_seconds <= 86400 or not .25 <= args.poll_seconds <= 60:
        parser.error("duration must be in (0, 86400], poll interval in [0.25, 60]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"capture_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.jsonl.gz"
    with gzip.open(path, "xt", compresslevel=3) as stream:
        counter = 0
        last_flush = time.monotonic()

        def write(row):
            nonlocal counter, last_flush
            counter += 1
            stream.write(json.dumps({"record_number": counter, "recorded_at": utc_now(),
                "recorded_monotonic_ns": time.monotonic_ns(), **row}, separators=(",", ":")) + "\n")
            if time.monotonic() - last_flush >= 1:
                stream.flush()
                last_flush = time.monotonic()

        write({"type": "capture_start", "tickers": args.ticker, "game_pks": args.game_pk,
               "schema_version": 1, "orders_enabled": False})
        try:
            asyncio.run(capture(args, write))
        finally:
            write({"type": "capture_end"})
    print(path)


if __name__ == "__main__":
    main()
