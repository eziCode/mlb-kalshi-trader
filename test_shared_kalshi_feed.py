from __future__ import annotations

from collections import defaultdict
import base64
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

import shared_kalshi_feed as feed


class SharedKalshiFeedTests(unittest.TestCase):
    def setUp(self) -> None:
        with feed.STATE.lock:
            feed.STATE.markets = defaultdict(feed.FeedMarket)
            feed.STATE.requested.clear()
            feed.STATE.subscribed.clear()

    def test_ticker_normalizes_backtest_compatible_top_of_book(self):
        feed._process({
            "type": "ticker",
            "msg": {
                "market_ticker": "TEST",
                "yes_bid_dollars": "0.43",
                "yes_ask_dollars": "0.45",
                "yes_bid_size_fp": "12",
                "yes_ask_size_fp": "9",
            },
        })
        snapshot = feed.STATE.markets["TEST"].snapshot
        self.assertEqual(snapshot["bid"], 0.43)
        self.assertEqual(snapshot["ask"], 0.45)
        self.assertEqual(snapshot["bid_size"], 12.0)
        self.assertEqual(snapshot["ask_size"], 9.0)

    def test_trade_normalizes_and_deduplicates_exact_tape(self):
        message = {
            "type": "trade",
            "msg": {
                "market_ticker": "TEST", "trade_id": "abc",
                "yes_price_dollars": "0.44", "count_fp": "3",
                "taker_outcome_side": "yes", "ts_ms": 1_700_000_000_123,
            },
        }
        feed._process(message)
        feed._process(message)
        trades = list(feed.STATE.markets["TEST"].trades)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["yes_price_dollars"], 0.44)
        self.assertEqual(trades[0]["taker_outcome_side"], "yes")
        self.assertIn(".123", trades[0]["created_time"])

    def test_trade_timestamp_always_has_one_parseable_format(self):
        for index, value in enumerate((
            "2026-07-21T00:58:33+00:00",
            "2026-07-21T00:58:33.123456Z",
        )):
            feed._process({
                "type": "trade",
                "msg": {
                    "market_ticker": "TEST", "trade_id": str(index),
                    "yes_price_dollars": "0.44", "count_fp": "1",
                    "taker_outcome_side": "yes", "created_time": value,
                },
            })
        values = [row["created_time"] for row in feed.STATE.markets["TEST"].trades]
        self.assertEqual(values, [
            "2026-07-21T00:58:33.000000+00:00",
            "2026-07-21T00:58:33.123456+00:00",
        ])

    def test_request_registers_ticker(self):
        payload = feed.STATE.payload("TEST")
        self.assertIn("TEST", feed.STATE.requested)
        self.assertIsNone(payload["snapshot"])

    def test_market_endpoint_fails_closed_while_websocket_is_disconnected(self):
        feed._set_snapshot("TEST", .40, .42, 10, 10)
        feed.STATE.connected = False
        handler = feed.Handler.__new__(feed.Handler)
        handler.path = "/markets/TEST"
        with patch.object(handler, "_reply") as reply:
            handler.do_GET()
        self.assertEqual(reply.call_args.args[0], 503)

    def test_auth_headers_sign_the_websocket_handshake(self):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.pem"
            path.write_bytes(pem)
            original = os.environ.copy()
            try:
                os.environ["KALSHI_API_KEY_ID"] = "test-id"
                os.environ["KALSHI_PRIVATE_KEY_PATH"] = str(path)
                headers = feed._headers()
            finally:
                os.environ.clear()
                os.environ.update(original)
        message = (
            headers["KALSHI-ACCESS-TIMESTAMP"] + "GET" + feed.WS_PATH
        ).encode()
        key.public_key().verify(
            base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        self.assertEqual(headers["KALSHI-ACCESS-KEY"], "test-id")


if __name__ == "__main__":
    unittest.main()
