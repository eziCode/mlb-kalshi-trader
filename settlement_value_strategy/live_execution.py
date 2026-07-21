"""Authenticated Kalshi execution with durable, conservative risk limits."""

from __future__ import annotations

import base64
import argparse
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import requests

from settlement_value_strategy.strategy import taker_fee


API_ROOT = "https://external-api.kalshi.com/trade-api/v2"
REAL_MONEY_ACK = "YES_I_UNDERSTAND_THIS_PLACES_REAL_ORDERS"


class KalshiAccountClient:
    def __init__(self) -> None:
        self.key_id = os.environ["KALSHI_API_KEY_ID"]
        key_path = Path(os.environ["KALSHI_PRIVATE_KEY_PATH"])
        self.private_key = serialization.load_pem_private_key(
            key_path.read_bytes(), password=None
        )

    def _headers(self, method: str, path: str) -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}/trade-api/v2{path}".encode()
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, **kwargs) -> dict:
        response = requests.request(
            method, f"{API_ROOT}{path}",
            headers=self._headers(method, path), timeout=10, **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def available_balance(self) -> float:
        return float(self.request("GET", "/portfolio/balance")["balance"]) / 100

    def positions(self) -> list[dict]:
        return list(self.request(
            "GET", "/portfolio/positions",
            params={"limit": 1000, "count_filter": "position"},
        ).get("market_positions") or [])

    def create_fill_or_kill(
        self, ticker: str, count: float, price: float, client_order_id: str,
    ) -> dict:
        return self.request(
            "POST", "/portfolio/events/orders",
            json={
                "ticker": ticker,
                "client_order_id": client_order_id,
                "side": "bid",
                "count": f"{count:.2f}",
                "price": f"{price:.4f}",
                "time_in_force": "fill_or_kill",
                "self_trade_prevention_type": "taker_at_cross",
                "post_only": False,
                "cancel_order_on_pause": True,
                "reduce_only": False,
                "subaccount": 0,
                "exchange_index": 0,
            },
        )


@dataclass(frozen=True)
class LiveFill:
    filled: bool
    client_order_id: str
    ticker: str
    contracts: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    reason: str = ""

    @property
    def capital(self) -> float:
        return self.contracts * self.price + self.fee


def contracts_for_budget(price: float, budget: float) -> float:
    """Largest 0.01-contract quantity whose principal plus fee fits budget."""
    count = float((Decimal(str(budget / price))).quantize(
        Decimal("0.01"), rounding=ROUND_DOWN
    ))
    while count > 0 and count * price + taker_fee(count, price) > budget + 1e-9:
        count = round(count - 0.01, 2)
    return max(count, 0.0)


class LiveRiskLedger:
    def __init__(self, path: Path, maximum_capital: float):
        self.path = Path(path)
        self.maximum_capital = maximum_capital
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS live_orders (
                client_order_id TEXT PRIMARY KEY,
                trigger_key TEXT NOT NULL UNIQUE,
                game_pk INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                reserved_capital REAL NOT NULL,
                committed_capital REAL NOT NULL DEFAULT 0,
                contracts REAL NOT NULL DEFAULT 0,
                price REAL NOT NULL DEFAULT 0,
                fee REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def committed(self, connection: sqlite3.Connection | None = None) -> float:
        owns = connection is None
        connection = connection or self._connect()
        try:
            row = connection.execute("""SELECT COALESCE(SUM(
                CASE WHEN status='filled' THEN committed_capital
                     WHEN status='pending' THEN reserved_capital ELSE 0 END
            ), 0) FROM live_orders""").fetchone()
            return float(row[0])
        finally:
            if owns:
                connection.close()

    def reserve(
        self, trigger_key: str, game_pk: int, ticker: str, budget: float,
        minimum_seconds_between_entries: float,
    ) -> str | None:
        digest = hashlib.sha256(f"{ticker}|{trigger_key}".encode()).hexdigest()[:24]
        client_id = f"sv-{digest}"
        now = datetime.now(timezone.utc)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM live_orders WHERE trigger_key=?", (trigger_key,)
            ).fetchone():
                return None
            latest = connection.execute(
                "SELECT created_at FROM live_orders WHERE game_pk=? AND "
                "status IN ('pending','filled') ORDER BY created_at DESC LIMIT 1",
                (game_pk,),
            ).fetchone()
            if latest and (
                now - datetime.fromisoformat(latest[0])
            ).total_seconds() < minimum_seconds_between_entries:
                return None
            if self.committed(connection) + budget > self.maximum_capital + 1e-9:
                return None
            stamp = now.isoformat()
            connection.execute(
                "INSERT INTO live_orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (client_id, trigger_key, game_pk, ticker, budget, 0.0, 0.0,
                 0.0, 0.0, "pending", stamp, stamp),
            )
        return client_id

    def finish(self, fill: LiveFill) -> None:
        status = "filled" if fill.filled else "not_filled"
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """UPDATE live_orders SET committed_capital=?, contracts=?,
                    price=?, fee=?, status=?, updated_at=?
                    WHERE client_order_id=?""",
                (fill.capital, fill.contracts, fill.price, fill.fee, status,
                 datetime.now(timezone.utc).isoformat(), fill.client_order_id),
            )

    def mark_error(self, client_order_id: str, definite_rejection: bool) -> None:
        # An ambiguous transport failure stays pending and consumes its full
        # reservation. This fails closed against accidental duplicate spend.
        status = "rejected" if definite_rejection else "pending"
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE live_orders SET status=?, updated_at=? WHERE client_order_id=?",
                (status, datetime.now(timezone.utc).isoformat(), client_order_id),
            )


class LiveExecutor:
    def __init__(self, ledger_path: Path):
        if os.getenv("LIVE_TRADING_ENABLED") != REAL_MONEY_ACK:
            raise RuntimeError("Real-money trading acknowledgement is missing")
        self.per_order_budget = float(os.getenv("LIVE_MAX_ORDER_CAPITAL", "0.75"))
        self.maximum_capital = float(os.getenv("LIVE_MAX_TOTAL_CAPITAL", "15.00"))
        if not 0 < self.per_order_budget <= self.maximum_capital <= 15.0:
            raise RuntimeError("Live capital limits must satisfy 0 < order <= total <= 15")
        self.client = KalshiAccountClient()
        self.ledger = LiveRiskLedger(ledger_path, self.maximum_capital)

    def account_status(self) -> dict:
        return {
            "available_balance": self.client.available_balance(),
            "strategy_committed": self.ledger.committed(),
            "strategy_remaining": self.maximum_capital - self.ledger.committed(),
            "account_positions": self.client.positions(),
        }

    def execute(
        self, *, trigger_key: str, game_pk: int, ticker: str,
        price: float, settlement_probability: float,
        original_bet_size: float, original_minimum_expected_pnl: float,
        minimum_seconds_between_entries: float,
    ) -> LiveFill:
        count = contracts_for_budget(price, self.per_order_budget)
        client_id = self.ledger.reserve(
            trigger_key, game_pk, ticker, self.per_order_budget,
            minimum_seconds_between_entries,
        )
        if client_id is None:
            return LiveFill(False, "", ticker, reason="risk_limit_cooldown_or_duplicate")
        fee = taker_fee(count, price)
        capital = count * price + fee
        scaled_minimum = original_minimum_expected_pnl * capital / original_bet_size
        expected = count * (settlement_probability - price) - fee
        if count <= 0 or expected + 1e-9 < scaled_minimum:
            fill = LiveFill(False, client_id, ticker, reason="scaled_value_check")
            self.ledger.finish(fill)
            return fill
        balance = self.client.available_balance()
        if balance + 1e-9 < capital:
            fill = LiveFill(False, client_id, ticker, reason="insufficient_account_balance")
            self.ledger.finish(fill)
            return fill
        try:
            result = self.client.create_fill_or_kill(ticker, count, price, client_id)
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else 0
            self.ledger.mark_error(
                client_id,
                definite_rejection=400 <= status < 500 and status != 409,
            )
            raise
        except requests.RequestException:
            self.ledger.mark_error(client_id, definite_rejection=False)
            raise
        filled = float(result.get("fill_count") or 0)
        if filled <= 0:
            fill = LiveFill(False, client_id, ticker, reason="fill_or_kill_not_filled")
        else:
            average_price = float(result["average_fill_price"])
            fee_per_contract = float(result.get("average_fee_paid") or 0)
            fill = LiveFill(
                True, client_id, ticker, filled, average_price,
                filled * fee_per_contract, "filled",
            )
            if fill.capital > self.per_order_budget + 1e-6:
                raise RuntimeError("Exchange fill exceeded configured per-order capital")
        self.ledger.finish(fill)
        return fill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["account-status"])
    parser.add_argument(
        "--ledger", type=Path,
        default=Path(os.getenv("LIVE_RISK_DB", "/app/live-state/risk.sqlite3")),
    )
    args = parser.parse_args()
    client = KalshiAccountClient()
    ledger = LiveRiskLedger(
        args.ledger, float(os.getenv("LIVE_MAX_TOTAL_CAPITAL", "15"))
    )
    print(json.dumps({
        "available_balance": client.available_balance(),
        "strategy_committed": ledger.committed(),
        "strategy_remaining": ledger.maximum_capital - ledger.committed(),
        "open_positions": client.positions(),
    }, indent=2))


if __name__ == "__main__":
    main()
