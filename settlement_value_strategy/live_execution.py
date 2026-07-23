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
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

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
        order_side: str = "bid", reduce_only: bool = False,
    ) -> dict:
        return self.request(
            "POST", "/portfolio/events/orders",
            json={
                "ticker": ticker,
                "client_order_id": client_order_id,
                "side": order_side,
                "count": f"{count:.2f}",
                "price": f"{price:.4f}",
                "time_in_force": "fill_or_kill",
                "self_trade_prevention_type": "taker_at_cross",
                "post_only": False,
                "cancel_order_on_pause": True,
                "reduce_only": reduce_only,
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
    def __init__(self, path: Path, maximum_capital: float | None):
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
                settlement_probability REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS live_exits (
                client_order_id TEXT PRIMARY KEY,
                trigger_key TEXT NOT NULL UNIQUE,
                entry_client_order_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                contracts REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS execution_attempts (
                attempt_id TEXT PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )""")
            columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(live_orders)"
                ).fetchall()
            }
            if "settlement_probability" not in columns:
                connection.execute(
                    "ALTER TABLE live_orders ADD COLUMN "
                    "settlement_probability REAL NOT NULL DEFAULT 0"
                )

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

    def pending(self, connection: sqlite3.Connection | None = None) -> float:
        owns = connection is None
        connection = connection or self._connect()
        try:
            row = connection.execute(
                "SELECT COALESCE(SUM(reserved_capital), 0) "
                "FROM live_orders WHERE status='pending'"
            ).fetchone()
            return float(row[0])
        finally:
            if owns:
                connection.close()

    def reserve(
        self, trigger_key: str, game_pk: int, ticker: str, budget: float,
        minimum_seconds_between_entries: float,
        settlement_probability: float = 0.0,
        available_cash: float | None = None,
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
            if self.maximum_capital is None:
                if (
                    available_cash is None
                    or not math.isfinite(available_cash)
                    or self.pending(connection) + budget > available_cash + 1e-9
                ):
                    return None
            elif self.committed(connection) + budget > self.maximum_capital + 1e-9:
                return None
            stamp = now.isoformat()
            connection.execute(
                """INSERT INTO live_orders (
                    client_order_id,trigger_key,game_pk,ticker,reserved_capital,
                    committed_capital,contracts,price,fee,
                    settlement_probability,status,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (client_id, trigger_key, game_pk, ticker, budget, 0.0, 0.0,
                 0.0, 0.0, settlement_probability, "pending", stamp, stamp),
            )
        return client_id

    def filled_for_game(self, game_pk: int) -> list[dict]:
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM live_orders WHERE game_pk=? AND status='filled'",
                (game_pk,),
            ).fetchall()
        return [dict(row) for row in rows]

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

    def reserve_exit(
        self, trigger_key: str, entry_client_order_id: str, ticker: str,
        contracts: float, price: float,
    ) -> str | None:
        digest = hashlib.sha256(f"exit|{ticker}|{trigger_key}".encode()).hexdigest()[:24]
        client_id = f"hx-{digest}"
        stamp = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT status FROM live_exits WHERE trigger_key=?", (trigger_key,)
            ).fetchone()
            if existing:
                if existing[0] == "not_filled":
                    connection.execute(
                        "DELETE FROM live_exits WHERE trigger_key=?", (trigger_key,)
                    )
                else:
                    return None
            connection.execute(
                "INSERT INTO live_exits VALUES (?,?,?,?,?,?,?,?,?,?)",
                (client_id, trigger_key, entry_client_order_id, ticker,
                 contracts, price, 0.0, "pending", stamp, stamp),
            )
        return client_id

    def finish_exit(self, client_id: str, filled: bool, fee: float) -> None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT entry_client_order_id FROM live_exits "
                "WHERE client_order_id=?", (client_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Unknown live exit reservation")
            status = "filled" if filled else "not_filled"
            stamp = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "UPDATE live_exits SET status=?,fee=?,updated_at=? "
                "WHERE client_order_id=?", (status, fee, stamp, client_id),
            )
            if filled:
                connection.execute(
                    "UPDATE live_orders SET status='closed',updated_at=? "
                    "WHERE client_order_id=?", (stamp, row[0]),
                )

    def close_entry(self, entry_client_order_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE live_orders SET status='closed',updated_at=? "
                "WHERE client_order_id=? AND status='filled'",
                (datetime.now(timezone.utc).isoformat(), entry_client_order_id),
            )

    def record_attempt(self, payload: dict) -> None:
        """Persist the latest state of an execution attempt for live auditing."""
        stamp = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO execution_attempts VALUES (?,?,?)",
                (payload["attempt_id"], stamp, json.dumps(payload, sort_keys=True)),
            )


class LiveExecutor:
    def __init__(self, ledger_path: Path):
        if os.getenv("LIVE_TRADING_ENABLED") != REAL_MONEY_ACK:
            raise RuntimeError("Real-money trading acknowledgement is missing")
        self.per_order_budget = float(os.getenv("LIVE_MAX_ORDER_CAPITAL", "0.75"))
        total_cap = os.getenv("LIVE_MAX_TOTAL_CAPITAL", "ALL_LIQUID_CASH")
        self.maximum_capital = (
            None if total_cap.upper() == "ALL_LIQUID_CASH" else float(total_cap)
        )
        if not (
            math.isfinite(self.per_order_budget)
            and 0 < self.per_order_budget
            and (
                self.maximum_capital is None
                or (
                    math.isfinite(self.maximum_capital)
                    and self.per_order_budget <= self.maximum_capital
                )
            )
        ):
            raise RuntimeError(
                "Live capital limits must be finite and satisfy "
                "0 < order <= total"
            )
        self.client = KalshiAccountClient()
        self.ledger = LiveRiskLedger(ledger_path, self.maximum_capital)

    def account_status(self) -> dict:
        available = self.client.available_balance()
        remaining = (
            max(0.0, available - self.ledger.pending())
            if self.maximum_capital is None
            else self.maximum_capital - self.ledger.committed()
        )
        return {
            "available_balance": available,
            "strategy_committed": self.ledger.committed(),
            "strategy_remaining": remaining,
            "account_positions": self.client.positions(),
        }

    def execute(
        self, *, trigger_key: str, game_pk: int, ticker: str,
        price: float, settlement_probability: float,
        original_bet_size: float, original_minimum_expected_pnl: float,
        minimum_seconds_between_entries: float,
        minimum_probability_edge: float = 0.0,
        strategy: str = "settlement_value",
        signal_time: datetime | None = None,
        signal_price: float | None = None,
        edge_at_submission: float | None = None,
        order_budget: float | None = None,
    ) -> LiveFill:
        effective_budget = (
            self.per_order_budget if order_budget is None else float(order_budget)
        )
        if not (
            math.isfinite(effective_budget)
            and 0 < effective_budget <= self.per_order_budget
        ):
            raise ValueError(
                "Order budget must be positive and no greater than the live cap"
            )
        decision_time = datetime.now(timezone.utc)
        attempt = {
            "attempt_id": str(uuid.uuid4()), "strategy": strategy,
            "trigger_key": trigger_key, "game_pk": game_pk, "ticker": ticker,
            "signal_time": signal_time.isoformat() if signal_time else None,
            "signal_price": signal_price, "decision_time": decision_time.isoformat(),
            "submission_time": None, "submission_latency_ms": None,
            "submission_price": price,
            "order_budget": effective_budget,
            "edge_at_submission": (
                settlement_probability - price
                if edge_at_submission is None else edge_at_submission
            ),
            "status": "evaluating", "fill_price": None, "contracts": 0.0,
            "fee": 0.0, "reason": None,
        }
        self.ledger.record_attempt(attempt)
        count = contracts_for_budget(price, effective_budget)
        available_cash = self.client.available_balance()
        client_id = self.ledger.reserve(
            trigger_key, game_pk, ticker, effective_budget,
            minimum_seconds_between_entries, settlement_probability,
            available_cash=available_cash,
        )
        if client_id is None:
            attempt.update(status="not_submitted", reason="risk_limit_cooldown_or_duplicate")
            self.ledger.record_attempt(attempt)
            return LiveFill(False, "", ticker, reason="risk_limit_cooldown_or_duplicate")
        fee = taker_fee(count, price)
        capital = count * price + fee
        scaled_minimum = original_minimum_expected_pnl * capital / original_bet_size
        expected = count * (settlement_probability - price) - fee
        if (
            count <= 0
            or settlement_probability - price < minimum_probability_edge
            or expected + 1e-9 < scaled_minimum
        ):
            fill = LiveFill(False, client_id, ticker, reason="scaled_value_check")
            self.ledger.finish(fill)
            attempt.update(status="not_submitted", reason=fill.reason)
            self.ledger.record_attempt(attempt)
            return fill
        balance = self.client.available_balance()
        if balance + 1e-9 < capital:
            fill = LiveFill(False, client_id, ticker, reason="insufficient_account_balance")
            self.ledger.finish(fill)
            attempt.update(status="not_submitted", reason=fill.reason)
            self.ledger.record_attempt(attempt)
            return fill
        submitted_at = datetime.now(timezone.utc)
        latency_origin = signal_time or decision_time
        attempt.update(
            status="submitted", submission_time=submitted_at.isoformat(),
            submission_latency_ms=max(
                0.0, (submitted_at - latency_origin).total_seconds() * 1000.0
            ), contracts=count,
        )
        self.ledger.record_attempt(attempt)
        try:
            result = self.client.create_fill_or_kill(ticker, count, price, client_id)
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else 0
            self.ledger.mark_error(
                client_id,
                definite_rejection=400 <= status < 500 and status != 409,
            )
            attempt.update(status="submission_error", reason=f"http_{status}")
            self.ledger.record_attempt(attempt)
            raise
        except requests.RequestException:
            self.ledger.mark_error(client_id, definite_rejection=False)
            attempt.update(status="submission_error", reason="transport_error")
            self.ledger.record_attempt(attempt)
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
            if fill.capital > effective_budget + 1e-6:
                raise RuntimeError("Exchange fill exceeded configured per-order capital")
        self.ledger.finish(fill)
        attempt.update(
            status="filled" if fill.filled else "not_filled",
            fill_price=fill.price if fill.filled else None,
            contracts=fill.contracts if fill.filled else count,
            fee=fill.fee, reason=fill.reason,
        )
        self.ledger.record_attempt(attempt)
        return fill

    def execute_exit(
        self, *, trigger_key: str, entry_client_order_id: str,
        ticker: str, contracts: float, price: float,
    ) -> LiveFill:
        client_id = self.ledger.reserve_exit(
            trigger_key, entry_client_order_id, ticker, contracts, price
        )
        if client_id is None:
            return LiveFill(False, "", ticker, reason="duplicate_exit")
        try:
            result = self.client.create_fill_or_kill(
                ticker, contracts, price, client_id,
                order_side="ask", reduce_only=True,
            )
        except requests.RequestException:
            # Leave an ambiguous exit pending. A duplicate will be rejected.
            raise
        filled = float(result.get("fill_count") or 0)
        if filled <= 0:
            fill = LiveFill(False, client_id, ticker, reason="fill_or_kill_not_filled")
            self.ledger.finish_exit(client_id, False, 0.0)
            return fill
        average_price = float(result["average_fill_price"])
        fee = filled * float(result.get("average_fee_paid") or 0)
        fill = LiveFill(True, client_id, ticker, filled, average_price, fee, "filled")
        self.ledger.finish_exit(client_id, True, fee)
        return fill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["account-status"])
    parser.add_argument(
        "--ledger", type=Path,
        default=Path(os.getenv("LIVE_RISK_DB", "/app/live-state/risk.sqlite3")),
    )
    args = parser.parse_args()
    executor = LiveExecutor(args.ledger)
    status = executor.account_status()
    status["open_positions"] = status.pop("account_positions")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
