from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from settlement_value_strategy.live_execution import (
    LiveExecutor, LiveRiskLedger, contracts_for_budget,
)
from settlement_value_strategy.strategy import taker_fee


class FakeClient:
    def __init__(self, balance: float = 15.0):
        self.balance = balance
        self.orders = []

    def available_balance(self):
        return self.balance

    def create_fill_or_kill(
        self, ticker, count, price, client_order_id,
        order_side="bid", reduce_only=False,
    ):
        self.orders.append((
            ticker, count, price, client_order_id, order_side, reduce_only,
        ))
        return {
            "fill_count": f"{count:.2f}",
            "average_fill_price": f"{price:.4f}",
            "average_fee_paid": f"{taker_fee(count, price) / count:.8f}",
        }


class LiveExecutionTests(unittest.TestCase):
    def test_executor_accepts_account_sized_total_with_two_dollar_orders(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {
                "LIVE_TRADING_ENABLED": "YES_I_UNDERSTAND_THIS_PLACES_REAL_ORDERS",
                "LIVE_MAX_ORDER_CAPITAL": "2",
                "LIVE_MAX_TOTAL_CAPITAL": "34.36",
            }),
            patch(
                "settlement_value_strategy.live_execution.KalshiAccountClient",
                return_value=FakeClient(balance=34.36),
            ),
        ):
            executor = LiveExecutor(Path(directory) / "risk.db")
        self.assertEqual(executor.per_order_budget, 2.0)
        self.assertEqual(executor.maximum_capital, 34.36)

    def test_contract_sizing_includes_fee_inside_75_cent_cap(self):
        count = contracts_for_budget(0.51, 0.75)
        self.assertLessEqual(count * 0.51 + taker_fee(count, 0.51), 0.75)
        larger = count + 0.01
        self.assertGreater(larger * 0.51 + taker_fee(larger, 0.51), 0.75)

    def test_ledger_hard_caps_total_reserved_capital(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = LiveRiskLedger(Path(directory) / "risk.db", 15.0)
            for game in range(20):
                self.assertIsNotNone(ledger.reserve(
                    f"trigger-{game}", game, f"ticker-{game}", 0.75, 200,
                ))
            self.assertIsNone(ledger.reserve(
                "trigger-21", 21, "ticker-21", 0.75, 200,
            ))
            self.assertAlmostEqual(ledger.committed(), 15.0)

    def test_execute_checks_balance_and_places_fill_or_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = LiveExecutor.__new__(LiveExecutor)
            executor.per_order_budget = 0.75
            executor.maximum_capital = 15.0
            executor.client = FakeClient()
            executor.ledger = LiveRiskLedger(Path(directory) / "risk.db", 15.0)
            fill = executor.execute(
                trigger_key="1:pitch", game_pk=1, ticker="TEST",
                price=0.51, settlement_probability=0.90,
                original_bet_size=10.0, original_minimum_expected_pnl=2.25,
                minimum_seconds_between_entries=200,
            )
            self.assertTrue(fill.filled)
            self.assertLessEqual(fill.capital, 0.75 + 1e-6)
            self.assertEqual(len(executor.client.orders), 1)

    def test_strategy_order_budget_can_be_lower_than_global_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = LiveExecutor.__new__(LiveExecutor)
            executor.per_order_budget = 2.0
            executor.maximum_capital = 34.0
            executor.client = FakeClient()
            executor.ledger = LiveRiskLedger(Path(directory) / "risk.db", 34.0)
            fill = executor.execute(
                trigger_key="hit:1", game_pk=1, ticker="TEST",
                price=0.51, settlement_probability=0.90,
                original_bet_size=10.0, original_minimum_expected_pnl=0.0,
                minimum_seconds_between_entries=0, order_budget=1.0,
                strategy="hit_reversion",
            )
            self.assertTrue(fill.filled)
            self.assertLessEqual(fill.capital, 1.0 + 1e-6)
            self.assertGreater(fill.capital, 0.90)

    def test_execute_refuses_when_real_balance_is_too_low(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = LiveExecutor.__new__(LiveExecutor)
            executor.per_order_budget = 0.75
            executor.maximum_capital = 15.0
            executor.client = FakeClient(balance=0.10)
            executor.ledger = LiveRiskLedger(Path(directory) / "risk.db", 15.0)
            fill = executor.execute(
                trigger_key="1:pitch", game_pk=1, ticker="TEST",
                price=0.51, settlement_probability=0.90,
                original_bet_size=10.0, original_minimum_expected_pnl=2.25,
                minimum_seconds_between_entries=200,
            )
            self.assertFalse(fill.filled)
            self.assertEqual(fill.reason, "insufficient_account_balance")
            self.assertFalse(executor.client.orders)

    def test_execute_rechecks_probability_edge_at_actual_price(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = LiveExecutor.__new__(LiveExecutor)
            executor.per_order_budget = 0.75
            executor.maximum_capital = 15.0
            executor.client = FakeClient()
            executor.ledger = LiveRiskLedger(Path(directory) / "risk.db", 15.0)
            fill = executor.execute(
                trigger_key="1:pitch", game_pk=1, ticker="TEST",
                price=0.76, settlement_probability=0.90,
                original_bet_size=10.0, original_minimum_expected_pnl=0.0,
                minimum_seconds_between_entries=200,
                minimum_probability_edge=0.15,
            )
            self.assertFalse(fill.filled)
            self.assertEqual(fill.reason, "scaled_value_check")
            self.assertFalse(executor.client.orders)

    def test_filled_orders_can_be_recovered_by_game(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = LiveRiskLedger(Path(directory) / "risk.db", 15.0)
            client_id = ledger.reserve("1:pitch", 1, "TEST", .75, 200, .9)
            ledger.finish(type("Fill", (), {
                "filled": True, "capital": .75, "contracts": 1.0,
                "price": .70, "fee": .05, "client_order_id": client_id,
            })())
            rows = ledger.filled_for_game(1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["settlement_probability"], .9)

    def test_reduce_only_exit_releases_shared_capital(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = LiveExecutor.__new__(LiveExecutor)
            executor.client = FakeClient()
            executor.ledger = LiveRiskLedger(Path(directory) / "risk.db", 15.0)
            entry_id = executor.ledger.reserve(
                "hr-entry", 1, "TEST", 2.0, 60, .8
            )
            executor.ledger.finish(type("Fill", (), {
                "filled": True, "capital": 2.0, "contracts": 3.0,
                "price": .60, "fee": .20, "client_order_id": entry_id,
            })())
            fill = executor.execute_exit(
                trigger_key="hr-exit", entry_client_order_id=entry_id,
                ticker="TEST", contracts=3.0, price=.70,
            )
            self.assertTrue(fill.filled)
            self.assertEqual(executor.client.orders[-1][-2:], ("ask", True))
            self.assertEqual(executor.ledger.committed(), 0.0)


if __name__ == "__main__":
    unittest.main()
