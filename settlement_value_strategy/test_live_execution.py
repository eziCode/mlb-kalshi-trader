from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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

    def create_fill_or_kill(self, ticker, count, price, client_order_id):
        self.orders.append((ticker, count, price, client_order_id))
        return {
            "fill_count": f"{count:.2f}",
            "average_fill_price": f"{price:.4f}",
            "average_fee_paid": f"{taker_fee(count, price) / count:.8f}",
        }


class LiveExecutionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
