import gzip
import json
from pathlib import Path
import tempfile
import unittest

from research.settle_forward import observed_settlements, reconcile


def position(quantity=1., pending=0):
    return dict(game_pk=1, inventory=quantity, outstanding_orders=pending,
        liquidation_marked_pnl=-.03, realized_pnl=0., inventory_cost_per_contract=.52,
        fills=[dict(action="buy", quantity=quantity, opening_quantity=quantity, price=.50,
                    fee=.02 * quantity, ticker="M", received_seconds=1.)])


class ForwardSettlementTests(unittest.TestCase):
    def terminal(self, value=1.):
        return {"M": dict(value=value, exchange_time=10., observed_monotonic_ns=20_000_000_000,
                          observed_at="1970-01-01T00:00:20Z")}

    def test_payout_uses_actual_fractional_inventory_and_does_not_double_charge_fees(self):
        result = reconcile(position(.4), self.terminal(), 0., 0)
        self.assertAlmostEqual(result["settlement_proceeds"], .4)
        self.assertAlmostEqual(result["pnl"], .192)
        self.assertTrue(result["settled"])

    def test_nonbinary_settlement_value_is_preserved(self):
        result = reconcile(position(), self.terminal(.5), 0., 0)
        self.assertAlmostEqual(result["pnl"], -.02)

    def test_missing_exchange_settlement_keeps_the_recorded_bid_mark(self):
        result = reconcile(position(), {}, 0., 0)
        self.assertFalse(result["settled"])
        self.assertEqual(result["pnl"], -.03)

    def test_settlement_does_not_remove_pending_order_uncertainty(self):
        result = reconcile(position(pending=1), self.terminal(), 0., 0)
        self.assertEqual(result["outstanding_orders"], 1)

    def test_entry_after_exchange_settlement_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "after exchange settlement"):
            reconcile(position(), self.terminal(), 10., 0)

    def test_corrupt_inventory_cost_is_rejected(self):
        game = position()
        game["inventory_cost_per_contract"] = .01
        with self.assertRaisesRegex(ValueError, "costs do not reconcile"):
            reconcile(game, self.terminal(), 0., 0)

    def observations(self, rows):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settlements.gz"
            with gzip.open(path, "wt") as stream:
                for i, market in enumerate(rows):
                    stream.write(json.dumps(dict(type="market_settlement_observation",
                        recorded_monotonic_ns=(20 + i) * 1_000_000_000,
                        recorded_at=f"1970-01-01T00:00:{20+i}Z", market=market)) + "\n")
            return observed_settlements(path)

    def test_only_finalized_exchange_observations_count(self):
        market = dict(ticker="M", status="determined", result="yes", settlement_ts="1970-01-01T00:00:10Z",
                      settlement_value_dollars="1.0000")
        result, errors = self.observations([market])
        self.assertEqual(result, {})
        self.assertEqual(errors, 0)

    def test_duplicate_finalized_value_is_credited_only_once(self):
        market = dict(ticker="M", status="finalized", settlement_ts="1970-01-01T00:00:10Z",
                      settlement_value_dollars="1.0000")
        result, _ = self.observations([market, market])
        self.assertEqual(len(result), 1)
        self.assertEqual(result["M"]["observed_monotonic_ns"], 20_000_000_000)

    def test_conflicting_terminal_values_are_not_silently_rewritten(self):
        market = dict(ticker="M", status="finalized", settlement_ts="1970-01-01T00:00:10Z",
                      settlement_value_dollars="1.0000")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.observations([market, {**market, "settlement_value_dollars": "0.0000"}])


if __name__ == "__main__":
    unittest.main()
