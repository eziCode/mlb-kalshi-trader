import sqlite3
from pathlib import Path
import tempfile
import unittest

from live_trading.portfolio_reporting import (
    format_live_portfolio_summary,
    read_strategy_book,
)


def make_book(path, *, starting=30.0, cash=25.0, positions=()):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE portfolio (id INTEGER PRIMARY KEY, "
            "starting_cash REAL, cash REAL)"
        )
        connection.execute("INSERT INTO portfolio VALUES (1, ?, ?)", (starting, cash))
        connection.execute(
            "CREATE TABLE positions (contracts REAL, mark_price REAL)"
        )
        connection.executemany("INSERT INTO positions VALUES (?, ?)", positions)


class FakeClient:
    def balance_snapshot(self):
        return {"cash": 26.0, "position_value": 17.24, "total": 43.24}


class PortfolioReportingTest(unittest.TestCase):
    def test_summary_uses_account_once_and_breaks_out_strategy_pnl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settlement = root / "settlement.sqlite3"
            hit = root / "hit.sqlite3"
            make_book(settlement, cash=35.0, positions=((2.0, 0.80),))
            make_book(hit, cash=26.0, positions=())
            summary = format_live_portfolio_summary(
                FakeClient(), settlement_value_db=settlement, hit_reversion_db=hit
            )
            self.assertIn("total=$43.24 cash=$26.00 positions=$17.24", summary)
            self.assertIn(
                "settlement_value: PnL=$+6.60 open_value=$1.60 open_positions=1",
                summary,
            )
            self.assertIn(
                "hit_reversion: PnL=$-4.00 open_value=$0.00 open_positions=0",
                summary,
            )
            self.assertEqual(summary.count("total="), 1)

    def test_missing_strategy_book_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "settlement.sqlite3"
            make_book(existing)
            summary = format_live_portfolio_summary(
                FakeClient(), settlement_value_db=existing,
                hit_reversion_db=root / "missing.sqlite3",
            )
            self.assertIn("hit_reversion: unavailable", summary)

    def test_strategy_snapshot_does_not_report_seed_cash_as_capital(self):
        with tempfile.TemporaryDirectory() as directory:
            book = Path(directory) / "book.sqlite3"
            make_book(book, starting=1000.0, cash=997.0, positions=((5.0, 0.70),))
            snapshot = read_strategy_book(book)
            self.assertIsNotNone(snapshot)
            self.assertAlmostEqual(snapshot.pnl, 0.5)
            self.assertAlmostEqual(snapshot.open_value, 3.5)
