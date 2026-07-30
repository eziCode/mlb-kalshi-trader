from datetime import date
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from live_trading.portfolio_paths import hit_reversion_path, settlement_value_path


class PortfolioPathTest(unittest.TestCase):
    def test_paths_rotate_with_slate_date_not_process_start_date(self):
        environment = {
            "SETTLEMENT_VALUE_PORTFOLIO_DIR": "/state",
            "HIT_REVERSION_PORTFOLIO_DIR": "/state/hit-reversion",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(
                settlement_value_path(date(2026, 7, 30), Path("/fallback")),
                Path("/state/settlement_value_portfolio_2026-07-30.sqlite3"),
            )
            self.assertEqual(
                hit_reversion_path(date(2026, 7, 31), Path("/fallback")),
                Path("/state/hit-reversion/hit_reversion_portfolio_2026-07-31.sqlite3"),
            )

    def test_explicit_database_remains_available_for_single_slate_runs(self):
        with patch.dict(
            os.environ, {"PAPER_PORTFOLIO_DB": "/custom/book.sqlite3"}, clear=True
        ):
            self.assertEqual(
                settlement_value_path(date(2026, 7, 30), Path("/fallback")),
                Path("/custom/book.sqlite3"),
            )
            self.assertEqual(
                hit_reversion_path(date(2026, 7, 30), Path("/fallback")),
                Path("/custom/book.sqlite3"),
            )
