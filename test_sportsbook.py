import argparse
import copy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from research.sportsbook import OddsTape, fetch, implied_probability, match_events, normalize, record, timestamp


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


BASE = timestamp("2026-09-27T19:00:00Z")
SLATE = {"games": [{"game_pk": 1, "home_code": "WSH", "away_code": "NYM", "scheduled_time": iso(BASE + 300)}]}


def odds_row(received=BASE, updated=None, home_price=1.9, away_price=1.9):
    updated = received if updated is None else updated
    books = [{"key": key, "last_update": iso(updated), "markets": [{"key": "h2h", "outcomes": [
        {"name": "Washington Nationals", "price": home_price}, {"name": "New York Mets", "price": away_price}]}]}
        for key in ("book1", "book2", "book3")]
    return {"type": "sportsbook_snapshot", "provider": "the_odds_api", "odds_format": "decimal",
        "request_sent_at": iso(received - 1), "received_at": iso(received), "recorded_at": iso(received),
        "events": [{"id": "E1", "sport_key": "baseball_mlb", "home_team": "Washington Nationals",
                    "away_team": "New York Mets", "commence_time": iso(BASE + 300), "bookmakers": books}]}


class SportsbookTests(unittest.TestCase):
    def test_remove_margin_with_both_outcomes(self):
        tape = OddsTape([odds_row(home_price=1.5, away_price=2.8)], SLATE)
        result = tape.reference(1, BASE + 1)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["home_probability"], (1 / 1.5) / (1 / 1.5 + 1 / 2.8))

    def test_american_and_decimal_conversion(self):
        self.assertAlmostEqual(implied_probability(-150, "american"), .6)
        self.assertAlmostEqual(implied_probability(200, "american"), 1 / 3)
        for value in (0, 1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                implied_probability(value, "decimal")

    def test_no_later_received_quote_can_rewrite_the_past(self):
        first, later = odds_row(), odds_row(BASE + 20, home_price=1.5, away_price=2.8)
        short, full = OddsTape([first], SLATE), OddsTape([first, later], SLATE)
        self.assertEqual(short.reference(1, BASE + 10), full.reference(1, BASE + 10))
        self.assertFalse(full.reference(1, BASE - 1)["valid"])
        self.assertNotEqual(full.reference(1, BASE + 20)["home_probability"], .5)

    def test_journal_write_delay_controls_availability(self):
        row = odds_row()
        row["recorded_at"] = iso(BASE + 5)
        tape = OddsTape([row], SLATE)
        self.assertFalse(tape.reference(1, BASE + 1)["valid"])
        self.assertTrue(tape.reference(1, BASE + 5)["valid"])
        self.assertFalse(tape.reference(1, BASE + 5, strictly_before=True)["valid"])

    def test_refetch_does_not_refresh_provider_age(self):
        tape = OddsTape([odds_row(), odds_row(BASE + 190, updated=BASE)], SLATE)
        self.assertEqual(tape.reference(1, BASE + 191)["reason"], "insufficient_fresh_books")

    def test_omitted_event_does_not_carry_forward(self):
        empty = odds_row(BASE + 10)
        empty["events"] = []
        tape = OddsTape([odds_row(), empty], SLATE)
        self.assertFalse(tape.reference(1, BASE + 11)["valid"])
        self.assertTrue(tape.reference(1, BASE + 9)["valid"])

    def test_omitted_books_do_not_carry_forward(self):
        latest = odds_row(BASE + 10)
        latest["events"][0]["bookmakers"].pop()
        tape = OddsTape([odds_row(), latest], SLATE)
        self.assertEqual(tape.reference(1, BASE + 11)["fresh_books"], 2)

    def test_incomplete_outcome_and_duplicate_book_are_rejected(self):
        row = odds_row()
        row["events"][0]["bookmakers"][0]["markets"][0]["outcomes"].pop()
        self.assertFalse(OddsTape([row], SLATE).reference(1, BASE)["valid"])
        row = odds_row()
        row["events"][0]["bookmakers"].append(copy.deepcopy(row["events"][0]["bookmakers"][0]))
        self.assertFalse(OddsTape([row], SLATE).reference(1, BASE)["valid"])

    def test_future_provider_clock_cannot_supply_consensus(self):
        self.assertFalse(OddsTape([odds_row(updated=BASE + 1)], SLATE).reference(1, BASE)["valid"])
        self.assertEqual(normalize(odds_row(updated=BASE + 1))["rejected"]["invalid_moneyline"], 3)

    def test_bad_envelope_and_backwards_time_fail_closed(self):
        row = odds_row()
        row["recorded_at"] = iso(BASE - 1)
        with self.assertRaises(ValueError):
            normalize(row)
        with self.assertRaises(ValueError):
            OddsTape([odds_row(BASE + 2), odds_row()], SLATE)
        with self.assertRaises(ValueError):
            timestamp("2026-09-27T19:00:00")

    def test_disagreement_disables_consensus(self):
        row = odds_row()
        outcomes = row["events"][0]["bookmakers"][0]["markets"][0]["outcomes"]
        outcomes[0]["price"], outcomes[1]["price"] = 1.5, 2.8
        self.assertEqual(OddsTape([row], SLATE).reference(1, BASE)["reason"], "books_disagree")

    def test_doubleheader_requires_mutually_unique_time_match(self):
        events = normalize(odds_row())["events"]
        ambiguous = {"games": SLATE["games"] + [{**SLATE["games"][0], "game_pk": 2, "scheduled_time": iso(BASE + 3600)}]}
        self.assertEqual(match_events(events, ambiguous), {})
        clear = {"games": SLATE["games"] + [{**SLATE["games"][0], "game_pk": 2, "scheduled_time": iso(BASE + 18000)}]}
        self.assertEqual(match_events(events, clear), {"E1": 1})
        events.append({**events[0], "event_id": "DUPLICATE_PROVIDER_EVENT"})
        self.assertEqual(match_events(events, SLATE), {})

    def test_wrong_date_or_unknown_team_cannot_match(self):
        row = odds_row()
        row["events"][0]["commence_time"] = iso(BASE - 86400)
        self.assertFalse(OddsTape([row], SLATE).reference(1, BASE)["valid"])
        row["events"][0]["home_team"] = "Something Similar Nationals"
        self.assertEqual(normalize(row)["rejected"]["invalid_event"], 1)

    def test_unmapped_schedule_game_still_blocks_ambiguous_identity(self):
        slate = {**SLATE, "schedule_identity_evidence": [{"gamePk": 2, "gameDate": iso(BASE + 3600),
            "teams": {"home": {"team": {"name": "Washington Nationals"}},
                      "away": {"team": {"name": "New York Mets"}}}}]}
        self.assertFalse(OddsTape([odds_row()], slate).reference(1, BASE)["valid"])

    def test_request_errors_do_not_leak_authenticated_url(self):
        class BrokenSession:
            def get(self, *args, **kwargs):
                raise requests.RequestException("https://provider?apiKey=SECRET")
        result = fetch(BrokenSession(), "SECRET")
        self.assertEqual(result["type"], "sportsbook_error")
        self.assertNotIn("SECRET", json.dumps(result))

    def test_unconfigured_collector_does_not_contact_provider(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {}, clear=True), patch("requests.Session") as session:
            args = argparse.Namespace(command="record", output_dir=Path(temp))
            with self.assertRaisesRegex(ValueError, "not configured"):
                record(args)
            session.assert_not_called()

    def test_raw_import_is_received_now_not_backdated(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "raw.json"
            source.write_text(json.dumps(odds_row()["events"]))
            args = argparse.Namespace(command="import", input=source, odds_format="decimal", output_dir=Path(temp) / "out")
            with patch("research.sportsbook.utc_now", return_value=iso(BASE + 60)):
                path = record(args)
            with gzip.open(path, "rt") as stream:
                rows = [json.loads(line) for line in stream]
            tape = OddsTape(rows, SLATE)
            self.assertFalse(tape.reference(1, BASE + 30)["valid"])
            self.assertTrue(tape.reference(1, BASE + 60)["valid"])


if __name__ == "__main__":
    unittest.main()
