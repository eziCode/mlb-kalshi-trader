"""Timestamped sportsbook observations and strictly as-of MLB consensus.

Read-only provider adapter; never places bets. Raw imports become available
NOW, not at the provider's historical quote timestamp. See NEXT_EXPERIMENT.md.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
from statistics import median
import time
import uuid
from zoneinfo import ZoneInfo

import requests


API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
EASTERN = ZoneInfo("America/New_York")
TEAM_NAMES = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH", "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}
ALIASES = {"ARI": "AZ", "CHW": "CWS", "OAK": "ATH", "WSN": "WSH", "SDP": "SD", "SFG": "SF", "TBR": "TB", "KCR": "KC"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("An explicit timezone-aware timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp has no timezone")
    return parsed.timestamp()


def team_code(name):
    code = TEAM_NAMES.get(name, name)
    code = ALIASES.get(code, code)
    if code not in set(TEAM_NAMES.values()):
        raise ValueError("Unrecognized MLB team")
    return code


def implied_probability(price, odds_format):
    value = float(price)
    if not math.isfinite(value):
        raise ValueError("Nonfinite odds")
    if odds_format == "decimal" and value > 1:
        return 1 / value
    if odds_format == "american" and abs(value) >= 100:
        return 100 / (value + 100) if value > 0 else -value / (100 - value)
    raise ValueError("Invalid odds or odds format")


def normalize(row):
    """One full snapshot replaces the previous snapshot, including omissions."""
    if row.get("provider") != "the_odds_api" or row.get("type") != "sportsbook_snapshot":
        raise ValueError("Unsupported sportsbook journal record")
    received, available = timestamp(row["received_at"]), timestamp(row["recorded_at"])
    if available < received or timestamp(row["request_sent_at"]) > received:
        raise ValueError("Invalid sportsbook receipt/journal clocks")
    events, rejected = [], Counter()
    if not isinstance(row.get("events"), list):
        raise ValueError("Expected a complete event list")
    ids = [e.get("id") for e in row["events"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate provider event identifiers")
    for raw in row["events"]:
        try:
            if raw.get("sport_key") != "baseball_mlb" or not raw.get("id"):
                raise ValueError("Wrong sport or missing event ID")
            home, away = team_code(raw["home_team"]), team_code(raw["away_team"])
            start = timestamp(raw["commence_time"])
            if home == away:
                raise ValueError("Identical home and away teams")
        except (ValueError, KeyError, TypeError):
            rejected["invalid_event"] += 1
            continue
        books = []
        keys = [b.get("key") for b in raw.get("bookmakers", [])]
        duplicated = {key for key in keys if keys.count(key) > 1}
        for book in raw.get("bookmakers", []):
            try:
                if not book.get("key") or book["key"] in duplicated:
                    raise ValueError("Duplicate or missing bookmaker")
                markets = [m for m in book.get("markets", []) if m.get("key") == "h2h"]
                if len(markets) != 1:
                    raise ValueError("Missing or ambiguous full-game moneyline")
                market = markets[0]
                # Prefer the market-specific update. Re-fetching an unchanged
                # quote never refreshes its provider timestamp.
                updated = timestamp(market.get("last_update") or book.get("last_update"))
                if updated > received:
                    raise ValueError("Provider timestamp is later than receipt")
                outcomes = market["outcomes"]
                names = [team_code(o["name"]) for o in outcomes]
                if len(names) != 2 or set(names) != {home, away}:
                    raise ValueError("Not a complete two-outcome moneyline")
                probabilities = {team_code(o["name"]): implied_probability(o["price"], row["odds_format"])
                                 for o in outcomes}
                overround = sum(probabilities.values())
                if not .98 <= overround <= 1.20:
                    raise ValueError("Implausible bookmaker margin")
                books.append({"bookmaker": book["key"], "updated_at": updated,
                              "home_probability": probabilities[home] / overround,
                              "overround": overround})
            except (ValueError, KeyError, TypeError):
                rejected["invalid_moneyline"] += 1
        events.append({"event_id": raw["id"], "home_code": home, "away_code": away,
                       "start": start, "books": books})
    return {"available_at": available, "received_at": received, "events": events,
            "rejected": dict(rejected)}


def match_events(events, slate, tolerance=90 * 60):
    """Mutually unique team/date/time matches; never zip doubleheaders."""
    # Unmapped schedule games must still compete for identity. Otherwise a
    # missing Kalshi leg could make the wrong doubleheader look unique.
    all_games = {g["game_pk"]: g for g in slate["games"]}
    mapped = set(all_games)
    for raw in slate.get("schedule_identity_evidence", []):
        if raw["gamePk"] in all_games:
            continue
        try:
            all_games[raw["gamePk"]] = {"game_pk": raw["gamePk"], "scheduled_time": raw["gameDate"],
                "home_code": team_code(raw["teams"]["home"]["team"]["name"]),
                "away_code": team_code(raw["teams"]["away"]["team"]["name"])}
        except (ValueError, KeyError, TypeError):
            raise ValueError("Incomplete schedule identity evidence") from None
    candidates = {}
    for event in events:
        possible = []
        for game in all_games.values():
            if (team_code(game["home_code"]), team_code(game["away_code"])) != (event["home_code"], event["away_code"]):
                continue
            scheduled = timestamp(game["scheduled_time"])
            same_date = datetime.fromtimestamp(scheduled, EASTERN).date() == datetime.fromtimestamp(event["start"], EASTERN).date()
            if same_date and abs(scheduled - event["start"]) <= tolerance:
                possible.append(game["game_pk"])
        candidates[event["event_id"]] = possible
    counts = Counter(pk for matches in candidates.values() for pk in matches)
    return {e["event_id"]: candidates[e["event_id"]][0] for e in events
            if len(candidates[e["event_id"]]) == 1 and counts[candidates[e["event_id"]][0]] == 1
            and candidates[e["event_id"]][0] in mapped}


class OddsTape:
    """Random-access as-of lookup; later snapshots cannot revise old features."""
    def __init__(self, rows, slate):
        self.snapshots, self.times = [], []
        self.rejected = Counter()
        for row in rows:
            if row.get("type") != "sportsbook_snapshot":
                continue
            snapshot = normalize(row)
            when = snapshot["available_at"]
            if self.times and when < self.times[-1]:
                raise ValueError("Sportsbook availability clock moved backwards")
            mapping = match_events(snapshot["events"], slate)
            snapshot["by_game"] = {mapping[e["event_id"]]: e for e in snapshot["events"] if e["event_id"] in mapping}
            self.rejected.update(snapshot["rejected"])
            self.rejected["unmatched_events"] += len(snapshot["events"]) - len(mapping)
            self.snapshots.append(snapshot)
            self.times.append(when)

    @classmethod
    def read(cls, path, slate):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as stream:
            return cls((json.loads(line) for line in stream), slate)

    def reference(self, game_pk, at, *, max_age=180., minimum_books=3, max_disagreement=.03, strictly_before=False):
        if not math.isfinite(at) or max_age <= 0 or minimum_books < 2 or not 0 <= max_disagreement < 1:
            raise ValueError("Invalid consensus controls")
        index = (bisect_left if strictly_before else bisect_right)(self.times, at) - 1
        if index < 0:
            return {"valid": False, "reason": "no_received_snapshot"}
        snapshot = self.snapshots[index]
        event = snapshot["by_game"].get(game_pk)
        if event is None:
            return {"valid": False, "reason": "event_missing_or_ambiguous"}
        fresh = [b for b in event["books"] if 0 <= at - b["updated_at"] <= max_age]
        if at - snapshot["received_at"] > max_age or len(fresh) < minimum_books:
            return {"valid": False, "reason": "insufficient_fresh_books", "fresh_books": len(fresh)}
        probabilities = [b["home_probability"] for b in fresh]
        disagreement = max(probabilities) - min(probabilities)
        if disagreement > max_disagreement + 1e-9:
            return {"valid": False, "reason": "books_disagree", "disagreement": disagreement}
        return {"valid": True, "home_probability": median(probabilities),
                "bookmakers": [b["bookmaker"] for b in fresh], "disagreement": disagreement,
                "oldest_quote_age": max(at - b["updated_at"] for b in fresh),
                "available_at": snapshot["available_at"], "provider_event_id": event["event_id"]}


def fetch(session, api_key, regions="us"):
    sent = utc_now()
    try:
        response = session.get(API_URL, params={"apiKey": api_key, "regions": regions,
            "markets": "h2h", "oddsFormat": "decimal", "dateFormat": "iso"}, timeout=15)
        received = utc_now()
        if response.status_code != 200:
            return {"type": "sportsbook_error", "error": "http_status", "status_code": response.status_code,
                    "request_sent_at": sent, "received_at": received}
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Invalid provider payload")
        return {"type": "sportsbook_snapshot", "provider": "the_odds_api", "odds_format": "decimal",
                "request_sent_at": sent, "received_at": received, "events": payload,
                "quota": {name: response.headers.get(name) for name in
                          ("x-requests-remaining", "x-requests-used", "x-requests-last")}}
    except (requests.RequestException, ValueError):
        # requests exceptions can contain the authenticated URL. Never log it.
        return {"type": "sportsbook_error", "error": "request_or_decode_failed",
                "request_sent_at": sent, "received_at": utc_now()}


def record(args):
    api_key = os.environ.get("ODDS_API_KEY")
    if args.command == "record" and not api_key:
        raise ValueError("ODDS_API_KEY is not configured; collector made no network request")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"odds_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.jsonl.gz"
    count = 0
    with gzip.open(path, "xt") as stream:
        def write(row):
            row = {**row, "recorded_at": utc_now(), "recorded_monotonic_ns": time.monotonic_ns()}
            if row["type"] == "sportsbook_snapshot":
                normalize(row)  # Validate the envelope before accepting it.
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
        write({"type": "capture_start", "provider": "the_odds_api", "orders_enabled": False,
               "source": "local_import_received_now" if args.command == "import" else "live_api"})
        if args.command == "import":
            now = utc_now()
            write({"type": "sportsbook_snapshot", "provider": "the_odds_api", "odds_format": args.odds_format,
                   "request_sent_at": now, "received_at": now, "events": json.loads(args.input.read_text())})
        else:
            with requests.Session() as session:
                for index in range(args.requests):
                    row = fetch(session, api_key, args.regions)
                    write(row)
                    count += 1
                    print(json.dumps({"file": str(path), "requests": count, "type": row["type"],
                                      "events": len(row.get("events", [])), "quota": row.get("quota")}), flush=True)
                    remaining = (row.get("quota") or {}).get("x-requests-remaining")
                    if row["type"] == "sportsbook_error" or (remaining is not None and int(remaining) <= args.minimum_remaining):
                        break
                    if index + 1 < args.requests:
                        deadline = time.monotonic() + args.interval
                        while time.monotonic() < deadline:
                            time.sleep(min(30., max(0., deadline - time.monotonic())))
        write({"type": "capture_end"})
    print(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["record", "import", "inspect"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--slate", type=Path)
    parser.add_argument("--at", help="Timezone-aware as-of time for inspection")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/sportsbook"))
    parser.add_argument("--odds-format", choices=["decimal", "american"], default="decimal")
    parser.add_argument("--requests", type=int, default=1, help="Hard request budget, at most 360")
    parser.add_argument("--interval", type=float, default=60.)
    parser.add_argument("--minimum-remaining", type=int, default=10)
    parser.add_argument("--regions", choices=["us", "uk", "eu", "au"], default="us")
    args = parser.parse_args()
    if not 1 <= args.requests <= 360 or not math.isfinite(args.interval) or args.interval < 15 or args.minimum_remaining < 0:
        parser.error("Invalid request budget, polling interval, or remaining quota")
    try:
        if args.command == "inspect":
            if not args.input or not args.slate or not args.at:
                parser.error("inspect requires --input, --slate and --at")
            slate = json.loads(args.slate.read_text())
            tape = OddsTape.read(args.input, slate)
            print(json.dumps({"snapshots": len(tape.times), "rejected": dict(tape.rejected),
                "games": {g["game_pk"]: tape.reference(g["game_pk"], timestamp(args.at)) for g in slate["games"]}}, indent=2))
        else:
            if args.command == "import" and not args.input:
                parser.error("import requires --input")
            record(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
