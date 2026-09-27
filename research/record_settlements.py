"""Observe public market settlement state without trading or using account APIs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
import uuid

import requests

FIELDS = ("ticker", "status", "result", "settlement_ts", "settlement_value_dollars", "close_time", "updated_time")


def fetch_market(ticker):
    sent = time.monotonic_ns()
    try:
        response = requests.get(f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}", timeout=15)
        response.raise_for_status()
        market = response.json()["market"]
        if market.get("ticker") != ticker:
            raise ValueError("Market response ticker mismatch")
        return {"type": "market_settlement_observation", "request_sent_monotonic_ns": sent,
                "received_monotonic_ns": time.monotonic_ns(),
                "received_at": datetime.now(timezone.utc).isoformat(),
                "market": {field: market.get(field) for field in FIELDS}}
    except (requests.RequestException, ValueError, KeyError) as error:
        return {"type": "market_settlement_error", "ticker": ticker,
                "request_sent_monotonic_ns": sent, "received_monotonic_ns": time.monotonic_ns(),
                "received_at": datetime.now(timezone.utc).isoformat(), "error": str(error)}


def record(args):
    slate = json.loads(args.slate.read_text())
    tickers = sorted({g[key] for g in slate["games"] for key in ("market_ticker", "away_market_ticker")})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"settlements_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.jsonl.gz"
    deadline, previous = time.monotonic() + args.duration, {}
    with gzip.open(path, "xt", compresslevel=3) as stream:
        count = 0

        def write(row):
            nonlocal count
            count += 1
            stream.write(json.dumps({"record_number": count, "recorded_at": datetime.now(timezone.utc).isoformat(),
                "recorded_monotonic_ns": time.monotonic_ns(), **row}, separators=(",", ":")) + "\n")

        write({"type": "capture_start", "stream": "public_market_settlements", "tickers": tickers,
               "orders_enabled": False})
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                while time.monotonic() < deadline:
                    # A fixed, low-rate polling batch. Timestamps preserve
                    # receipt separately from the later journal write time.
                    rows = list(pool.map(fetch_market, tickers))
                    for row in sorted(rows, key=lambda r: r["received_monotonic_ns"]):
                        if row["type"] == "market_settlement_observation":
                            ticker = row["market"]["ticker"]
                            identity = json.dumps(row["market"], sort_keys=True)
                            if previous.get(ticker) == identity:
                                row = {**row, "market": {"ticker": ticker}, "unchanged": True}
                            previous[ticker] = identity
                        write(row)
                    stream.flush()
                    print(json.dumps({"file": str(path), "observed_markets": len(rows),
                        "errors": sum(r["type"] == "market_settlement_error" for r in rows),
                        "finalized": sum((r.get("market") or {}).get("status") == "finalized" for r in rows)}), flush=True)
                    next_poll = min(time.monotonic() + args.interval, deadline)
                    while time.monotonic() < next_poll:
                        time.sleep(min(30., max(0., next_poll - time.monotonic())))
        finally:
            write({"type": "capture_end"})
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/forward_settlements"))
    parser.add_argument("--duration", type=float, default=21600)
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if not 60 <= args.duration <= 86400 or not 60 <= args.interval <= 3600:
        parser.error("Duration must be 60–86400 seconds and interval 60–3600 seconds")
    record(args)


if __name__ == "__main__":
    main()
