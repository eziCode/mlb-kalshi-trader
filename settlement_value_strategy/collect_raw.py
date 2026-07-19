"""Collect public Kalshi trades and MLB live-feed snapshots for explicit IDs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent


def get_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(
        url, params=params, timeout=20,
        headers={"User-Agent": "mispricing-strategy-data-collector/1.0"},
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-pk", required=True, type=int)
    parser.add_argument("--market-ticker", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "raw/api_snapshots")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output / str(args.game_pk) / timestamp
    destination.mkdir(parents=True, exist_ok=True)
    mlb = get_json(
        f"https://statsapi.mlb.com/api/v1.1/game/{args.game_pk}/feed/live"
    )
    trades = get_json(
        "https://external-api.kalshi.com/trade-api/v2/markets/trades",
        {"ticker": args.market_ticker, "limit": 1000},
    )
    orderbook = get_json(
        "https://external-api.kalshi.com/trade-api/v2/markets/"
        f"{args.market_ticker}/orderbook"
    )
    (destination / "mlb_feed.json").write_text(json.dumps(mlb))
    (destination / "kalshi_trades.json").write_text(json.dumps(trades))
    (destination / "kalshi_orderbook.json").write_text(json.dumps(orderbook))
    print(destination)


if __name__ == "__main__":
    main()

