"""Download trade-level Kalshi MLB market history.

This downloader uses ``GET /markets/trades`` (and ``GET /historical/trades``
when the requested fills are older than Kalshi's current cutoff) for the
``KXMLBGAME`` series. Each output row is one executed trade, preserving its
exact API timestamp, price, size, trade ID, and market metadata.

Trades provide much finer timestamps than one-minute candles, but they are
still an irregular event stream: a row exists only when a trade executes.
They do not reconstruct the bid/ask spread or order book between executions.

By default the script downloads games from the latest 60 UTC dates, excludes
block trades, caches each market separately for resumability, and writes a
single Parquet file under::

    data/raw/kalshi_live_market_logs/

Examples
--------
Download the default rolling window::

    .venv/bin/python data/raw/scripts/download_kalshi_high_fidelity_market_logs.py

Smoke-test two games without reusing cached trades::

    .venv/bin/python data/raw/scripts/download_kalshi_high_fidelity_market_logs.py --max-games 2 --refresh-trades --verbose

Download a fixed, reproducible game-date window (both dates inclusive)::

    .venv/bin/python data/raw/scripts/download_kalshi_high_fidelity_market_logs.py --start-date 2026-05-15 --end-date 2026-07-12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXMLBGAME"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/raw/kalshi_live_market_logs"

DEFAULT_DAYS = 60
PAGE_LIMIT = 1000
EVENT_PAGE_LIMIT = 200
REQUEST_SLEEP_SECONDS = 0.10
FINAL_MARKET_STATUSES = {"closed", "settled", "finalized"}

MONTH_ABBR_TO_NUMBER = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
EVENT_DATE_PATTERN = re.compile(r"^KXMLBGAME-(\d{2})([A-Z]{3})(\d{2})")

OUTPUT_COLUMNS = [
    "season",
    "game_date",
    "event_ticker",
    "market_ticker",
    "market_title",
    "yes_sub_title",
    "no_sub_title",
    "market_open_time",
    "market_close_time",
    "market_status",
    "market_result",
    "trade_id",
    "created_time",
    "created_time_raw",
    "created_ts",
    "yes_price_dollars",
    "no_price_dollars",
    "count_fp",
    "taker_side",
    "taker_outcome_side",
    "taker_book_side",
    "is_block_trade",
    "source_endpoint",
]


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "mlb-kalshi-trade-downloader/1.0"})
    return session


SESSION = make_session()


def api_get(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
) -> dict[str, Any] | None:
    url = f"{BASE_URL}{path}"
    if verbose:
        print(f"    GET {path} params={params}")
    response = SESSION.get(url, params=params, timeout=30)
    time.sleep(REQUEST_SLEEP_SECONDS)
    if response.status_code == 404:
        return None
    if not response.ok:
        print(
            f"    !! {response.status_code} {path}: {response.text[:500]}",
            file=sys.stderr,
        )
    response.raise_for_status()
    return response.json()


def fetch_all_pages(
    path: str,
    response_key: str,
    params: dict[str, Any],
    *,
    page_limit: int = PAGE_LIMIT,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    page = 0

    while True:
        page_params = {**params, "limit": page_limit}
        if cursor:
            page_params["cursor"] = cursor
        payload = api_get(path, page_params, verbose=verbose)
        if not payload:
            break

        batch = payload.get(response_key) or []
        rows.extend(batch)
        page += 1
        if verbose:
            print(f"      page {page}: +{len(batch):,} ({len(rows):,} total)")

        next_cursor = payload.get("cursor") or ""
        if not batch or not next_cursor:
            break
        if next_cursor in seen_cursors:
            raise RuntimeError(f"Pagination cursor repeated for {path}")
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return rows


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_game_date(event_ticker: str | None) -> date | None:
    if not event_ticker:
        return None
    match = EVENT_DATE_PATTERN.match(event_ticker)
    if not match:
        return None
    year, month_name, day = match.groups()
    month = MONTH_ABBR_TO_NUMBER.get(month_name)
    if month is None:
        return None
    return date(2000 + int(year), month, int(day))


def utc_day_start(value: date) -> datetime:
    return datetime.combine(value, datetime_time.min, tzinfo=timezone.utc)


def get_historical_cutoffs(verbose: bool) -> dict[str, Any]:
    return api_get("/historical/cutoff", verbose=verbose) or {}


def discover_events(
    start_date: date,
    end_date: date,
    *,
    verbose: bool,
) -> list[dict[str, Any]]:
    # Kalshi's events endpoint supports min_close_ts but not max_close_ts.
    # The ticker-date filter below applies the requested upper bound and is
    # authoritative for the game-date semantics.
    params = {
        "series_ticker": SERIES_TICKER,
        "with_nested_markets": "true",
        "min_close_ts": int((utc_day_start(start_date) - timedelta(days=2)).timestamp()),
    }
    events = fetch_all_pages(
        "/events",
        "events",
        params,
        page_limit=EVENT_PAGE_LIMIT,
        verbose=verbose,
    )
    selected = []
    for event in events:
        game_date = parse_game_date(event.get("event_ticker"))
        if game_date is not None and start_date <= game_date <= end_date:
            selected.append(event)
    return sorted(selected, key=lambda item: item.get("event_ticker", ""))


def fetch_event_markets(
    event: dict[str, Any],
    *,
    verbose: bool,
) -> list[dict[str, Any]]:
    nested = event.get("markets") or []
    if nested:
        return nested

    event_ticker = event["event_ticker"]
    for path in ("/historical/markets", "/markets"):
        markets = fetch_all_pages(
            path,
            "markets",
            {"event_ticker": event_ticker, "mve_filter": "exclude"},
            verbose=verbose,
        )
        if markets:
            return markets
    return []


def market_trade_window(market: dict[str, Any], game_date: date) -> tuple[int, int]:
    open_time = parse_iso_timestamp(market.get("open_time"))
    close_time = (
        parse_iso_timestamp(market.get("close_time"))
        or parse_iso_timestamp(market.get("latest_expiration_time"))
        or parse_iso_timestamp(market.get("expiration_time"))
    )
    if open_time is None:
        open_time = utc_day_start(game_date) - timedelta(days=7)
    if close_time is None:
        close_time = utc_day_start(game_date + timedelta(days=2))
    if close_time < open_time:
        raise ValueError(f"Market closes before it opens: {market.get('ticker')}")
    return int(open_time.timestamp()), int(close_time.timestamp())


def trade_cache_path(
    cache_dir: Path,
    ticker: str,
    include_block_trades: bool,
) -> Path:
    suffix = "all" if include_block_trades else "non_block"
    return cache_dir / f"{ticker}_{suffix}.json"


def fetch_market_trades(
    market: dict[str, Any],
    game_date: date,
    trades_cutoff: datetime | None,
    cache_dir: Path,
    *,
    include_block_trades: bool,
    refresh: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    ticker = market["ticker"]
    cache_file = trade_cache_path(cache_dir, ticker, include_block_trades)
    is_final = str(market.get("status", "")).lower() in FINAL_MARKET_STATUSES
    if cache_file.exists() and not refresh and is_final:
        cached = json.loads(cache_file.read_text())
        return cached.get("trades", [])

    min_ts, max_ts = market_trade_window(market, game_date)
    base_params: dict[str, Any] = {"ticker": ticker}
    if not include_block_trades:
        # Omitting this filter returns both regular and block trades.
        base_params["is_block_trade"] = "false"
    segments: list[tuple[str, str, int, int]] = []
    if trades_cutoff is None:
        segments.append(("/markets/trades", "live", min_ts, max_ts))
    else:
        cutoff_ts = int(trades_cutoff.timestamp())
        if min_ts <= cutoff_ts:
            segments.append(
                ("/historical/trades", "historical", min_ts, min(max_ts, cutoff_ts))
            )
        if max_ts >= cutoff_ts:
            segments.append(
                ("/markets/trades", "live", max(min_ts, cutoff_ts), max_ts)
            )

    trades_by_id: dict[str, dict[str, Any]] = {}
    for path, source, segment_start, segment_end in segments:
        if segment_start > segment_end:
            continue
        trades = fetch_all_pages(
            path,
            "trades",
            {
                **base_params,
                "min_ts": segment_start,
                "max_ts": segment_end,
            },
            verbose=verbose,
        )
        for trade in trades:
            trade = {**trade, "_source_endpoint": source}
            trade_id = str(trade.get("trade_id") or "")
            if not trade_id:
                # Defensive fallback; current Kalshi responses include IDs.
                trade_id = "|".join(
                    str(trade.get(field) or "")
                    for field in ("ticker", "created_time", "yes_price_dollars", "count_fp")
                )
                trade["trade_id"] = trade_id
            trades_by_id[trade_id] = trade

    trades = sorted(
        trades_by_id.values(),
        key=lambda item: (item.get("created_time", ""), item.get("trade_id", "")),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "ticker": ticker,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "include_block_trades": include_block_trades,
        "trades": trades,
    }))
    return trades


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def trades_to_rows(
    trades: list[dict[str, Any]],
    market: dict[str, Any],
    game_date: date,
) -> list[dict[str, Any]]:
    rows = []
    open_time = parse_iso_timestamp(market.get("open_time"))
    close_time = (
        parse_iso_timestamp(market.get("close_time"))
        or parse_iso_timestamp(market.get("latest_expiration_time"))
    )
    for trade in trades:
        raw_created_time = trade.get("created_time")
        created_time = parse_iso_timestamp(raw_created_time)
        rows.append({
            "season": game_date.year,
            "game_date": game_date.isoformat(),
            "event_ticker": market.get("event_ticker"),
            "market_ticker": market.get("ticker"),
            "market_title": market.get("title"),
            "yes_sub_title": market.get("yes_sub_title"),
            "no_sub_title": market.get("no_sub_title"),
            "market_open_time": open_time,
            "market_close_time": close_time,
            "market_status": market.get("status"),
            "market_result": market.get("result"),
            "trade_id": trade.get("trade_id"),
            "created_time": created_time,
            "created_time_raw": raw_created_time,
            "created_ts": created_time.timestamp() if created_time else None,
            "yes_price_dollars": to_float(trade.get("yes_price_dollars")),
            "no_price_dollars": to_float(trade.get("no_price_dollars")),
            "count_fp": to_float(trade.get("count_fp")),
            "taker_side": trade.get("taker_side"),
            "taker_outcome_side": trade.get("taker_outcome_side"),
            "taker_book_side": trade.get("taker_book_side"),
            "is_block_trade": trade.get("is_block_trade"),
            "source_endpoint": trade.get("_source_endpoint"),
        })
    return rows


def output_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame[OUTPUT_COLUMNS]
    return frame.sort_values(
        ["game_date", "event_ticker", "market_ticker", "created_time", "trade_id"]
    ).drop_duplicates("trade_id").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument(
        "--max-games",
        type=int,
        help="Limit the number of events after date filtering (for smoke tests).",
    )
    parser.add_argument(
        "--include-block-trades",
        action="store_true",
        help="Include block trades, which may not represent the public order book.",
    )
    parser.add_argument("--refresh-markets", action="store_true")
    parser.add_argument("--refresh-trades", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be at least 1")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    today = datetime.now(timezone.utc).date()
    end_date = args.end_date or today
    start_date = args.start_date or (end_date - timedelta(days=args.days - 1))
    if start_date > end_date:
        raise ValueError("start date must be on or before end date")

    output_dir: Path = args.output_dir
    cache_dir = output_dir / "cache"
    trades_cache_dir = cache_dir / "trades"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    window_name = f"{start_date.isoformat()}_{end_date.isoformat()}"
    events_cache = cache_dir / f"events_{window_name}.json"
    if events_cache.exists() and not args.refresh_markets:
        events = json.loads(events_cache.read_text())
        print(f"Using {len(events):,} cached MLB events from {events_cache}")
    else:
        print(f"Discovering {SERIES_TICKER} events from {start_date} through {end_date}...")
        events = discover_events(start_date, end_date, verbose=args.verbose)
        events_cache.write_text(json.dumps(events))
        print(f"Discovered {len(events):,} MLB game events")

    if args.max_games is not None:
        events = events[:args.max_games]
        print(f"Smoke-test limit: processing {len(events):,} events")

    cutoffs = get_historical_cutoffs(args.verbose)
    trades_cutoff = parse_iso_timestamp(cutoffs.get("trades_created_ts"))
    print(f"Kalshi trades cutoff: {trades_cutoff}")

    rows: list[dict[str, Any]] = []
    market_count = 0
    for event_index, event in enumerate(events, start=1):
        game_date = parse_game_date(event.get("event_ticker"))
        if game_date is None:
            continue
        markets = fetch_event_markets(event, verbose=args.verbose)
        print(
            f"[{event_index:>4}/{len(events)}] {event.get('event_ticker')}: "
            f"{len(markets)} market(s)"
        )
        for market in markets:
            market = {**market}
            market.setdefault("event_ticker", event.get("event_ticker"))
            market.setdefault("status", event.get("status"))
            market_count += 1
            try:
                trades = fetch_market_trades(
                    market,
                    game_date,
                    trades_cutoff,
                    trades_cache_dir,
                    include_block_trades=args.include_block_trades,
                    refresh=args.refresh_trades,
                    verbose=args.verbose,
                )
                rows.extend(trades_to_rows(trades, market, game_date))
                print(f"    {market.get('ticker')}: {len(trades):,} trades")
            except (requests.HTTPError, ValueError, RuntimeError) as error:
                print(f"    !! {market.get('ticker')}: {error}", file=sys.stderr)

    frame = output_frame(rows)
    qualifier = "all" if args.include_block_trades else "non_block"
    suffix = f"_{args.max_games}games" if args.max_games is not None else ""
    output_path = output_dir / (
        f"kalshi_mlb_trades_{window_name}_{qualifier}{suffix}.parquet"
    )
    frame.to_parquet(output_path, index=False)

    print("\nDownload complete")
    print(f"  Game events: {len(events):,}")
    print(f"  Markets:     {market_count:,}")
    print(f"  Trades:      {len(frame):,}")
    if not frame.empty:
        print(f"  First trade: {frame['created_time'].min()}")
        print(f"  Last trade:  {frame['created_time'].max()}")
    print(f"  Output:      {output_path}")


if __name__ == "__main__":
    main()
