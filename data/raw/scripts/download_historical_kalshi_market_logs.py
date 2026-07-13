"""
download_kalshi_market_logs.py

Pulls Kalshi single-game MLB moneyline market data (series ticker KXMLBGAME)
and writes one Parquet file per MLB season:

    kalshi_mlb_2025.parquet
    kalshi_mlb_2026.parquet

Each row is one *market-minute*: a single team's contract, for a single
1-minute candlestick window, during a single game. Both season files share
an identical schema, so they can be concatenated directly or read with a
single pd.read_parquet(glob) call.

WHY THIS SHAPE:
Kalshi's finest public granularity is 1-minute OHLC candlesticks (see
GET /markets/{ticker}/candlesticks). That's the natural join target for
lining up against Statcast pitch timestamps (via the `sv_id` field), since
it gives a continuous implied-probability series rather than the sparse,
irregular ticks you'd get from raw trade fills alone.

Background this script assumes (see conversation for sourcing):
- Kalshi launched single-game MLB moneyline markets (KXMLBGAME) on
  2025-04-16. There is no usable data before that date.
- Kalshi's public market-data endpoints (events/markets/trades/candlesticks)
  do not require an API key.
- Kalshi splits data into "live" (~rolling 3 month window) and "historical"
  tiers for markets/candlesticks/trades. This script checks
  GET /historical/cutoff and routes each market to the correct endpoint
  automatically.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
 
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
 
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXMLBGAME"
CANDLE_INTERVAL_MIN = 1  # finest granularity Kalshi offers (1, 60, or 1440)
 
# Kalshi launched single-game MLB markets on this date. Anything before it
# doesn't exist and shouldn't be requested.
KALSHI_MLB_LAUNCH = datetime(2025, 4, 16, tzinfo=timezone.utc)
 
DEFAULT_OUTPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/kalshi_mlb"
)
 
REQUEST_SLEEP_SEC = 0.15  # be polite between calls
PAGE_LIMIT = 200          # events/markets page size
TRADE_PAGE_LIMIT = 1000   # trades page size (max allowed)
MAX_CANDLES_PER_REQUEST = 5000  # Kalshi's cap on candlesticks per request
 
# Best-effort full-name -> standard abbreviation map, so this joins cleanly
# against a Statcast frame's home_team / away_team columns. Kalshi's
# yes_sub_title / no_sub_title fields are typically the team name only
# (e.g. "Yankees"), so this dict is keyed on the short forms Kalshi uses,
# with full names included defensively in case that ever changes.
TEAM_NAME_TO_ABBR = {
    "Diamondbacks": "ARI", "Arizona Diamondbacks": "ARI",
    "Braves": "ATL", "Atlanta Braves": "ATL",
    "Orioles": "BAL", "Baltimore Orioles": "BAL",
    "Red Sox": "BOS", "Boston Red Sox": "BOS",
    "Cubs": "CHC", "Chicago Cubs": "CHC",
    "White Sox": "CHW", "Chicago White Sox": "CHW",
    "Reds": "CIN", "Cincinnati Reds": "CIN",
    "Guardians": "CLE", "Cleveland Guardians": "CLE",
    "Rockies": "COL", "Colorado Rockies": "COL",
    "Tigers": "DET", "Detroit Tigers": "DET",
    "Astros": "HOU", "Houston Astros": "HOU",
    "Royals": "KC", "Kansas City Royals": "KC",
    "Angels": "LAA", "Los Angeles Angels": "LAA",
    "Dodgers": "LAD", "Los Angeles Dodgers": "LAD",
    "Marlins": "MIA", "Miami Marlins": "MIA",
    "Brewers": "MIL", "Milwaukee Brewers": "MIL",
    "Twins": "MIN", "Minnesota Twins": "MIN",
    "Mets": "NYM", "New York Mets": "NYM",
    "Yankees": "NYY", "New York Yankees": "NYY",
    "Athletics": "ATH", "Oakland Athletics": "ATH",
    "Phillies": "PHI", "Philadelphia Phillies": "PHI",
    "Pirates": "PIT", "Pittsburgh Pirates": "PIT",
    "Padres": "SD", "San Diego Padres": "SD",
    "Giants": "SF", "San Francisco Giants": "SF",
    "Mariners": "SEA", "Seattle Mariners": "SEA",
    "Cardinals": "STL", "St. Louis Cardinals": "STL",
    "Rays": "TB", "Tampa Bay Rays": "TB",
    "Rangers": "TEX", "Texas Rangers": "TEX",
    "Blue Jays": "TOR", "Toronto Blue Jays": "TOR",
    "Nationals": "WSH", "Washington Nationals": "WSH",
}
 
MONTH_ABBR_TO_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
# Event tickers look like KXMLBGAME-25APR16NYYBOS -> yy / month abbr / dd.
# This is the most reliable way to get a game date: it's baked into the
# ticker string itself and doesn't depend on nested markets being present
# (which the API omits for events whose markets are old enough to have
# rolled into the historical tier -- see get_historical_cutoff()).
TICKER_DATE_RE = re.compile(r"^KXMLBGAME-(\d{2})([A-Z]{3})(\d{2})")
 
COLUMNS = [
    "season", "game_date", "event_ticker", "market_ticker",
    "team", "team_abbr", "opponent", "opponent_abbr",
    "open_time", "close_time",
    "period_end_ts", "period_end_time",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "price_open", "price_high", "price_low", "price_close", "price_mean",
    "volume", "open_interest",
]
 
 
# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------
 
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session
 
 
SESSION = make_session()
 
 
def api_get(path: str, params: dict | None = None, verbose: bool = False):
    """GET against the Kalshi public API. Returns parsed JSON, or None on 404."""
    url = f"{BASE_URL}{path}"
    if verbose:
        print(f"    GET {url} params={params}")
    resp = SESSION.get(url, params=params, timeout=30)
    time.sleep(REQUEST_SLEEP_SEC)
    if resp.status_code == 404:
        # A 404 can legitimately mean "no candles yet for this market" but
        # it can also mean the URL path itself is wrong. Surface it (even
        # without --verbose) so a path mistake doesn't silently masquerade
        # as "zero rows" the way the live-candlesticks path bug did.
        print(f"    (404, treated as no-data) {url} params={params}", file=sys.stderr)
        return None
    if not resp.ok:
        print(f"    !! {resp.status_code} on {url} params={params}: {resp.text[:300]}",
              file=sys.stderr)
    resp.raise_for_status()
    return resp.json()
 
 
def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
 
 
def parse_iso(ts: str | None):
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
 
 
# --------------------------------------------------------------------------
# Historical cutoff
# --------------------------------------------------------------------------
 
def get_historical_cutoff() -> datetime | None:
    data = api_get("/historical/cutoff")
    if not data:
        return None
    return parse_iso(data.get("market_settled_ts"))
 
 
# --------------------------------------------------------------------------
# Events / markets discovery
# --------------------------------------------------------------------------
 
def fetch_all_events(cache_path: Path, refresh: bool, verbose: bool) -> list[dict]:
    """Fetch every KXMLBGAME event (any status), with nested markets included."""
    if cache_path.exists() and not refresh:
        print(f"Using cached event list: {cache_path}")
        return json.loads(cache_path.read_text())
 
    events = []
    cursor = None
    page = 0
    while True:
        params = {
            "series_ticker": SERIES_TICKER,
            "with_nested_markets": "true",
            "limit": PAGE_LIMIT,
        }
        if cursor:
            params["cursor"] = cursor
        data = api_get("/events", params=params, verbose=verbose)
        if not data:
            break
        batch = data.get("events", [])
        events.extend(batch)
        page += 1
        print(f"  events page {page}: +{len(batch)} (total {len(events)})")
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
 
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(events))
    return events
 
 
def parse_ticker_date(event_ticker: str | None) -> datetime | None:
    if not event_ticker:
        return None
    m = TICKER_DATE_RE.match(event_ticker)
    if not m:
        return None
    yy, mon, dd = m.groups()
    month = MONTH_ABBR_TO_NUM.get(mon)
    if not month:
        return None
    return datetime(2000 + int(yy), month, int(dd), tzinfo=timezone.utc)
 
 
def event_game_date(event: dict) -> datetime | None:
    """
    Best-effort game date for an event.
 
    NOTE: this deliberately tries the ticker first, *not* strike_date or
    nested market open_time. Kalshi's /events endpoint silently omits
    nested markets for events whose markets are old enough to have rolled
    into the historical tier (see the "Impacted Live Endpoints" table in
    Kalshi's historical-data docs: "GET /events with with_nested_markets=true
    -- Nested markets older than the cutoff will not be included"). That
    means for most/all of the 2025 season, event.get("markets") comes back
    empty and event.get("strike_date") may also be unset for sports events,
    so both of those would silently drop every 2025 game. Ticker parsing
    has no such dependency.
    """
    dt = parse_ticker_date(event.get("event_ticker"))
    if dt:
        return dt
    dt = parse_iso(event.get("strike_date"))
    if dt:
        return dt
    markets = event.get("markets") or []
    open_times = [parse_iso(m.get("open_time")) for m in markets]
    open_times = [t for t in open_times if t]
    return min(open_times) if open_times else None
 
 
def season_of(dt: datetime) -> int:
    return dt.year
 
 
def fetch_markets_for_event(event_ticker: str, cache_dir: Path,
                             verbose: bool) -> list[dict]:
    """
    Backfill markets for an event when they weren't nested in the /events
    response (see event_game_date docstring above for why). Tries the
    historical endpoint first since this only happens for older events;
    falls back to the live endpoint as a safety net.
    """
    cache_file = cache_dir / f"{event_ticker}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
 
    markets: list[dict] = []
    for path in (
        "/historical/markets",
        "/markets",  # fallback safety net
    ):
        cursor = None
        while True:
            params = {"event_ticker": event_ticker, "limit": PAGE_LIMIT}
            if cursor:
                params["cursor"] = cursor
            data = api_get(path, params=params, verbose=verbose)
            if not data:
                break
            batch = data.get("markets", [])
            markets.extend(batch)
            cursor = data.get("cursor")
            if not cursor or not batch:
                break
        if markets:
            break
 
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(markets))
    return markets
 
 
# --------------------------------------------------------------------------
# Per-market candlestick fetch
# --------------------------------------------------------------------------
 
def fetch_candlesticks(ticker: str, start_ts: int, end_ts: int,
                        historical: bool, verbose: bool) -> dict | None:
    # NOTE: the live endpoint requires the series ticker in the path
    # (/series/{series_ticker}/markets/{ticker}/candlesticks); the historical
    # endpoint does not (/historical/markets/{ticker}/candlesticks). Mixing
    # these up makes the live path 404 silently -- which is exactly what was
    # happening here, producing 0 rows for every 2026 market with no visible
    # error (api_get treats 404 as "no data" rather than raising).
    path = (f"/historical/markets/{ticker}/candlesticks" if historical
            else f"/series/{SERIES_TICKER}/markets/{ticker}/candlesticks")
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": CANDLE_INTERVAL_MIN,
    }
    data = api_get(path, params=params, verbose=verbose)
    if data is None and not historical:
        # Market may have rolled into the historical tier since we checked
        # the cutoff (e.g. a long-running script). Fall back once.
        data = api_get(f"/historical/markets/{ticker}/candlesticks",
                        params=params, verbose=verbose)
    return data
 
 
def fetch_candlesticks_chunked(ticker: str, start_ts: int, end_ts: int,
                                historical: bool, verbose: bool) -> dict:
    """
    Kalshi caps a single candlesticks request at MAX_CANDLES_PER_REQUEST
    periods. Kalshi opens MLB moneyline markets well before first pitch
    (sometimes days ahead), so open_time-to-close_time can easily exceed
    that cap at 1-minute resolution. Page through the window in chunks and
    concatenate, rather than requesting the whole span at once.
    """
    max_span_sec = (MAX_CANDLES_PER_REQUEST - 1) * CANDLE_INTERVAL_MIN * 60
    all_candlesticks = []
    cur_start = start_ts
    while cur_start < end_ts:
        cur_end = min(cur_start + max_span_sec, end_ts)
        chunk = fetch_candlesticks(ticker, cur_start, cur_end, historical, verbose)
        if chunk:
            all_candlesticks.extend(chunk.get("candlesticks", []))
        cur_start = cur_end + 1
    return {"ticker": ticker, "candlesticks": all_candlesticks}
 
 
def candles_to_rows(candles: dict, meta: dict) -> list[dict]:
    rows = []
    for c in (candles or {}).get("candlesticks", []):
        ts = c.get("end_period_ts")
        yes_bid = c.get("yes_bid") or {}
        yes_ask = c.get("yes_ask") or {}
        price = c.get("price") or {}
        rows.append({
            **meta,
            "period_end_ts": ts,
            "period_end_time": (
                datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
            ),
            "yes_bid_open": to_float(yes_bid.get("open_dollars")),
            "yes_bid_high": to_float(yes_bid.get("high_dollars")),
            "yes_bid_low": to_float(yes_bid.get("low_dollars")),
            "yes_bid_close": to_float(yes_bid.get("close_dollars")),
            "yes_ask_open": to_float(yes_ask.get("open_dollars")),
            "yes_ask_high": to_float(yes_ask.get("high_dollars")),
            "yes_ask_low": to_float(yes_ask.get("low_dollars")),
            "yes_ask_close": to_float(yes_ask.get("close_dollars")),
            "price_open": to_float(price.get("open_dollars")),
            "price_high": to_float(price.get("high_dollars")),
            "price_low": to_float(price.get("low_dollars")),
            "price_close": to_float(price.get("close_dollars")),
            "price_mean": to_float(price.get("mean_dollars")),
            "volume": to_float(c.get("volume_fp")),
            "open_interest": to_float(c.get("open_interest_fp")),
        })
    return rows
 
 
# --------------------------------------------------------------------------
# Main pull
# --------------------------------------------------------------------------
 
def process_market(market: dict, event: dict, game_date: datetime,
                    season: int, cutoff: datetime | None,
                    cache_dir: Path, verbose: bool) -> list[dict]:
    ticker = market["ticker"]
    event_ticker = event.get("event_ticker")
    team = market.get("yes_sub_title") or ""
    opponent = market.get("no_sub_title") or ""
 
    open_time = parse_iso(market.get("open_time"))
    close_time = parse_iso(market.get("close_time")) or parse_iso(
        market.get("latest_expiration_time")
    )
    if not open_time or not close_time:
        return []
 
    cache_file = cache_dir / f"{ticker}.json"
    if cache_file.exists():
        candles = json.loads(cache_file.read_text())
    else:
        historical = bool(cutoff and close_time < cutoff)
        # Small buffer around the game window to catch pre-game price
        # settling and the final settlement tick. Chunked because the
        # open-to-close span (market opens well before first pitch) can
        # exceed Kalshi's max-candlesticks-per-request cap.
        start_ts = int(open_time.timestamp()) - 300
        end_ts = int(close_time.timestamp()) + 900
        candles = fetch_candlesticks_chunked(ticker, start_ts, end_ts, historical, verbose)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(candles or {}))
 
    meta = {
        "season": season,
        "game_date": game_date.date().isoformat() if game_date else None,
        "event_ticker": event_ticker,
        "market_ticker": ticker,
        "team": team,
        "team_abbr": TEAM_NAME_TO_ABBR.get(team, team),
        "opponent": opponent,
        "opponent_abbr": TEAM_NAME_TO_ABBR.get(opponent, opponent),
        "open_time": open_time,
        "close_time": close_time,
    }
    return candles_to_rows(candles, meta)
 
 
def build_season_frame(events: list[dict], season: int, cutoff: datetime | None,
                        cache_dir: Path, event_markets_cache_dir: Path,
                        max_games: int | None, verbose: bool) -> pd.DataFrame:
    rows = []
    n_games = 0
    n_backfilled = 0
    for event in events:
        game_date = event_game_date(event)
        if not game_date or season_of(game_date) != season:
            continue
        if game_date < KALSHI_MLB_LAUNCH:
            continue
 
        markets = event.get("markets") or []
        if not markets:
            # /events omits nested markets once they're old enough to have
            # rolled into the historical tier -- fetch them directly.
            markets = fetch_markets_for_event(
                event.get("event_ticker"), event_markets_cache_dir, verbose
            )
            n_backfilled += 1
        if not markets:
            continue
 
        n_games += 1
        if max_games and n_games > max_games:
            break
 
        for market in markets:
            try:
                rows.extend(
                    process_market(market, event, game_date, season, cutoff,
                                    cache_dir, verbose)
                )
            except requests.HTTPError as e:
                print(f"  !! failed {market.get('ticker')}: {e}", file=sys.stderr)
 
        if n_games % 25 == 0:
            print(f"  ...{season}: processed {n_games} games, {len(rows)} rows so far")
 
    print(f"  {season}: {n_games} games processed "
          f"({n_backfilled} needed a markets backfill call), "
          f"{len(rows)} candlestick rows total")
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[COLUMNS].sort_values(
        ["game_date", "event_ticker", "market_ticker", "period_end_ts"]
    ).reset_index(drop=True)
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--refresh-events", action="store_true",
                         help="Ignore cached event list and refetch from Kalshi.")
    parser.add_argument("--max-games", type=int, default=None,
                         help="Cap games per season (smoke testing).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
 
    output_dir = args.output_dir
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print("Fetching event list (all statuses, all seasons)...")
    events = fetch_all_events(
        cache_path=cache_dir / "events.json",
        refresh=args.refresh_events,
        verbose=args.verbose,
    )
    print(f"Total KXMLBGAME events discovered: {len(events)}")
 
    print("Fetching historical/live cutoff...")
    cutoff = get_historical_cutoff()
    print(f"Historical cutoff (market_settled_ts): {cutoff}")
 
    candle_cache_dir = cache_dir / "candles"
    event_markets_cache_dir = cache_dir / "event_markets"
 
    for season in args.seasons:
        print(f"\n=== Season {season} ===")
        df = build_season_frame(
            events, season, cutoff, candle_cache_dir, event_markets_cache_dir,
            args.max_games, args.verbose
        )
        out_path = output_dir / f"kalshi_mlb_{season}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Wrote {len(df):,} rows -> {out_path}")
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()