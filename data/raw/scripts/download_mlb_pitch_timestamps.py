"""
download_mlb_pitch_timestamps.py

Pulls wall-clock timestamps for every pitch in a season directly from the
MLB Stats API live feed endpoint:

    GET https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live

For each game the feed returns liveData.plays.allPlays[*].playEvents[*].
Each pitch event carries:
    isPitch:     true
    pitchNumber: int  (1-indexed within the at-bat, same as Statcast pitch_number)
    startTime:   ISO-8601 wall-clock timestamp  ← what we're after

Each plate appearance (play) has:
    atBatIndex:  int  (0-indexed)  → at_bat_number = atBatIndex + 1  (Statcast is 1-indexed)

The resulting parquet files are keyed on (game_pk, at_bat_number, pitch_number),
which is an exact-join key against the existing Statcast parquet data.

Output
------
    data/raw/mlb_timestamps/pitch_timestamps_2025.parquet
    data/raw/mlb_timestamps/pitch_timestamps_2026.parquet

Resumable
---------
Each game's raw API response is cached to disk under
    data/raw/mlb_timestamps/cache/{season}/{game_pk}.json.gz

Re-running the script skips any game whose cache file already exists, so
it is safe to kill and restart at any point.

Usage
-----
    python download_mlb_pitch_timestamps.py                    # both seasons
    python download_mlb_pitch_timestamps.py --seasons 2026     # 2026 only
    python download_mlb_pitch_timestamps.py --max-games 5      # smoke test
    python download_mlb_pitch_timestamps.py --verbose          # print every URL
    python download_mlb_pitch_timestamps.py --refresh          # ignore all caches

Schema sanity-check
-------------------
On the FIRST game of each run the script prints the raw structure of the
feed response so you can immediately verify that the field names match what
the API actually returns.  If they differ, the extraction logic is in
extract_pitches_from_feed() below — a single, easy-to-update function.
"""

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MLB_API_BASE = "https://statsapi.mlb.com/api"

# Statcast parquets we read to discover game_pk values.
STATCAST_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_statcast"
)

DEFAULT_OUTPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_timestamps"
)

# Polite delay between HTTP requests (seconds).  The MLB Stats API is a
# public, unauthenticated service — be a good citizen.
REQUEST_SLEEP_SEC = 0.20

OUTPUT_COLUMNS = [
    "season",
    "game_pk",
    "at_bat_number",   # 1-indexed (Statcast convention)
    "pitch_number",    # 1-indexed within the at-bat (Statcast convention)
    "start_time",      # ISO-8601 wall-clock string from the API (UTC)
    "start_time_utc",  # parsed as a tz-aware datetime (UTC)
]

# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

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
    session.mount("http://", adapter)
    return session


SESSION = make_session()


def api_get(url: str, params: dict | None = None, verbose: bool = False) -> dict | None:
    """
    GET the given URL.  Returns parsed JSON dict, or None on 404.
    Raises on any other non-2xx status after the retry policy is exhausted.
    """
    if verbose:
        print(f"    GET {url}  params={params}")
    resp = SESSION.get(url, params=params, timeout=60)
    time.sleep(REQUEST_SLEEP_SEC)
    if resp.status_code == 404:
        print(f"    (404, no data) {url}", file=sys.stderr)
        return None
    if not resp.ok:
        print(
            f"    !! HTTP {resp.status_code} on {url}: {resp.text[:300]}",
            file=sys.stderr,
        )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Game PK discovery from Statcast parquets
# ---------------------------------------------------------------------------

def discover_game_pks(seasons: list[int]) -> dict[int, list[int]]:
    """
    Read game_pk values directly out of the existing Statcast parquet files.
    Returns {season: sorted list of unique game_pk ints}.
    """
    result: dict[int, list[int]] = {}
    for season in seasons:
        path = STATCAST_DIR / f"{season}.parquet"
        if not path.exists():
            print(f"  [warn] Statcast file not found, skipping: {path}", file=sys.stderr)
            result[season] = []
            continue
        # Only read the game_pk column — these files can be ~100 MB+.
        df = pd.read_parquet(path, columns=["game_pk"])
        pks = sorted(df["game_pk"].dropna().astype(int).unique().tolist())
        print(f"  {season}: found {len(pks):,} unique game_pk values in {path.name}")
        result[season] = pks
    return result


# ---------------------------------------------------------------------------
# Live feed fetch & extraction
# ---------------------------------------------------------------------------

def fetch_live_feed(game_pk: int, verbose: bool = False) -> dict | None:
    """
    Fetch the full play-by-play live feed for a single game.
    Returns the parsed JSON dict (the full response), or None on failure.
    """
    url = f"{MLB_API_BASE}/v1.1/game/{game_pk}/feed/live"
    return api_get(url, verbose=verbose)


def _print_schema_sample(feed: dict) -> None:
    """
    Pretty-print enough of the feed structure for a first-game sanity check.
    Shows: top-level keys, liveData keys, and the first pitch event in full.
    """
    print("\n" + "=" * 70)
    print("SCHEMA SANITY CHECK — first game raw structure")
    print("=" * 70)
    print(f"Top-level keys: {list(feed.keys())}")

    live_data = feed.get("liveData") or {}
    print(f"liveData keys:  {list(live_data.keys())}")

    plays = live_data.get("plays") or {}
    print(f"liveData.plays keys: {list(plays.keys())}")

    all_plays = plays.get("allPlays") or []
    print(f"Number of plays (allPlays): {len(all_plays)}")

    if all_plays:
        play0 = all_plays[0]
        print(f"\nFirst play (allPlays[0]) top-level keys: {list(play0.keys())}")
        print(f"  atBatIndex: {play0.get('atBatIndex')}")
        events = play0.get("playEvents") or []
        print(f"  playEvents count: {len(events)}")
        # Find first pitch event
        for ev in events:
            if ev.get("isPitch"):
                print(f"\nFirst pitch event (playEvents[?]):")
                print(json.dumps(ev, indent=2, default=str))
                break
        else:
            if events:
                print(f"\nFirst playEvent (no isPitch=true found in play 0):")
                print(json.dumps(events[0], indent=2, default=str))
    print("=" * 70 + "\n")


def extract_pitches_from_feed(
    feed: dict, game_pk: int, season: int
) -> list[dict]:
    """
    Walk liveData.plays.allPlays[*].playEvents[*] and return one row dict
    per pitch event.

    Key mapping
    -----------
    Statcast field      MLB API source
    ------------------- ---------------------------------------------------
    game_pk             (parameter — already known)
    at_bat_number       allPlays[i].atBatIndex + 1    (0-indexed -> 1-indexed)
    pitch_number        playEvents[j].pitchNumber
    start_time          playEvents[j].startTime       (ISO-8601 string)
    """
    rows: list[dict] = []

    live_data = feed.get("liveData") or {}
    plays = live_data.get("plays") or {}
    all_plays = plays.get("allPlays") or []

    for play in all_plays:
        # atBatIndex is 0-indexed in the API; Statcast at_bat_number is 1-indexed.
        at_bat_idx = play.get("atBatIndex")
        if at_bat_idx is None:
            continue
        at_bat_number = int(at_bat_idx) + 1

        play_events = play.get("playEvents") or []
        for event in play_events:
            if not event.get("isPitch"):
                continue

            pitch_number_raw = event.get("pitchNumber")
            start_time_raw = event.get("startTime")

            if pitch_number_raw is None or not start_time_raw:
                # Skip events with no usable data (rare, but be defensive).
                continue

            # Parse the ISO-8601 timestamp.  The API typically returns
            # something like "2025-04-16T19:23:41.000Z".
            try:
                start_time_utc = datetime.fromisoformat(
                    start_time_raw.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                start_time_utc = None

            rows.append(
                {
                    "season": season,
                    "game_pk": game_pk,
                    "at_bat_number": at_bat_number,
                    "pitch_number": int(pitch_number_raw),
                    "start_time": start_time_raw,
                    "start_time_utc": start_time_utc,
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Per-game caching
# ---------------------------------------------------------------------------

def cache_path_for(game_pk: int, cache_dir: Path) -> Path:
    """Return the gzip-compressed JSON cache path for a given game_pk."""
    return cache_dir / f"{game_pk}.json.gz"


def load_cached_feed(game_pk: int, cache_dir: Path) -> dict | None:
    p = cache_path_for(game_pk, cache_dir)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_feed_cache(game_pk: int, feed: dict, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_path_for(game_pk, cache_dir)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(feed, f)


# ---------------------------------------------------------------------------
# Season-level processing
# ---------------------------------------------------------------------------

def process_season(
    season: int,
    game_pks: list[int],
    cache_dir: Path,
    output_path: Path,
    max_games: int | None,
    refresh: bool,
    verbose: bool,
) -> pd.DataFrame:
    """
    For each game_pk in game_pks:
      1. Load from cache, or fetch from the MLB API (and cache the result).
      2. Extract pitch rows.
    Returns the combined DataFrame for the season.
    """
    all_rows: list[dict] = []
    schema_printed = False
    n_fetched = 0
    n_cached = 0
    n_failed = 0

    if max_games is not None:
        game_pks = game_pks[:max_games]

    total = len(game_pks)
    for idx, game_pk in enumerate(game_pks, start=1):
        if idx % 100 == 0 or idx == 1 or idx == total:
            print(
                f"  [{season}] {idx}/{total}  fetched={n_fetched}  "
                f"cached={n_cached}  failed={n_failed}  "
                f"rows_so_far={len(all_rows):,}"
            )

        feed = None

        if not refresh:
            feed = load_cached_feed(game_pk, cache_dir)
            if feed is not None:
                n_cached += 1

        if feed is None:
            try:
                feed = fetch_live_feed(game_pk, verbose=verbose)
            except Exception as exc:
                print(
                    f"  !! [{season}] game_pk={game_pk} fetch error: {exc}",
                    file=sys.stderr,
                )
                n_failed += 1
                continue

            if feed is None:
                # 404 or empty — the game might not exist yet.
                n_failed += 1
                continue

            save_feed_cache(game_pk, feed, cache_dir)
            n_fetched += 1

        # Print schema info on the very first game (fetched or cached).
        if not schema_printed:
            _print_schema_sample(feed)
            schema_printed = True

        rows = extract_pitches_from_feed(feed, game_pk, season)
        all_rows.extend(rows)

    print(
        f"\n  {season} complete: {total} games  "
        f"({n_fetched} fetched, {n_cached} from cache, {n_failed} failed)  "
        f"{len(all_rows):,} pitch rows"
    )

    if not all_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(all_rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = (
        df[OUTPUT_COLUMNS]
        .sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2025, 2026],
        help="Seasons to process (default: 2025 2026).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write pitch_timestamps_<year>.parquet files.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Process at most N games per season (smoke testing).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore all cached game feeds and re-fetch from the MLB API.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every HTTP request URL.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    # ------------------------------------------------------------------
    # 1. Discover game_pk values from existing Statcast parquets
    # ------------------------------------------------------------------
    print("Discovering game_pk values from Statcast parquets...")
    season_pks = discover_game_pks(args.seasons)

    # ------------------------------------------------------------------
    # 2. Process each season
    # ------------------------------------------------------------------
    for season in args.seasons:
        game_pks = season_pks.get(season, [])
        if not game_pks:
            print(f"\n=== Season {season}: no game_pks found, skipping ===")
            continue

        print(f"\n=== Season {season}: {len(game_pks):,} games ===")

        df = process_season(
            season=season,
            game_pks=game_pks,
            cache_dir=cache_dir / str(season),
            output_path=output_dir / f"pitch_timestamps_{season}.parquet",
            max_games=args.max_games,
            refresh=args.refresh,
            verbose=args.verbose,
        )

        out_path = output_dir / f"pitch_timestamps_{season}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Wrote {len(df):,} rows -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
