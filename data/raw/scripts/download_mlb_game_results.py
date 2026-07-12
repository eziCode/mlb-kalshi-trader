#!/usr/bin/env python3
"""
data/raw/scripts/pull_mlb_game_results.py

Pulls the AUTHORITATIVE final score for every game from MLB's live feed
(the same API used for pitch timestamps), and derives home_win directly
from it -- rather than inferring it from the last Statcast pitch's
score_diff, which misses walk-off finishes (the winning run scores on a
play with no subsequent pitch, so score_diff on the "last" pitch never
reflects it) and is fragile against any row with a missing
pitch_timestamp_utc affecting which pitch sorts last.

Output:
    data/raw/mlb_game_results/game_results_2025.parquet
    data/raw/mlb_game_results/game_results_2026.parquet

Each row: game_pk, home_runs_final, away_runs_final, home_win

If you already have per-game live-feed JSON cached on disk from your
existing timestamp puller, point --cache-dir at that folder to reuse it
and avoid re-fetching ~2,200+ games from scratch.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STATCAST_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_statcast"
)
OUTPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_game_results"
)
DEFAULT_CACHE_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_pitch_timestamps/cache"
)

BASE_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
REQUEST_SLEEP_SEC = 0.2


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = make_session()


def fetch_game_feed(game_pk: int, cache_dir: Path) -> dict | None:
    cache_file = cache_dir / f"{game_pk}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    url = BASE_URL.format(game_pk=game_pk)
    resp = SESSION.get(url, timeout=30)
    time.sleep(REQUEST_SLEEP_SEC)

    if resp.status_code == 404:
        print(f"    !! 404 for game_pk={game_pk}", file=sys.stderr)
        return None
    if not resp.ok:
        print(f"    !! {resp.status_code} for game_pk={game_pk}", file=sys.stderr)
        resp.raise_for_status()

    data = resp.json()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def extract_result(game_pk: int, feed: dict) -> dict | None:
    try:
        linescore = feed["liveData"]["linescore"]["teams"]
        home_runs = linescore["home"]["runs"]
        away_runs = linescore["away"]["runs"]
    except KeyError:
        print(f"    !! no linescore found for game_pk={game_pk} "
              f"(postponed/suspended/in-progress?)", file=sys.stderr)
        return None

    if home_runs is None or away_runs is None:
        return None

    return {
        "game_pk": game_pk,
        "home_runs_final": home_runs,
        "away_runs_final": away_runs,
        "home_win": int(home_runs > away_runs),
    }


def get_game_pks_by_season() -> dict[int, list[int]]:
    by_season = {}
    for path in sorted(STATCAST_DIR.glob("*.parquet")):
        year = int(path.stem)
        df = pd.read_parquet(path, columns=["game_pk"])
        by_season[year] = sorted(df["game_pk"].dropna().unique().tolist())
    return by_season


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                         help="Reuse cached live-feed JSON from here if present "
                              "(e.g. point at your existing timestamp puller's "
                              "cache to avoid re-fetching every game).")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    game_pks_by_season = get_game_pks_by_season()

    n_pulled = 0
    for season, game_pks in game_pks_by_season.items():
        print(f"\n=== Season {season}: {len(game_pks):,} games ===")
        rows = []

        for i, game_pk in enumerate(game_pks, start=1):
            if args.max_games and n_pulled >= args.max_games:
                break
            feed = fetch_game_feed(int(game_pk), args.cache_dir)
            n_pulled += 1
            if feed is None:
                continue
            result = extract_result(int(game_pk), feed)
            if result:
                rows.append(result)
            if i % 200 == 0:
                print(f"  ...{i:,}/{len(game_pks):,} games processed")

        df = pd.DataFrame(rows, columns=[
            "game_pk", "home_runs_final", "away_runs_final", "home_win"
        ])
        out_path = OUTPUT_DIR / f"game_results_{season}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Wrote {len(df):,} / {len(game_pks):,} game results -> {out_path}")

        if args.max_games and n_pulled >= args.max_games:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()