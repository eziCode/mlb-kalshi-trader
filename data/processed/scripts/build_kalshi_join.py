"""
data/processed/scripts/build_kalshi_join.py

Joins Kalshi KXMLBGAME market data onto the per-pitch feature file, so each
pitch carries the market-implied home-win probability (and related fields)
as of the most recent fully-settled price update BEFORE that pitch.

Output:
    data/processed/mlb_game_state/training_dataset.parquet

INPUT ASSUMPTIONS (verify against your own diagnostics output below):
    - pitch_state_features.parquet has: game_pk, game_date, home_team,
      away_team, at_bat_number, pitch_number, pitch_timestamp_utc (tz-aware UTC).
    - kalshi_mlb_2025.parquet / kalshi_mlb_2026.parquet have identical
      schemas: event_ticker, market_ticker, team_abbr, opponent_abbr,
      game_date, open_time, close_time, period_end_time, price_close,
      yes_bid_close, yes_ask_close, volume, open_interest, ...

DATA LEAKAGE PREVENTION:
    Kalshi candles are 1-minute OHLC windows.  period_end_time is the
    CLOSE of that window.  A pitch that occurs inside a candle window has
    NOT yet seen that candle's closing price -- it only becomes observable
    after the window ends.  Therefore the as-of join uses strict
    direction="backward" with allow_exact_matches=False, meaning a pitch
    at time T only sees candles whose period_end_time < T (strictly less
    than).  A pitch that falls exactly on a candle boundary gets the
    previous candle's price, not the concurrent one.

WHY THIS IS TRICKY (read before trusting the output blindly):
    Statcast's team abbreviations and the abbreviations produced by the
    Kalshi pull script's team-name lookup are NOT guaranteed to agree
    (e.g. Statcast commonly uses "AZ" / "CWS", while the Kalshi script's
    dict produced "ARI" / "CHW"). Rather than silently dropping games on
    a failed string match, this script routes both sides through an
    explicit TEAM_CODE_CROSSWALK and prints anything it can't normalize,
    so a real mismatch is visible instead of quietly losing rows.

    It also does NOT assume the Kalshi event_ticker's date/time encoding
    is fully reliable for disambiguating doubleheaders -- instead it
    matches games chronologically against each game's own pitch-timestamp
    range, which uses data you've already verified rather than a guessed
    ticker grammar.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

GAME_STATE_DIR = Path(
    "data/processed/mlb_game_state"
)

KALSHI_DIR = Path(
    "data/raw/kalshi_market_logs"
)

TRAIN_DIR = Path(
    "data/processed/train"
)

TEST_DIR = Path(
    "data/processed/test"
)

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Team code normalization
# --------------------------------------------------
# Maps every known variant spelling/abbreviation to ONE canonical code.
# This is deliberately generous / redundant -- better to have an unused
# entry than to silently fail to match a real one. Anything encountered
# in your actual data that ISN'T covered here will be printed at runtime
# so you can add it rather than have it fail silently.

import re

TEAM_CODE_CROSSWALK = {
    # abbreviation-style variants
    "ARI": "ARI", "AZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC", "CHN": "CHC",
    "CHW": "CHW", "CWS": "CHW", "CHA": "CHW",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KC", "KCR": "KC",
    "LAA": "LAA", "ANA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA", "FLA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "OAK": "ATH", "ATH": "ATH",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SD", "SDP": "SD",
    "SF": "SF", "SFG": "SF",
    "SEA": "SEA",
    "STL": "STL",
    "TB": "TB", "TBR": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSH", "WAS": "WSH",

    # full nickname variants
    "DIAMONDBACKS": "ARI",
    "BRAVES": "ATL",
    "ORIOLES": "BAL",
    "REDSOX": "BOS", "RED SOX": "BOS",
    "CUBS": "CHC",
    "WHITESOX": "CHW", "WHITE SOX": "CHW",
    "REDS": "CIN",
    "GUARDIANS": "CLE",
    "ROCKIES": "COL",
    "TIGERS": "DET",
    "ASTROS": "HOU",
    "ROYALS": "KC",
    "ANGELS": "LAA",
    "DODGERS": "LAD",
    "MARLINS": "MIA",
    "BREWERS": "MIL",
    "TWINS": "MIN",
    "METS": "NYM",
    "YANKEES": "NYY",
    "ATHLETICS": "ATH", "AS": "ATH",  # "A's" -> cleaned to "AS"
    "PHILLIES": "PHI",
    "PIRATES": "PIT",
    "PADRES": "SD",
    "GIANTS": "SF",
    "MARINERS": "SEA",
    "CARDINALS": "STL",
    "RAYS": "TB",
    "RANGERS": "TEX",
    "BLUEJAYS": "TOR", "BLUE JAYS": "TOR",
    "NATIONALS": "WSH",

    # city-name variants -- this is what Kalshi's team_abbr/opponent_abbr
    # fields actually contain (confirmed from a real run): plain city name
    # for single-team cities, and a truncated disambiguator for the three
    # two-team cities (Chicago/LA/NY). "Chicago W" and "Chicago WS" both
    # showed up for the White Sox across different events -- Kalshi wasn't
    # even internally consistent about the truncation, so both are mapped.
    "ARIZONA": "ARI",
    "ATLANTA": "ATL",
    "BALTIMORE": "BAL",
    "BOSTON": "BOS",
    "CHICAGO C": "CHC",
    "CHICAGO W": "CHW", "CHICAGO WS": "CHW",
    "CINCINNATI": "CIN",
    "CLEVELAND": "CLE",
    "COLORADO": "COL",
    "DETROIT": "DET",
    "HOUSTON": "HOU",
    "KANSAS CITY": "KC",
    "LOS ANGELES A": "LAA",
    "LOS ANGELES D": "LAD",
    "MIAMI": "MIA",
    "MILWAUKEE": "MIL",
    "MINNESOTA": "MIN",
    "NEW YORK M": "NYM",
    "NEW YORK Y": "NYY",
    "PHILADELPHIA": "PHI",
    "PITTSBURGH": "PIT",
    "SAN DIEGO": "SD",
    "SAN FRANCISCO": "SF",
    "SEATTLE": "SEA",
    "ST LOUIS": "STL",  # period stripped by _clean() before lookup
    "TAMPA BAY": "TB",
    "TEXAS": "TEX",
    "TORONTO": "TOR",
    "WASHINGTON": "WSH",
}


def _clean(s: str) -> str:
    # Strip punctuation (periods, straight/curly apostrophes) so variants
    # like "St. Louis" / "ST LOUIS" and "A's" / "AS" normalize the same way,
    # regardless of exactly which quote character or spacing Kalshi used.
    return re.sub(r"[.'\u2019]", "", s).strip().upper()


def normalize_team(code) -> str | None:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    return TEAM_CODE_CROSSWALK.get(_clean(str(code)))


def report_unmapped(series: pd.Series, label: str) -> None:
    raw_values = series.dropna().unique()
    unmapped = sorted({v for v in raw_values if normalize_team(v) is None})
    if unmapped:
        print(f"  WARNING: {label} has {len(unmapped)} value(s) not in "
              f"TEAM_CODE_CROSSWALK: {unmapped}")
        print("           Add these to TEAM_CODE_CROSSWALK before trusting "
              "match coverage.")


# --------------------------------------------------
# Load inputs
# --------------------------------------------------

def load_pitches() -> pd.DataFrame:
    path = GAME_STATE_DIR / "pitch_state_features.parquet"
    print(f"Loading {path}")
    df = pd.read_parquet(path)

    required = {"game_pk", "game_date", "home_team", "away_team", "pitch_timestamp_utc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"pitch_state_features.parquet is missing required columns: {missing}. "
            "Make sure you've rerun build_event_state_features.py with the "
            "updated version that keeps home_team/away_team/pitch_timestamp_utc."
        )

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    report_unmapped(df["home_team"], "pitch_state_features.home_team")
    report_unmapped(df["away_team"], "pitch_state_features.away_team")

    df["home_team_canon"] = df["home_team"].map(normalize_team)
    df["away_team_canon"] = df["away_team"].map(normalize_team)

    n_missing_ts = df["pitch_timestamp_utc"].isna().sum()
    if n_missing_ts:
        print(f"  {n_missing_ts:,} / {len(df):,} pitches have no "
              f"pitch_timestamp_utc and cannot be price-joined "
              f"(will appear in output with NaN Kalshi columns).")

    # ------------------------------------------------------------------
    # Calculate home_win using the final pitch of each game
    # ------------------------------------------------------------------
    # Sort by time, group by game, and grab the very last pitch
    last_pitches = df.sort_values("pitch_timestamp_utc").groupby("game_pk").tail(1)
    
    # If the home team has a positive score differential at the end, they won
    home_winners = last_pitches[last_pitches["score_diff"] > 0]["game_pk"]
    
    # Map this boolean back to every pitch in the game as an integer (1 or 0)
    df["home_win"] = df["game_pk"].isin(home_winners).astype(int)
    
    print(f"  Calculated home_win for {df['game_pk'].nunique():,} games.")

    return df


def load_kalshi() -> pd.DataFrame:
    frames = []
    for path in sorted(KALSHI_DIR.glob("kalshi_mlb_*.parquet")):
        print(f"Loading {path}")
        frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(f"No kalshi_mlb_*.parquet files found in {KALSHI_DIR}")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} Kalshi candlestick rows across {df['event_ticker'].nunique():,} events")

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    report_unmapped(df["team_abbr"], "kalshi.team_abbr")
    report_unmapped(df["opponent_abbr"], "kalshi.opponent_abbr")

    df["team_canon"] = df["team_abbr"].map(normalize_team)
    df["opponent_canon"] = df["opponent_abbr"].map(normalize_team)
    df["period_end_time"] = pd.to_datetime(df["period_end_time"], utc=True)

    return df


# --------------------------------------------------
# Build game <-> event/market mapping
# --------------------------------------------------

def build_statcast_games(pitches: pd.DataFrame) -> pd.DataFrame:
    games = pitches.groupby("game_pk").agg(
        game_date=("game_date", "first"),
        home_team_canon=("home_team_canon", "first"),
        away_team_canon=("away_team_canon", "first"),
        min_ts=("pitch_timestamp_utc", "min"),
        max_ts=("pitch_timestamp_utc", "max"),
    ).reset_index()

    games["matchup"] = games.apply(
        lambda r: frozenset({r["home_team_canon"], r["away_team_canon"]}), axis=1
    )
    return games


def build_kalshi_events(kalshi: pd.DataFrame) -> pd.DataFrame:
    def team_set(s):
        return frozenset(s.dropna().unique())

    events = kalshi.groupby("event_ticker").agg(
        game_date=("game_date", "first"),
        open_time=("open_time", "min"),
        close_time=("close_time", "max"),
    ).reset_index()

    teams = kalshi.groupby("event_ticker")["team_canon"].apply(team_set)
    events = events.merge(teams.rename("matchup"), on="event_ticker")

    return events


def match_games(statcast_games: pd.DataFrame, kalshi_events: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per matched game_pk with the event_ticker it corresponds
    to. Handles doubleheaders (same date + same matchup appearing more than
    once) by pairing games and events in chronological order rather than
    relying on ticker string parsing.
    """
    matches = []
    unmatched_games = []

    for (date, matchup), s_group in statcast_games.groupby(["game_date", "matchup"]):
        k_group = kalshi_events[
            (kalshi_events["game_date"] == date) & (kalshi_events["matchup"] == matchup)
        ]

        if len(matchup) != 2:
            # Missing team code(s) after normalization -- can't match at all.
            unmatched_games.extend(s_group["game_pk"].tolist())
            continue

        if k_group.empty:
            unmatched_games.extend(s_group["game_pk"].tolist())
            continue

        s_sorted = s_group.sort_values("min_ts")
        k_sorted = k_group.sort_values("open_time")

        # Pair chronologically. Works for the common case (1 game <-> 1
        # event) and for doubleheaders (2 games <-> 2 events), as long as
        # both sides are in the same chronological order -- true almost
        # always since neither Statcast nor Kalshi shuffles same-day games.
        n = min(len(s_sorted), len(k_sorted))
        for i in range(n):
            matches.append({
                "game_pk": s_sorted.iloc[i]["game_pk"],
                "event_ticker": k_sorted.iloc[i]["event_ticker"],
                "home_team_canon": s_sorted.iloc[i]["home_team_canon"],
            })
        if len(s_sorted) != len(k_sorted):
            # Leftover games/events on this date+matchup with no counterpart.
            leftover = s_sorted.iloc[n:]["game_pk"].tolist()
            unmatched_games.extend(leftover)

    match_df = pd.DataFrame(matches, columns=["game_pk", "event_ticker", "home_team_canon"])

    print(f"\nGame matching: {len(match_df):,} / {len(statcast_games):,} "
          f"games matched to a Kalshi event "
          f"({len(match_df) / len(statcast_games):.1%})")
    if unmatched_games:
        sample = statcast_games[statcast_games["game_pk"].isin(unmatched_games)]
        print(f"  {len(unmatched_games):,} unmatched games. Sample:")
        print(sample[["game_pk", "game_date", "home_team_canon", "away_team_canon"]]
              .head(10).to_string(index=False))

    return match_df


def attach_home_market(match_df: pd.DataFrame, kalshi: pd.DataFrame) -> pd.DataFrame:
    """For each matched game, pick the market_ticker whose team_canon equals
    the home team -- that market's price is the implied P(home team wins)."""
    if match_df.empty:
        print("  No games matched -- skipping home-market attachment. "
              "Check the TEAM_CODE_CROSSWALK warnings above.")
        return pd.DataFrame(columns=["game_pk", "event_ticker", "market_ticker"])

    market_lookup = kalshi[["event_ticker", "market_ticker", "team_canon"]].drop_duplicates()

    merged = match_df.merge(
        market_lookup,
        left_on=["event_ticker", "home_team_canon"],
        right_on=["event_ticker", "team_canon"],
        how="left",
    )

    n_missing = merged["market_ticker"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing:,} matched games had no corresponding "
              f"home-team market_ticker (check team_canon alignment).")

    return merged[["game_pk", "event_ticker", "market_ticker"]].dropna(subset=["market_ticker"])


# --------------------------------------------------
# Per-pitch as-of price join
# --------------------------------------------------

KALSHI_PRICE_COLS = [
    "market_ticker", "period_end_time",
    "price_open", "price_high", "price_low", "price_close",
    "yes_bid_close", "yes_ask_close",
    "volume", "open_interest",
]


def join_prices(pitches: pd.DataFrame, game_market_map: pd.DataFrame,
                 kalshi: pd.DataFrame) -> pd.DataFrame:
    """
    As-of join: for each pitch, attach the most recently *fully settled*
    Kalshi candle, i.e. the latest candle whose period_end_time is
    STRICTLY LESS THAN the pitch timestamp.

    A 1-minute candle closing at T encodes prices from (T-60s, T].  A pitch
    happening at time T has not yet observed that candle's closing price --
    it becomes observable only after the window ends.  Using allow_exact_
    matches=False enforces this strict inequality and eliminates the
    boundary data-leakage case.

    kalshi_price column
    -------------------
    Set to price_close (the last-trade price from the candlestick).
    Rows where price_close is NaN -- meaning no trade occurred in that
    candle window, or the 2025 historical API tier which returned zero
    price data -- are dropped entirely.  The output only contains pitches
    with an observed Kalshi market price.
    """
    pitches = pitches.merge(game_market_map, on="game_pk", how="left")

    has_ts = pitches["pitch_timestamp_utc"].notna()
    has_market = pitches["market_ticker"].notna()
    joinable = pitches[has_ts & has_market].copy()
    not_joinable = pitches[~(has_ts & has_market)].copy()

    print(f"\nPrice join: {len(joinable):,} / {len(pitches):,} pitches "
          f"have both a timestamp and a matched market "
          f"({len(joinable) / len(pitches):.1%})")

    if joinable.empty:
        return not_joinable

    kalshi_prices = kalshi[KALSHI_PRICE_COLS].dropna(subset=["period_end_time"])

    joinable = joinable.sort_values("pitch_timestamp_utc")
    kalshi_prices = kalshi_prices.sort_values("period_end_time")

    merged = pd.merge_asof(
        joinable,
        kalshi_prices,
        left_on="pitch_timestamp_utc",
        right_on="period_end_time",
        by="market_ticker",
        direction="backward",      # only look backwards in time
        allow_exact_matches=False, # strict <: a pitch at exactly T does NOT
                                   # see the candle that closed at T
    )

    merged["seconds_since_price_update"] = (
        (merged["pitch_timestamp_utc"] - merged["period_end_time"])
        .dt.total_seconds()
    )

    # Sanity check: all values must be strictly positive (candle is in the past).
    negative = (merged["seconds_since_price_update"] <= 0).sum()
    if negative:
        print(f"  WARNING: {negative:,} pitches have seconds_since_price_update <= 0 "
              f"-- this indicates a data leakage bug. Investigate before training.")

    # Diagnostic: flag suspiciously stale prices (> 30 min since last candle).
    stale = (merged["seconds_since_price_update"] > 1800).sum()
    if stale:
        print(f"  NOTE: {stale:,} pitches have a price that is >30 min old "
              f"(market may have been inactive / pre-game). Consider filtering "
              f"these in training.")

    # ------------------------------------------------------------------
    # kalshi_price = price_close (original candlestick logic).
    # Drop rows with no price -- these are entirely the 2025 historical-
    # tier candles (which returned zero price data) plus any 2026 minutes
    # with no trades.  Only keep pitches with an actual observed price.
    # ------------------------------------------------------------------
    merged["kalshi_price"] = merged["price_close"]

    n_before = len(merged)
    merged = merged[merged["kalshi_price"].notna()].copy()
    n_dropped = n_before - len(merged)
    print(f"  Dropped {n_dropped:,} pitches with no Kalshi price "
          f"(zero-price candles / 2025 historical-tier data).")
    print(f"  Retained {len(merged):,} pitches with a valid kalshi_price.")

    return merged


# --------------------------------------------------
# Chronological train / test split
# --------------------------------------------------

TRAIN_FRAC = 0.80


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df into train (first 80% of game dates) and test (last 20%).

    Splitting is done at the GAME DATE level, not the row level, so every
    pitch from a given game ends up entirely in one set.  A row-level
    random split would mix pre- and post-event context from the same game
    across the boundary, which is a subtle form of data leakage.
    """
    all_dates = sorted(df["game_date"].dropna().unique())
    cutoff_idx = int(len(all_dates) * TRAIN_FRAC)
    cutoff_date = all_dates[cutoff_idx]  # first date that goes into test

    train = df[df["game_date"] < cutoff_date].copy()
    test  = df[df["game_date"] >= cutoff_date].copy()

    print(f"\nChronological 80/20 split ({TRAIN_FRAC:.0%} train):")
    print(f"  Train: {len(all_dates[:cutoff_idx]):>3} dates  "
          f"({all_dates[0]} → {all_dates[cutoff_idx - 1]})  "
          f"{len(train):,} pitches")
    print(f"  Test:  {len(all_dates[cutoff_idx:]):>3} dates  "
          f"({cutoff_date} → {all_dates[-1]})  "
          f"{len(test):,} pitches")

    return train, test


# --------------------------------------------------
# Columns to drop from the final model dataset
# --------------------------------------------------
# These are identifiers, timestamps, and redundant fields that carry
# no learnable signal for the model.
DROP_COLS = [
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "pitch_timestamp_utc",
    "home_team_canon",
    "away_team_canon",
    "event_ticker",
    "market_ticker",
    "period_end_time",
    "price_close",
    "delta_home_win_exp"
]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    pitches = load_pitches()
    kalshi = load_kalshi()

    statcast_games = build_statcast_games(pitches)
    kalshi_events = build_kalshi_events(kalshi)

    match_df = match_games(statcast_games, kalshi_events)
    game_market_map = attach_home_market(match_df, kalshi)

    final = join_prices(pitches, game_market_map, kalshi)

    train, test = chronological_split(final)

    # Drop metadata / identifier columns before saving.
    cols_to_drop = [c for c in DROP_COLS if c in train.columns]
    train = train.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop)
    
    print(f"\nDropped {len(cols_to_drop)} metadata columns: {cols_to_drop}")
    print(f"Remaining columns ({len(train.columns)}): {train.columns.tolist()}")

    train_path = TRAIN_DIR / "training_dataset.parquet"
    test_path  = TEST_DIR  / "test_dataset.parquet"

    print(f"\nSaving train ({len(train):,} rows) -> {train_path}")
    train.to_parquet(train_path, index=False)

    print(f"Saving test  ({len(test):,} rows) -> {test_path}")
    test.to_parquet(test_path, index=False)

    print("Done!")


if __name__ == "__main__":
    main()