"""
data/processed/scripts/build_kalshi_join.py

Joins Kalshi KXMLBGAME market data onto the per-pitch feature file, so each
pitch carries the market-implied home-win probability (and related fields)
as of the most recent fully-settled price update BEFORE that pitch.

Output:
    data/processed/train/training_dataset.parquet
    data/processed/test/test_dataset.parquet

INPUT ASSUMPTIONS (verify against your own diagnostics output below):
    - pitch_state_features.parquet has: game_pk, game_date, home_team,
      away_team, at_bat_number, pitch_number, pitch_timestamp_utc (tz-aware UTC).
    - kalshi_mlb_2025.parquet / kalshi_mlb_2026.parquet have identical
      schemas: event_ticker, market_ticker, team_abbr, opponent_abbr,
      game_date, open_time, close_time, period_end_time, price_close,
      yes_bid_close, yes_ask_close, volume, open_interest, ...
    - game_results_2025.parquet / game_results_2026.parquet (from
      pull_mlb_game_results.py) have: game_pk, home_runs_final,
      away_runs_final, home_win.

DATA LEAKAGE PREVENTION:
    Kalshi candles are 1-minute OHLC windows.  period_end_time is the
    CLOSE of that window.  A pitch that occurs inside a candle window has
    NOT yet seen that candle's closing price -- it only becomes observable
    after the window ends.  Therefore the as-of join uses strict
    direction="backward" with allow_exact_matches=False, meaning a pitch
    at time T only sees candles whose period_end_time < T (strictly less
    than).  A pitch that falls exactly on a candle boundary gets the
    previous candle's price, not the concurrent one.

CHANGE LOG:
    - home_win is now loaded from pull_mlb_game_results.py's authoritative
      linescore-derived output, NOT inferred from the last-by-timestamp
      Statcast pitch's score_diff. The old approach silently mis-scored
      walk-off finishes (the winning run scores on a play with no
      subsequent pitch, so no pitch's score_diff ever reflects the final
      score) and was fragile against any row with a missing
      pitch_timestamp_utc affecting which pitch sorted "last".
    - Output is now explicitly sorted by pitch_timestamp_utc before the
      train/test split and save. join_prices() sorts the joinable subset
      internally but appends the not-joinable subset afterward via concat,
      so without this final sort the saved row order wasn't reliably
      temporal -- which matters because both training scripts do a naive
      positional 80/20 split (int(len(df)*0.8)) for their internal eval
      set, silently assuming row order == time order.

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
import re


# --------------------------------------------------
# Paths
# --------------------------------------------------

GAME_STATE_DIR = Path("data/processed/mlb_game_state")
KALSHI_DIR = Path("data/raw/kalshi_historical_market_logs")
GAME_RESULTS_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_game_results"
)

TRAIN_DIR = Path("data/processed/train")
TEST_DIR = Path("data/processed/test")

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Team code normalization
# --------------------------------------------------

TEAM_CODE_CROSSWALK = {
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
    "ATHLETICS": "ATH", "AS": "ATH",
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
    "ST LOUIS": "STL",
    "TAMPA BAY": "TB",
    "TEXAS": "TEX",
    "TORONTO": "TOR",
    "WASHINGTON": "WSH",
}


def _clean(s: str) -> str:
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

    return df


def load_game_results() -> pd.DataFrame:
    """
    Authoritative home_win, from MLB's live-feed linescore (see
    pull_mlb_game_results.py) -- NOT derived from the last Statcast pitch's
    score_diff. That approach misses walk-off finishes (the winning run
    scores on a play with no subsequent pitch, so no pitch's score_diff
    ever reflects the final score) and is fragile against any row with a
    missing pitch_timestamp_utc affecting which pitch sorts "last".
    """
    frames = []
    for path in sorted(GAME_RESULTS_DIR.glob("game_results_*.parquet")):
        print(f"Loading {path}")
        frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(
            f"No game_results_*.parquet files found in {GAME_RESULTS_DIR}. "
            "Run pull_mlb_game_results.py before build_kalshi_join.py."
        )

    results = pd.concat(frames, ignore_index=True)
    dupes = results["game_pk"].duplicated().sum()
    if dupes:
        print(f"  WARNING: {dupes} duplicate game_pk in game results "
              f"(keeping first occurrence).")
        results = results.drop_duplicates(subset="game_pk", keep="first")

    print(f"Loaded {len(results):,} game results "
          f"({results['home_win'].mean():.1%} home win rate)")
    return results[["game_pk", "home_win"]]


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
    matches = []
    unmatched_games = []

    for (date, matchup), s_group in statcast_games.groupby(["game_date", "matchup"]):
        k_group = kalshi_events[
            (kalshi_events["game_date"] == date) & (kalshi_events["matchup"] == matchup)
        ]

        if len(matchup) != 2:
            unmatched_games.extend(s_group["game_pk"].tolist())
            continue

        if k_group.empty:
            unmatched_games.extend(s_group["game_pk"].tolist())
            continue

        s_sorted = s_group.sort_values("min_ts")
        k_sorted = k_group.sort_values("open_time")

        n = min(len(s_sorted), len(k_sorted))
        for i in range(n):
            matches.append({
                "game_pk": s_sorted.iloc[i]["game_pk"],
                "event_ticker": k_sorted.iloc[i]["event_ticker"],
                "home_team_canon": s_sorted.iloc[i]["home_team_canon"],
            })
        if len(s_sorted) != len(k_sorted):
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
    """Create causal decision rows at candle-close observation times.

    The old pitch-centric join combined a newly updated MLB state with an
    earlier candle and then treated that stale candle as executable. Here the
    market observation is the decision clock. It sees only a pitch state whose
    pitch start is no later than the candle close, and it retains the candle's
    actual closing bid and ask.
    """
    pitches = pitches.merge(game_market_map, on="game_pk", how="inner")
    pitches = pitches.dropna(subset=["pitch_timestamp_utc", "market_ticker"])
    pitches = pitches.sort_values(["game_pk", "pitch_timestamp_utc"])
    windows = pitches.groupby(["game_pk", "market_ticker"], as_index=False).agg(
        game_date=("game_date", "first"),
        first_pitch_time=("pitch_timestamp_utc", "min"),
        last_pitch_time=("pitch_timestamp_utc", "max"),
        pregame_home_win_exp=("home_win_exp", "first"),
    )

    candles = kalshi[KALSHI_PRICE_COLS].copy()
    candles = candles.merge(
        game_market_map[["game_pk", "market_ticker"]],
        on="market_ticker",
        how="inner",
    ).merge(windows, on=["game_pk", "market_ticker"], how="inner")
    numeric = [
        "yes_bid_close", "yes_ask_close", "volume", "open_interest",
    ]
    for column in numeric:
        candles[column] = pd.to_numeric(candles[column], errors="coerce")
    candles = candles.dropna(subset=[
        "period_end_time", "yes_bid_close", "yes_ask_close",
    ])
    valid = (
        candles["yes_bid_close"].between(0.01, 0.99)
        & candles["yes_ask_close"].between(0.01, 0.99)
        & (candles["yes_ask_close"] > candles["yes_bid_close"])
    )
    candles = candles[valid].copy()
    candles["kalshi_price"] = (
        candles["yes_bid_close"] + candles["yes_ask_close"]
    ) / 2.0
    candles["spread"] = (
        candles["yes_ask_close"] - candles["yes_bid_close"]
    )

    anchors = (
        candles[candles["period_end_time"] < candles["first_pitch_time"]]
        .sort_values(["game_pk", "period_end_time"])
        .groupby("game_pk", as_index=False)
        .tail(1)[["game_pk", "kalshi_price"]]
        .rename(columns={"kalshi_price": "pregame_prob"})
    )
    decisions = candles[
        (candles["period_end_time"] >= candles["first_pitch_time"])
        & (candles["period_end_time"] <= candles["last_pitch_time"])
    ].copy()
    decisions = decisions.merge(anchors, on="game_pk", how="inner")
    decisions["decision_time"] = decisions["period_end_time"]
    decisions["game_pk"] = decisions["game_pk"].astype("int64")

    state_drop = [
        column for column in ["market_ticker", "game_date"]
        if column in pitches.columns
    ]
    states = pitches.drop(columns=state_drop).sort_values("pitch_timestamp_utc")
    states["game_pk"] = states["game_pk"].astype("int64")
    decisions = decisions.sort_values("decision_time")
    merged = pd.merge_asof(
        decisions,
        states,
        left_on="decision_time",
        right_on="pitch_timestamp_utc",
        by="game_pk",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_state"),
    )
    merged = merged.dropna(subset=["home_win_exp"]).copy()
    merged["state_age_seconds"] = (
        merged["decision_time"] - merged["pitch_timestamp_utc"]
    ).dt.total_seconds()
    if (merged["state_age_seconds"] < 0).any():
        raise AssertionError("A decision row contains a future MLB state")
    print(
        f"\nCausal market/state join: {len(merged):,} candle decisions across "
        f"{merged['game_pk'].nunique():,} games using actual closing bid/ask."
    )
    return merged.sort_values("decision_time").reset_index(drop=True)


# --------------------------------------------------
# Chronological train / test split
# --------------------------------------------------

TRAIN_FRAC = 0.80


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = sorted(df["game_date"].dropna().unique())
    cutoff_idx = int(len(all_dates) * TRAIN_FRAC)
    cutoff_date = all_dates[cutoff_idx]

    train = df[df["game_date"] < cutoff_date].copy()
    test = df[df["game_date"] >= cutoff_date].copy()

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

DROP_COLS = [
    "home_team",
    "away_team",
    "home_team_canon",
    "away_team_canon",
    "event_ticker",
    "price_close",
    "delta_home_win_exp",
]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    pitches = load_pitches()
    game_results = load_game_results()
    kalshi = load_kalshi()

    n_before = len(pitches)
    pitches = pitches.merge(game_results, on="game_pk", how="left")
    n_missing_result = pitches["home_win"].isna().sum()
    if n_missing_result:
        print(f"  WARNING: {n_missing_result:,} / {n_before:,} pitches have no "
              f"matching game result and will be dropped.")
        pitches = pitches[pitches["home_win"].notna()].copy()
    pitches["home_win"] = pitches["home_win"].astype(int)

    statcast_games = build_statcast_games(pitches)
    kalshi_events = build_kalshi_events(kalshi)

    match_df = match_games(statcast_games, kalshi_events)
    game_market_map = attach_home_market(match_df, kalshi)

    final = join_prices(pitches, game_market_map, kalshi)

    # Sort chronologically before splitting/saving. This matters beyond
    # tidiness: both training scripts do a naive positional 80/20 split
    # (int(len(df)*0.8)) for their internal eval set, which silently
    # assumes row order == time order. join_prices sorts the joinable
    # subset but appends the not-joinable subset afterward via concat,
    # so without this the saved file's row order isn't reliably temporal.
    final = final.sort_values("decision_time").reset_index(drop=True)

    train, test = chronological_split(final)

    cols_to_drop = [c for c in DROP_COLS if c in train.columns]
    train = train.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop)

    print(f"\nDropped {len(cols_to_drop)} metadata columns: {cols_to_drop}")
    print(f"Remaining columns ({len(train.columns)}): {train.columns.tolist()}")

    train_path = TRAIN_DIR / "training_dataset.parquet"
    test_path = TEST_DIR / "test_dataset.parquet"

    print(f"\nSaving train ({len(train):,} rows) -> {train_path}")
    train.to_parquet(train_path, index=False)

    print(f"Saving test  ({len(test):,} rows) -> {test_path}")
    test.to_parquet(test_path, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
