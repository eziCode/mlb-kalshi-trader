"""
data/processed/scripts/build_event_state_features.py

--------------------------------------------------------------------------
CHANGE LOG:

v1 — Kalshi join scaffolding
  Kept home_team / away_team in select_features() output so each pitch
  carries the team identity needed to look up its Kalshi market.

v2 — Authoritative MLB API timestamps
  Loads pitch_timestamps_{year}.parquet produced by
  download_mlb_pitch_timestamps.py and left-joins on
  (game_pk, at_bat_number, pitch_number) to attach a real UTC wall-clock
  timestamp (pitch_timestamp_utc) to every pitch.
--------------------------------------------------------------------------
"""

from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

STATCAST_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_statcast"
)

TIMESTAMP_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_timestamps"
)

HITTER_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/processed/mlb_hitter_data"
)

PITCHER_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/processed/mlb_pitcher_data"
)

OUTPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/processed/mlb_game_state"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load hitter rolling features
# --------------------------------------------------

def load_hitter_features():

    path = HITTER_DIR / "batter_rolling_form.parquet"
    print(f"Loading {path}")

    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    return df

def load_pitcher_features():
    path = PITCHER_DIR / "pitcher_game_logs.parquet"
    print(f"Loading {path}")
    df = pd.read_parquet(path)
    return df


# --------------------------------------------------
# Load statcast
# --------------------------------------------------

def load_statcast():

    dfs = []

    for year in [2025, 2026]:
        path = STATCAST_DIR / f"{year}.parquet"
        print(f"Loading {path}")

        df = pd.read_parquet(path)
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["batter"] = pd.to_numeric(df["batter"], errors="coerce")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} pitches")

    return df


# --------------------------------------------------
# Load MLB API pitch timestamps
# --------------------------------------------------

def load_mlb_timestamps() -> pd.DataFrame:
    """
    Load the authoritative per-pitch wall-clock timestamps produced by
    download_mlb_pitch_timestamps.py.  Returns a DataFrame with columns:
        game_pk, at_bat_number, pitch_number, start_time_utc
    ready to left-join onto the Statcast frame.
    """
    dfs = []
    for year in [2025, 2026]:
        path = TIMESTAMP_DIR / f"pitch_timestamps_{year}.parquet"
        if not path.exists():
            print(f"  WARNING: timestamp file not found, skipping: {path}")
            continue
        print(f"Loading {path}")
        df = pd.read_parquet(
            path,
            columns=["game_pk", "at_bat_number", "pitch_number", "start_time_utc"],
        )
        dfs.append(df)

    if not dfs:
        print("  WARNING: no MLB timestamp files found -- pitch_timestamp_utc "
              "will fall back entirely to sv_id.")
        return pd.DataFrame(
            columns=["game_pk", "at_bat_number", "pitch_number", "start_time_utc"]
        )

    ts = pd.concat(dfs, ignore_index=True)
    # Ensure the join keys are the same types used in the Statcast frame.
    ts["game_pk"] = ts["game_pk"].astype("int64")
    ts["at_bat_number"] = ts["at_bat_number"].astype("int64")
    ts["pitch_number"] = ts["pitch_number"].astype("int64")
    # start_time_utc comes out of parquet as tz-aware UTC datetime already;
    # make sure it is, just in case it was serialised without tz info.
    if ts["start_time_utc"].dt.tz is None:
        ts["start_time_utc"] = ts["start_time_utc"].dt.tz_localize("UTC")
    else:
        ts["start_time_utc"] = ts["start_time_utc"].dt.tz_convert("UTC")
    print(f"Loaded {len(ts):,} MLB API pitch timestamps")
    return ts


# --------------------------------------------------
# Merge authoritative timestamps
# --------------------------------------------------

def merge_pitch_timestamps(df: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the MLB API timestamps onto the Statcast frame on
    (game_pk, at_bat_number, pitch_number) and assign the result as
    pitch_timestamp_utc.  Pitches with no match get NaT.
    """
    print("Merging MLB API pitch timestamps...")

    # Coerce join keys to the same type on both sides.
    for col in ["game_pk", "at_bat_number", "pitch_number"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if ts.empty:
        print("  WARNING: no API timestamps loaded -- pitch_timestamp_utc will be all-NaT.")
        df["pitch_timestamp_utc"] = pd.NaT
        return df

    ts = ts.copy()
    for col in ["game_pk", "at_bat_number", "pitch_number"]:
        ts[col] = ts[col].astype("Int64")

    before = len(df)
    df = df.merge(
        ts.rename(columns={"start_time_utc": "pitch_timestamp_utc"}),
        on=["game_pk", "at_bat_number", "pitch_number"],
        how="left",
    )
    assert len(df) == before, (
        f"merge_pitch_timestamps produced {len(df)} rows from {before} — "
        "check for duplicate (game_pk, at_bat_number, pitch_number) in timestamp file"
    )

    matched = df["pitch_timestamp_utc"].notna().sum()
    total = len(df)
    print(
        f"  matched {matched:,} / {total:,} pitches ({matched / total:.1%}); "
        f"{total - matched:,} unmatched (pitch_timestamp_utc=NaT)"
    )

    return df


def add_completed_event_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Expose plate-appearance results no earlier than the next pitch.

    The historical MLB timestamp files contain pitch *start* times only.  A
    Statcast ``events`` value belongs to the result of that pitch, so using it
    at the pitch start would leak the future.  The next pitch start is the
    first timestamp in this dataset at which the result is certainly known.
    Final plate appearances with no subsequent pitch are intentionally absent.
    """
    df = df.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).copy()
    grouped = df.groupby("game_pk", sort=False)
    previous_event = grouped["events"].shift(1)
    event_available = previous_event.notna()

    df["completed_event"] = previous_event
    df["completed_event_at_bat"] = grouped["at_bat_number"].shift(1).where(
        event_available
    )
    df["completed_event_batting_home"] = (
        grouped["inning_topbot"].shift(1).eq("Bot").where(event_available)
    )
    df["completed_event_time"] = df["pitch_timestamp_utc"].where(
        event_available
    )
    df["completed_event_pitch_start"] = grouped[
        "pitch_timestamp_utc"
    ].shift(1).where(event_available)
    df["completed_event_sequence"] = event_available.groupby(
        df["game_pk"]
    ).cumsum()

    carry = [
        "completed_event",
        "completed_event_at_bat",
        "completed_event_batting_home",
        "completed_event_time",
        "completed_event_pitch_start",
    ]
    df[carry] = df.groupby("game_pk", sort=False)[carry].ffill()
    df["completed_event_sequence"] = (
        df["completed_event_sequence"].fillna(0).astype("int64")
    )
    return df


# --------------------------------------------------
# Create game state features
# --------------------------------------------------

def build_game_state(df):

    print("Building game state features...")

    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    df["score_diff"] = df["home_score"] - df["away_score"]

    # Vectorized instead of the previous df.apply(axis=1): equivalent output,
    # much faster at millions of rows.
    df["runner_state"] = (
        df["on_1b"].notna().astype(int).astype(str)
        + df["on_2b"].notna().astype(int).astype(str)
        + df["on_3b"].notna().astype(int).astype(str)
    )

    return df


# --------------------------------------------------
# Load hitter rolling features
# --------------------------------------------------

def merge_hitter_rolling(pitches, hitter):

    print("Joining hitter rolling form...")

    pitches["batter"] = pitches["batter"].astype("Int64")
    hitter["batter_id"] = hitter["batter_id"].astype("Int64")

    pitches = pitches.merge(
        hitter,
        how="left",
        left_on=["game_pk", "batter", "at_bat_number"],
        right_on=["game_pk", "batter_id", "at_bat_number"],
        suffixes=("", "_hitter"),
    )

    print("Hitter rolling merge complete")

    return pitches


def create_hitter_form_metrics(df):

    print("Creating hitter form metrics...")

    def build_metric(prefix):
        HR_rate = df[f"{prefix}_HR"]
        hit_rate = df[f"{prefix}_hits"]
        K_rate = df[f"{prefix}_K"]

        metric = (
            0.40 * df[f"{prefix}_wOBA"].fillna(0)
            + 0.25 * HR_rate.fillna(0)
            + 0.20 * hit_rate.fillna(0)
            + 0.10 * (df[f"{prefix}_EV"].fillna(0) / 100)
            - 0.05 * K_rate.fillna(0)
        )
        return metric

    df["hitter_form_7d"] = build_metric("last_7")
    df["hitter_form_21d"] = build_metric("last_21")

    return df


# --------------------------------------------------
# Add hitter game performance
# --------------------------------------------------

def add_game_hitter_stats(df):

    print("Adding hitter in-game stats...")

    df = df.sort_values(["game_pk", "batter", "at_bat_number", "pitch_number"])

    df["PA_hit"] = df["events"].isin(
        ["single", "double", "triple", "home_run"]
    ).astype(int)

    df["PA_HR"] = (df["events"] == "home_run").astype(int)

    df["PA_K"] = df["events"].isin(
        ["strikeout", "strikeout_double_play"]
    ).astype(int)

    df["PA_BB"] = (df["events"] == "walk").astype(int)

    groups = ["game_pk", "batter"]

    df["PA_before"] = (
        df.groupby(groups)["events"]
        .transform(lambda x: x.notna().shift(fill_value=False).astype(int).cumsum())
    )

    for source, target in [
        ("PA_hit", "hits_before"),
        ("PA_HR", "HR_before"),
        ("PA_K", "K_before"),
        ("PA_BB", "BB_before"),
    ]:
        df[target] = (
            df.groupby(groups)[source]
            .transform(lambda x: x.shift().fillna(0).cumsum())
        )

    for col in ["PA_before", "hits_before", "HR_before", "K_before", "BB_before"]:
        df[col] = df[col].astype("int64")

    df = df.rename(columns={
        "PA_before": "game_PA_before",
        "hits_before": "game_hits_before",
        "HR_before": "game_HR_before",
        "K_before": "game_K_before",
        "BB_before": "game_BB_before",
    })

    return df


# --------------------------------------------------
# Select final columns
# --------------------------------------------------

def select_features(df):

    cols = [
        # identity / join keys
        "game_pk",
        "game_date",
        "home_team",
        "away_team",

        # completed plate appearance, delayed to its first safe timestamp
        "at_bat_number",
        "completed_event",
        "completed_event_at_bat",
        "completed_event_batting_home",
        "completed_event_time",
        "completed_event_pitch_start",
        "completed_event_sequence",

        # timestamp for the Kalshi join
        "pitch_timestamp_utc",

        # game state
        "inning",
        "inning_topbot",
        "outs_when_up",
        "score_diff",
        "runner_state",
        "balls",
        "strikes",

        # hitter rolling
        "hitter_form_7d",
        "hitter_form_21d",

        # hitter game
        # "game_PA_before",
        # "game_hits_before",
        # "game_HR_before",
        # "game_K_before",
        # "game_BB_before",

        # pitch
        "pitch_number",

        # pitcher
        "pitcher_is_starter",
        "pitcher_game_pitch_count",
        "hist_entry_score_diff",

        # MLB's pre-calculated Win Probability
        "home_win_exp",

        # target
        "delta_home_win_exp",
    ]

    cols = [c for c in cols if c in df.columns]
    return df[cols]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    statcast = load_statcast()

    mlb_ts = load_mlb_timestamps()
    statcast = merge_pitch_timestamps(statcast, mlb_ts)

    statcast = add_completed_event_availability(statcast)

    statcast = build_game_state(statcast)

    hitter = load_hitter_features()
    statcast = merge_hitter_rolling(statcast, hitter)
    statcast = create_hitter_form_metrics(statcast)
    
    pitcher = load_pitcher_features()
    print("Merging pitcher features...")
    statcast = statcast.merge(pitcher, on=["game_pk", "at_bat_number", "pitch_number"], how="left")

    statcast["hit"] = statcast["events"].isin(
        ["single", "double", "triple", "home_run"]
    ).astype(int)

    statcast["home_run"] = (statcast["events"] == "home_run").astype(int)

    statcast["strikeout"] = statcast["events"].isin(
        ["strikeout", "strikeout_double_play"]
    ).astype(int)

    statcast["walk"] = (statcast["events"] == "walk").astype(int)

    statcast = add_game_hitter_stats(statcast)

    final = select_features(statcast)

    output = OUTPUT_DIR / "pitch_state_features.parquet"
    print(f"Saving {len(final):,} rows")
    final.to_parquet(output, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
