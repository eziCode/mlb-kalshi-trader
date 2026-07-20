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


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATCAST_DIR = PROJECT_ROOT / "data/raw/mlb_statcast"
TIMESTAMP_DIR = PROJECT_ROOT / "data/raw/mlb_timestamps"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/mlb_game_state"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load statcast
# --------------------------------------------------

def load_statcast():

    dfs = []

    for path in sorted(STATCAST_DIR.glob("*.parquet")):
        print(f"Loading {path}")

        df = pd.read_parquet(path)
        df["game_date"] = pd.to_datetime(df["game_date"])
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
    for path in sorted(TIMESTAMP_DIR.glob("pitch_timestamps_*.parquet")):
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
        "home_score",
        "away_score",
        "post_home_score",
        "post_away_score",
        "runner_state",
        "balls",
        "strikes",

        # pitch
        "pitch_number",

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

    final = select_features(statcast)

    output = OUTPUT_DIR / "pitch_state_features.parquet"
    print(f"Saving {len(final):,} rows")
    final.to_parquet(output, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
