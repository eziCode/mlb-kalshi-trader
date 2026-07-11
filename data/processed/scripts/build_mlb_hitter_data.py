"""
data/processed/scripts/build_mlb_hitter_data.py

Output Files:

1. batter_game_logs.parquet
   - Grain: one row per batter plate appearance.
   - Contains what happened during each at-bat:
       * game information
       * batter ID
       * inning / PA number
       * result (single, HR, strikeout, walk, etc.)
       * exit velocity, launch angle, wOBA
   - Used to track a player's performance within a game.

2. batter_rolling_form.parquet
   - Grain: one row per batter plate appearance.
   - Contains hitter performance metrics BEFORE each PA.
   - Includes rolling 7, 14, 21, and 30 day stats:
       * PA count
       * hits
       * HR
       * walks
       * strikeouts
       * wOBA
       * average exit velocity
   - Used as model features to measure recent hitter form
     without leaking future information.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_statcast"
)

OUTPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/processed/mlb_player_data/hitters"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load Statcast data
# --------------------------------------------------

def load_statcast():

    files = [
        INPUT_DIR / "2025.parquet",
        INPUT_DIR / "2026.parquet",
    ]

    dfs = []

    for file in files:
        print(f"Loading {file}")

        df = pd.read_parquet(file)

        # Normalize types immediately
        df["game_date"] = pd.to_datetime(
            df["game_date"]
        )

        df["batter"] = pd.to_numeric(
            df["batter"],
            errors="coerce"
        )

        dfs.append(df)

    df = pd.concat(
        dfs,
        ignore_index=True
    )

    print(f"Loaded {len(df):,} pitches")

    return df



# --------------------------------------------------
# Convert pitch data -> plate appearances
# --------------------------------------------------

def build_plate_appearances(df):

    print("Building plate appearances...")


    df = df.sort_values(
        [
            "game_date",
            "game_pk",
            "at_bat_number",
            "pitch_number",
        ]
    )


    # Only final pitch of plate appearance
    pa = df[
        df["events"].notna()
    ].copy()


    pa["batter_id"] = pa["batter"].astype(
        "int64"
    )


    # Outcome indicators

    pa["hit"] = pa["events"].isin(
        [
            "single",
            "double",
            "triple",
            "home_run",
        ]
    ).astype(int)


    pa["single"] = (
        pa["events"] == "single"
    ).astype(int)


    pa["double"] = (
        pa["events"] == "double"
    ).astype(int)


    pa["triple"] = (
        pa["events"] == "triple"
    ).astype(int)


    pa["home_run"] = (
        pa["events"] == "home_run"
    ).astype(int)


    pa["walk"] = (
        pa["events"] == "walk"
    ).astype(int)


    pa["strikeout"] = pa["events"].isin(
        [
            "strikeout",
            "strikeout_double_play",
        ]
    ).astype(int)


    return pa



# --------------------------------------------------
# Batter game logs
# --------------------------------------------------

def create_game_logs(pa):

    print("Creating batter game logs...")


    cols = [

        "game_pk",
        "game_date",

        "batter_id",
        "player_name",

        "inning",
        "inning_topbot",
        "at_bat_number",

        "events",

        "hit",
        "single",
        "double",
        "triple",
        "home_run",

        "walk",
        "strikeout",

        "launch_speed",
        "launch_angle",

        "woba_value",
        "estimated_woba_using_speedangle",

    ]


    game_logs = pa[cols].copy()


    game_logs = game_logs.rename(
        columns={
            "player_name": "batter_name"
        }
    )


    # Normalize types
    game_logs["game_date"] = pd.to_datetime(
        game_logs["game_date"]
    )


    game_logs["batter_id"] = (
        game_logs["batter_id"]
        .astype("int64")
    )


    return game_logs



# --------------------------------------------------
# Rolling hitter form
# --------------------------------------------------

def calculate_rolling_form(game_logs):

    print("Calculating rolling hitter form...")


    game_logs = game_logs.sort_values(
        [
            "batter_id",
            "game_date",
            "game_pk",
            "at_bat_number",
        ]
    )


    windows = {
        "7": 7,
        "14": 14,
        "21": 21,
        "30": 30,
    }


    results = []


    for batter_id, group in game_logs.groupby(
        "batter_id"
    ):

        history = []


        for _, row in group.iterrows():

            current_date = row["game_date"]


            feature = {

                "game_pk": row["game_pk"],
                "game_date": current_date,

                "batter_id": batter_id,
                "batter_name": row["batter_name"],

                "at_bat_number": row["at_bat_number"],

            }


            if history:

                history_df = pd.DataFrame(history)

                history_df["game_date"] = pd.to_datetime(
                    history_df["game_date"]
                )

            else:

                history_df = pd.DataFrame()



            for window_name, days in windows.items():

                if history_df.empty:

                    recent = history_df

                else:

                    recent = history_df[
                        history_df["game_date"]
                        >=
                        current_date -
                        pd.Timedelta(days=days)
                    ]


                feature[
                    f"last_{window_name}_PA"
                ] = len(recent)


                feature[
                    f"last_{window_name}_hits"
                ] = (
                    recent["hit"].sum()
                    if len(recent)
                    else 0
                )


                feature[
                    f"last_{window_name}_HR"
                ] = (
                    recent["home_run"].sum()
                    if len(recent)
                    else 0
                )


                feature[
                    f"last_{window_name}_BB"
                ] = (
                    recent["walk"].sum()
                    if len(recent)
                    else 0
                )


                feature[
                    f"last_{window_name}_K"
                ] = (
                    recent["strikeout"].sum()
                    if len(recent)
                    else 0
                )


                feature[
                    f"last_{window_name}_wOBA"
                ] = (
                    recent["woba_value"].mean()
                    if len(recent)
                    else np.nan
                )


                feature[
                    f"last_{window_name}_EV"
                ] = (
                    recent["launch_speed"].mean()
                    if len(recent)
                    else np.nan
                )


            results.append(feature)


            # Add this PA AFTER calculating features
            history.append(row)


    return pd.DataFrame(results)



# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    statcast = load_statcast()


    pa = build_plate_appearances(
        statcast
    )


    game_logs = create_game_logs(
        pa
    )


    print(game_logs.dtypes)


    rolling = calculate_rolling_form(
        game_logs
    )


    print("Saving parquet files...")


    game_logs.to_parquet(
        OUTPUT_DIR /
        "batter_game_logs.parquet",
        index=False
    )


    rolling.to_parquet(
        OUTPUT_DIR /
        "batter_rolling_form.parquet",
        index=False
    )


    print("\nFinished!")
    print(
        f"Game logs: {len(game_logs):,} rows"
    )
    print(
        f"Rolling form: {len(rolling):,} rows"
    )



if __name__ == "__main__":
    main()