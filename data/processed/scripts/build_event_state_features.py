from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

STATCAST_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_statcast"
)

HITTER_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/processed/mlb_hitter_data"
)

OUTPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/processed/mlb_game_state"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# --------------------------------------------------
# Load hitter rolling features
# --------------------------------------------------

# --------------------------------------------------
# Load hitter rolling features
# --------------------------------------------------

def load_hitter_features():

    path = (
        HITTER_DIR /
        "batter_rolling_form.parquet"
    )

    print(
        f"Loading {path}"
    )

    df = pd.read_parquet(path)

    df["game_date"] = pd.to_datetime(
        df["game_date"]
    )

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


    print(
        f"Loaded {len(df):,} pitches"
    )

    return df



# --------------------------------------------------
# Create game state features
# --------------------------------------------------

def build_game_state(df):

    print("Building game state features...")


    df = df.sort_values(
        [
            "game_pk",
            "at_bat_number",
            "pitch_number"
        ]
    )


    df["score_diff"] = (
        df["home_score"]
        -
        df["away_score"]
    )


    def encode_runners(row):

        return (
            str(int(pd.notna(row["on_1b"])))
            +
            str(int(pd.notna(row["on_2b"])))
            +
            str(int(pd.notna(row["on_3b"])))
        )


    df["runner_state"] = df.apply(
        encode_runners,
        axis=1
    )


    return df



# --------------------------------------------------
# Load hitter rolling features
# --------------------------------------------------

def merge_hitter_rolling(
    pitches,
    hitter
):

    print(
        "Joining hitter rolling form..."
    )


    pitches["batter"] = (
        pitches["batter"]
        .astype("Int64")
    )

    hitter["batter_id"] = (
        hitter["batter_id"]
        .astype("Int64")
    )


    pitches = pitches.merge(
        hitter,
        how="left",
        left_on=[
            "game_pk",
            "batter",
            "at_bat_number"
        ],
        right_on=[
            "game_pk",
            "batter_id",
            "at_bat_number"
        ],
        suffixes=("", "_hitter")
    )


    print(
        "Hitter rolling merge complete"
    )


    return pitches


def create_hitter_form_metrics(df):

    print(
        "Creating hitter form metrics..."
    )


    def build_metric(prefix):

        PA = df[f"{prefix}_PA"].replace(
            0,
            np.nan
        )

        HR_rate = (
            df[f"{prefix}_HR"]
            /
            PA
        )

        hit_rate = (
            df[f"{prefix}_hits"]
            /
            PA
        )

        K_rate = (
            df[f"{prefix}_K"]
            /
            PA
        )


        metric = (
            0.40 *
            df[f"{prefix}_wOBA"].fillna(0)
            +
            0.25 *
            HR_rate.fillna(0)
            +
            0.20 *
            hit_rate.fillna(0)
            +
            0.10 *
            (
                df[f"{prefix}_EV"]
                .fillna(0)
                /
                100
            )
            -
            0.05 *
            K_rate.fillna(0)
        )


        return metric



    df["hitter_form_7d"] = (
        build_metric("last_7")
    )


    df["hitter_form_21d"] = (
        build_metric("last_21")
    )


    return df


# --------------------------------------------------
# Add hitter game performance
# --------------------------------------------------

def add_game_hitter_stats(df):

    print(
        "Adding hitter in-game stats..."
    )


    df = df.sort_values(
        [
            "game_pk",
            "batter",
            "at_bat_number",
            "pitch_number"
        ]
    )


    # Create PA outcome indicators only
    df["PA_hit"] = (
        df["events"]
        .isin(
            [
                "single",
                "double",
                "triple",
                "home_run"
            ]
        )
    ).astype(int)


    df["PA_HR"] = (
        df["events"]
        ==
        "home_run"
    ).astype(int)


    df["PA_K"] = (
        df["events"]
        .isin(
            [
                "strikeout",
                "strikeout_double_play"
            ]
        )
    ).astype(int)


    df["PA_BB"] = (
        df["events"]
        ==
        "walk"
    ).astype(int)


    # Previous completed PA stats
    groups = [
        "game_pk",
        "batter"
    ]


    df["PA_before"] = (
        df.groupby(groups)
        ["events"]
        .transform(
            lambda x:
            x.notna()
            .shift(fill_value=False)
            .astype(int)
            .cumsum()
        )
    )


    for source, target in [
        ("PA_hit", "hits_before"),
        ("PA_HR", "HR_before"),
        ("PA_K", "K_before"),
        ("PA_BB", "BB_before"),
    ]:

        df[target] = (
            df.groupby(groups)[source]
            .transform(
                lambda x:
                x.shift()
                .fillna(0)
                .cumsum()
            )
        )

    for col in [
        "PA_before",
        "hits_before",
        "HR_before",
        "K_before",
        "BB_before"
    ]:
        df[col] = (
            df[col]
            .astype("int64")
        )


    return df



# --------------------------------------------------
# Select final columns
# --------------------------------------------------

def select_features(df):

    cols = [

        # game state
        "game_pk",
        "game_date",

        "inning",
        "inning_topbot",

        "outs_when_up",

        "score_diff",
        "runner_state",

        "balls",
        "strikes",

        # hitter rolling
        *[
            "hitter_form_7d",
            "hitter_form_21d"
        ],

        # hitter game
        "game_PA_before",
        "game_hits_before",
        "game_HR_before",
        "game_K_before",
        "game_BB_before",

        # pitch
        "pitch_number",
        "description",
        "events",

        # target
        "delta_home_win_exp"

    ]


    cols = [
        c for c in cols
        if c in df.columns
    ]


    return df[cols]



# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    statcast = load_statcast()


    statcast = build_game_state(
        statcast
    )


    hitter = load_hitter_features()


    statcast = merge_hitter_rolling(
        statcast,
        hitter
    )
    
    statcast = create_hitter_form_metrics(
        statcast
    )


    # create outcome indicators
    statcast["hit"] = statcast["events"].isin(
        [
            "single",
            "double",
            "triple",
            "home_run"
        ]
    ).astype(int)


    statcast["home_run"] = (
        statcast["events"]
        ==
        "home_run"
    ).astype(int)


    statcast["strikeout"] = (
        statcast["events"]
        .isin(
            [
                "strikeout",
                "strikeout_double_play"
            ]
        )
        .astype(int)
    )


    statcast["walk"] = (
        statcast["events"]
        ==
        "walk"
    ).astype(int)


    statcast = add_game_hitter_stats(
        statcast
    )


    final = select_features(
        statcast
    )


    output = (
        OUTPUT_DIR /
        "pitch_state_features.parquet"
    )


    print(
        f"Saving {len(final):,} rows"
    )


    final.to_parquet(
        output,
        index=False
    )


    print("Done!")



if __name__ == "__main__":
    main()