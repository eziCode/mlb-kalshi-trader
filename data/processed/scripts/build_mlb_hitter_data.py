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

--------------------------------------------------------------------------
CHANGE LOG (fix for in-sample league-average leakage):

Previously, the shrinkage prior (league_woba, league_hit_rate, etc.) was
computed ONCE as a single mean over the entire combined 2025+2026 dataset,
before the per-batter loop ran. That means a plate appearance from April
2025 was being shrunk toward a league average that already "knew about"
every game through the end of the loaded 2026 data -- future information
a real-time model would never actually have.

Fixed by replacing that single global mean with a per-date, EXPANDING
league-average table: for any given date, the league rate used is
computed only from PAs whose game_date is strictly before that date.
This uses only the data already loaded (no external historical file),
and satisfies "don't fail at the start of the season" by falling back to
fixed neutral constants (see NEUTRAL_PRIORS below) only for the very
first date in the entire loaded dataset, where zero prior PAs of any
kind exist yet. That fallback never depends on future data either --
it's a fixed constant, not something computed from the dataset.

Also rewrote the per-batter rolling-window computation to use expiring
sliding windows (append/pop with running sums) instead of rebuilding a
full DataFrame from a growing Python list every row. Same output, but
O(n) instead of O(n^2) per batter, which matters once you're re-running
this across a full season repeatedly.
--------------------------------------------------------------------------
"""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_statcast"
)

OUTPUT_DIR = Path(
    "/Users/ezraakresh/Documents/mlb-kalshi-trader/data/processed/mlb_hitter_data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rolling windows, in days
WINDOWS = {
    "7": 7,
    "14": 14,
    "21": 21,
    "30": 30,
}

# Fixed, non-data-derived fallback used ONLY when zero prior PAs exist
# anywhere in the loaded dataset (i.e. the very first date of the whole
# sample, before any game has been played). These are just reasonable
# modern-MLB ballpark figures, not fit from your data, so they can't leak
# anything -- they're a bootstrap constant, not a historical file.
NEUTRAL_PRIORS = {
    "hit_rate": 0.240,
    "hr_rate": 0.035,
    "bb_rate": 0.085,
    "k_rate": 0.225,
    "woba": 0.310,
    "ev": 88.0,
}


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
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["batter"] = pd.to_numeric(df["batter"], errors="coerce")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} pitches")
    return df


# --------------------------------------------------
# Convert pitch data -> plate appearances
# --------------------------------------------------

def build_plate_appearances(df):

    print("Building plate appearances...")

    df = df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )

    # Only final pitch of plate appearance
    pa = df[df["events"].notna()].copy()

    pa["batter_id"] = pa["batter"].astype("int64")

    pa["hit"] = pa["events"].isin(
        ["single", "double", "triple", "home_run"]
    ).astype(int)

    pa["single"] = (pa["events"] == "single").astype(int)
    pa["double"] = (pa["events"] == "double").astype(int)
    pa["triple"] = (pa["events"] == "triple").astype(int)
    pa["home_run"] = (pa["events"] == "home_run").astype(int)
    pa["walk"] = (pa["events"] == "walk").astype(int)

    pa["strikeout"] = pa["events"].isin(
        ["strikeout", "strikeout_double_play"]
    ).astype(int)

    return pa


# --------------------------------------------------
# Batter game logs
# --------------------------------------------------

def create_game_logs(pa):

    print("Creating batter game logs...")

    cols = [
        "game_pk", "game_date",
        "batter_id", "player_name",
        "inning", "inning_topbot", "at_bat_number",
        "events",
        "hit", "single", "double", "triple", "home_run",
        "walk", "strikeout",
        "launch_speed", "launch_angle",
        "woba_value", "estimated_woba_using_speedangle",
    ]

    game_logs = pa[cols].copy()
    game_logs = game_logs.rename(columns={"player_name": "batter_name"})
    game_logs["game_date"] = pd.to_datetime(game_logs["game_date"])
    game_logs["batter_id"] = game_logs["batter_id"].astype("int64")

    return game_logs


# --------------------------------------------------
# Causal (non-leaking) league-average prior table
# --------------------------------------------------

def build_league_prior_table(game_logs):
    """
    Returns a dict: game_date -> {hit_rate, hr_rate, bb_rate, k_rate, woba, ev}
    where each rate is computed only from PAs strictly BEFORE that date.

    For the earliest date in the dataset (no prior PAs exist at all), falls
    back to NEUTRAL_PRIORS -- a fixed constant, not derived from any data,
    so there's no lookahead even on day one.
    """
    print("Building causal league-average prior table...")

    daily = game_logs.groupby("game_date").agg(
        pa=("hit", "size"),
        hits=("hit", "sum"),
        hr=("home_run", "sum"),
        bb=("walk", "sum"),
        k=("strikeout", "sum"),
        woba_sum=("woba_value", "sum"),
        woba_count=("woba_value", "count"),
        ev_sum=("launch_speed", "sum"),
        ev_count=("launch_speed", "count"),
    ).sort_index()

    # Cumulative totals THROUGH each date, then shift by one row so that
    # the value attached to date d reflects everything strictly BEFORE d
    # (dates are one row per distinct game_date, so a positional shift(1)
    # is exactly "as of the previous game date").
    cum_through = daily.cumsum()
    cum_before = cum_through.shift(1).fillna(0)

    prior_table = {}
    for date, row in cum_before.iterrows():
        if row["pa"] > 0:
            hit_rate = row["hits"] / row["pa"]
            hr_rate = row["hr"] / row["pa"]
            bb_rate = row["bb"] / row["pa"]
            k_rate = row["k"] / row["pa"]
            woba = (row["woba_sum"] / row["woba_count"]
                    if row["woba_count"] > 0 else NEUTRAL_PRIORS["woba"])
            ev = (row["ev_sum"] / row["ev_count"]
                  if row["ev_count"] > 0 else NEUTRAL_PRIORS["ev"])
        else:
            # True start of the whole dataset: nothing has happened yet.
            hit_rate = NEUTRAL_PRIORS["hit_rate"]
            hr_rate = NEUTRAL_PRIORS["hr_rate"]
            bb_rate = NEUTRAL_PRIORS["bb_rate"]
            k_rate = NEUTRAL_PRIORS["k_rate"]
            woba = NEUTRAL_PRIORS["woba"]
            ev = NEUTRAL_PRIORS["ev"]

        prior_table[date] = {
            "hit_rate": hit_rate, "hr_rate": hr_rate,
            "bb_rate": bb_rate, "k_rate": k_rate,
            "woba": woba, "ev": ev,
        }

    return prior_table


# --------------------------------------------------
# Rolling hitter form (efficient, causal)
# --------------------------------------------------

class _WindowState:
    """Running sums for one rolling window, backed by a deque so
    entries can be popped off the front in O(1) as they age out."""

    __slots__ = ("days", "buf", "pa", "hit", "hr", "bb", "k",
                 "woba_sum", "woba_n", "ev_sum", "ev_n")

    def __init__(self, days):
        self.days = days
        self.buf = deque()  # (date, hit, hr, bb, k, woba, ev)
        self.pa = 0
        self.hit = 0
        self.hr = 0
        self.bb = 0
        self.k = 0
        self.woba_sum = 0.0
        self.woba_n = 0
        self.ev_sum = 0.0
        self.ev_n = 0

    def expire(self, current_date):
        cutoff = current_date - pd.Timedelta(days=self.days)
        while self.buf and self.buf[0][0] < cutoff:
            _, hit, hr, bb, k, woba, ev = self.buf.popleft()
            self.pa -= 1
            self.hit -= hit
            self.hr -= hr
            self.bb -= bb
            self.k -= k
            if not pd.isna(woba):
                self.woba_sum -= woba
                self.woba_n -= 1
            if not pd.isna(ev):
                self.ev_sum -= ev
                self.ev_n -= 1

    def push(self, date, hit, hr, bb, k, woba, ev):
        self.buf.append((date, hit, hr, bb, k, woba, ev))
        self.pa += 1
        self.hit += hit
        self.hr += hr
        self.bb += bb
        self.k += k
        if not pd.isna(woba):
            self.woba_sum += woba
            self.woba_n += 1
        if not pd.isna(ev):
            self.ev_sum += ev
            self.ev_n += 1

    def rates(self):
        if self.pa > 0:
            hit_rate = self.hit / self.pa
            hr_rate = self.hr / self.pa
            bb_rate = self.bb / self.pa
            k_rate = self.k / self.pa
        else:
            hit_rate = hr_rate = bb_rate = k_rate = 0.0
        woba = self.woba_sum / self.woba_n if self.woba_n > 0 else 0.0
        ev = self.ev_sum / self.ev_n if self.ev_n > 0 else 0.0
        return self.pa, hit_rate, hr_rate, bb_rate, k_rate, woba, ev


def calculate_rolling_form(game_logs):

    print("Calculating rolling hitter form...")

    game_logs = game_logs.sort_values(
        ["batter_id", "game_date", "game_pk", "at_bat_number"]
    )

    league_prior = build_league_prior_table(game_logs)

    results = []

    for batter_id, group in game_logs.groupby("batter_id"):

        windows = {name: _WindowState(days) for name, days in WINDOWS.items()}

        for row in group.itertuples(index=False):
            current_date = row.game_date
            priors = league_prior[current_date]

            feature = {
                "game_pk": row.game_pk,
                "game_date": current_date,
                "batter_id": batter_id,
                "batter_name": row.batter_name,
                "at_bat_number": row.at_bat_number,
            }

            woba_val = row.woba_value if row.woba_value is not None else np.nan
            ev_val = row.launch_speed if row.launch_speed is not None else np.nan

            for window_name, state in windows.items():
                state.expire(current_date)

                pa, hit_rate, hr_rate, bb_rate, k_rate, woba, ev = state.rates()
                weight = min(pa / 50.0, 1.0)

                if pa == 0:
                    hit_rate, hr_rate, bb_rate, k_rate = (
                        priors["hit_rate"], priors["hr_rate"],
                        priors["bb_rate"], priors["k_rate"],
                    )
                    woba, ev = priors["woba"], priors["ev"]

                # Empirical Bayes shrinkage toward the CAUSAL (as-of-date)
                # league prior, not a full-sample mean.
                hit_rate = weight * hit_rate + (1 - weight) * priors["hit_rate"]
                hr_rate = weight * hr_rate + (1 - weight) * priors["hr_rate"]
                bb_rate = weight * bb_rate + (1 - weight) * priors["bb_rate"]
                k_rate = weight * k_rate + (1 - weight) * priors["k_rate"]
                woba = weight * woba + (1 - weight) * priors["woba"]
                ev = weight * ev + (1 - weight) * priors["ev"]

                feature[f"last_{window_name}_PA"] = pa
                feature[f"last_{window_name}_hits"] = hit_rate
                feature[f"last_{window_name}_HR"] = hr_rate
                feature[f"last_{window_name}_BB"] = bb_rate
                feature[f"last_{window_name}_K"] = k_rate
                feature[f"last_{window_name}_wOBA"] = woba
                feature[f"last_{window_name}_EV"] = ev

                # Push this PA into the window AFTER computing the feature,
                # so the current PA never leaks into its own feature.
                state.push(
                    current_date, row.hit, row.home_run, row.walk,
                    row.strikeout, woba_val, ev_val,
                )

            results.append(feature)

    return pd.DataFrame(results)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    statcast = load_statcast()
    pa = build_plate_appearances(statcast)
    game_logs = create_game_logs(pa)

    print(game_logs.dtypes)

    rolling = calculate_rolling_form(game_logs)

    print("Saving parquet files...")

    game_logs.to_parquet(OUTPUT_DIR / "batter_game_logs.parquet", index=False)
    rolling.to_parquet(OUTPUT_DIR / "batter_rolling_form.parquet", index=False)

    print("\nFinished!")
    print(f"Game logs: {len(game_logs):,} rows")
    print(f"Rolling form: {len(rolling):,} rows")


if __name__ == "__main__":
    main()