# scripts/download_statcast.py

from pathlib import Path
from datetime import datetime
from pybaseball import statcast
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

START_YEAR = 2019
END_YEAR = 2025

OUTPUT_DIR = Path("data/raw/statcast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def month_ranges(year):
    """
    Generate (start_date, end_date) tuples for each month.
    """
    months = []

    for month in range(1, 13):

        start = datetime(year, month, 1)

        if month == 12:
            end = datetime(year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            end = datetime(year, month + 1, 1) - pd.Timedelta(days=1)

        months.append(
            (
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
        )

    return months


def download_season(year):

    print(f"\n========== {year} ==========")

    monthly_data = []

    for start_dt, end_dt in month_ranges(year):

        print(f"Downloading {start_dt} -> {end_dt}")

        try:
            df = statcast(
                start_dt=start_dt,
                end_dt=end_dt,
                verbose=True
            )

            if len(df) == 0:
                continue

            monthly_data.append(df)

        except Exception as e:
            print(f"Failed: {start_dt} - {end_dt}")
            print(e)

    if len(monthly_data) == 0:
        print("No data found.")
        return

    season_df = pd.concat(monthly_data, ignore_index=True)

    output_file = OUTPUT_DIR / f"{year}.parquet"

    season_df.to_parquet(output_file, index=False)

    print(f"Saved {len(season_df):,} pitches")
    print(output_file)


def main():

    for year in range(START_YEAR, END_YEAR + 1):
        download_season(year)


if __name__ == "__main__":
    main()