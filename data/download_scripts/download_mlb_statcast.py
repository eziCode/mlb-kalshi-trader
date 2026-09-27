# scripts/download_mlb_statcast.py

from pathlib import Path
import argparse
from datetime import datetime
from pybaseball import statcast
import pandas as pd

# ----------------------------------------
# Configuration
# ----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data/raw/mlb_statcast"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TODAY = datetime.today()


def month_ranges(start_date, end_date):
    """
    Generate month-sized date ranges from start_date to end_date.
    """

    ranges = []

    current = start_date

    while current <= end_date:

        # First day of next month
        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1)
        else:
            next_month = datetime(current.year, current.month + 1, 1)

        # End of current chunk
        chunk_end = min(next_month - pd.Timedelta(days=1), end_date)

        ranges.append(
            (
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
        )

        current = next_month

    return ranges


def download_season(year, start_date, end_date):

    print(f"\n========== {year} ==========")

    monthly_data = []
    failures = []

    for start_dt, end_dt in month_ranges(start_date, end_date):

        print(f"Downloading {start_dt} -> {end_dt}")

        try:

            df = statcast(
                start_dt=start_dt,
                end_dt=end_dt,
                verbose=True
            )

            if df.empty:
                continue

            print(f"Downloaded {len(df):,} pitches")

            monthly_data.append(df)

        except Exception as e:

            print(f"Failed: {start_dt} -> {end_dt}")
            print(e)
            failures.append((start_dt, end_dt))

    if failures:
        raise RuntimeError(f"Incomplete Statcast download; existing data preserved: {failures}")

    if not monthly_data:
        print(f"No data found for {year}")
        return

    season_df = pd.concat(monthly_data, ignore_index=True)

    output_file = OUTPUT_DIR / f"{year}.parquet"
    # Incremental windows must not erase previously acquired seasons.
    if output_file.exists():
        season_df = pd.concat([pd.read_parquet(output_file), season_df], ignore_index=True)
    season_df = season_df.drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"], keep="last"
    ).sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    temporary = output_file.with_suffix(".parquet.tmp")
    season_df.to_parquet(temporary, index=False)
    temporary.replace(output_file)

    print(f"\nSaved {len(season_df):,} pitches")
    print(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2023)
    parser.add_argument("--end-season", type=int, default=TODAY.year)
    parser.add_argument("--start-date", type=datetime.fromisoformat)
    parser.add_argument("--end-date", type=datetime.fromisoformat)
    args = parser.parse_args()
    start_year = args.start_date.year if args.start_date else args.start_season
    end_year = args.end_date.year if args.end_date else args.end_season
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error("--start-date must not follow --end-date")
    for year in range(start_year, end_year + 1):
        start = max(args.start_date or datetime(year, 3, 1), datetime(year, 1, 1))
        end = min(args.end_date or TODAY, TODAY, datetime(year, 12, 31))
        if end < start:
            continue
        download_season(year, start, end)


if __name__ == "__main__":
    main()
