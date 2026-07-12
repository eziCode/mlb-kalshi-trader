"""
data/processed/scripts/build_mlb_pitcher_data.py

Output Files:
1. pitcher_game_logs.parquet
   - Pitch-by-pitch tracking of game pitch count and starter flag.
   - Contains historical entry score differential to determine if a reliever is a closer or mop-up guy.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = Path("/Users/ezraakresh/Documents/mlb-kalshi-trader/data/raw/mlb_statcast")
OUTPUT_DIR = Path("/Users/ezraakresh/Documents/mlb-kalshi-trader/data/processed/mlb_pitcher_data")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_statcast():
    files = [
        INPUT_DIR / "2025.parquet",
        INPUT_DIR / "2026.parquet",
    ]
    dfs = []
    for file in files:
        if file.exists():
            print(f"Loading {file}")
            df = pd.read_parquet(file)
            df["game_date"] = pd.to_datetime(df["game_date"])
            df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce")
            dfs.append(df)
        else:
            print(f"Warning: {file} not found.")
            
    if not dfs:
        raise FileNotFoundError("No statcast parquet files found.")
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} pitches")
    return df

def build_pitcher_features(df):
    print("Sorting pitches by game and time...")
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    
    print("Building game-level pitch counts and starter flags...")
    
    # Who was the first pitcher for each team in each game?
    first_pitches = df.drop_duplicates(subset=["game_pk", "inning_topbot"])
    starters = first_pitches[["game_pk", "pitcher"]].copy()
    starters["pitcher_is_starter"] = 1
    
    # Merge starter flag back onto every pitch
    df = df.merge(starters, on=["game_pk", "pitcher"], how="left")
    df["pitcher_is_starter"] = df["pitcher_is_starter"].fillna(0).astype(int)
    
    # Calculate pitch count for this pitcher in this game
    df["pitcher_game_pitch_count"] = df.groupby(["game_pk", "pitcher"]).cumcount() + 1
    
    # ---------------------------------------------------------
    # Game Script / Reliever Quality: Average Entry Score Diff
    # ---------------------------------------------------------
    print("Calculating rolling reliever entry contexts...")
    
    # We only care about when a reliever ENTERS the game.
    # The first pitch thrown by a pitcher in a game is their entry point.
    entries = df.drop_duplicates(subset=["game_pk", "pitcher"]).copy()
    
    # Filter to only relievers
    reliever_entries = entries[entries["pitcher_is_starter"] == 0].copy()
    
    # Calculate the score differential from the PITCHING team's perspective
    # 'fld_score' is the fielding (pitching) team's score, 'bat_score' is the hitting team's score.
    if "bat_score" in reliever_entries.columns and "fld_score" in reliever_entries.columns:
        reliever_entries["entry_score_diff"] = reliever_entries["fld_score"] - reliever_entries["bat_score"]
    else:
        # Fallback if bat/fld score isn't there (though it should be in statcast)
        reliever_entries["entry_score_diff"] = 0 
        
    # We want a historical expanding average of entry_score_diff per pitcher.
    reliever_entries = reliever_entries.sort_values(["game_date", "game_pk"])
    
    # Shift 1 ensures we only average games strictly BEFORE the current game!
    reliever_entries["hist_entry_score_diff"] = reliever_entries.groupby("pitcher")["entry_score_diff"].transform(
        lambda x: x.expanding().mean().shift(1) 
    )
    
    # Fill NaN for first-time relievers with 0 (neutral leverage)
    reliever_entries["hist_entry_score_diff"] = reliever_entries["hist_entry_score_diff"].fillna(0.0)
    
    # Select columns to merge back
    entry_features = reliever_entries[["game_pk", "pitcher", "hist_entry_score_diff"]]
    
    # Merge back to the main df
    df = df.merge(entry_features, on=["game_pk", "pitcher"], how="left")
    
    # Starters get a neutral 0 for hist_entry_score_diff since the score is always 0-0
    df["hist_entry_score_diff"] = df["hist_entry_score_diff"].fillna(0.0)
    
    # Export pitcher features
    out_cols = [
        "game_pk", "at_bat_number", "pitch_number", "pitcher", 
        "pitcher_is_starter", "pitcher_game_pitch_count", "hist_entry_score_diff"
    ]
    
    # Ensure all required columns exist before filtering
    out_cols = [c for c in out_cols if c in df.columns]
    
    out_df = df[out_cols]
    return out_df

def main():
    df = load_statcast()
    pitcher_df = build_pitcher_features(df)
    
    out_path = OUTPUT_DIR / "pitcher_game_logs.parquet"
    print(f"Saving pitcher features to {out_path}...")
    pitcher_df.to_parquet(out_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
