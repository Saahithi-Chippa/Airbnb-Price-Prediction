"""Pipeline runner: load cleaned data → engineer features → select features → save."""

import time
from pathlib import Path

import pandas as pd

from engineering import create_features, select_features

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INPUT_PATH = DATA_DIR / "cleaned.csv"
OUTPUT_PATH = DATA_DIR / "features.csv"

NON_FEATURE_COLS = [
    "Unnamed: 0", "track_id", "artists", "album_name", "track_name", "track_genre"
]


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"[1/4] Loading {INPUT_PATH.name} ...")
    df_raw = pd.read_csv(INPUT_PATH)
    print(f"      Shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

    # ------------------------------------------------------------------
    # Engineer
    # ------------------------------------------------------------------
    print(f"\n[2/4] Engineering features ...")
    t_eng = time.time()
    df_engineered = create_features(df_raw)
    new_cols = [c for c in df_engineered.columns if c not in df_raw.columns]
    print(f"      Added {len(new_cols)} features in {time.time() - t_eng:.2f}s")
    print(f"      Shape: {df_engineered.shape[0]:,} rows x {df_engineered.shape[1]} columns")

    # ------------------------------------------------------------------
    # Select
    # ------------------------------------------------------------------
    print(f"\n[3/4] Selecting features ...")
    t_sel = time.time()
    selected_names, df_selected = select_features(
        df_engineered,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        exclude_cols=NON_FEATURE_COLS,
    )
    dropped_count = df_engineered.select_dtypes(include="number").shape[1] - len(selected_names)
    print(f"      Dropped {dropped_count} numeric features in {time.time() - t_sel:.2f}s")
    print(f"      Shape: {df_selected.shape[0]:,} rows x {df_selected.shape[1]} columns")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving to {OUTPUT_PATH.name} ...")
    df_selected.to_csv(OUTPUT_PATH, index=False)
    print(f"      Saved {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*52}")
    print(f"  FEATURE PIPELINE COMPLETE  ({elapsed:.2f}s)")
    print(f"{'='*52}")
    print(f"  Input shape   : {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    print(f"  Output shape  : {df_selected.shape[0]:,} rows x {df_selected.shape[1]} columns")
    print(f"\n  Kept features ({len(selected_names)}):")
    for name in selected_names:
        print(f"    {name}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
