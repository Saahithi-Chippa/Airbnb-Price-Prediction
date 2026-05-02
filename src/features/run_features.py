import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data.loader import load_csv  # noqa: E402
from features.engineering import create_features, select_features  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def run_feature_pipeline() -> None:
    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading data/cleaned.csv ...")
    df_clean = load_csv("cleaned.csv")
    print(f"  Input shape:  {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")

    # ── Engineer ──────────────────────────────────────────────────────────
    print("\nRunning create_features ...")
    t1 = time.time()
    df_feat = create_features(df_clean)
    new_cols = [c for c in df_feat.columns if c not in df_clean.columns]
    print(f"  {len(new_cols)} features added in {time.time() - t1:.2f}s")
    print(f"  Shape after engineering: {df_feat.shape[0]:,} x {df_feat.shape[1]}")

    # ── Select ────────────────────────────────────────────────────────────
    print("\nRunning select_features ...")
    t2 = time.time()
    selected, df_selected = select_features(df_feat)
    print(f"  Selection completed in {time.time() - t2:.2f}s")
    print(f"  Shape after selection:   {df_selected.shape[0]:,} x {df_selected.shape[1]}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = DATA_DIR / "features.csv"
    df_selected.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    dropped = [c for c in df_feat.columns if c not in df_selected.columns]

    print(f"\n{'='*60}")
    print(f"Feature pipeline complete in {elapsed:.2f}s")
    print(f"  {df_clean.shape[1]} raw columns")
    print(f"  +{len(new_cols)} engineered  ->  {df_feat.shape[1]} total")
    print(f"  -{len(dropped)} dropped    ->  {df_selected.shape[1]} final")
    print(f"\nKept features ({len(selected)} numeric + non-numeric passthrough):")
    for col in selected:
        print(f"  {col}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run_feature_pipeline()
