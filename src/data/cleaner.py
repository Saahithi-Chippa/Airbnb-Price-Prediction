import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLEANED_PATH = DATA_DIR / "cleaned.csv"

FLOAT_COLS = ["latitude", "longitude", "price", "reviews_per_month"]
INT_COLS = [
    "id", "host_id", "minimum_nights", "number_of_reviews",
    "calculated_host_listings_count", "availability_365", "number_of_reviews_ltm",
]
STR_COLS = ["name", "host_name", "neighbourhood", "room_type", "last_review"]

# Listings with no reviews have null last_review + reviews_per_month — both are valid
# data points for a host pricing model, so impute rather than drop them.
_NULL_FILLS = {
    "reviews_per_month": 0.0,
    "last_review":       "",       # feature engineering treats "" as "never reviewed"
    "host_name":         "Unknown",
}


def _drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    null_rates = df.isnull().mean()
    to_drop = null_rates[null_rates > threshold].index.tolist()
    if to_drop:
        print(f"  Dropped columns (>{threshold:.0%} nulls): {to_drop}")
    return df.drop(columns=to_drop)


def _drop_target_nulls(df: pd.DataFrame, target: str = "price") -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[target])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows where target '{target}' is null")
    return df


def _impute_known_nulls(df: pd.DataFrame) -> pd.DataFrame:
    for col, fill_value in _NULL_FILLS.items():
        if col in df.columns:
            n = int(df[col].isnull().sum())
            if n:
                df[col] = df[col].fillna(fill_value)
                print(f"  Filled {n:,} nulls in '{col}' with {fill_value!r}")
    return df


def _drop_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with remaining nulls")
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} exact duplicate rows")
    else:
        print("  No duplicate rows found")
    return df


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in STR_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from data.quality import check_data_quality

    print("Cleaning steps:")
    df = _drop_high_null_columns(df)
    df = _drop_target_nulls(df)
    df = _impute_known_nulls(df)
    df = _drop_remaining_nulls(df)
    df = _drop_duplicates(df)
    df = _coerce_dtypes(df)
    df = df.reset_index(drop=True)

    df.to_csv(CLEANED_PATH, index=False)
    print(f"\nSaved cleaned data to {CLEANED_PATH}")

    print("\nRe-running quality gate on cleaned data...")
    quality_result = check_data_quality(df)

    return df, quality_result


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from data.loader import load_csv

    csvs = [p for p in DATA_DIR.glob("*.csv") if p.name != "cleaned.csv"]
    if not csvs:
        raise FileNotFoundError(f"No raw CSV files found in {DATA_DIR}")
    filename = sys.argv[1] if len(sys.argv) > 1 else csvs[0].name
    if len(sys.argv) <= 1:
        print(f"No filename given — using {filename}\n")

    raw_df = load_csv(filename)
    print(f"Raw data:  {len(raw_df):,} rows × {len(raw_df.columns)} columns\n")

    cleaned_df, quality = clean_data(raw_df)

    print(f"\nCleaned data: {len(cleaned_df):,} rows × {len(cleaned_df.columns)} columns")
    print(f"Rows retained: {len(cleaned_df)/len(raw_df):.1%}")

    status = "PASSED" if quality["success"] else "FAILED"
    print(f"\nQuality gate: {status}")

    if quality["failures"]:
        print(f"\nFailures ({len(quality['failures'])}):")
        for msg in quality["failures"]:
            print(f"  [FAIL] {msg}")

    if quality["warnings"]:
        print(f"\nWarnings ({len(quality['warnings'])}):")
        for msg in quality["warnings"]:
            print(f"  [WARN] {msg}")

    if quality["statistics"].get("target_stats"):
        t = quality["statistics"]["target_stats"]
        print("\nCleaned target 'price':")
        print(f"  mean={t['mean']:,.2f}  median={t['median']:,.2f}  "
              f"std={t['std']:,.2f}  min={t['min']:,.2f}  "
              f"max={t['max']:,.2f}  skew={t['skew']:.4f}")
