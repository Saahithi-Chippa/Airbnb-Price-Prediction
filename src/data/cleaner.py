import pandas as pd
from pathlib import Path

from loader import load_csv
from quality import check_data_quality, _print_report

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLEANED_PATH = DATA_DIR / "cleaned.csv"
NULL_DROP_THRESHOLD = 0.50

# Tokens used to detect time-series-like columns
_DATETIME_TOKENS = ("date", "time", "timestamp", "datetime", "dt", "period", "year", "month", "week")


def _is_time_series(df: pd.DataFrame) -> bool:
    """Return True if the dataframe looks like a time series.

    Heuristics:
    - Any column has a datetime dtype, OR
    - Any column name contains a datetime-like token.
    """
    if any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns):
        return True
    col_lower = [c.lower() for c in df.columns]
    return any(token in name for name in col_lower for token in _DATETIME_TOKENS)


def _drop_high_null_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns where > 50% of values are null."""
    null_rates = df.isnull().mean()
    cols_to_drop = null_rates[null_rates > NULL_DROP_THRESHOLD].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop


def _handle_nulls(
    df: pd.DataFrame,
    target_column: str | None,
    is_ts: bool,
) -> tuple[pd.DataFrame, int]:
    """Handle nulls according to data type (time series vs tabular).

    Returns the cleaned dataframe and the number of rows dropped.
    """
    initial_rows = len(df)

    # Drop rows where the target is null (only relevant when target_column is set)
    if target_column and target_column in df.columns:
        df = df.dropna(subset=[target_column])

    if is_ts:
        # Forward-fill remaining nulls (preserves row count)
        df = df.ffill()
        # Back-fill any leading nulls that ffill couldn't cover
        df = df.bfill()
    else:
        # Drop rows with any remaining nulls
        df = df.dropna()

    return df, initial_rows - len(df)


def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(keep="first")
    return df, before - len(df)


def _convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to clean dtypes.

    - Columns already numeric stay as int64 / float64.
    - Object columns that can be fully parsed as numeric are converted.
    - Remaining object columns are cast to string.
    - Bool columns are left as bool.
    """
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # Downcast floats that are whole numbers to int where possible
            if pd.api.types.is_float_dtype(df[col]):
                if (df[col].dropna() % 1 == 0).all():
                    df[col] = df[col].astype("Int64")  # nullable int
            continue

        if df[col].dtype == object:
            # Try numeric coercion first
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() / max(len(df), 1) > 0.95:
                df[col] = coerced
            else:
                df[col] = df[col].astype(str)

    return df


def clean_data(
    df: pd.DataFrame,
    target_column: str | None = None,
    save_path: Path = CLEANED_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Clean a raw dataframe and return (cleaned_df, quality_report).

    Steps
    -----
    1. Drop columns with > 50% nulls.
    2. Drop rows where target is null (if target_column is set).
    3. Forward-fill nulls (time series) or drop null rows (tabular).
    4. Remove exact duplicate rows.
    5. Coerce dtypes to numeric / string.
    6. Save cleaned CSV to save_path.
    7. Run quality gate and return result.
    """
    df = df.copy()
    is_ts = _is_time_series(df)

    # Step 1 — drop high-null columns
    df, dropped_cols = _drop_high_null_columns(df)
    if dropped_cols:
        print(f"  [clean] Dropped {len(dropped_cols)} high-null column(s): {dropped_cols}")

    # Step 2 & 3 — handle nulls
    df, rows_dropped_nulls = _handle_nulls(df, target_column, is_ts)
    fill_strategy = "forward-filled" if is_ts else "dropped"
    strategy_label = f"time-series ({fill_strategy})" if is_ts else f"tabular ({fill_strategy})"
    print(f"  [clean] Null strategy : {strategy_label:<35} rows removed: {rows_dropped_nulls:,}")

    # Step 4 — remove duplicates
    df, dupes_removed = _remove_duplicates(df)
    print(f"  [clean] Duplicate rows removed: {dupes_removed:,}")

    # Step 5 — convert dtypes
    df = _convert_dtypes(df)
    print(f"  [clean] Dtype coercion complete")

    # Step 6 — save
    df.to_csv(save_path, index=False)
    print(f"  [clean] Saved cleaned data to: {save_path}")

    # Step 7 — quality gate
    print(f"  [clean] Re-running quality gate on cleaned data...")
    quality_result = check_data_quality(df, target_column=target_column)

    return df, quality_result


if __name__ == "__main__":
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else None
    if not filename:
        import glob as _glob
        csv_files = [p for p in DATA_DIR.glob("*.csv") if p.name != "cleaned.csv"]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
        filename = csv_files[0].name
        print(f"No filename provided — using: {filename}\n")

    raw_df = load_csv(filename)
    print(f"Before cleaning: {len(raw_df):,} rows x {raw_df.shape[1]} columns")
    print()

    cleaned_df, report = clean_data(raw_df, target_column=None)

    print(f"\nAfter cleaning : {len(cleaned_df):,} rows x {cleaned_df.shape[1]} columns")
    print(f"Rows removed   : {len(raw_df) - len(cleaned_df):,}")

    _print_report(report)
