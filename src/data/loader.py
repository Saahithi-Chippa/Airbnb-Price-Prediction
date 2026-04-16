import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df


def print_shape(df: pd.DataFrame) -> None:
    rows, cols = df.shape
    print(f"Shape: {rows} rows x {cols} columns")


def print_dtypes(df: pd.DataFrame) -> None:
    print("\nColumn Names and Data Types:")
    print(df.dtypes.to_string())


def print_summary_stats(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("\nNo numeric columns found.")
        return
    stats = numeric_df.agg(["mean", "std", "min", "max"])
    print("\nSummary Statistics (numeric columns):")
    print(stats.to_string())


def print_missing_values(df: pd.DataFrame) -> None:
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing = pd.DataFrame({"missing_count": missing_counts, "missing_%": missing_pct})
    missing = missing[missing["missing_count"] > 0]
    if missing.empty:
        print("\nNo missing values found.")
    else:
        print("\nMissing Value Counts and Percentages:")
        print(missing.to_string())


def profile(filename: str) -> pd.DataFrame:
    df = load_csv(filename)
    print_shape(df)
    print_dtypes(df)
    print_summary_stats(df)
    print_missing_values(df)
    return df


if __name__ == "__main__":
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else None
    if not filename:
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
        filename = csv_files[0].name
        print(f"No filename provided — using: {filename}\n")

    profile(filename)
