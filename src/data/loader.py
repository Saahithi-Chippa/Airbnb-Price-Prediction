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


def print_schema(df: pd.DataFrame) -> None:
    print("\nColumn names and data types:")
    print(df.dtypes.to_string())


def print_summary_stats(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        print("\nNo numeric columns found.")
        return
    stats = numeric.agg(["mean", "std", "min", "max"])
    print("\nSummary statistics (numeric columns):")
    print(stats.to_string())


def print_missing(df: pd.DataFrame) -> None:
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing_counts, "missing_%": missing_pct})
    report = report[report["missing_count"] > 0]
    if report.empty:
        print("\nNo missing values.")
    else:
        print("\nMissing value report:")
        print(report.to_string())


def profile(df: pd.DataFrame) -> None:
    print_shape(df)
    print_schema(df)
    print_summary_stats(df)
    print_missing(df)


if __name__ == "__main__":
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else None
    if filename is None:
        csvs = [p for p in DATA_DIR.glob("*.csv") if p.name != "cleaned.csv"]
        if not csvs:
            raise FileNotFoundError(f"No raw CSV files found in {DATA_DIR}")
        filename = csvs[0].name
        print(f"No filename given — using {filename}\n")

    df = load_csv(filename)
    profile(df)
