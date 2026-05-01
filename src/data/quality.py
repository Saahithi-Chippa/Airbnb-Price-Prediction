import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# dtype.kind codes: i=integer, f=float, O=object/string, M=datetime
REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "id":               ("i", "f"),
    "name":             ("O",),
    "host_id":          ("i", "f"),
    "neighbourhood":    ("O",),
    "latitude":         ("f",),
    "longitude":        ("f",),
    "room_type":        ("O",),
    "price":            ("f", "i"),
    "minimum_nights":   ("i", "f"),
    "availability_365": ("i", "f"),
}

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "latitude":                       (-90.0,  90.0),
    "longitude":                      (-180.0, 180.0),
    "price":                          (0.0,    float("inf")),
    "minimum_nights":                 (0,      1_825),
    "availability_365":               (0,      365),
    "number_of_reviews":              (0,      float("inf")),
    "reviews_per_month":              (0.0,    100.0),
    "calculated_host_listings_count": (0,      float("inf")),
    "number_of_reviews_ltm":          (0,      float("inf")),
}

TARGET_COLUMN = "price"
MIN_ROWS = 100
WARN_ROWS = 1_000
CRITICAL_NULL_RATE = 0.50
WARN_NULL_RATE = 0.20
HIGH_SKEW_THRESHOLD = 2.0
OUTLIER_IQR_MULTIPLIER = 3.0


def _check_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    for col, allowed_kinds in REQUIRED_SCHEMA.items():
        if col not in df.columns:
            failures.append(f"Missing required column: '{col}'")
        elif df[col].dtype.kind not in allowed_kinds:
            warnings.append(
                f"Column '{col}' dtype is '{df[col].dtype}' "
                f"(expected kind in {allowed_kinds})"
            )
    return failures, warnings


def _check_row_count(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    n = len(df)
    if n < MIN_ROWS:
        failures.append(f"Row count {n:,} is below minimum threshold of {MIN_ROWS:,}")
    elif n < WARN_ROWS:
        warnings.append(f"Row count {n:,} is low (< {WARN_ROWS:,}); model may underfit")
    return failures, warnings


def _check_null_rates(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    for col, rate in df.isnull().mean().items():
        if rate > CRITICAL_NULL_RATE:
            failures.append(
                f"Column '{col}' has {rate:.1%} null rate "
                f"(critical threshold: {CRITICAL_NULL_RATE:.0%})"
            )
        elif rate > WARN_NULL_RATE:
            warnings.append(
                f"Column '{col}' has {rate:.1%} null rate "
                f"(warn threshold: {WARN_NULL_RATE:.0%})"
            )
    return failures, warnings


def _check_value_ranges(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures, warnings = [], []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        actual_min, actual_max = series.min(), series.max()
        if actual_min < lo:
            failures.append(
                f"Column '{col}' has values below minimum bound "
                f"{lo} (found {actual_min})"
            )
        if hi != float("inf") and actual_max > hi:
            failures.append(
                f"Column '{col}' has values above maximum bound "
                f"{hi} (found {actual_max})"
            )
    return failures, warnings


def _check_target_distribution(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Regression checks on price: existence, variance, skew, extreme outliers."""
    failures, warnings = [], []

    if TARGET_COLUMN not in df.columns:
        failures.append(f"Target column '{TARGET_COLUMN}' not found in dataset")
        return failures, warnings

    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        failures.append(f"Target '{TARGET_COLUMN}' must be numeric for price regression")
        return failures, warnings

    series = df[TARGET_COLUMN].dropna()

    if series.std() == 0:
        failures.append(
            f"Target '{TARGET_COLUMN}' has zero variance — all values are identical"
        )
        return failures, warnings

    n_negative = int((series <= 0).sum())
    if n_negative > 0:
        failures.append(
            f"Target '{TARGET_COLUMN}' has {n_negative:,} non-positive values "
            "(prices must be > 0)"
        )

    skew = series.skew()
    if abs(skew) > HIGH_SKEW_THRESHOLD:
        warnings.append(
            f"Target '{TARGET_COLUMN}' is highly skewed (skewness={skew:.2f}); "
            "a log-transform is strongly recommended before modelling"
        )

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    n_outliers = int(((series < q1 - OUTLIER_IQR_MULTIPLIER * iqr) |
                      (series > q3 + OUTLIER_IQR_MULTIPLIER * iqr)).sum())
    if n_outliers > 0:
        warnings.append(
            f"Target '{TARGET_COLUMN}' has {n_outliers:,} extreme outliers ({n_outliers/len(series):.1%}) "
            f"beyond {OUTLIER_IQR_MULTIPLIER:.0f}×IQR — consider capping before modelling"
        )

    null_pct = df[TARGET_COLUMN].isnull().mean()
    if null_pct > 0:
        warnings.append(
            f"Target '{TARGET_COLUMN}' has {null_pct:.1%} missing values — "
            "these rows will be dropped during training"
        )

    return failures, warnings


def check_data_quality(df: pd.DataFrame) -> dict:
    failures: list[str] = []
    warnings: list[str] = []

    for check in (
        _check_schema,
        _check_row_count,
        _check_null_rates,
        _check_value_ranges,
        _check_target_distribution,
    ):
        f, w = check(df)
        failures.extend(f)
        warnings.extend(w)

    null_counts = df.isnull().sum()
    statistics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls": int(null_counts.sum()),
        "total_nulls_by_column": null_counts[null_counts > 0].to_dict(),
        "null_rate_by_column": {
            col: round(rate, 4)
            for col, rate in df.isnull().mean().items()
            if rate > 0
        },
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(include="object").columns),
        "target_stats": (
            {
                "mean":   round(df[TARGET_COLUMN].mean(), 2),
                "median": round(df[TARGET_COLUMN].median(), 2),
                "std":    round(df[TARGET_COLUMN].std(), 2),
                "min":    round(df[TARGET_COLUMN].min(), 2),
                "max":    round(df[TARGET_COLUMN].max(), 2),
                "skew":   round(df[TARGET_COLUMN].skew(), 4),
            }
            if TARGET_COLUMN in df.columns and pd.api.types.is_numeric_dtype(df[TARGET_COLUMN])
            else {}
        ),
    }

    return {
        "success": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from data.loader import load_csv

    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    filename = sys.argv[1] if len(sys.argv) > 1 else csvs[0].name
    if len(sys.argv) <= 1:
        print(f"No filename given — using {filename}\n")

    df = load_csv(filename)
    result = check_data_quality(df)

    status = "PASSED" if result["success"] else "FAILED"
    print(f"Quality gate: {status}")

    if result["failures"]:
        print(f"\nFailures ({len(result['failures'])}):")
        for msg in result["failures"]:
            print(f"  [FAIL] {msg}")

    if result["warnings"]:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for msg in result["warnings"]:
            print(f"  [WARN] {msg}")

    stats = result["statistics"]
    print(f"\nStatistics:")
    print(f"  total_rows:    {stats['total_rows']:,}")
    print(f"  total_columns: {stats['total_columns']}")
    print(f"  total_nulls:   {stats['total_nulls']:,}")
    if stats["total_nulls_by_column"]:
        print("  nulls by column:")
        for col, count in stats["total_nulls_by_column"].items():
            rate = stats["null_rate_by_column"].get(col, 0)
            print(f"    {col}: {count:,}  ({rate:.1%})")
    if stats["target_stats"]:
        t = stats["target_stats"]
        print(f"\nTarget '{TARGET_COLUMN}':")
        print(f"  mean={t['mean']:,.2f}  median={t['median']:,.2f}  "
              f"std={t['std']:,.2f}  min={t['min']:,.2f}  max={t['max']:,.2f}  "
              f"skew={t['skew']:.4f}")
