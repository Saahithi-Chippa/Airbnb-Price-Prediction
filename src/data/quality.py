import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Thresholds
MIN_ROWS_CRITICAL = 100
MIN_ROWS_WARN = 1000
NULL_RATE_CRITICAL = 0.50
NULL_RATE_WARN = 0.20
MIN_CLASS_SHARE = 0.05
IMBALANCE_WARN = 0.10


def _check_schema(
    df: pd.DataFrame,
    required_columns: list[str] | None,
    expected_dtypes: dict[str, str] | None,
    failures: list,
    warnings: list,
) -> None:
    """Check 1 — required columns exist and have expected dtypes."""
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            failures.append(f"Schema: missing required columns: {missing}")

    if expected_dtypes:
        for col, expected in expected_dtypes.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            if not actual.startswith(expected):
                warnings.append(
                    f"Schema: column '{col}' expected dtype '{expected}', got '{actual}'"
                )


def _check_row_count(
    df: pd.DataFrame, failures: list, warnings: list
) -> None:
    """Check 2 — dataset has enough rows."""
    n = len(df)
    if n < MIN_ROWS_CRITICAL:
        failures.append(
            f"Row count: only {n} rows (minimum {MIN_ROWS_CRITICAL} required)"
        )
    elif n < MIN_ROWS_WARN:
        warnings.append(
            f"Row count: {n} rows is below recommended threshold of {MIN_ROWS_WARN}"
        )


def _check_null_rates(
    df: pd.DataFrame, failures: list, warnings: list
) -> dict[str, float]:
    """Check 3 — null rates per column."""
    null_rates = (df.isnull().mean()).to_dict()
    for col, rate in null_rates.items():
        if rate > NULL_RATE_CRITICAL:
            failures.append(
                f"Nulls: column '{col}' has {rate:.1%} missing values (>{NULL_RATE_CRITICAL:.0%} critical threshold)"
            )
        elif rate > NULL_RATE_WARN:
            warnings.append(
                f"Nulls: column '{col}' has {rate:.1%} missing values (>{NULL_RATE_WARN:.0%} warn threshold)"
            )
    return null_rates


def _check_value_ranges(
    df: pd.DataFrame, failures: list, warnings: list
) -> None:
    """Check 4 — numeric columns within sensible bounds.

    Rules applied automatically by column name heuristics:
    - *_pct / *_rate / *_ratio / *_score columns: expected in [0, 1] or [0, 100]
    - Columns whose name suggests a count (*_count, *_num, n_*): no negatives
    - Any numeric column: flag if max > 10 000% of mean (extreme outlier signal)
    """
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        col_lower = col.lower()

        # Count-like columns must be non-negative
        is_count = any(
            token in col_lower
            for token in ("count", "_num", "n_", "qty", "quantity")
        )
        if is_count and (series < 0).any():
            failures.append(
                f"Range: count column '{col}' contains negative values"
            )

        # Ratio/percentage columns bounded to [0, 1] or [0, 100]
        ratio_tokens = ("_pct", "_rate", "_ratio", "_score", "_probability", "_prob",
                        "pct_", "rate_", "ratio_", "score_")
        is_ratio = any(token in col_lower for token in ratio_tokens)
        if is_ratio:
            col_max = series.max()
            col_min = series.min()
            if col_min < 0:
                failures.append(
                    f"Range: ratio/pct column '{col}' has values below 0 (min={col_min:.4f})"
                )
            elif col_max > 100:
                failures.append(
                    f"Range: ratio/pct column '{col}' has values above 100 (max={col_max:.4f})"
                )

        # Generic extreme-outlier check: max > 10 000% of mean
        col_mean = series.mean()
        if col_mean > 0:
            ratio = series.max() / col_mean
            if ratio > 100:
                warnings.append(
                    f"Range: column '{col}' has extreme spread — max is {ratio:.0f}x the mean"
                )


def _check_target_distribution(
    df: pd.DataFrame,
    target_column: str | None,
    failures: list,
    warnings: list,
) -> dict | None:
    """Check 5 — classification target has enough classes and balance."""
    if not target_column:
        return None
    if target_column not in df.columns:
        failures.append(
            f"Target: column '{target_column}' not found in dataframe"
        )
        return None

    value_counts = df[target_column].value_counts(normalize=True)
    n_classes = len(value_counts)

    if n_classes < 2:
        failures.append(
            f"Target: '{target_column}' has only {n_classes} class — need at least 2"
        )
        return {"n_classes": n_classes, "class_shares": value_counts.to_dict()}

    rare_classes = value_counts[value_counts < MIN_CLASS_SHARE]
    if not rare_classes.empty:
        failures.append(
            f"Target: {len(rare_classes)} class(es) in '{target_column}' "
            f"represent <{MIN_CLASS_SHARE:.0%} of data: {rare_classes.index.tolist()}"
        )

    imbalanced = value_counts[value_counts < IMBALANCE_WARN]
    non_critical_imbalance = imbalanced[imbalanced >= MIN_CLASS_SHARE]
    if not non_critical_imbalance.empty:
        warnings.append(
            f"Target: '{target_column}' has imbalanced classes "
            f"(<{IMBALANCE_WARN:.0%}): {non_critical_imbalance.index.tolist()}"
        )

    return {"n_classes": n_classes, "class_shares": value_counts.round(4).to_dict()}


def check_data_quality(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    expected_dtypes: dict[str, str] | None = None,
    target_column: str | None = None,
) -> dict:
    """Run 5 data quality checks and return a structured report.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to check.
    required_columns : list[str], optional
        Column names that must be present.
    expected_dtypes : dict[str, str], optional
        Mapping of column name to expected dtype prefix, e.g. {'age': 'int', 'price': 'float'}.
    target_column : str, optional
        Name of the classification target column for distribution checks.

    Returns
    -------
    dict with keys: success, failures, warnings, statistics
    """
    failures: list[str] = []
    warnings: list[str] = []

    # --- Run checks ---
    _check_schema(df, required_columns, expected_dtypes, failures, warnings)
    _check_row_count(df, failures, warnings)
    null_rates = _check_null_rates(df, failures, warnings)
    _check_value_ranges(df, failures, warnings)
    target_stats = _check_target_distribution(df, target_column, failures, warnings)

    # --- Build statistics ---
    total_nulls_by_column = df.isnull().sum().to_dict()
    statistics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls": int(df.isnull().sum().sum()),
        "total_nulls_by_column": {
            col: int(count)
            for col, count in total_nulls_by_column.items()
            if count > 0
        },
        "null_rates_by_column": {
            col: round(rate, 4)
            for col, rate in null_rates.items()
            if rate > 0
        },
        "numeric_column_count": int(df.select_dtypes(include="number").shape[1]),
        "categorical_column_count": int(df.select_dtypes(include="object").shape[1]),
    }
    if target_stats:
        statistics["target_distribution"] = target_stats

    return {
        "success": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }


def _print_report(report: dict) -> None:
    status = "PASSED" if report["success"] else "FAILED"
    print(f"\n{'='*52}")
    print(f"  DATA QUALITY GATE: {status}")
    print(f"{'='*52}")

    stats = report["statistics"]
    print(f"\nStatistics:")
    print(f"  Rows            : {stats['total_rows']:,}")
    print(f"  Columns         : {stats['total_columns']}")
    print(f"  Total nulls     : {stats['total_nulls']:,}")
    print(f"  Numeric columns : {stats['numeric_column_count']}")
    print(f"  Object columns  : {stats['categorical_column_count']}")

    if stats.get("null_rates_by_column"):
        print(f"\n  Null rates (affected columns only):")
        for col, rate in stats["null_rates_by_column"].items():
            print(f"    {col:<30} {rate:.2%}")

    if stats.get("target_distribution"):
        td = stats["target_distribution"]
        print(f"\n  Target distribution ({td['n_classes']} classes):")
        for cls, share in list(td["class_shares"].items())[:10]:
            print(f"    {str(cls):<30} {share:.2%}")
        if td["n_classes"] > 10:
            print(f"    ... and {td['n_classes'] - 10} more classes")

    if report["failures"]:
        print(f"\nFailures ({len(report['failures'])}):")
        for f in report["failures"]:
            print(f"  [FAIL] {f}")

    if report["warnings"]:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  [WARN] {w}")

    if not report["failures"] and not report["warnings"]:
        print("\n  No issues found.")

    print(f"\n{'='*52}\n")


if __name__ == "__main__":
    import sys
    from loader import load_csv

    filename = sys.argv[1] if len(sys.argv) > 1 else None
    if not filename:
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
        filename = csv_files[0].name
        print(f"No filename provided — using: {filename}")

    df = load_csv(filename)
    report = check_data_quality(df, target_column=None)
    _print_report(report)
