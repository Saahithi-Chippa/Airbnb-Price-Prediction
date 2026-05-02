import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.quality import check_data_quality


# ── Shared fixture: minimal valid dataframe ────────────────────────────────

def _make_valid_df(n: int = 200) -> pd.DataFrame:
    """Minimal dataframe that satisfies every quality check."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id":                           np.arange(1, n + 1),
        "name":                         [f"Listing {i}" for i in range(n)],
        "host_id":                      rng.integers(1_000, 9_999, n),
        "neighbourhood":                ["Vadhana"] * n,
        "latitude":                     rng.uniform(13.6, 13.9, n),
        "longitude":                    rng.uniform(100.4, 100.7, n),
        "room_type":                    ["Entire home/apt"] * n,
        "price":                        rng.uniform(500, 5_000, n),
        "minimum_nights":               rng.integers(1, 30, n),
        "availability_365":             rng.integers(1, 365, n),
        "number_of_reviews":            rng.integers(0, 200, n),
        "reviews_per_month":            rng.uniform(0, 5, n),
        "calculated_host_listings_count": rng.integers(1, 10, n),
        "number_of_reviews_ltm":        rng.integers(0, 50, n),
        "last_review":                  ["2024-01-01"] * n,
        "host_name":                    ["Host"] * n,
    })


# ── Tests: quality gate passes on clean data ──────────────────────────────

class TestQualityGatePasses:

    def test_success_flag_is_true(self):
        result = check_data_quality(_make_valid_df())
        assert result["success"] is True

    def test_no_failures(self):
        result = check_data_quality(_make_valid_df())
        assert result["failures"] == []

    def test_statistics_keys_present(self):
        result = check_data_quality(_make_valid_df())
        stats = result["statistics"]
        for key in ("total_rows", "total_columns", "total_nulls", "target_stats"):
            assert key in stats, f"Missing statistics key: {key}"

    def test_statistics_row_count(self):
        df = _make_valid_df(n=200)
        result = check_data_quality(df)
        assert result["statistics"]["total_rows"] == 200

    def test_target_stats_populated(self):
        result = check_data_quality(_make_valid_df())
        t = result["statistics"]["target_stats"]
        assert t["mean"] > 0
        assert t["median"] > 0
        assert t["min"] > 0

    def test_passes_on_real_cleaned_csv(self):
        """Integration: actual cleaned.csv must pass the quality gate."""
        cleaned = Path(__file__).resolve().parents[1] / "data" / "cleaned.csv"
        if not cleaned.exists():
            pytest.skip("data/cleaned.csv not present")
        df = pd.read_csv(cleaned)
        result = check_data_quality(df)
        assert result["success"] is True, result["failures"]


# ── Tests: quality gate catches broken data ───────────────────────────────

class TestQualityGateFailures:

    def test_fails_on_missing_required_column(self):
        df = _make_valid_df().drop(columns=["price"])
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("price" in f for f in result["failures"])

    def test_fails_below_minimum_row_count(self):
        df = _make_valid_df(n=50)  # MIN_ROWS = 100
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("Row count" in f for f in result["failures"])

    def test_fails_on_negative_prices(self):
        df = _make_valid_df()
        # Only flip 10 rows so price retains variance (all-same triggers the
        # zero-variance early-return before the non-positive check runs).
        df.loc[:9, "price"] = -100.0
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("non-positive" in f for f in result["failures"])

    def test_fails_on_critical_null_rate(self):
        df = _make_valid_df(n=200)
        # Make 60 % of price null  (> CRITICAL_NULL_RATE = 0.50)
        df.loc[:119, "price"] = np.nan
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("price" in f and "null rate" in f for f in result["failures"])

    def test_fails_on_latitude_out_of_bounds(self):
        df = _make_valid_df()
        df["latitude"] = 200.0  # > 90
        result = check_data_quality(df)
        assert result["success"] is False
        assert any("latitude" in f for f in result["failures"])

    def test_warns_on_high_skew(self):
        df = _make_valid_df(n=500)
        # Inject a handful of extreme outliers to push skewness > 2
        df.loc[:4, "price"] = 1_000_000
        result = check_data_quality(df)
        assert any("skew" in w.lower() for w in result["warnings"])

    def test_warns_on_moderate_null_rate(self):
        df = _make_valid_df(n=200)
        # 25 % nulls on a non-required column  (> WARN_NULL_RATE = 0.20)
        df.loc[:49, "reviews_per_month"] = np.nan
        result = check_data_quality(df)
        assert any("reviews_per_month" in w for w in result["warnings"])

    def test_result_always_has_required_keys(self):
        df = _make_valid_df(n=10)  # tiny broken df
        result = check_data_quality(df)
        for key in ("success", "failures", "warnings", "statistics"):
            assert key in result
