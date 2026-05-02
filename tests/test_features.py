import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.engineering import create_features, select_features

# Features added by create_features (17 new columns)
ENGINEERED_FEATURES = [
    "is_entire_home", "is_private_room", "has_reviews", "is_professional_host",
    "high_availability", "availability_rate", "days_since_last_review",
    "name_length", "name_word_count", "name_has_luxury_kw", "name_has_transport_kw",
    "log_minimum_nights", "log_number_of_reviews", "review_recency_ratio",
    "demand_index", "host_scale_score", "min_nights_availability_ratio",
]

BINARY_FEATURES = [
    "is_entire_home", "is_private_room", "has_reviews", "is_professional_host",
    "high_availability", "name_has_luxury_kw", "name_has_transport_kw",
]


# ── Shared fixture ─────────────────────────────────────────────────────────

def _make_cleaned_df(n: int = 100) -> pd.DataFrame:
    """Minimal dataframe matching the schema output by cleaner.py."""
    rng = np.random.default_rng(42)
    room_types = rng.choice(
        ["Entire home/apt", "Private room", "Shared room", "Hotel room"], n
    )
    return pd.DataFrame({
        "id":                           np.arange(1, n + 1),
        "name":                         [f"BTS Sukhumvit Suite {i}" for i in range(n)],
        "host_id":                      rng.integers(1_000, 9_999, n),
        "host_name":                    ["Host"] * n,
        "neighbourhood":                ["Vadhana"] * n,
        "latitude":                     rng.uniform(13.6, 13.9, n),
        "longitude":                    rng.uniform(100.4, 100.7, n),
        "room_type":                    room_types,
        "price":                        rng.uniform(500, 5_000, n),
        "minimum_nights":               rng.integers(1, 30, n),
        "number_of_reviews":            rng.integers(0, 200, n),
        "last_review":                  ["2024-06-01"] * n,
        "reviews_per_month":            rng.uniform(0, 5, n),
        "calculated_host_listings_count": rng.integers(1, 20, n),
        "availability_365":             rng.integers(0, 365, n),
        "number_of_reviews_ltm":        rng.integers(0, 50, n),
    })


@pytest.fixture(scope="module")
def featured_df():
    return create_features(_make_cleaned_df(n=200))


# ── Tests: column count ────────────────────────────────────────────────────

class TestColumnCount:

    def test_adds_17_new_columns(self):
        df_in = _make_cleaned_df()
        df_out = create_features(df_in)
        assert len(df_out.columns) == len(df_in.columns) + 17

    def test_all_engineered_columns_present(self, featured_df):
        for col in ENGINEERED_FEATURES:
            assert col in featured_df.columns, f"Missing engineered column: {col}"

    def test_input_columns_preserved(self):
        df_in = _make_cleaned_df()
        df_out = create_features(df_in)
        for col in df_in.columns:
            assert col in df_out.columns, f"Input column dropped: {col}"

    def test_row_count_unchanged(self):
        df_in = _make_cleaned_df(n=150)
        df_out = create_features(df_in)
        assert len(df_out) == len(df_in)


# ── Tests: no NaN values in engineered columns ────────────────────────────

class TestNoNulls:

    def test_no_nans_in_engineered_features(self, featured_df):
        null_counts = featured_df[ENGINEERED_FEATURES].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        assert cols_with_nulls == {}, f"NaN found in: {cols_with_nulls}"

    def test_handles_null_last_review(self):
        df = _make_cleaned_df()
        df["last_review"] = None
        out = create_features(df)
        assert out["days_since_last_review"].isnull().sum() == 0

    def test_handles_null_name(self):
        df = _make_cleaned_df()
        df["name"] = None
        out = create_features(df)
        assert out["name_length"].isnull().sum() == 0
        assert out["name_word_count"].isnull().sum() == 0

    def test_handles_zero_reviews(self):
        df = _make_cleaned_df()
        df["number_of_reviews"] = 0
        out = create_features(df)
        assert out["review_recency_ratio"].isnull().sum() == 0

    def test_handles_zero_availability(self):
        df = _make_cleaned_df()
        df["availability_365"] = 0
        out = create_features(df)
        assert out["min_nights_availability_ratio"].isnull().sum() == 0


# ── Tests: value ranges ───────────────────────────────────────────────────

class TestValueRanges:

    def test_binary_features_are_zero_or_one(self, featured_df):
        for col in BINARY_FEATURES:
            vals = featured_df[col].unique()
            assert set(vals).issubset({0, 1}), f"{col} contains values outside {{0, 1}}: {vals}"

    def test_availability_rate_between_0_and_1(self, featured_df):
        assert featured_df["availability_rate"].between(0, 1).all()

    def test_days_since_last_review_capped_at_730(self, featured_df):
        assert featured_df["days_since_last_review"].max() <= 730
        assert featured_df["days_since_last_review"].min() >= 0

    def test_review_recency_ratio_between_0_and_1(self, featured_df):
        assert featured_df["review_recency_ratio"].between(0, 1).all()

    def test_log_features_are_non_negative(self, featured_df):
        assert (featured_df["log_minimum_nights"] >= 0).all()
        assert (featured_df["log_number_of_reviews"] >= 0).all()

    def test_demand_index_is_non_negative(self, featured_df):
        assert (featured_df["demand_index"] >= 0).all()

    def test_is_entire_home_reflects_room_type(self):
        df = _make_cleaned_df()
        df["room_type"] = "Entire home/apt"
        out = create_features(df)
        assert (out["is_entire_home"] == 1).all()
        assert (out["is_private_room"] == 0).all()

    def test_luxury_keyword_detected(self):
        df = _make_cleaned_df(n=5)
        df["name"] = ["Luxury Pool Villa"] * 5
        out = create_features(df)
        assert (out["name_has_luxury_kw"] == 1).all()

    def test_transport_keyword_detected(self):
        df = _make_cleaned_df(n=5)
        df["name"] = ["Near BTS Asok station"] * 5
        out = create_features(df)
        assert (out["name_has_transport_kw"] == 1).all()

    def test_no_keywords_gives_zero(self):
        df = _make_cleaned_df(n=5)
        df["name"] = ["Quiet room in the suburbs"] * 5
        out = create_features(df)
        assert (out["name_has_luxury_kw"] == 0).all()
        assert (out["name_has_transport_kw"] == 0).all()


# ── Tests: select_features ────────────────────────────────────────────────

class TestSelectFeatures:

    def test_returns_fewer_or_equal_features(self, featured_df):
        numeric_before = len(featured_df.select_dtypes(include="number").columns) - 2  # minus id, price
        selected, _ = select_features(featured_df)
        assert len(selected) <= numeric_before

    def test_target_not_in_selected(self, featured_df):
        selected, _ = select_features(featured_df)
        assert "price" not in selected

    def test_reduced_df_has_same_rows(self, featured_df):
        _, reduced = select_features(featured_df)
        assert len(reduced) == len(featured_df)

    def test_selected_cols_exist_in_reduced_df(self, featured_df):
        selected, reduced = select_features(featured_df)
        for col in selected:
            assert col in reduced.columns
