import numpy as np
import pandas as pd
import pytest
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "production_model.pkl"

FEATURE_COLS = [
    "minimum_nights", "number_of_reviews", "reviews_per_month",
    "number_of_reviews_ltm", "is_entire_home", "is_private_room",
    "has_reviews", "is_professional_host", "high_availability",
    "availability_rate", "days_since_last_review", "name_length",
    "name_word_count", "name_has_luxury_kw", "name_has_transport_kw",
    "log_minimum_nights", "log_number_of_reviews", "review_recency_ratio",
    "demand_index", "host_scale_score", "min_nights_availability_ratio",
]

# Reasonable Bangkok price bounds in THB
PRICE_MIN =    50
PRICE_MAX = 50_000


def _make_feature_row(**overrides) -> pd.DataFrame:
    """Single-row dataframe with sensible Bangkok listing defaults."""
    base = {
        "minimum_nights":               2,
        "number_of_reviews":            30,
        "reviews_per_month":            1.5,
        "number_of_reviews_ltm":        12,
        "is_entire_home":               1,
        "is_private_room":              0,
        "has_reviews":                  1,
        "is_professional_host":         0,
        "high_availability":            1,
        "availability_rate":            0.55,
        "days_since_last_review":       45,
        "name_length":                  35,
        "name_word_count":              6,
        "name_has_luxury_kw":           0,
        "name_has_transport_kw":        1,
        "log_minimum_nights":           np.log1p(2),
        "log_number_of_reviews":        np.log1p(30),
        "review_recency_ratio":         0.4,
        "demand_index":                 0.55 * 0.5 + 1.5 * 0.3 + 0.2,
        "host_scale_score":             np.log1p(3),
        "min_nights_availability_ratio": 2 / 0.55,
    }
    base.update(overrides)
    return pd.DataFrame([base])[FEATURE_COLS]


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("models/production_model.pkl not present")
    return joblib.load(MODEL_PATH)


# ── Tests: model loads correctly ──────────────────────────────────────────

class TestModelLoads:

    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"

    def test_model_loads_without_error(self, model):
        assert model is not None

    def test_model_has_predict_method(self, model):
        assert callable(getattr(model, "predict", None))

    def test_model_has_feature_importances(self, model):
        importances = model.regressor_.feature_importances_
        assert len(importances) == len(FEATURE_COLS)
        assert (importances >= 0).all()

    def test_feature_importances_sum_is_positive(self, model):
        assert model.regressor_.feature_importances_.sum() > 0


# ── Tests: predictions are in expected range ──────────────────────────────

class TestPredictionRange:

    def test_single_row_prediction_is_scalar(self, model):
        preds = model.predict(_make_feature_row())
        assert preds.shape == (1,)

    def test_prediction_is_positive(self, model):
        pred = float(model.predict(_make_feature_row())[0])
        assert pred > 0, f"Prediction {pred} is not positive"

    def test_prediction_within_realistic_bounds(self, model):
        pred = float(model.predict(_make_feature_row())[0])
        assert PRICE_MIN <= pred <= PRICE_MAX, (
            f"Prediction ฿{pred:,.0f} is outside [{PRICE_MIN}, {PRICE_MAX}]"
        )

    def test_entire_home_predicts_higher_than_private_room(self, model):
        entire  = float(model.predict(_make_feature_row(is_entire_home=1, is_private_room=0))[0])
        private = float(model.predict(_make_feature_row(is_entire_home=0, is_private_room=1))[0])
        assert entire > private, (
            f"Entire home (฿{entire:,.0f}) should predict higher than private room (฿{private:,.0f})"
        )

    def test_luxury_listing_predicts_higher(self, model):
        standard = float(model.predict(_make_feature_row(name_has_luxury_kw=0))[0])
        luxury   = float(model.predict(_make_feature_row(name_has_luxury_kw=1))[0])
        assert luxury > standard, (
            f"Luxury listing (฿{luxury:,.0f}) should predict higher than standard (฿{standard:,.0f})"
        )

    def test_batch_prediction_shape(self, model):
        rows = pd.concat([_make_feature_row() for _ in range(10)], ignore_index=True)
        preds = model.predict(rows)
        assert preds.shape == (10,)

    def test_all_batch_predictions_positive(self, model):
        rows = pd.concat([_make_feature_row() for _ in range(20)], ignore_index=True)
        preds = model.predict(rows)
        assert (preds > 0).all()

    def test_predictions_on_real_test_set(self):
        """Integration: predicted_price in predictions.csv is always positive."""
        pred_path = Path(__file__).resolve().parents[1] / "data" / "predictions.csv"
        if not pred_path.exists():
            pytest.skip("data/predictions.csv not present")
        df = pd.read_csv(pred_path)
        assert (df["predicted_price"] > 0).all()
        assert df["predicted_price"].between(PRICE_MIN, PRICE_MAX * 2).mean() > 0.99
