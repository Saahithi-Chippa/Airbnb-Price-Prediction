"""
Generate data/predictions.csv — test-set predictions from the production model.
Must be run before launching the Streamlit dashboard.
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

TARGET       = "price"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

_EXCLUDE = {"id", "host_id", "name", "host_name", "neighbourhood",
            "room_type", "last_review", TARGET}


def generate() -> None:
    df = pd.read_csv(DATA_DIR / "features.csv")

    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _EXCLUDE
    ]
    X = df[feature_cols]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = joblib.load(MODELS_DIR / "production_model.pkl")
    y_pred = model.predict(X_test)

    out = df.loc[X_test.index].copy()
    out["predicted_price"] = np.round(y_pred, 2)
    out["residual"]        = out[TARGET] - out["predicted_price"]
    out["abs_error"]       = out["residual"].abs()
    out["pct_error"]       = (out["abs_error"] / out[TARGET].replace(0, np.nan) * 100).round(2)

    path = DATA_DIR / "predictions.csv"
    out.to_csv(path, index=False)
    print(f"Saved {len(out):,} predictions -> {path}")


if __name__ == "__main__":
    generate()
