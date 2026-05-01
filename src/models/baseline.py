"""
Baseline model: LinearRegression on log1p(price).

Target transformation rationale
---------------------------------
Price has a raw skewness of ~53. LinearRegression assumes normally distributed
residuals and a linear relationship between features and the target. Training
directly on raw price violates both: a handful of luxury outliers dominate the
loss function, and the signal from mid-range listings is swamped. log1p(price)
reduces skewness to near-zero, satisfies the normality assumption in practice,
and lets the model treat relative price differences equally across the range.
Predictions are inverse-transformed with expm1 before metric evaluation so all
reported numbers are in the original THB scale.

Model assumptions
------------------
1. Linearity          — log(price) is a linear function of each input feature.
2. Independence       — residuals are independent across listings (reasonable;
                        listings from the same host introduce mild dependence,
                        but host_scale_score partially absorbs this).
3. Homoscedasticity   — residual variance is constant across fitted values.
                        The log transform substantially reduces heteroscedasticity,
                        but some remains at the extreme price tiers.
4. Normality of res.  — residuals approximately follow N(0, σ²) after the log
                        transform; validated by the near-normal log-price histogram
                        in the EDA notebook.
5. No multicollinearity — no pair of features has |r| > 0.95 after select_features,
                        so coefficient estimates are stable.

This baseline sets the performance floor: any subsequent model (XGBoost, LightGBM)
that cannot exceed these metrics on the held-out test set adds no value.
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

TARGET       = "price"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Columns present in features.csv that are not model inputs
_EXCLUDE = {"id", "host_id", "name", "host_name", "neighbourhood",
            "room_type", "last_review", TARGET}


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "features.csv")
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _EXCLUDE
    ]
    return df[feature_cols], df[TARGET]


def build_pipeline() -> Pipeline:
    # StandardScaler ensures coefficients are on comparable scales, which
    # also makes the coefficient magnitude meaningful for feature importance.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, n_features: int = 0
) -> dict[str, float]:
    n  = len(y_true)
    r2 = r2_score(y_true, y_pred)
    # Adjusted R² penalises adding features that do not improve the fit.
    # Requires n_features > 0 and n > n_features + 1 to be defined.
    adj_r2 = (
        1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        if n_features > 0 and n > n_features + 1
        else float("nan")
    )
    return {
        "MAE":    mean_absolute_error(y_true, y_pred),
        "RMSE":   float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":     r2,
        "Adj_R2": adj_r2,
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"  {label}")
    print(f"    MAE     {metrics['MAE']:>12,.2f} THB")
    print(f"    RMSE    {metrics['RMSE']:>12,.2f} THB")
    print(f"    R2      {metrics['R2']:>12.4f}")
    print(f"    Adj R2  {metrics['Adj_R2']:>12.4f}")


if __name__ == "__main__":
    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading data/features.csv ...")
    X, y = load_features()
    print(f"  {X.shape[0]:,} rows  |  {X.shape[1]} features  |  target: '{TARGET}'")

    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  (80/20, seed={RANDOM_STATE})")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\nTraining LinearRegression (on log1p-transformed target) ...")
    t1 = time.time()

    pipeline = build_pipeline()
    pipeline.fit(X_train, np.log1p(y_train))

    print(f"  Fit completed in {time.time() - t1:.2f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────
    train_pred = np.expm1(pipeline.predict(X_train))
    test_pred  = np.expm1(pipeline.predict(X_test))

    n_feat = X_train.shape[1]
    train_metrics = compute_metrics(y_train, train_pred, n_features=n_feat)
    test_metrics  = compute_metrics(y_test,  test_pred,  n_features=n_feat)

    print("\nMetrics (original THB scale):")
    print_metrics("Train", train_metrics)
    print_metrics("Test ", test_metrics)

    gap = test_metrics["R2"] - train_metrics["R2"]
    if abs(gap) < 0.03:
        print("\n  Generalisation: good (train/test R2 within 0.03)")
    elif gap < -0.05:
        print(f"\n  Generalisation: possible overfit (R2 gap = {gap:.4f})")

    # ── Feature coefficients ──────────────────────────────────────────────
    coefs = pd.Series(
        pipeline.named_steps["model"].coef_,
        index=X.columns,
    ).abs().sort_values(ascending=False)

    print("\nTop 10 features by |coefficient| (scaled space, log-price target):")
    for feat, coef in coefs.head(10).items():
        print(f"  {feat:40s}  {coef:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "baseline.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved -> {model_path}")
    print(f"Total time:  {time.time() - t0:.2f}s")
