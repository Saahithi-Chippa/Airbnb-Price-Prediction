"""
Model selection for Airbnb price regression.

Candidate models
-----------------
RandomForest
    Bagging ensemble of decorrelated trees. No feature scaling needed; majority
    voting dampens the impact of the extreme luxury-price outliers that crushed
    linear regression. Provides reliable feature importance as a by-product.

XGBoost
    Sequential gradient boosting — each tree corrects the residuals of the last.
    Built-in L1/L2 regularisation prevents overfitting on 21 engineered features.
    Handles binary flags and continuous features in the same tree without encoding
    tricks. The reference model for structured/tabular regression.

LightGBM
    Leaf-wise gradient boosting with histogram-based splits. 5-10x faster inference
    than XGBoost — critical for an automated pricing system scoring new listings in
    real time. Often marginally better on datasets with many continuous interaction
    features (demand_index, host_scale_score).

All three models train on log1p(price) via TransformedTargetRegressor so the
objective matches the log-normal price distribution. CV and test metrics are
inverse-transformed (expm1) to the original THB scale for interpretability.

Baseline reference (LinearRegression): MAE 1,693 THB | R² 0.0006
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

TARGET       = "price"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 5

_EXCLUDE = {"id", "host_id", "name", "host_name", "neighbourhood",
            "room_type", "last_review", TARGET}

CANDIDATE_MODELS: dict = {
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",      # standard variance-reduction trick
        min_samples_leaf=5,       # prevents memorising single listings
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,            # row sampling per tree (reduces overfit)
        colsample_bytree=0.8,     # feature sampling per tree
        min_child_weight=5,       # min sum of instance weight in a leaf
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,            # leaf-wise growth; more expressive than max_depth alone
        min_child_samples=20,     # equivalent to min_samples_leaf
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ),
}

MODEL_RATIONALE: dict[str, str] = {
    "RandomForest": (
        "Bagging + feature subsampling creates decorrelated trees whose average "
        "cancels individual errors. No feature scaling required. The majority-vote "
        "mechanism limits how much a single 1M-THB outlier can skew predictions — "
        "a direct fix for linear regression's core failure on this dataset."
    ),
    "XGBoost": (
        "Boosting iteratively fits new trees to the current residuals in log-price "
        "space. Built-in L1/L2 regularisation keeps the 21-feature model from "
        "overfitting. Captures non-linear interactions (e.g. professional host AND "
        "high availability) that linear models and single trees cannot."
    ),
    "LightGBM": (
        "Leaf-wise growth finds deeper, more targeted splits than XGBoost's "
        "level-wise strategy, often yielding better accuracy on datasets with "
        "many continuous interaction features (demand_index, host_scale_score). "
        "Histogram binning makes training and inference fast — essential for "
        "scoring new listings in an automated pricing pipeline."
    ),
}


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "features.csv")
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _EXCLUDE
    ]
    return df[feature_cols], df[TARGET]


def wrap_log_target(model) -> TransformedTargetRegressor:
    """Train on log1p(y); predict returns expm1 values on original scale."""
    return TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, n_features: int = 0
) -> dict[str, float]:
    n  = len(y_true)
    r2 = r2_score(y_true, y_pred)
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


if __name__ == "__main__":
    t_total = time.time()

    # ── Load & split ──────────────────────────────────────────────────────
    print("Loading data/features.csv ...")
    X, y = load_features()
    print(f"  {X.shape[0]:,} rows  |  {X.shape[1]} features  |  target: '{TARGET}'")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  (80/20, seed={RANDOM_STATE})\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw_results: list[dict] = []

    # ── Train, CV, evaluate, save ─────────────────────────────────────────
    for name, model in CANDIDATE_MODELS.items():
        print("-" * 60)
        print(f"  {name}")
        print(f"  {MODEL_RATIONALE[name]}\n")

        wrapped = wrap_log_target(model)

        # 5-fold cross-validation on training set (original scale scoring)
        print(f"  {CV_FOLDS}-fold CV ...")
        t_cv = time.time()
        cv_out = cross_validate(
            wrapped, X_train, y_train,
            cv=CV_FOLDS,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1,           # model-level parallelism already used
        )
        cv_time = time.time() - t_cv
        cv_mae  = -cv_out["test_mae"]
        cv_r2   =  cv_out["test_r2"]
        print(f"  CV  MAE  {cv_mae.mean():>10,.0f} ± {cv_mae.std():,.0f} THB   ({cv_time:.1f}s)")
        print(f"  CV  R²   {cv_r2.mean():>10.4f} ± {cv_r2.std():.4f}")

        # Final fit on full training split
        t_fit = time.time()
        wrapped.fit(X_train, y_train)
        fit_time = time.time() - t_fit

        # Test-set evaluation
        y_pred  = wrapped.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, n_features=X_train.shape[1])
        print(f"  Test MAE    {metrics['MAE']:>10,.0f} THB")
        print(f"  Test RMSE   {metrics['RMSE']:>10,.0f} THB")
        print(f"  Test R2     {metrics['R2']:>10.4f}")
        print(f"  Test Adj R2 {metrics['Adj_R2']:>10.4f}")
        print(f"  Fit time    {fit_time:.1f}s")

        # Save
        save_path = MODELS_DIR / f"{name.lower()}.pkl"
        joblib.dump(wrapped, save_path)
        print(f"  Saved -> {save_path}\n")

        raw_results.append({
            "model":        name,
            "cv_mae_mean":  cv_mae.mean(),
            "cv_mae_std":   cv_mae.std(),
            "cv_r2_mean":   cv_r2.mean(),
            "test_mae":     metrics["MAE"],
            "test_r2":      metrics["R2"],
            "test_adj_r2":  metrics["Adj_R2"],
            "fit_time":     fit_time,
        })

    # ── Comparison table ──────────────────────────────────────────────────
    n_feat = X_train.shape[1]
    baseline_r2 = 0.0006
    baseline_adj_r2 = 1 - (1 - baseline_r2) * (len(X_train) - 1) / (len(X_train) - n_feat - 1)
    baseline = {
        "model": "LinearRegression (baseline)",
        "cv_mae_mean": None, "cv_mae_std": None, "cv_r2_mean": None,
        "test_mae": 1693.0, "test_r2": baseline_r2,
        "test_adj_r2": baseline_adj_r2, "fit_time": 0.25,
    }

    def fmt_row(r: dict) -> dict:
        cv_mae = (
            f"{r['cv_mae_mean']:,.0f} +/- {r['cv_mae_std']:,.0f}"
            if r["cv_mae_mean"] is not None else "-"
        )
        cv_r2 = f"{r['cv_r2_mean']:.4f}" if r["cv_r2_mean"] is not None else "-"
        return {
            "Model":           r["model"],
            "CV MAE (THB)":    cv_mae,
            "CV R2":           cv_r2,
            "Test MAE (THB)":  f"{r['test_mae']:,.0f}",
            "Test R2":         f"{r['test_r2']:.4f}",
            "Test Adj R2":     f"{r['test_adj_r2']:.4f}",
            "Fit Time (s)":    f"{r['fit_time']:.1f}",
        }

    table = pd.DataFrame([fmt_row(baseline)] + [fmt_row(r) for r in raw_results])

    print("=" * 70)
    print("Comparison Table")
    print("=" * 70)
    print(table.to_string(index=False))

    # ── Best model ────────────────────────────────────────────────────────
    best = min(raw_results, key=lambda r: r["test_mae"])
    improvement = (1693 - best["test_mae"]) / 1693 * 100

    print(f"\n{'='*70}")
    print(f"Best model:  {best['model']}")
    print(f"  Test MAE   {best['test_mae']:,.0f} THB  (baseline: 1,693 THB)")
    print(f"  Test R²    {best['test_r2']:.4f}       (baseline: 0.0006)")
    print(f"  MAE improvement over baseline: {improvement:.1f}%")

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
