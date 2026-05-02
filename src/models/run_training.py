"""
MLflow-tracked training pipeline for Airbnb price prediction.

Trains and logs two model configurations:
  1. LinearRegression  — performance floor / baseline
  2. LightGBM (tuned) — best model from Optuna tuning (models/best_params.json)

Each run logs: hyperparameters, train + test metrics (MAE, RMSE, R2, Adj R2),
fit time, and the model artifact (both sklearn mlflow flavour and joblib file).

The best model by test MAE is saved to models/production_model.pkl.

To inspect all runs in the MLflow UI:
  mlflow server --host 127.0.0.1 --port 5000
Then open http://localhost:5000
"""

import json
import time
import warnings
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

# Store mlruns outside OneDrive — OneDrive's sync client locks files mid-write,
# causing PermissionError when MLflow reads params it just logged.
MLRUNS_DIR = Path.home() / "mlruns"

TARGET            = "price"
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
EXPERIMENT_NAME   = "airbnb_price_prediction"

_EXCLUDE = {"id", "host_id", "name", "host_name", "neighbourhood",
            "room_type", "last_review", TARGET}


# ── Helpers ────────────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "features.csv")
    cols = [c for c in df.select_dtypes(include="number").columns if c not in _EXCLUDE]
    return df[cols], df[TARGET]


def load_best_params() -> dict:
    path = MODELS_DIR / "best_params.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run src/models/tuning.py first")
    with open(path) as f:
        return json.load(f)


def wrap_log_target(model) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=model, func=np.log1p, inverse_func=np.expm1
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
        "mae":    mean_absolute_error(y_true, y_pred),
        "rmse":   float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":     r2,
        "adj_r2": adj_r2,
    }


# ── Model configs ──────────────────────────────────────────────────────────

def build_configs(best_params: dict) -> list[dict]:
    return [
        {
            "run_name":   "LinearRegression_baseline",
            "model_name": "LinearRegression",
            "model": wrap_log_target(
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("model",  LinearRegression()),
                ])
            ),
            "logged_params": {
                "model_type":       "LinearRegression",
                "scaler":           "StandardScaler",
                "target_transform": "log1p",
            },
        },
        {
            "run_name":   "LightGBM_tuned",
            "model_name": "LightGBM",
            "model": wrap_log_target(
                LGBMRegressor(
                    **best_params,
                    subsample_freq=1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                )
            ),
            "logged_params": {
                "model_type":       "LightGBM",
                "target_transform": "log1p",
                **best_params,
            },
        },
    ]


# ── Single MLflow run ──────────────────────────────────────────────────────

def train_and_log(
    config: dict,
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series,    y_test:  pd.Series,
) -> tuple[object, dict]:
    """Train one model config, log everything to MLflow, return (model, test_metrics)."""
    n_feat = X_train.shape[1]

    with mlflow.start_run(run_name=config["run_name"]) as run:

        # -- Params ----------------------------------------------------------
        mlflow.log_param("model_name", config["model_name"])
        for k, v in config["logged_params"].items():
            mlflow.log_param(k, v)
        mlflow.log_param("n_features",    n_feat)
        mlflow.log_param("train_rows",    len(X_train))
        mlflow.log_param("test_rows",     len(X_test))
        mlflow.log_param("random_state",  RANDOM_STATE)

        # -- Train -----------------------------------------------------------
        t0 = time.time()
        config["model"].fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)
        mlflow.log_metric("fit_time_s", fit_time)

        # -- Metrics ---------------------------------------------------------
        train_pred = config["model"].predict(X_train)
        test_pred  = config["model"].predict(X_test)

        train_m = compute_metrics(y_train, train_pred, n_features=n_feat)
        test_m  = compute_metrics(y_test,  test_pred,  n_features=n_feat)

        for split, m in [("train", train_m), ("test", test_m)]:
            for name, val in m.items():
                if not np.isnan(val):
                    mlflow.log_metric(f"{split}_{name}", round(val, 4))

        # -- Artifacts -------------------------------------------------------
        # sklearn mlflow flavour — loadable with mlflow.sklearn.load_model()
        mlflow.sklearn.log_model(config["model"], artifact_path="model")

        # joblib file — direct binary for production inference
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_path = MODELS_DIR / f"{config['run_name'].lower()}.pkl"
        joblib.dump(config["model"], pkl_path)
        mlflow.log_artifact(str(pkl_path), artifact_path="joblib")

        # -- Console summary -------------------------------------------------
        print(f"  Run ID : {run.info.run_id}")
        print(f"  Fit    : {fit_time}s")
        print(f"  Train  ->  MAE {train_m['mae']:>9,.0f} THB  |  R2 {train_m['r2']:.4f}  |  Adj R2 {train_m['adj_r2']:.4f}")  # noqa: E501
        print(f"  Test   ->  MAE {test_m['mae']:>9,.0f} THB  |  R2 {test_m['r2']:.4f}  |  Adj R2 {test_m['adj_r2']:.4f}")  # noqa: E501
        print(f"  Artifact -> {pkl_path.name}")

    return config["model"], test_m


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_total = time.time()

    # ── Setup ──────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment : '{EXPERIMENT_NAME}'")
    print(f"Tracking URI      : {mlflow.get_tracking_uri()}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading data/features.csv ...")
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {X.shape[1]}\n")

    # ── Run configs ────────────────────────────────────────────────────────
    best_params = load_best_params()
    configs = build_configs(best_params)
    run_results: list[dict] = []

    for cfg in configs:
        print(f"{'='*60}")
        print(f"Run: {cfg['run_name']}")
        model, test_m = train_and_log(cfg, X_train, X_test, y_train, y_test)
        run_results.append({"config": cfg, "model": model, "test_mae": test_m["mae"], "metrics": test_m})
        print()

    # ── Comparison summary ─────────────────────────────────────────────────
    print("=" * 60)
    print("Run summary:")
    print(f"  {'Run':<35} {'Test MAE':>10}  {'Test R2':>8}  {'Adj R2':>8}")
    print(f"  {'-'*65}")
    for r in run_results:
        m = r["metrics"]
        print(f"  {r['config']['run_name']:<35} {m['mae']:>10,.0f}  {m['r2']:>8.4f}  {m['adj_r2']:>8.4f}")

    # ── Save production model ──────────────────────────────────────────────
    best_run = min(run_results, key=lambda r: r["test_mae"])
    prod_path = MODELS_DIR / "production_model.pkl"
    joblib.dump(best_run["model"], prod_path)

    print(f"\nProduction model : {best_run['config']['run_name']}")
    print(f"  Test MAE       : {best_run['test_mae']:,.0f} THB")
    print(f"  Saved          : {prod_path}")

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
    print("\nTo view all runs in the MLflow UI:")
    print(f"  mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri {MLRUNS_DIR.as_uri()}")
    print("  Then open http://localhost:5000")
