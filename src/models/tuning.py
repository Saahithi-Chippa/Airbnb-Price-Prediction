"""
Hyperparameter tuning for LightGBM using Optuna.

LightGBM was selected as the best model in train.py:
  LinearRegression (baseline) : MAE 1,693 THB | R2 0.0006
  LightGBM (default params)   : MAE 1,560 THB | R2 0.0036

Strategy
---------
TPE (Tree-structured Parzen Estimator) sampler builds a probabilistic model
of the objective surface and proposes candidates more likely to improve than
random or grid search.  30 trials, each scored with 5-fold CV on the training
split so the test set stays completely held-out until the final evaluation.

Search space
-------------
Log-scale for learning_rate, reg_alpha, reg_lambda — these span several orders
of magnitude and uniform sampling clusters proposals near the upper bound.
subsample_freq is fixed at 1 (required by LightGBM whenever subsample < 1.0).
"""

import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import logging
import optuna
from pathlib import Path
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(logging.WARNING)

DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

TARGET       = "price"
TEST_SIZE    = 0.20
RANDOM_STATE = 42
N_TRIALS     = 30
CV_FOLDS     = 5

_BASELINE_MAE = 1_693
_PRETUNE_MAE  = 1_560

_EXCLUDE = {"id", "host_id", "name", "host_name", "neighbourhood",
            "room_type", "last_review", TARGET}


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_DIR / "features.csv")
    feature_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _EXCLUDE
    ]
    return df[feature_cols], df[TARGET]


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


def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    params = {
        "n_estimators":      trial.suggest_int(  "n_estimators",       100,  800),
        "learning_rate":     trial.suggest_float("learning_rate",      0.01,  0.3,  log=True),
        "num_leaves":        trial.suggest_int(  "num_leaves",           15,  127),
        "max_depth":         trial.suggest_int(  "max_depth",             3,    9),
        "min_child_samples": trial.suggest_int(  "min_child_samples",    10,  100),
        "subsample":         trial.suggest_float("subsample",            0.6,  1.0),
        "subsample_freq":    1,
        "colsample_bytree":  trial.suggest_float("colsample_bytree",    0.6,  1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha",           1e-6,  1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",          1e-6,  1.0, log=True),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    model = TransformedTargetRegressor(
        regressor=LGBMRegressor(**params),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=CV_FOLDS,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    return float(-cv_scores.mean())


def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """Callback: one line per completed trial; marks when a new best is found."""
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    p      = trial.params
    is_best = trial.number == study.best_trial.number
    tag    = " <-- best" if is_best else ""
    print(
        f"  Trial {trial.number + 1:3d}/{N_TRIALS} | "
        f"MAE {trial.value:>9,.0f} THB | "
        f"lr={p['learning_rate']:.4f}  "
        f"leaves={p['num_leaves']:3d}  "
        f"depth={p['max_depth']}  "
        f"n_est={p['n_estimators']:4d}  "
        f"sub={p['subsample']:.2f}  "
        f"col={p['colsample_bytree']:.2f}"
        f"{tag}"
    )


if __name__ == "__main__":
    t_total = time.time()

    # ── Load & split ──────────────────────────────────────────────────────
    print("Loading data/features.csv ...")
    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Features: {X.shape[1]}  |  Target: '{TARGET}'\n")

    # ── Optuna study ──────────────────────────────────────────────────────
    print(f"Tuning LightGBM: {N_TRIALS} trials x {CV_FOLDS}-fold CV  (TPE sampler)")
    print("-" * 75)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name="lightgbm_airbnb_price",
    )

    t_tune = time.time()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        callbacks=[_log_trial],
        show_progress_bar=False,
    )
    tune_time = time.time() - t_tune

    print("-" * 75)
    best_trial = study.best_trial
    print(f"Tuning complete in {tune_time:.1f}s")
    print(f"\nBest trial:   #{best_trial.number + 1}")
    print(f"Best CV MAE:  {study.best_value:,.0f} THB")
    print("\nBest hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k:25s} {v}")

    # ── Save best params ──────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    params_path = MODELS_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"\nBest params saved -> {params_path}")

    # ── Train final model with best params ────────────────────────────────
    print("\nTraining final model on full training split ...")
    final_params = {
        **best_trial.params,
        "subsample_freq": 1,
        "random_state":   RANDOM_STATE,
        "n_jobs":         -1,
        "verbose":        -1,
    }
    tuned_model = TransformedTargetRegressor(
        regressor=LGBMRegressor(**final_params),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    t_fit = time.time()
    tuned_model.fit(X_train, y_train)
    print(f"  Fit completed in {time.time() - t_fit:.2f}s")

    # ── Evaluate on held-out test set ─────────────────────────────────────
    y_pred  = tuned_model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, n_features=X_train.shape[1])

    print("\nTest set metrics (original THB scale):")
    print(f"  MAE     {metrics['MAE']:>10,.0f} THB")
    print(f"  RMSE    {metrics['RMSE']:>10,.0f} THB")
    print(f"  R2      {metrics['R2']:>10.4f}")
    print(f"  Adj R2  {metrics['Adj_R2']:>10.4f}")

    # ── Progression summary ───────────────────────────────────────────────
    vs_baseline = (_BASELINE_MAE - metrics["MAE"]) / _BASELINE_MAE * 100
    vs_pretune  = (_PRETUNE_MAE  - metrics["MAE"]) / _PRETUNE_MAE  * 100

    print("\nProgression:")
    print(f"  {'Model':<35} {'Test MAE':>10}   {'vs baseline':>12}")
    print(f"  {'-'*60}")
    print(f"  {'LinearRegression (baseline)':<35} {_BASELINE_MAE:>10,}   {'':>12}")
    print(f"  {'LightGBM (default params)':<35} {_PRETUNE_MAE:>10,}   {(_PRETUNE_MAE - _BASELINE_MAE) / _BASELINE_MAE * 100:>+11.1f}%")
    print(f"  {'LightGBM (tuned)':<35} {metrics['MAE']:>10,.0f}   {vs_baseline:>+11.1f}%")
    print(f"\n  Tuning gain over default params: {vs_pretune:+.1f}%")

    # ── Save tuned model ──────────────────────────────────────────────────
    model_path = MODELS_DIR / "tuned_model.pkl"
    joblib.dump(tuned_model, model_path)
    print(f"\nTuned model saved -> {model_path}")
    print(f"Total time: {time.time() - t_total:.1f}s")
