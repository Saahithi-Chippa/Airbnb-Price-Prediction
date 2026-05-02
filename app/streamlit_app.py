"""
Airbnb Price Prediction — Bangkok
Multi-page Streamlit portfolio dashboard.

Run:
    streamlit run app/streamlit_app.py

Works without training artifacts: if data/predictions.csv,
data/model_results.json, data/features.csv, or
models/production_model.pkl are missing, the app generates
synthetic demo data so every page is always viewable.
"""

import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"

# ── Demo mode: True when any artifact produced by training is absent ───────
_DEMO_MODE = not all([
    (DATA_DIR   / "features.csv").exists(),
    (DATA_DIR   / "predictions.csv").exists(),
    (DATA_DIR   / "model_results.json").exists(),
    (MODELS_DIR / "production_model.pkl").exists(),
])

# ── Theme colours ──────────────────────────────────────────────────────────
PRIMARY  = "#FF5A5F"
TEAL     = "#00A699"
ORANGE   = "#FC642D"
DARK     = "#484848"
LIGHT_BG = "#F7F7F7"

# ── Feature order (must match training) ───────────────────────────────────
FEATURE_COLS = [
    "minimum_nights", "number_of_reviews", "reviews_per_month",
    "number_of_reviews_ltm", "is_entire_home", "is_private_room",
    "has_reviews", "is_professional_host", "high_availability",
    "availability_rate", "days_since_last_review", "name_length",
    "name_word_count", "name_has_luxury_kw", "name_has_transport_kw",
    "log_minimum_nights", "log_number_of_reviews", "review_recency_ratio",
    "demand_index", "host_scale_score", "min_nights_availability_ratio",
]

LUXURY_KW    = re.compile(r"luxury|penthouse|villa|pool|rooftop|sky|suite|premium|deluxe|spa", re.I)
TRANSPORT_KW = re.compile(r"bts|mrt|skytrain|metro|station|airport|transit|asok|silom|sukhumvit", re.I)


# ── Demo data generators ───────────────────────────────────────────────────

def _make_demo_features() -> pd.DataFrame:
    """Synthetic Bangkok-like Airbnb dataset (5,000 rows) for demo mode."""
    rng = np.random.default_rng(42)
    n   = 5_000

    room_types = rng.choice(
        ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
        size=n, p=[0.71, 0.27, 0.014, 0.006],
    )
    neighbourhoods = [
        "Vadhana", "Khlong Toei", "Huai Khwang", "Ratchathewi", "Sathon",
        "Phra Khanong", "Phra Nakhon", "Bang Rak", "Suanluang", "Chatu Chak",
        "Pathum Wan", "Din Daeng", "Yan Nawa", "Bang Na", "Lat Phrao",
        "Khlong San", "Thon Buri", "Bang Kho Laem", "Saphan Sung", "Wang Thong Lang",
    ]
    hood_weights = np.array([3709, 3119, 3033, 1243, 1092, 1001, 924, 756, 728, 681,
                             600, 550, 500, 450, 400, 350, 300, 250, 200, 150], dtype=float)
    hood_weights /= hood_weights.sum()
    hoods = rng.choice(neighbourhoods, size=n, p=hood_weights)

    # Right-skewed price: log-normal base + a handful of luxury outliers
    price = np.exp(rng.normal(7.2, 0.8, n))
    price[room_types == "Entire home/apt"]  *= 1.2
    price[room_types == "Private room"]     *= 0.6
    price[room_types == "Shared room"]      *= 0.4
    luxury_idx = rng.choice(n, int(n * 0.002), replace=False)
    price[luxury_idx] = rng.uniform(50_000, 1_000_000, len(luxury_idx))
    price = np.clip(price, 100, 1_000_000).round(0)

    min_nights      = rng.integers(1, 30, n)
    n_reviews       = rng.integers(0, 500, n)
    rpm             = np.where(n_reviews > 0, rng.uniform(0.1, 10, n).round(2), 0.0)
    n_reviews_ltm   = rng.integers(0, 100, n)
    availability    = rng.integers(0, 365, n)
    avail_rate      = (availability / 365.0).round(4)
    days_since      = rng.integers(0, 730, n)

    return pd.DataFrame({
        "id":                           np.arange(1, n + 1),
        "name":                         [f"Demo Listing {i}" for i in range(n)],
        "host_id":                      rng.integers(1_000, 99_999, n),
        "host_name":                    ["Demo Host"] * n,
        "neighbourhood":                hoods,
        "room_type":                    room_types,
        "price":                        price,
        "minimum_nights":               min_nights,
        "number_of_reviews":            n_reviews,
        "last_review":                  "",
        "reviews_per_month":            rpm,
        "number_of_reviews_ltm":        n_reviews_ltm,
        "is_entire_home":               (room_types == "Entire home/apt").astype(int),
        "is_private_room":              (room_types == "Private room").astype(int),
        "has_reviews":                  (n_reviews > 0).astype(int),
        "is_professional_host":         rng.binomial(1, 0.15, n),
        "high_availability":            (avail_rate > 0.5).astype(int),
        "availability_rate":            avail_rate,
        "days_since_last_review":       days_since,
        "name_length":                  rng.integers(10, 80, n),
        "name_word_count":              rng.integers(2, 15, n),
        "name_has_luxury_kw":           rng.binomial(1, 0.10, n),
        "name_has_transport_kw":        rng.binomial(1, 0.20, n),
        "log_minimum_nights":           np.log1p(min_nights).round(4),
        "log_number_of_reviews":        np.log1p(n_reviews).round(4),
        "review_recency_ratio":         (n_reviews / np.maximum(days_since, 1)).round(4),
        "demand_index":                 (avail_rate * 0.5 + rpm * 0.3 + 0.2).round(4),
        "host_scale_score":             np.log1p(rng.integers(1, 50, n)).round(4),
        "min_nights_availability_ratio": (min_nights / np.maximum(avail_rate, 0.001)).round(4),
    })


def _make_demo_predictions() -> pd.DataFrame:
    """Synthetic predictions for demo mode (1,000 rows)."""
    rng = np.random.default_rng(99)
    n   = 1_000

    room_types = rng.choice(
        ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
        size=n, p=[0.71, 0.27, 0.014, 0.006],
    )
    price = np.exp(rng.normal(7.2, 0.8, n))
    price[room_types == "Entire home/apt"] *= 1.2
    price[room_types == "Private room"]    *= 0.6
    price = np.clip(price, 100, 50_000).round(0)

    # Simulate realistic model behaviour: predict in log space, expm1 back
    log_pred  = np.log1p(price) + rng.normal(0, 0.25, n)
    predicted = np.expm1(log_pred).round(2)
    residual  = price - predicted
    abs_error = np.abs(residual).round(2)
    pct_error = (abs_error / np.maximum(price, 1) * 100).round(2)

    return pd.DataFrame({
        "price":           price,
        "predicted_price": predicted,
        "residual":        residual.round(2),
        "abs_error":       abs_error,
        "pct_error":       pct_error,
        "room_type":       room_types,
    })


def _make_demo_model_results() -> dict:
    """Hardcoded results matching the real training run."""
    return {
        "data_points":    23_273,
        "feature_count":  17,
        "baseline_mae":   1_693,
        "winner_mae":     1_573,
        "improvement_pct": 7.1,
        "target": "price",
        "city":   "Bangkok",
        "models": [
            {
                "name": "Linear Regression", "type": "baseline",
                "cv_mae": None, "cv_mae_std": None,
                "test_mae": 1_693, "test_rmse": 19_824,
                "test_r2": 0.0006, "test_adj_r2": -0.0039,
                "fit_time_s": 0.25, "is_winner": False,
                "note": "Performance floor. Log1p target transform required due to skewness 53.",
            },
            {
                "name": "Random Forest", "type": "candidate",
                "cv_mae": 1_424, "cv_mae_std": 161,
                "test_mae": 1_580, "test_rmse": 19_802,
                "test_r2": 0.0028, "test_adj_r2": -0.0017,
                "fit_time_s": 1.8, "is_winner": False,
                "note": "Majority voting dampens luxury-price outlier impact. Solid tree baseline.",
            },
            {
                "name": "XGBoost", "type": "candidate",
                "cv_mae": 1_416, "cv_mae_std": 159,
                "test_mae": 1_575, "test_rmse": 19_778,
                "test_r2": 0.0052, "test_adj_r2": 0.0007,
                "fit_time_s": 1.3, "is_winner": False,
                "note": "Best R2 among default-param models. Built-in L1/L2 regularisation.",
            },
            {
                "name": "LightGBM (default)", "type": "candidate",
                "cv_mae": 1_406, "cv_mae_std": 158,
                "test_mae": 1_560, "test_rmse": 19_794,
                "test_r2": 0.0036, "test_adj_r2": -0.0009,
                "fit_time_s": 0.7, "is_winner": False,
                "note": "Fastest training. Best CV MAE. Selected for Optuna tuning.",
            },
            {
                "name": "LightGBM (tuned)", "type": "winner",
                "cv_mae": 1_415, "cv_mae_std": None,
                "test_mae": 1_573, "test_rmse": 19_770,
                "test_r2": 0.0061, "test_adj_r2": 0.0016,
                "fit_time_s": 2.24, "is_winner": True,
                "note": "30-trial Optuna TPE search. Best R2, Adj R2, and RMSE overall.",
            },
        ],
    }


class _DemoModel:
    """
    Lightweight stand-in when production_model.pkl is absent.
    Exposes the same interface the app uses: .predict() and
    .regressor_.feature_importances_.
    """

    # Plausible gain-based importances in FEATURE_COLS order
    _IMPORTANCES = np.array([
        300,    # minimum_nights
        1_000,  # number_of_reviews
        900,    # reviews_per_month
        600,    # number_of_reviews_ltm
        8_000,  # is_entire_home        <- dominant signal
        3_000,  # is_private_room
        500,    # has_reviews
        350,    # is_professional_host
        400,    # high_availability
        4_500,  # availability_rate
        1_200,  # days_since_last_review
        250,    # name_length
        200,    # name_word_count
        1_800,  # name_has_luxury_kw
        450,    # name_has_transport_kw
        1_500,  # log_minimum_nights
        700,    # log_number_of_reviews
        800,    # review_recency_ratio
        5_000,  # demand_index
        2_000,  # host_scale_score
        3_500,  # min_nights_availability_ratio
    ], dtype=float)

    class _Regressor:
        pass

    def __init__(self) -> None:
        self.regressor_ = self._Regressor()
        self.regressor_.feature_importances_ = self._IMPORTANCES

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Log-linear approximation calibrated to Bangkok price range."""
        score = (
            7.20
            + X["is_entire_home"].values          *  0.45
            - X["is_private_room"].values          *  0.35
            + X["demand_index"].values             *  0.30
            + X["host_scale_score"].values         *  0.10
            + X["name_has_luxury_kw"].values       *  0.40
            + X["name_has_transport_kw"].values    *  0.15
            - X["log_minimum_nights"].values       *  0.05
            + X["log_number_of_reviews"].values    *  0.02
        )
        return np.expm1(score)


# ── Data loaders (all @st.cache_data; fall back to demo when files missing) ─

@st.cache_data
def load_features() -> pd.DataFrame:
    path = DATA_DIR / "features.csv"
    if path.exists():
        return pd.read_csv(path)
    return _make_demo_features()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = DATA_DIR / "predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return _make_demo_predictions()


@st.cache_data
def load_model_results() -> dict:
    path = DATA_DIR / "model_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _make_demo_model_results()


@st.cache_resource
def load_model():
    path = MODELS_DIR / "production_model.pkl"
    if path.exists():
        return joblib.load(path)
    return _DemoModel()


# ── Shared helpers ─────────────────────────────────────────────────────────

def _demo_banner() -> None:
    """Show once per page when running without training artifacts."""
    if _DEMO_MODE:
        st.info(
            "**Demo mode** — training artifacts not found. "
            "Charts and predictions use synthetic data that mirrors the real Bangkok "
            "dataset distribution. Run the pipeline to load real results:\n\n"
            "`python src/features/run_features.py` → "
            "`python src/models/run_training.py` → "
            "`python src/models/predict.py`",
            icon="ℹ️",
        )


def styled_metric(label: str, value: str, delta: str = "", color: str = PRIMARY) -> None:
    st.markdown(
        f"""
        <div style="background:{LIGHT_BG};border-left:4px solid {color};
                    padding:12px 16px;border-radius:6px;margin-bottom:8px">
            <div style="font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:0.05em">{label}</div>
            <div style="font-size:1.6rem;font-weight:700;color:{DARK}">{value}</div>
            {"<div style='font-size:0.8rem;color:#888'>"+delta+"</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<span style='color:#888'>{subtitle}</span>", unsafe_allow_html=True)
    st.markdown("---")


def badge(text: str, color: str = TEAL) -> str:
    return (
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:0.78rem;font-weight:600;margin:2px;display:inline-block'>"
        f"{text}</span>"
    )


# ── Page 1 — Project Overview ──────────────────────────────────────────────

def page_overview() -> None:
    _demo_banner()

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{PRIMARY},{ORANGE});
                    padding:40px 32px;border-radius:12px;margin-bottom:32px;color:white">
            <h1 style="margin:0;font-size:2.2rem">Bangkok Airbnb Price Predictor</h1>
            <p style="margin:8px 0 0;font-size:1.05rem;opacity:0.9">
                End-to-end ML pipeline &nbsp;·&nbsp; 23,273 listings &nbsp;·&nbsp;
                LightGBM + Optuna &nbsp;·&nbsp; MLflow tracked
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = load_model_results()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        styled_metric("Listings analysed", f"{results['data_points']:,}", color=PRIMARY)
    with c2:
        styled_metric("Features engineered", str(results["feature_count"]), color=TEAL)
    with c3:
        styled_metric("Best MAE", f"฿{results['winner_mae']:,} THB", color=ORANGE)
    with c4:
        styled_metric(
            "Improvement vs baseline",
            f"{results['improvement_pct']}%",
            "over Linear Regression",
            color=DARK,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        section_header("What This Project Does")
        st.markdown(
            """
            This project builds a **regression model** to predict nightly Airbnb prices
            for Bangkok listings. It walks through every stage of a production ML workflow:

            - **Data loading & quality gates** — 9 automated checks, structured pass/fail report
            - **Cleaning pipeline** — 28,806 → 23,273 rows (80.8% retained)
            - **Exploratory analysis** — price distribution, neighbourhood heatmaps, correlation study
            - **Feature engineering** — 16 raw → 33 engineered → 21 selected features
            - **Model comparison** — Linear Regression baseline + Random Forest, XGBoost, LightGBM
            - **Hyperparameter tuning** — 30-trial Optuna TPE search on LightGBM
            - **Experiment tracking** — MLflow logs every run (params, metrics, artifacts)
            """
        )

    with col_right:
        section_header("Tech Stack")
        techs = [
            ("Python 3.11", PRIMARY), ("Pandas / NumPy", TEAL), ("scikit-learn", ORANGE),
            ("LightGBM", PRIMARY),    ("XGBoost", TEAL),         ("Optuna", ORANGE),
            ("MLflow", PRIMARY),      ("Streamlit", TEAL),        ("Plotly", ORANGE),
            ("Jupyter", PRIMARY),
        ]
        st.markdown(" ".join(badge(t, c) for t, c in techs), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("Dataset")
        st.markdown(
            """
            **Inside Airbnb — Bangkok, Thailand**

            - 28,806 raw listings (post-cleaning: 23,273)
            - Target: nightly price in Thai Baht (THB)
            - Median price: ฿1,379 · Range: ฿4 – ฿1,000,000
            - Price skewness: 53.24 → log1p transform applied
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Key Challenge: Near-Zero R²")
    st.info(
        "All models show R² ≈ 0.001–0.006 on the raw THB scale. This is expected — "
        "roughly 50 luxury listings priced at ฿1,000,000 dominate the total sum of squares, "
        "making R² a misleading metric here. **MAE (mean absolute error) is the honest signal**: "
        "our best model errs by ฿1,573 against a median price of ฿1,379."
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#aaa;font-size:0.8rem'>"
        "Built by Saahithi Chippa &nbsp;·&nbsp; Bangkok Airbnb Price Prediction Portfolio Project"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page 2 — Explore the Data ─────────────────────────────────────────────

def page_eda() -> None:
    _demo_banner()
    section_header("Explore the Data", "Interactive visualisations of the Bangkok Airbnb dataset")

    df = load_features()

    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            room_types  = ["All"] + sorted(df["room_type"].unique().tolist())
            room_filter = st.selectbox("Room type", room_types)
        with fc2:
            price_cap = st.slider(
                "Price cap (THB) — excludes luxury outliers",
                min_value=1_000, max_value=30_000, value=15_000, step=500,
            )
        with fc3:
            top_n_hoods = st.slider("Top N neighbourhoods", 5, 20, 10)

    filtered = df[df["price"] <= price_cap].copy()
    if room_filter != "All":
        filtered = filtered[filtered["room_type"] == room_filter]

    st.caption(f"Showing {len(filtered):,} listings after filters (raw: {len(df):,})")

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**Price Distribution**")
        fig = px.histogram(
            filtered, x="price", nbins=80,
            color_discrete_sequence=[PRIMARY],
            labels={"price": "Nightly price (THB)"},
        )
        fig.update_layout(
            showlegend=False, plot_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Nightly price (THB)", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown("**Median Price by Room Type**")
        room_stats = (
            filtered.groupby("room_type")["price"]
            .median().reset_index()
            .sort_values("price", ascending=True)
        )
        fig2 = px.bar(
            room_stats, x="price", y="room_type", orientation="h",
            color_discrete_sequence=[TEAL],
            labels={"price": "Median price (THB)", "room_type": ""},
        )
        fig2.update_layout(plot_bgcolor="white", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Price Distribution by Top Neighbourhoods**")
    top_hoods = (
        filtered.groupby("neighbourhood")["price"]
        .median().nlargest(top_n_hoods).index.tolist()
    )
    hood_df = filtered[filtered["neighbourhood"].isin(top_hoods)]
    fig3 = px.box(
        hood_df, x="neighbourhood", y="price",
        color="neighbourhood",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"price": "Nightly price (THB)", "neighbourhood": ""},
    )
    fig3.update_layout(showlegend=False, plot_bgcolor="white", margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig3, use_container_width=True)

    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("**Correlation with Price (top features)**")
        num_cols = [c for c in FEATURE_COLS if c in filtered.columns] + ["price"]
        corr_series = (
            filtered[num_cols].corr()["price"]
            .drop("price").abs()
            .sort_values(ascending=False).head(12)
        )
        fig4 = px.bar(
            x=corr_series.values, y=corr_series.index, orientation="h",
            color=corr_series.values,
            color_continuous_scale=[[0, LIGHT_BG], [1, PRIMARY]],
            labels={"x": "|Correlation|", "y": ""},
        )
        fig4.update_layout(
            coloraxis_showscale=False, plot_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with r3c2:
        st.markdown("**Reviews vs Price**")
        sample = filtered.sample(min(2_000, len(filtered)), random_state=42)
        fig5 = px.scatter(
            sample, x="number_of_reviews", y="price",
            color="room_type",
            color_discrete_sequence=[PRIMARY, TEAL, ORANGE, DARK],
            opacity=0.5,
            labels={"number_of_reviews": "Number of reviews", "price": "Price (THB)"},
        )
        fig5.update_layout(
            plot_bgcolor="white", margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(title="", orientation="h", y=-0.2),
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Key EDA Findings")
    kf1, kf2, kf3 = st.columns(3)
    with kf1:
        st.success(
            "**Entire homes dominate** — 71% of listings; median price ฿1,600 vs ฿800 for private rooms"
        )
    with kf2:
        st.warning(
            "**Extreme right skew** — Skewness = 53.24. A handful of luxury villas at ฿1,000,000 "
            "make log1p transform mandatory before any linear model."
        )
    with kf3:
        st.info(
            "**Reviews proxy demand** — High-review listings cluster in a tighter price band, "
            "suggesting market efficiency for well-reviewed properties."
        )


# ── Page 3 — Model Results ─────────────────────────────────────────────────

def page_models() -> None:
    _demo_banner()
    section_header("Model Results", "Comparison, feature importance, residual analysis, and live prediction")

    results = load_model_results()
    pred_df = load_predictions()
    model   = load_model()

    # ── Model comparison table ─────────────────────────────────────────────
    st.markdown("#### Model Comparison")
    rows = []
    for m in results["models"]:
        rows.append({
            "Model":     m["name"],
            "Type":      m["type"].capitalize(),
            "CV MAE":    f"฿{m['cv_mae']:,}" if m["cv_mae"] else "—",
            "Test MAE":  f"฿{m['test_mae']:,}",
            "Test RMSE": f"฿{m['test_rmse']:,}",
            "Test R²":   f"{m['test_r2']:.4f}",
            "Adj R²":    f"{m['test_adj_r2']:.4f}",
            "Fit (s)":   m["fit_time_s"],
            "Winner":    "🏆" if m["is_winner"] else "",
        })
    comp_df = pd.DataFrame(rows)

    def highlight_winner(row):
        color = "#FFF3CD" if row["Winner"] == "🏆" else ""
        return [f"background-color:{color}"] * len(row)

    st.dataframe(
        comp_df.style.apply(highlight_winner, axis=1),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Why LightGBM (tuned)?"):
        st.markdown(
            """
            - **Fastest default training** (0.7 s) and best cross-validated MAE among default models → selected for Optuna tuning
            - **30-trial Optuna TPE search** explored 9 hyperparameters; best trial (#27) MAE = ฿1,415 CV
            - **Best test R² and Adj R²** across all runs (R² = 0.0061)
            - Tuned model's test MAE (฿1,573) is within noise of default (฿1,560) — confirming default params were already near-optimal for this dataset
            - All models wrapped with `TransformedTargetRegressor(log1p / expm1)` to handle price skewness = 53.24
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance + residual plot ────────────────────────────────
    fi_col, res_col = st.columns(2)

    with fi_col:
        st.markdown("#### Feature Importance (Top 15)")
        imp = model.regressor_.feature_importances_
        fi_df = (
            pd.DataFrame({"feature": FEATURE_COLS, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        fig_fi = px.bar(
            fi_df.sort_values("importance"),
            x="importance", y="feature", orientation="h",
            color="importance",
            color_continuous_scale=[[0, LIGHT_BG], [1, PRIMARY]],
            labels={"importance": "Importance (gain)", "feature": ""},
        )
        fig_fi.update_layout(
            coloraxis_showscale=False, plot_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with res_col:
        st.markdown("#### Residuals: Actual vs Predicted")
        price_cap = st.slider(
            "Price cap for residual plot (THB)",
            1_000, 20_000, 10_000, 1_000, key="res_cap",
        )
        plot_df = pred_df[pred_df["price"] <= price_cap]
        plot_df = plot_df.sample(min(2_000, len(plot_df)), random_state=42)
        fig_res = px.scatter(
            plot_df, x="price", y="predicted_price",
            color="abs_error",
            color_continuous_scale=[[0, TEAL], [0.5, ORANGE], [1, PRIMARY]],
            opacity=0.5,
            labels={"price": "Actual price (THB)", "predicted_price": "Predicted price (THB)"},
        )
        lo, hi = plot_df["price"].min(), plot_df["price"].max()
        fig_res.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color=DARK, dash="dash", width=1),
            name="Perfect prediction", showlegend=True,
        ))
        fig_res.update_layout(
            plot_bgcolor="white", margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_colorbar=dict(title="Abs error"),
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Error distribution ─────────────────────────────────────────────────
    st.markdown("#### Prediction Error Distribution")
    ec1, ec2 = st.columns(2)
    with ec1:
        err_capped = pred_df[pred_df["abs_error"] <= 10_000]
        fig_err = px.histogram(
            err_capped, x="abs_error", nbins=60,
            color_discrete_sequence=[TEAL],
            labels={"abs_error": "Absolute error (THB)"},
        )
        fig_err.update_layout(plot_bgcolor="white", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_err, use_container_width=True)
    with ec2:
        mae_by_room = (
            pred_df.groupby("room_type")["abs_error"]
            .median().reset_index().sort_values("abs_error")
        )
        fig_room = px.bar(
            mae_by_room, x="abs_error", y="room_type", orientation="h",
            color_discrete_sequence=[ORANGE],
            labels={"abs_error": "Median absolute error (THB)", "room_type": ""},
        )
        fig_room.update_layout(plot_bgcolor="white", margin=dict(t=10, b=10, l=10, r=10))
        st.markdown("**Median Error by Room Type**")
        st.plotly_chart(fig_room, use_container_width=True)

    # ── Try it yourself ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Try It Yourself", "Enter listing details to get a price prediction")

    with st.form("prediction_form"):
        p1, p2, p3 = st.columns(3)
        with p1:
            listing_name  = st.text_input("Listing name", "Cozy BTS Asok Suite with pool")
            room_type_sel = st.selectbox(
                "Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
            )
        with p2:
            min_nights    = st.number_input("Minimum nights", 1, 365, 2)
            n_reviews     = st.number_input("Number of reviews", 0, 1_000, 25)
            rpm           = st.number_input("Reviews per month", 0.0, 20.0, 1.5, 0.1)
        with p3:
            availability  = st.slider("Availability (days/year)", 0, 365, 200)
            n_reviews_ltm = st.number_input("Reviews last 12 months", 0, 500, 10)
            host_listings = st.number_input("Host total listings", 1, 500, 3)

        adv1, adv2 = st.columns(2)
        with adv1:
            days_inactive = st.slider("Days since last review", 0, 730, 30)
        with adv2:
            is_pro_host = st.checkbox("Professional host (10+ listings)", value=False)

        submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

    if submitted:
        feats = _build_input_features(
            listing_name=listing_name, room_type=room_type_sel,
            min_nights=min_nights, n_reviews=n_reviews,
            reviews_per_month=rpm, availability=availability,
            n_reviews_ltm=n_reviews_ltm, host_listings=host_listings,
            days_since_last_review=days_inactive, is_pro_host=is_pro_host,
        )
        X    = pd.DataFrame([feats])[FEATURE_COLS]
        pred = float(model.predict(X)[0])

        pr1, pr2, pr3 = st.columns(3)
        with pr1:
            st.metric("Predicted nightly price", f"฿{pred:,.0f} THB")
        with pr2:
            label = "Model test MAE" if not _DEMO_MODE else "Demo model MAE (approx)"
            st.metric(label, "฿1,573 THB", help="Typical prediction error on holdout set")
        with pr3:
            st.metric("Likely range", f"฿{max(0, pred-1573):,.0f} – ฿{pred+1573:,.0f}")


def _build_input_features(
    listing_name: str, room_type: str,
    min_nights: int, n_reviews: int, reviews_per_month: float,
    availability: int, n_reviews_ltm: int, host_listings: int,
    days_since_last_review: int, is_pro_host: bool,
) -> dict:
    avail_rate = availability / 365.0
    return {
        "minimum_nights":                min_nights,
        "number_of_reviews":             n_reviews,
        "reviews_per_month":             reviews_per_month,
        "number_of_reviews_ltm":         n_reviews_ltm,
        "is_entire_home":                int(room_type == "Entire home/apt"),
        "is_private_room":               int(room_type == "Private room"),
        "has_reviews":                   int(n_reviews > 0),
        "is_professional_host":          int(is_pro_host),
        "high_availability":             int(avail_rate > 0.5),
        "availability_rate":             avail_rate,
        "days_since_last_review":        min(days_since_last_review, 730),
        "name_length":                   len(listing_name),
        "name_word_count":               len(listing_name.split()),
        "name_has_luxury_kw":            int(bool(LUXURY_KW.search(listing_name))),
        "name_has_transport_kw":         int(bool(TRANSPORT_KW.search(listing_name))),
        "log_minimum_nights":            np.log1p(min_nights),
        "log_number_of_reviews":         np.log1p(n_reviews),
        "review_recency_ratio":          n_reviews / max(days_since_last_review, 1),
        "demand_index":                  avail_rate * 0.5 + reviews_per_month * 0.3 + 0.2,
        "host_scale_score":              np.log1p(host_listings),
        "min_nights_availability_ratio": min_nights / max(avail_rate, 0.001),
    }


# ── Page 4 — How I Built This ─────────────────────────────────────────────

def page_how_i_built() -> None:
    section_header("How I Built This", "Architecture, design decisions, and lessons learned")

    st.markdown("#### Pipeline Architecture")
    st.graphviz_chart(
        """
        digraph pipeline {
            rankdir=LR
            node [shape=box style=filled fontname="Helvetica" fontsize=11]

            raw   [label="Raw CSV\\n(28,806 rows)"       fillcolor="#FFE0E0"]
            qual  [label="Quality Gate\\n(9 checks)"      fillcolor="#FFE0E0"]
            clean [label="Cleaner\\n(23,273 rows)"        fillcolor="#FFE0E0"]
            eda   [label="EDA Notebook\\n(7 sections)"    fillcolor="#E0F0FF"]
            eng   [label="Feature Engineering\\n(16->33 cols)" fillcolor="#E8F5E9"]
            sel   [label="Feature Selection\\n(33->21 cols)"   fillcolor="#E8F5E9"]
            base  [label="Baseline\\nLinearRegression"    fillcolor="#FFF3E0"]
            cand  [label="Candidates\\nRF / XGB / LGBM"   fillcolor="#FFF3E0"]
            tune  [label="Optuna Tuning\\n(30 trials)"    fillcolor="#FFF3E0"]
            mlf   [label="MLflow Tracking\\n(params+metrics)" fillcolor="#F3E5F5"]
            prod  [label="production_model.pkl"           fillcolor="#E8F5E9" shape=cylinder]
            app   [label="Streamlit Dashboard"            fillcolor="#FF5A5F" fontcolor=white]

            raw -> qual -> clean -> eda
            clean -> eng -> sel
            sel -> base -> mlf
            sel -> cand -> tune -> mlf
            mlf -> prod -> app
        }
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Build Timeline")
    timeline = [
        ("Stage 1", "Data Loading & Quality", "loader.py, quality.py",
         "Built 9 automated checks: null rates, type coercion, target distribution, IQR outlier flagging. "
         "Quality gate returns structured pass/fail/warn dict so downstream scripts can react programmatically."),
        ("Stage 2", "Data Cleaning", "cleaner.py",
         "Dropped 100%-null columns (neighbourhood_group, license), removed price-null rows, "
         "imputed reviews_per_month=0 for never-reviewed listings. 80.8% row retention."),
        ("Stage 3", "Exploratory Analysis", "notebooks/eda.ipynb",
         "7-section notebook: price skewness (53.24), room-type breakdown, neighbourhood heatmap, "
         "correlation study. Key insight: log1p transform is mandatory before any linear model."),
        ("Stage 4", "Feature Engineering", "engineering.py",
         "16 raw -> 33 engineered features. Domain features: is_entire_home, availability_rate, "
         "demand_index. NLP flags: luxury keywords, transport proximity. Log transforms for counts."),
        ("Stage 5", "Feature Selection", "engineering.py: select_features()",
         "Correlation filter (|r|>0.95) dropped availability_365 and calculated_host_listings_count. "
         "Median-based variance filter dropped latitude and longitude. Final: 21 features."),
        ("Stage 6", "Model Comparison", "baseline.py, train.py",
         "Baseline LinearRegression (MAE=1,693). Candidates: RandomForest (-6.7%), XGBoost (-7.0%), "
         "LightGBM (-7.9%). All wrapped in TransformedTargetRegressor(log1p/expm1)."),
        ("Stage 7", "Hyperparameter Tuning", "tuning.py",
         "30-trial Optuna TPE search on LightGBM (9 hyperparameters, 5-fold CV). "
         "Best trial #27: n_est=740, lr=0.037, num_leaves=125. Tuned MAE=1,573."),
        ("Stage 8", "MLflow Tracking", "run_training.py",
         "Logs params, train/test metrics (MAE, RMSE, R2, Adj R2), fit time, and model artifacts "
         "for every run. MLflow store placed outside OneDrive to avoid sync file-locking on Windows."),
    ]
    for stage, title, files, desc in timeline:
        with st.expander(f"**{stage} — {title}** · `{files}`"):
            st.markdown(desc)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Key Design Decisions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            f"""
            <div style="background:{LIGHT_BG};border-radius:8px;padding:16px;margin-bottom:12px">
                <strong style="color:{PRIMARY}">Log1p Target Transform</strong><br>
                Price skewness of 53.24 makes linear models train on a near-constant target
                otherwise. Wrapping all models in <code>TransformedTargetRegressor</code> keeps
                the pipeline interface clean — <code>predict()</code> always returns THB.
            </div>
            <div style="background:{LIGHT_BG};border-radius:8px;padding:16px;margin-bottom:12px">
                <strong style="color:{TEAL}">Median Variance Threshold</strong><br>
                The initial sentinel value (9999) for <code>days_since_last_review</code>
                inflated mean variance by 50,000x, causing the variance filter to drop every
                other feature. Switched to median-based reference + capped the feature at 730 days.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"""
            <div style="background:{LIGHT_BG};border-radius:8px;padding:16px;margin-bottom:12px">
                <strong style="color:{ORANGE}">MAE over R2</strong><br>
                ~50 luxury listings at ฿1,000,000 dominate sum of squares, making R2 near zero
                even for useful models. MAE on the original scale is the honest KPI — it directly
                answers "how wrong is the prediction in Thai Baht?"
            </div>
            <div style="background:{LIGHT_BG};border-radius:8px;padding:16px;margin-bottom:12px">
                <strong style="color:{DARK}">MLflow Outside OneDrive</strong><br>
                OneDrive's sync client locks files mid-write, causing <code>PermissionError</code>
                when MLflow reads params it just logged. Solution: set
                <code>MLRUNS_DIR = Path.home() / "mlruns"</code> (outside the synced folder).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Source Code")
    st.code(
        "git clone https://github.com/Saahithi-Chippa/airbnb-price-prediction\n"
        "pip install -r requirements.txt\n"
        "pip install -e .\n"
        "python src/data/loader.py            # load & profile\n"
        "python src/features/run_features.py  # engineer + select\n"
        "python src/models/run_training.py    # MLflow training run\n"
        "python src/models/predict.py         # generate predictions.csv\n"
        "streamlit run app/streamlit_app.py   # launch dashboard",
        language="bash",
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#aaa;font-size:0.8rem'>"
        "Built by Saahithi Chippa &nbsp;·&nbsp; Bangkok Airbnb Price Prediction Portfolio Project"
        "</div>",
        unsafe_allow_html=True,
    )


# ── App entry point ────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Bangkok Airbnb Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{background-color:#FFFFFF}}
        [data-testid="stSidebar"] {{background-color:{LIGHT_BG}}}
        h1,h2,h3 {{color:{DARK}}}
        .stButton > button {{background-color:{PRIMARY};color:white;border:none;border-radius:6px}}
        .stButton > button:hover {{background-color:{ORANGE}}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{PRIMARY};margin-bottom:4px'>Bangkok Airbnb</h2>"
            f"<p style='color:#888;font-size:0.85rem'>Price Prediction Portfolio</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["Project Overview", "Explore the Data", "Model Results", "How I Built This"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        if _DEMO_MODE:
            st.warning("Demo mode — synthetic data", icon="⚠️")
        else:
            st.markdown(
                "<div style='font-size:0.78rem;color:#aaa'>"
                "Model: LightGBM (tuned)<br>"
                "Test MAE: ฿1,573 THB<br>"
                "Dataset: 23,273 listings<br>"
                "City: Bangkok, Thailand"
                "</div>",
                unsafe_allow_html=True,
            )

    if page == "Project Overview":
        page_overview()
    elif page == "Explore the Data":
        page_eda()
    elif page == "Model Results":
        page_models()
    else:
        page_how_i_built()


if __name__ == "__main__":
    main()
