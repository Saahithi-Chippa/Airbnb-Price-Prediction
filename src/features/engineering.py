import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ------------------------------------------------------------------ #
    # Category 1: Listing & Host Characteristics (domain-specific)        #
    # ------------------------------------------------------------------ #

    # Entire home/apt commands the highest price tier; a binary flag is
    # cleaner than passing the raw string to tree models or linear encoders.
    df["is_entire_home"] = (df["room_type"] == "Entire home/apt").astype(int)

    # Private rooms are mid-tier; two flags together represent all three
    # room-type price tiers without needing one-hot encoding downstream.
    df["is_private_room"] = (df["room_type"] == "Private room").astype(int)

    # Listings with zero reviews carry unknown-quality risk for guests and
    # must often price below the market to attract their first booking.
    df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)

    # Hosts managing 5+ listings operate commercially with strategic pricing,
    # professional photography, and managed calendars — a different supply
    # dynamic than a casual single-property host.
    df["is_professional_host"] = (df["calculated_host_listings_count"] >= 5).astype(int)

    # A listing available >180 days/year is actively marketed and kept
    # guest-ready; passive listings with low availability are often priced
    # inconsistently or not updated.
    df["high_availability"] = (df["availability_365"] > 180).astype(int)

    # Continuous availability rate (0–1) captures the granularity lost in
    # the binary flag above and scales naturally for tree splits.
    df["availability_rate"] = df["availability_365"] / 365

    # Recency of the last review proxies how actively the host manages the
    # listing; stale listings are less competitive regardless of price.
    # Capped at 730 days (2 years) — beyond that the signal is the same:
    # "not recently reviewed". Using 9999 as a sentinel inflates variance
    # and skews distance-based models, so a finite cap is cleaner.
    today = pd.Timestamp.today().normalize()
    last_review_dt = pd.to_datetime(df["last_review"], errors="coerce")
    df["days_since_last_review"] = (
        (today - last_review_dt).dt.days.fillna(730).clip(upper=730).astype(int)
    )

    # ------------------------------------------------------------------ #
    # Category 2: Text & NLP Features (engineered from listing name)      #
    # ------------------------------------------------------------------ #

    name_lower = df["name"].fillna("").str.lower()

    # Longer, more descriptive names reflect host effort and listing quality;
    # marketplace research consistently links name richness to higher prices.
    df["name_length"] = df["name"].str.len().fillna(0).astype(int)
    df["name_word_count"] = df["name"].str.split().str.len().fillna(0).astype(int)

    # Luxury-signal keywords in the title directly indicate premium
    # positioning — hosts who mention pools, views, or villas are targeting
    # a higher willingness-to-pay segment.
    _luxury_kw = (
        "luxury|pool|villa|penthouse|view|rooftop|sky|resort|suite|jacuzzi|infinity"
    )
    df["name_has_luxury_kw"] = name_lower.str.contains(_luxury_kw, regex=True).astype(int)

    # BTS/MRT proximity is Bangkok guests' top search filter; listings that
    # advertise transit access or central neighbourhoods command a clear
    # location premium over otherwise similar properties.
    _transport_kw = (
        "bts|mrt|sukhumvit|silom|cbd|central|asok|nana|ploenchit|siam|thonglor|ekkamai"
    )
    df["name_has_transport_kw"] = name_lower.str.contains(_transport_kw, regex=True).astype(int)

    # ------------------------------------------------------------------ #
    # Category 3: Statistical Transforms                                  #
    # ------------------------------------------------------------------ #

    # minimum_nights is heavily right-skewed (max 1,115); log1p compresses
    # the tail so the model does not over-weight extreme stay requirements.
    df["log_minimum_nights"] = np.log1p(df["minimum_nights"])

    # number_of_reviews follows a similar long-tail distribution; log1p
    # normalises it before tree splits or any linear scaling step.
    df["log_number_of_reviews"] = np.log1p(df["number_of_reviews"])

    # Fraction of reviews received in the last 12 months — a high ratio
    # means the listing is currently popular, not coasting on old reviews.
    # Clipped to [0, 1] to handle edge cases from integer rounding.
    df["review_recency_ratio"] = (
        df["number_of_reviews_ltm"]
        / df["number_of_reviews"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    # ------------------------------------------------------------------ #
    # Category 4: Interaction Features                                    #
    # ------------------------------------------------------------------ #

    # High monthly reviews AND high availability = an in-demand listing
    # with an open calendar. The product captures "achievable bookings"
    # better than either feature alone — a listing can have many reviews
    # but be fully blocked, or be available but unbooked.
    df["demand_index"] = df["reviews_per_month"] * df["availability_rate"]

    # Professional hosts (many listings) who maintain high availability
    # operate like small hotels. This interaction flags that commercial
    # supply dynamic, which correlates with strategic above-market pricing.
    df["host_scale_score"] = df["calculated_host_listings_count"] * df["availability_rate"]

    # A high minimum-stay relative to actual availability signals a host
    # targeting long-term renters rather than tourists — a fundamentally
    # different price elasticity and competitive set.
    df["min_nights_availability_ratio"] = (
        df["minimum_nights"]
        / df["availability_365"].replace(0, np.nan)
    ).fillna(0)

    return df


CORR_THRESHOLD = 0.95
VARIANCE_FACTOR = 0.01   # feature must have >= 1% of the mean feature variance


def select_features(
    df: pd.DataFrame,
    target: str = "price",
    corr_threshold: float = CORR_THRESHOLD,
    variance_factor: float = VARIANCE_FACTOR,
) -> tuple[list[str], pd.DataFrame]:
    """
    Two-pass feature selection:
      1. Drop columns whose |Pearson r| > corr_threshold with an earlier column.
      2. Drop columns whose variance < variance_factor * mean(all variances).
    Returns (selected_feature_names, reduced_dataframe).
    """
    _exclude = {target, "id", "host_id"}
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in _exclude
    ]

    dropped_corr: list[tuple[str, str, float]] = []   # (dropped, kept, r)
    dropped_var:  list[tuple[str, float, float]] = []  # (dropped, var, threshold)

    # ------------------------------------------------------------------ #
    # Pass 1: Correlation filter                                          #
    # ------------------------------------------------------------------ #
    corr_matrix = df[numeric_cols].corr().abs()

    # Upper triangle only — avoid double-counting each pair
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    corr_drop: set[str] = set()
    for col in upper.columns:
        if col in corr_drop:
            continue  # already flagged; don't use it as the "keeper"
        redundant = upper[col][upper[col] > corr_threshold].index.tolist()
        for partner in redundant:
            if partner not in corr_drop:
                corr_drop.add(partner)
                dropped_corr.append((partner, col, corr_matrix.loc[col, partner]))

    after_corr = [c for c in numeric_cols if c not in corr_drop]

    # ------------------------------------------------------------------ #
    # Pass 2: Variance filter                                             #
    # ------------------------------------------------------------------ #
    variances = df[after_corr].var()
    # Median is robust against one or two columns with extreme variance
    # (e.g. a date-difference column) inflating the reference and wiping
    # out legitimately useful low-scale features like binary flags.
    median_variance = variances.median()
    var_threshold = variance_factor * median_variance

    var_drop: set[str] = set()
    for col, var in variances.items():
        if var < var_threshold:
            var_drop.add(col)
            dropped_var.append((col, float(var), var_threshold))

    selected = [c for c in after_corr if c not in var_drop]

    # ------------------------------------------------------------------ #
    # Logging                                                             #
    # ------------------------------------------------------------------ #
    print("Feature selection log:")

    if dropped_corr:
        print(f"  Dropped for high correlation (|r| > {corr_threshold}):")
        for dropped, kept, r in dropped_corr:
            print(f"    - '{dropped}'  |r|={r:.4f} with '{kept}' (kept)")
    else:
        print(f"  No features dropped for high correlation (threshold {corr_threshold})")

    if dropped_var:
        print(
            f"  Dropped for low variance "
            f"(< {var_threshold:.6f} = {variance_factor:.0%} of median variance {median_variance:.4f}):"
        )
        for col, var, thr in dropped_var:
            print(f"    - '{col}'  var={var:.6f}")
    else:
        print(f"  No features dropped for low variance (threshold {var_threshold:.6f})")

    print(
        f"\nSummary: {len(numeric_cols)} numeric features"
        f" -> {len(selected)} selected"
        f"  (dropped {len(dropped_corr)} correlated, {len(dropped_var)} low-variance)"
    )

    # Return selected names + df with redundant numeric columns removed
    all_dropped = corr_drop | var_drop
    reduced_df = df.drop(columns=list(all_dropped))
    return selected, reduced_df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from data.loader import load_csv

    cleaned_csvs = [p for p in DATA_DIR.glob("*.csv") if p.name == "cleaned.csv"]
    if not cleaned_csvs:
        raise FileNotFoundError(
            "cleaned.csv not found — run src/data/cleaner.py first"
        )

    df_raw = load_csv("cleaned.csv")
    print(f"Input:  {len(df_raw):,} rows x {df_raw.shape[1]} columns")

    df_feat = create_features(df_raw)

    new_cols = [c for c in df_feat.columns if c not in df_raw.columns]
    print(f"Output: {len(df_feat):,} rows x {df_feat.shape[1]} columns")
    print(f"\n{len(new_cols)} new features engineered:")

    categories = {
        "Listing & Host": [
            "is_entire_home", "is_private_room", "has_reviews",
            "is_professional_host", "high_availability",
            "availability_rate", "days_since_last_review",
        ],
        "Text & NLP": [
            "name_length", "name_word_count",
            "name_has_luxury_kw", "name_has_transport_kw",
        ],
        "Statistical Transforms": [
            "log_minimum_nights", "log_number_of_reviews", "review_recency_ratio",
        ],
        "Interactions": [
            "demand_index", "host_scale_score", "min_nights_availability_ratio",
        ],
    }
    for cat, cols in categories.items():
        print(f"\n  [{cat}]")
        for col in cols:
            if col in df_feat.columns:
                print(f"    {col:40s}  mean={df_feat[col].mean():.4f}")

    out_path = DATA_DIR / "featured.csv"
    df_feat.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 60)
    selected_cols, df_selected = select_features(df_feat)
    print(f"\nSelected features ({len(selected_cols)}):")
    for col in selected_cols:
        print(f"  {col}")

    sel_path = DATA_DIR / "selected.csv"
    df_selected.to_csv(sel_path, index=False)
    print(f"\nSaved to {sel_path}")
