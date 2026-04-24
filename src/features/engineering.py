"""Feature engineering for the Spotify genre-classification / recommendation pipeline."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide two series, replacing division-by-zero with `fill`."""
    return numerator.div(denominator.replace(0, np.nan)).fillna(fill)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 10+ new features and return a copy with originals preserved.

    Categories
    ----------
    A. Domain-specific  — capture Spotify listener psychology and audio perception
    B. Statistical      — transforms that correct skew or normalise scale
    C. Interaction      — products / ratios of paired features that matter jointly

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Spotify dataset (output of cleaner.clean_data).

    Returns
    -------
    pd.DataFrame
        Original columns plus all engineered features.
    """
    out = df.copy()

    # -----------------------------------------------------------------------
    # A. Domain-specific features
    # -----------------------------------------------------------------------

    # A1: energy_loudness_product
    # Loudness and energy both measure perceived intensity but on different scales
    # (energy: 0-1, loudness: dB).  Their product surfaces tracks that are BOTH
    # high-energy AND loud — the defining signature of hard rock, EDM, and metal —
    # while down-weighting tracks that are loud but low-energy (e.g., compressed
    # spoken-word) or energetic but quiet (e.g., acoustic fingerpicking).
    out["energy_loudness_product"] = out["energy"] * (out["loudness"] + 60) / 60

    # A2: danceability_valence_vibe
    # Recommendations live or die on mood fit.  Danceability captures rhythmic
    # drive; valence captures emotional positivity.  Their product isolates the
    # "feel-good party" quadrant (high dance × high valence) vs. "dark club" (high
    # dance × low valence), a distinction pure genre labels often blur.
    out["danceability_valence_vibe"] = out["danceability"] * out["valence"]

    # A3: acoustic_instrumental_purity
    # Tracks that are simultaneously high in acousticness AND instrumentalness are
    # almost exclusively classical, jazz solo, or ambient — a tight cluster that
    # resists separation by other features alone.  The product creates a dedicated
    # "pure acoustic instrumental" axis for the recommendation engine.
    out["acoustic_instrumental_purity"] = out["acousticness"] * out["instrumentalness"]

    # A4: speechiness_bin
    # Spotify's own documentation partitions speechiness into three meaningful
    # ranges: <0.33 = music, 0.33–0.66 = music+speech (e.g., rap), >0.66 =
    # speech-dominant (podcasts, spoken word).  Encoding this directly prevents
    # linear models from treating the three zones as a continuous gradient.
    out["speechiness_bin"] = pd.cut(
        out["speechiness"],
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=[0, 1, 2],
    ).astype(int)

    # A5: tempo_zone
    # Human perception of tempo clusters naturally around BPM zones that align
    # with musical feel: slow (<80 BPM), moderate (80-120), upbeat (120-160),
    # fast (>160).  Genre classifiers trained on raw BPM struggle to generalise
    # across songs where 90 BPM and 180 BPM are double-time variations of the
    # same groove; binning removes that harmonic-aliasing noise.
    out["tempo_zone"] = pd.cut(
        out["tempo"],
        bins=[-0.001, 80, 120, 160, 1000],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # A6: duration_minutes
    # duration_ms is hard to interpret and has extreme outliers (see skewness = 11).
    # Converting to minutes makes the scale human-readable and reduces the raw
    # integer magnitude, which benefits distance-based models.
    out["duration_minutes"] = out["duration_ms"] / 60_000

    # A7: is_live_recording
    # Liveness > 0.8 is Spotify's own threshold for "likely a live performance".
    # Live recordings have distinct audio characteristics (crowd noise, slight
    # pitch instability) that confuse genre classifiers if treated as studio tracks.
    # Flagging them explicitly lets models handle them as a separate case.
    out["is_live_recording"] = (out["liveness"] > 0.8).astype(int)

    # -----------------------------------------------------------------------
    # B. Statistical features
    # -----------------------------------------------------------------------

    # B1: log_speechiness
    # Speechiness is right-skewed (skewness ≈ 4.6): ~90 % of values sit near zero
    # with a long tail toward 1.0.  log1p compresses the tail and spreads the
    # dense near-zero region — essential for KNN, SVM (RBF), and logistic
    # regression where feature scale and distribution shape matter.
    out["log_speechiness"] = np.log1p(out["speechiness"])

    # B2: log_instrumentalness
    # Same skew problem (skewness ≈ 1.7): most tracks are not instrumental, so the
    # raw column has a spike at 0 and a long tail.  The log transform gives linear
    # models a fair chance to use gradations in the 0.001–0.1 range, which encode
    # the difference between "mostly vocal" and "mostly instrumental".
    out["log_instrumentalness"] = np.log1p(out["instrumentalness"])

    # B3: log_duration_ms
    # duration_ms has extreme positive skew (skewness ≈ 11), driven by a handful
    # of very long tracks (podcasts / DJ sets).  The log transform prevents those
    # outliers from dominating distance-based and regularised models.
    out["log_duration_ms"] = np.log1p(out["duration_ms"])

    # -----------------------------------------------------------------------
    # C. Interaction features
    # -----------------------------------------------------------------------

    # C1: energy_to_acousticness_ratio
    # Energy and acousticness are strongly anti-correlated (r = -0.73), meaning
    # they sit on opposite ends of the same "electric vs. acoustic" axis.  Their
    # ratio collapses two collinear columns into a single bipolar scale: high
    # values = electric/produced, low values = acoustic/organic.  This eliminates
    # redundancy in linear models while retaining the discriminating information.
    out["energy_to_acousticness_ratio"] = _safe_ratio(out["energy"], out["acousticness"])

    # C2: danceability_tempo_drive
    # A track can be fast but hard to dance to (complex polyrhythm) or slow but
    # highly danceable (reggaeton ~90 BPM with a strong grid).  Multiplying the
    # two captures tracks that combine rhythmic drive (danceability) WITH physical
    # pace (tempo) — the "high-octane dance floor" signal that neither feature
    # encodes alone.  Normalised by 250 (approx. max tempo) to keep the range in
    # [0, 1].
    out["danceability_tempo_drive"] = out["danceability"] * (out["tempo"] / 250)

    # C3: loudness_norm
    # Raw loudness ranges from -49.5 to +4.5 dB.  Normalising to [0, 1] via
    # min-max over the observed range makes it directly comparable to the other
    # [0, 1]-bounded audio features and prevents it from dominating Euclidean
    # distance calculations purely because of its larger absolute scale.
    _loud_min, _loud_range = -49.531, 54.063  # observed min, (max - min)
    out["loudness_norm"] = (out["loudness"] - _loud_min) / _loud_range

    return out


def select_features(
    df: pd.DataFrame,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    exclude_cols: list[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Remove low-variance and highly-correlated numeric features.

    Steps
    -----
    1. Restrict candidate set to numeric columns (non-numeric passthrough).
    2. Drop features whose variance is below ``variance_threshold`` × overall
       mean variance — near-constant columns carry no discriminating signal.
    3. Build the correlation matrix; for each pair with |r| > ``correlation_threshold``
       drop the *second* feature encountered (keeps the one that appeared first
       in the column order, which is typically the original / more interpretable
       column).
    4. Return the selected feature names and the reduced dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with original + engineered features (output of create_features).
    variance_threshold : float
        Fraction of mean variance below which a feature is dropped (default 0.01).
    correlation_threshold : float
        Absolute correlation above which a feature is considered redundant (default 0.95).
    exclude_cols : list[str], optional
        Columns to exclude from selection entirely (e.g. target, identifiers).
        They are passed through unchanged and not counted as selected features.

    Returns
    -------
    tuple[list[str], pd.DataFrame]
        (selected_feature_names, reduced_dataframe)
        selected_feature_names contains only the kept *numeric* features.
        reduced_dataframe contains selected numeric columns + all non-numeric
        passthrough columns + any excluded columns.
    """
    exclude_cols = set(exclude_cols or [])

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude_cols
    ]
    passthrough_cols = [c for c in df.columns if c not in numeric_cols and c not in exclude_cols]

    candidates = df[numeric_cols].copy()
    dropped: dict[str, str] = {}  # col -> reason

    # ------------------------------------------------------------------
    # Step 1 — variance filter
    # ------------------------------------------------------------------
    # Variance is only comparable across features when they are on the same
    # scale.  Raw features span wildly different ranges ([0,1] audio features
    # vs. duration_ms in the billions), so we min-max normalise each column
    # to [0,1] before comparing variances.  The threshold is then applied
    # as a fraction of the mean *normalised* variance, keeping the cutoff
    # scale-invariant and preventing a single large-magnitude column from
    # inflating the mean and wiping out everything else.
    col_min = candidates.min()
    col_range = (candidates.max() - col_min).replace(0, np.nan)
    normalised = (candidates - col_min) / col_range

    norm_variances = normalised.var()
    overall_mean_var = norm_variances.mean()
    cutoff = variance_threshold * overall_mean_var

    low_var_cols = norm_variances[norm_variances < cutoff].index.tolist()
    for col in low_var_cols:
        dropped[col] = (
            f"low variance  norm_var={norm_variances[col]:.6f}  "
            f"(threshold={cutoff:.6f}, {variance_threshold:.0%} of mean_norm_var={overall_mean_var:.6f})"
        )

    candidates = candidates.drop(columns=low_var_cols)

    # ------------------------------------------------------------------
    # Step 2 — correlation filter
    # ------------------------------------------------------------------
    corr_matrix = candidates.corr().abs()

    # Upper triangle only — avoid double-counting pairs
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    corr_drop: set[str] = set()
    for col in upper.columns:
        if col in corr_drop:
            continue
        # All *other* candidates that correlate above threshold with this col
        partners = upper[col][upper[col] > correlation_threshold].index.tolist()
        for partner in partners:
            if partner not in corr_drop:
                corr_drop.add(partner)
                dropped[partner] = (
                    f"high correlation  |r|={corr_matrix.loc[col, partner]:.4f}  "
                    f"with '{col}'  (threshold={correlation_threshold})"
                )

    candidates = candidates.drop(columns=list(corr_drop))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    total_in = len(numeric_cols)
    total_kept = len(candidates.columns)

    print(f"\n{'='*60}")
    print(f"  FEATURE SELECTION REPORT")
    print(f"{'='*60}")
    print(f"  Numeric features in   : {total_in}")
    print(f"  Dropped               : {len(dropped)}")
    print(f"  Numeric features kept : {total_kept}")
    print(f"  Variance cutoff       : {cutoff:.6f}  "
          f"({variance_threshold:.0%} × mean_var {overall_mean_var:.6f})")
    print(f"  Correlation threshold : {correlation_threshold}")

    if dropped:
        print(f"\n  Dropped features:")
        for col, reason in dropped.items():
            print(f"    [-] {col:<40} {reason}")
    else:
        print(f"\n  No features dropped.")

    print(f"\n  Kept features ({total_kept}):")
    for col in candidates.columns:
        print(f"    [+] {col}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Assemble output dataframe
    # ------------------------------------------------------------------
    selected_names = candidates.columns.tolist()

    kept_cols_ordered = [
        c for c in df.columns
        if c in selected_names or c in passthrough_cols or c in exclude_cols
    ]
    reduced_df = df[kept_cols_ordered].copy()

    return selected_names, reduced_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else "cleaned.csv"
    input_path = DATA_DIR / filename

    print(f"Loading: {input_path}")
    raw = pd.read_csv(input_path)
    print(f"Input  : {raw.shape[0]:,} rows x {raw.shape[1]} columns")

    engineered = create_features(raw)

    new_cols = [c for c in engineered.columns if c not in raw.columns]
    print(f"Output : {engineered.shape[0]:,} rows x {engineered.shape[1]} columns")
    print(f"New features ({len(new_cols)}):")
    for col in new_cols:
        print(f"  {col:<40} dtype={engineered[col].dtype}  "
              f"min={engineered[col].min():.4f}  max={engineered[col].max():.4f}")

    out_path = DATA_DIR / "engineered.csv"
    engineered.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    # --- feature selection ---
    non_feature_cols = ["Unnamed: 0", "track_id", "artists", "album_name",
                        "track_name", "track_genre"]
    selected, reduced = select_features(
        engineered,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        exclude_cols=non_feature_cols,
    )
    print(f"Reduced dataframe: {reduced.shape[0]:,} rows x {reduced.shape[1]} columns")

    sel_path = DATA_DIR / "selected.csv"
    reduced.to_csv(sel_path, index=False)
    print(f"Saved to: {sel_path}")
