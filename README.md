# Spotify Track Genre Classifier

A machine-learning pipeline that predicts a Spotify track's genre from its audio features.

---

## Project Structure

```
├── data/
│   ├── dataset.csv          # Raw Spotify tracks data
│   └── cleaned.csv          # Output of the cleaning pipeline
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── src/
│   └── data/
│       ├── loader.py        # CSV loading and profiling utilities
│       ├── quality.py       # Data quality gate (5 checks)
│       └── cleaner.py       # Cleaning pipeline (nulls, dupes, dtypes)
├── tests/
├── requirements.txt
└── setup.py
```

---

## Exploratory Data Analysis

Full analysis: [`notebooks/eda.ipynb`](notebooks/eda.ipynb)

### Dataset Dimensions

| Property | Value |
|---|---|
| Rows | 114,000 |
| Columns | 21 |
| Target | `track_genre` (114 classes, 1,000 tracks each) |
| Missing values | 0 (after cleaning) |

**Feature types**

| Type | Columns |
|---|---|
| Continuous audio features | `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo` |
| Discrete / ordinal | `key`, `mode`, `time_signature`, `popularity` |
| Track metadata | `duration_ms`, `explicit` |
| Identifiers (excluded from modelling) | `track_id`, `track_name`, `album_name`, `artists` |

---

### Key Findings

- **Perfectly balanced classes.** Every one of the 114 genres contains exactly 1,000 tracks. No oversampling or class-weighting is needed; standard accuracy is a valid evaluation metric.

- **Energy and loudness are highly redundant (r = 0.76).** Acousticness anti-correlates with both (r = −0.73 vs energy, r = −0.59 vs loudness). Including all three in a linear model inflates multicollinearity; keeping one of {energy, loudness} and acousticness covers the same variance.

- **Speechiness, instrumentalness, and duration are heavily right-skewed** (skewness: 4.6, 1.7, 11.2 respectively). These will need a log or power transform before any distance-based or regularised linear model, or before feature scaling.

- **Danceability and valence move together (r = 0.48).** Genres such as dance/electronic cluster at high danceability + high valence, while classical/acoustic genres occupy the low end of both axes — making this pair a useful combined signal for genre separation.

- **Popularity carries little genre signal.** Its distribution is nearly symmetric (skewness ≈ 0.05) and its correlations with all audio features are below 0.15. It should be treated as a weak auxiliary feature rather than a primary predictor.

---

### Modeling Implications

- **Drop or merge the energy/loudness/acousticness triplet.** Because energy and loudness are 0.76 correlated and acousticness mirrors both, using all three together in regularised models (Ridge, Lasso, logistic regression) will compete for the same coefficient budget. Prefer: keep `energy` + `acousticness`; drop `loudness`, or apply PCA to the trio.

- **Log-transform before scaling.** Apply `log1p` to `speechiness`, `instrumentalness`, and `duration_ms` before fitting any model that uses distance metrics (KNN, SVM with RBF kernel) or that relies on normally distributed inputs. Tree-based models (Random Forest, XGBoost) are robust to the skew and do not require this transform.
