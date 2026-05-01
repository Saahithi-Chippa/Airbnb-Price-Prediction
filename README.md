# Airbnb Price Prediction

Predicting Bangkok Airbnb listing prices using engineered features from text descriptions, amenity lists, and neighbourhood data — built to help hosts price their listings competitively.

## Exploratory Data Analysis

**Dataset:** 28,806 listings × 18 columns (raw) → 23,273 × 16 after cleaning.

Numeric features: `latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`, `number_of_reviews_ltm`.

Categorical features: `name`, `host_name`, `neighbourhood`, `room_type`, `last_review`.

**Key findings:**

- `neighbourhood_group` and `license` are 100% null across all listings and were dropped — they carry no signal for this city.
- ~35% of listings have no reviews; `last_review` and `reviews_per_month` are jointly null for these. `reviews_per_month` was imputed to 0; a `has_reviews` binary flag will be engineered.
- `room_type` drives a clear price tier — entire homes command a strong premium over private and shared rooms.
- Latitude and longitude show spatially distinct price bands, making a `distance_to_city_centre` or neighbourhood-cluster feature worthwhile.

**Modelling implications:**

- `price` has raw skewness of 53.24 (mean 2,529 THB vs median 1,379 THB) — **log1p transform is mandatory** before training and evaluation.
- 1,145 listings (4.9%) are extreme price outliers beyond 3×IQR — cap or Winsorise before fitting.
