import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.max_columns", None)

# -----------------------
# Load
# -----------------------
df = pd.read_parquet("ml_dataset.parquet")
print(df.head(1))
print(df.shape)

TIME_COL = "time_hour"
TARGET   = "trips"

# -----------------------
# Ensure proper datetime + sort
# -----------------------
df = df.copy()
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

# -----------------------
# FIX A: aggregate to ONE ROW PER HOUR (citywide)
# -----------------------
# Keep all non-trip columns as hourly means (weather/calendar), and sum trips.
non_trip_cols = [c for c in df.columns if c not in [TIME_COL, TARGET]]

hourly = (
    df.groupby(TIME_COL, as_index=False)
      .agg({TARGET: "sum", **{c: "mean" for c in non_trip_cols}})
      .sort_values(TIME_COL)
      .reset_index(drop=True)
)

# -----------------------
# Add lag features (now this is a true time-series lag)
# -----------------------
hourly["trips_lag_1"]  = hourly[TARGET].shift(1)
hourly["trips_lag_24"] = hourly[TARGET].shift(24)

hourly = hourly.dropna(subset=["trips_lag_1", "trips_lag_24"]).reset_index(drop=True)

# -----------------------
# Proper time-based split
# -----------------------
split_idx = int(len(hourly) * 0.8)

train = hourly.iloc[:split_idx].copy()
test  = hourly.iloc[split_idx:].copy()

X_train = train.drop(columns=[TARGET, TIME_COL])
y_train = train[TARGET]

X_test  = test.drop(columns=[TARGET, TIME_COL])
y_test  = test[TARGET]

# bool -> int (helps sklearn)
for c in X_train.columns:
    if X_train[c].dtype == "bool":
        X_train[c] = X_train[c].astype(int)
        X_test[c]  = X_test[c].astype(int)

# -----------------------
# Train linear regression
# -----------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------
# Evaluate: MAE, RMSE, Adj R^2
# -----------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # version-safe

r2 = r2_score(y_test, y_pred)
n = len(y_test)
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\nRESULTS (Fix A: hourly aggregate + lags + time split)")
print("Train time range:", train[TIME_COL].iloc[0], "->", train[TIME_COL].iloc[-1])
print("Test  time range:", test[TIME_COL].iloc[0],  "->", test[TIME_COL].iloc[-1])

print(f"MAE     : {mae:.2f}")
print(f"RMSE    : {rmse:.2f}")
print(f"R²      : {r2:.6f}")
print(f"Adj. R² : {adj_r2:.6f}")

# Optional: inspect which features matter most
coef_df = (
    pd.DataFrame({"feature": X_train.columns, "coefficient": model.coef_})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns="abs_coef")
)
print("\nTop coefficients:\n", coef_df.head(15))
