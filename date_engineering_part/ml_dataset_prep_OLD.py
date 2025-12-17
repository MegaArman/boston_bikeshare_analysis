import duckdb
import pandas as pd
# ~ import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

con = duckdb.connect("bluebikes_2015_2025.duckdb")

df_bike = con.execute("""
    SELECT start_station_id, start_station_name, start_lat, start_lng,
    end_station_id, end_station_name, end_lat, end_lng, weather_time,
    wind_speed, total_precipitation, precipitation_type, "2m_temperature", ride_id
    FROM trips_weather_interpolated
    WHERE weather_time >= '2024-08-01'
      AND weather_time <  '2025-08-01';
""").df()

con.close()

print(df_bike.head(1))


# ~ #============================================================
# ~ # Select start station columns and rename them to common names
start_stations = df_bike[[
    "start_station_id", "start_station_name", "start_lat", "start_lng"
]].rename(columns={
    "start_station_id": "station_id",       # rename to a general station ID
    "start_station_name": "station_name",   # rename to a general station name
    "start_lat": "lat",                     # rename latitude
    "start_lng": "lng"                      # rename longitude
})

# Select end station columns and rename them to the same common names
end_stations = df_bike[[
    "end_station_id", "end_station_name", "end_lat", "end_lng"
]].rename(columns={
    "end_station_id": "station_id",         # rename to match start table
    "end_station_name": "station_name",     # rename to match start table
    "end_lat": "lat",                       # rename latitude
    "end_lng": "lng"                        # rename longitude
})

# Combine start and end station tables together
all_stations = pd.concat([start_stations, end_stations])

# Remove duplicate stations based on station_id (keep only unique ones)
unique_stations = all_stations.drop_duplicates(subset=["station_id"])

# Print the total number of unique stations
print("Total number of unique stations:", len(unique_stations))

# Show the first few rows of unique stations
unique_stations.head()
# ~ #====================================================================

# Convert the weather_time column to a real datetime type
df_bike["time_hour"] = pd.to_datetime(df_bike["weather_time"]).dt.floor("h")

# Extract the hour of the day (0–23)
df_bike["hour"] = df_bike["time_hour"].dt.hour

# Extract the day of the month (1–31)
df_bike["day"] = df_bike["time_hour"].dt.day

# Extract the weekday number (0 = Monday, 6 = Sunday)
df_bike["weekday"] = df_bike["time_hour"].dt.dayofweek

# Create a weekend flag (1 = weekend, 0 = weekday)
df_bike["is_weekend"] = df_bike["weekday"].isin([5, 6]).astype(int)

# Create a windy day flag (1 = wind speed > 6 m/s)
df_bike["windy_day"] = (df_bike["wind_speed"] > 6).astype(int)

# Create a rainy day flag (1 = total_precipitation > 0.5 mm)
df_bike["rainy_day"] = (df_bike["total_precipitation"] > 0.5).astype(int)

# Create a snowy day flag (1 = precipitation_type == 3)
df_bike["snowy_day"] = df_bike["precipitation_type"].isin([3]).astype(int)

# Show example rows
# ~ df_bike[[
    # ~ "time_hour", "hour", "day", "weekday", "is_weekend",
    # ~ "wind_speed", "windy_day",
    # ~ "total_precipitation", "rainy_day",
    # ~ "precipitation_type", "snowy_day"
# ~ ]].head(1)
# ~ #====================================================================
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# Group the bike data by each hour (time_hour)
df_hourly = df_bike.groupby("time_hour").agg({
    "ride_id": "count",          # count how many trips happened in this hour
    "hour": "first",             # keep the hour value
    "day": "first",              # keep the day of the month
    "weekday": "first",          # keep the weekday number
    "is_weekend": "first",       # keep weekend flag
    "wind_speed": "mean",        # average wind speed in this hour
    "windy_day": "max",          # if ANY minute was windy → mark hour as windy
    "total_precipitation": "mean",  # average precipitation in this hour
    "rainy_day": "max",          # if it rained at any time in this hour → mark as rainy
    "precipitation_type": "max", # highest precipitation type in the hour
    "snowy_day": "max",           # if it snowed at any point → mark as snowy
    "2m_temperature": "mean"
   

}).reset_index()

# Rename the trip count column
df_hourly = df_hourly.rename(columns={"ride_id": "trips"})

# ---- Add Holiday Feature (after groupby) ----
cal = calendar()
holidays = cal.holidays(
    start=df_hourly["time_hour"].min(),
    end=df_hourly["time_hour"].max()
)
# 1 = holiday, 0 = normal day
df_hourly["is_holiday"] = df_hourly["time_hour"].dt.normalize().isin(holidays).astype(int)


# Make sure month exists
df_hourly["month"] = df_hourly["time_hour"].dt.month
df_hourly["season"] = pd.cut(
    df_hourly["month"],
    bins=[0, 2, 5, 8, 11, 12],
    labels=["Winter", "Spring", "Summer", "Fall", "Winter"],
    right=True,
    ordered=False   
)

df_hourly = pd.get_dummies(df_hourly, columns=["season"])
df_hourly = df_hourly.dropna()

# Show the first rows
# ~ head = df_hourly.head(1)
# ~ print(head)

#===================================================================
# Compute threshold based on bottom 25% of all temperatures
cold_threshold = df_hourly["2m_temperature"].quantile(0.25)

print("Cold threshold =", cold_threshold)

df_hourly["cold_hour"] = (df_hourly["2m_temperature"] <= cold_threshold).astype(int)

# ----- 1) Use your real cold threshold -----
cold_threshold = 3.2256

# Mark cold hours based on actual temperature threshold
df_hourly["cold_hour"] = (df_hourly["2m_temperature"] <= cold_threshold).astype(int)

# ----- 2) Compute average trips for cold vs normal hours -----
mean_trips_by_temp = df_hourly.groupby("cold_hour")["trips"].mean()

print("Average trips (normal vs cold temperature):")
print(mean_trips_by_temp)
#======================================================================
def get_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

# We already have 'month' column
df_hourly["season"] = df_hourly["month"].apply(get_season)

#======================================================================
# --- Create df with station_id + time_hour ---
station_df = df_bike[["time_hour", "start_station_id","end_station_id","start_lat","start_lng"]]

# --- Merge df_hourly with station info ---
station_hourly = df_hourly.merge(
    station_df,
    on="time_hour",
    how="left"
)

print("STATION HOURLY")
print(station_hourly.head(1))

station_hourly["time_hour"] = pd.to_datetime(station_hourly["time_hour"])
station_hourly = station_hourly.sort_values("time_hour")

# Time-based split (80% train, 20% test)
train_ratio = 0.8
split_idx = int(len(station_hourly) * train_ratio)

train = station_hourly.iloc[:split_idx]
test  = station_hourly.iloc[split_idx:]

# Feature columns
feature_cols = [
    "hour",
    "day",
    "weekday",
    "is_weekend",
    "wind_speed",
    "windy_day",
    "total_precipitation",
    "rainy_day",
    "precipitation_type",
    "snowy_day",
    "2m_temperature",
    "is_holiday",
    "month",
    "season_Fall",
    "season_Spring",
    "season_Summer",
    "season_Winter",
    "time_hour"
]

#==============================================
# WRITE TO FILE
con = duckdb.connect()

df = station_hourly[feature_cols]
df["trips"] = station_hourly["trips"]
con.register("df_view", df)

con.execute("""
    COPY df_view TO 'ml_dataset.parquet' (FORMAT PARQUET);
""")

con.close()


#==============================================

# Prepare X and y
# ~ X_train = train[feature_cols]
# ~ y_train = train["trips"]

# ~ X_test  = test[feature_cols]
# ~ y_test  = test["trips"]

# ~ train_head = X_train.head()

# ~ print(X_train.head(1))
