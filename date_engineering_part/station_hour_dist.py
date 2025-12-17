import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

pd.set_option("display.precision", 3)

# --- 0. Load parquet ---
df = pd.read_parquet("network_data_2024_08_to_2025_08.parquet")

# Ensure datetime
df["started_at"] = pd.to_datetime(df["started_at"])

# Extract hour bucket
df["hour_bucket"] = df["started_at"].dt.floor("H")

# --- 1. Compute trips per station per hour ---
trips_per_hour = (
    df.groupby(["start_station_id", "hour_bucket"])
      .size()
      .reset_index(name="trips_per_hour")
)

print(trips_per_hour["trips_per_hour"].describe())

# --- 2. Plot distribution ---
# ~ plt.figure(figsize=(8, 5))
# ~ plt.hist(trips_per_hour["trips_per_hour"], bins=50)
# ~ plt.xlabel("Trips per station per hour")
# ~ plt.ylabel("Frequency")
# ~ plt.title("Distribution of Hourly Trips per Station")
# ~ plt.tight_layout()
# ~ plt.show()


import matplotlib.pyplot as plt

# --- Original histogram (linear scale) ---
plt.figure(figsize=(8, 5))
plt.hist(trips_per_hour["trips_per_hour"], bins=50)
plt.xlabel("Trips per station per hour")
plt.ylabel("Frequency")
plt.title("Distribution of Hourly Trips per Station (Linear Scale)")
plt.tight_layout()
plt.show()

# --- Log-scaled histogram ---
plt.figure(figsize=(8, 5))
plt.hist(trips_per_hour["trips_per_hour"], bins=50)
plt.yscale("log")   # <-- key addition
plt.xlabel("Trips per station per hour")
plt.ylabel("Frequency (log scale)")
plt.title("Distribution of Hourly Trips per Station (Log Scale)")
plt.tight_layout()
plt.show()
