import pandas as pd

df = pd.read_parquet("network_data_2024_08_to_2025_08.parquet")

print(df.head(1))
print(df.shape)
