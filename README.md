# boston_bikeshare_analysis
boston blue bike analysis. Focusing on august 2024-august 2025. Network Science and Urban Computing.

A full technical report was written based on these findings. With temporal features (including lag demand) and weather features, our model yields exceptional .96 adj r-squared. Initially, we did not use lag demand so it was in the more commonly cited ~.85 range. Lag demand is not only possible, but realistic for modern bikeshare systems since near real-time data is often available. 

The parquet file in the data engineering directory is our final dataset after cleanup and feature engineering.

