import duckdb

# Connect to the DuckDB database
con = duckdb.connect("bluebikes_2015_2025.duckdb")

# Export selected columns for the 1-year window to Parquet
con.execute("""
    COPY (
        SELECT
            start_station_id,
            end_station_id,
            started_at,
            start_station_name,
            end_station_name,
            tripduration,
            start_lat,
            start_lng,
            end_lat,
            end_lng
        FROM trips_weather_interpolated
        WHERE weather_time >= '2024-08-01'
          AND weather_time <  '2025-08-01'
    )
    TO 'network_data_2024_08_to_2025_08.parquet'
    (FORMAT PARQUET);
""")

# Close the connection
con.close()
