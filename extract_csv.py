# ~ import duckdb

# ~ con = duckdb.connect("bluebikes_2015_2025.duckdb")

# ~ con.execute("""
    # ~ COPY (
        # ~ SELECT *
        # ~ FROM trips_weather_interpolated
        # ~ WHERE weather_time >= '2024-08-01'
          # ~ AND weather_time <  '2025-08-01'
    # ~ )
    # ~ TO 'bluebikes_2024_08_to_2025_08.csv'
    # ~ (HEADER, DELIMITER ',');
# ~ """)

# ~ con.close()

# ~ import duckdb

# ~ con = duckdb.connect("bluebikes_2015_2025.duckdb")

# ~ con.execute("""
    # ~ COPY (
        # ~ SELECT
            # ~ start_station_id,
            # ~ end_station_id,
            # ~ started_at,
            # ~ start_station_name,
            # ~ end_station_name,
            # ~ tripduration
        # ~ FROM trips_weather_interpolated
        # ~ WHERE weather_time >= '2024-08-01'
          # ~ AND weather_time <  '2025-08-01'
    # ~ )
    # ~ TO 'network_data_2024_08_to_2025_08.csv'
    # ~ (HEADER, DELIMITER ',');
# ~ """)

# ~ con.close()

con.execute("""
    COPY (
        SELECT
            start_station_id,
            end_station_id,
            started_at,
            start_station_name,
            end_station_name,
            tripduration
        FROM trips_weather_interpolated
        WHERE weather_time >= '2024-08-01'
          AND weather_time <  '2025-08-01'
    )
    TO 'network_data_2024_08_to_2025_08.parquet'
    (FORMAT PARQUET);
""")
