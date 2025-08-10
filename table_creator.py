def create_weather_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS manteca_data.daily_weather_2023 (
            date DATE PRIMARY KEY,
            max_temp_f REAL,
            avg_temp_f REAL,
            min_temp_f REAL,
            max_dew_point_f REAL,
            avg_dew_point_f REAL,
            min_dew_point_f REAL,
            max_humidity_pct REAL,
            avg_humidity_pct REAL,
            min_humidity_pct REAL,
            max_wind_mph REAL,
            avg_wind_mph REAL,
            min_wind_mph REAL,
            max_pressure_in REAL,
            avg_pressure_in REAL,
            min_pressure_in REAL,
            precip_in REAL
        );
    """)
    conn.commit()
    print("Table created in manteca_data schema!")