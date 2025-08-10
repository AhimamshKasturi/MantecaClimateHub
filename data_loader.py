import pandas as pd

def load_excel_data(filepath):
    df = pd.read_excel(filepath, engine="openpyxl")
    df.columns = [
        "date", "max_temp_f", "avg_temp_f", "min_temp_f",
        "max_dew_point_f", "avg_dew_point_f", "min_dew_point_f",
        "max_humidity_pct", "avg_humidity_pct", "min_humidity_pct",
        "max_wind_mph", "avg_wind_mph", "min_wind_mph",
        "max_pressure_in", "avg_pressure_in", "min_pressure_in",
        "precip_in"
    ]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df