import pandas as pd
from sqlalchemy import create_engine
from db_config import get_sqlalchemy_url

def upload_to_postgres(df, table_name="daily_weather_2023", schema="manteca_data"):
    engine = create_engine(get_sqlalchemy_url())

    # Step 1: Fetch existing dates from the database
    try:
        existing_dates = pd.read_sql(
            f"SELECT date FROM {schema}.{table_name}",
            con=engine
        )
    except Exception as e:
        print(f"Error fetching existing dates: {e}")
        return

    # Step 2: Filter out rows that already exist
    df_filtered = df[~df["date"].isin(existing_dates["date"])]

    # Step 3: Check if there's anything new to upload
    if df_filtered.empty:
        print("No new data to upload. All dates already exist in the database.")
        return

    # Step 4: Upload only new rows
    try:
        df_filtered.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists="append",
            index=False
        )
        print(f"Uploaded {len(df_filtered)} new rows to {schema}.{table_name}")
    except Exception as e:
        print(f"Upload failed: {e}")
