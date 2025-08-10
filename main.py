from db_connection import get_connection
from schema_inspector import inspect_schema
from table_creator import create_weather_table
from data_loader import load_excel_data
from data_uploader import upload_to_postgres

def main():
    conn = get_connection()
    if not conn:
        return

    inspect_schema(conn)
    create_weather_table(conn)

    df = load_excel_data("/Users/ahimamshkasturi/job/Job/restart/Projects/CityTemperatures/Manteca2023.xlsx")
    upload_to_postgres(df)

    conn.close()

if __name__ == "__main__":
    main()