def inspect_schema(conn):
    cur = conn.cursor()

    cur.execute("SELECT current_database();")
    print("Connected to:", cur.fetchone())

    cur.execute("SELECT schema_name FROM information_schema.schemata;")
    print("Schemas:", cur.fetchall())

    cur.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE';
    """)
    print("All tables:", cur.fetchall())

    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'daily_weather_2023';
    """)
    print("Columns:", cur.fetchall())

    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public';
    """)
    print("Public tables:", cur.fetchall())