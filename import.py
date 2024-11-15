import sqlite3
import pandas as pd
from IPython.display import display


# Connect to the database
conn = sqlite3.connect('C:/Users/popova/Documents/Bartab/2024-11_MIRL_batch-5176_LIV.db')

# Query to list all tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)

for table_name in tables:
    table_name = table_name[0]  # Extract table name from tuple
    print(f"\nDeleting first 2 rows from table: {table_name}")
    cursor.execute(f"DELETE FROM {table_name} WHERE rowid IN (SELECT rowid FROM {table_name} LIMIT 2)")
    conn.commit()  # Commit the changes to the database

# Loop through each table and display its content
for table_name in tables:
    table_name = table_name[0]  # Extract table name from tuple
    print(f"\nData from table: {table_name}")
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    display(df)

    