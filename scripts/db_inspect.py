import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    database=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    connect_timeout=5,
)

app_df = pd.read_sql(
    "SELECT * FROM app_feedback ORDER BY timestamp DESC LIMIT 10;",
    conn
)
print("------ Table: app_feedback ------")
print(app_df)

clip_df = pd.read_sql(
    "SELECT * FROM clip_predict ORDER BY timestamp DESC LIMIT 10;",
    conn
)
print("------ Table: clip_predict ------")
print(clip_df)
