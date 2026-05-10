import os
import psycopg2
from psycopg2.extras import Json
# stores Python dict as JSONB
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# Note: Later "image_url" could point to a s3 bucket storing the saved image, uploaded by user.

def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        connect_timeout=5,
    )

def check_db() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
    finally:
        conn.close()

def init_db():
    print("[info] - Start init db")
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS clip_predict (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            image_url TEXT,
            model_name TEXT NOT NULL,
            device_type TEXT NOT NULL,
            predictions JSONB NOT NULL,
            chatbot_ans TEXT,
            user_rating INTEGER,
            user_feedback TEXT
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("[info] - End init db")

def log_prediction(
    image_url: Optional[str],
    model_name: str,
    device_type: str,
    predictions: dict,
    chatbot_ans: str = None,
    user_rating: int = None,
    user_feedback: str = None
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO clip_predict (
            timestamp,
            image_url,
            model_name,
            device_type,
            predictions,
            chatbot_ans,
            user_rating,
            user_feedback
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        datetime.utcnow(),
        image_url,
        model_name,
        device_type,
        Json(predictions),
        chatbot_ans,
        user_rating,
        user_feedback
    ))

    conn.commit()
    cur.close()
    conn.close()
