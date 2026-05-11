import os
from datetime import datetime
from typing import Optional

import psycopg2

# stores Python dict as JSONB
from dotenv import load_dotenv
from psycopg2.extras import Json

load_dotenv()

# Note: Later "image_url" could point to a s3 bucket storing the saved image, uploaded by user.


def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_feedback (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            app_rating INTEGER NOT NULL,
            app_comment TEXT
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
    chatbot_ans: str,
    user_rating: Optional[int],
    user_feedback: Optional[str],
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
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
        """,
        (
            datetime.utcnow(),
            image_url,
            model_name,
            device_type,
            Json(predictions),
            chatbot_ans,
            user_rating,
            user_feedback,
        ),
    )

    conn.commit()
    cur.close()
    conn.close()


def log_app_feedback(
    app_rating: int,
    app_comment: Optional[str] = None,
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO app_feedback (
            timestamp,
            app_rating,
            app_comment
        )
        VALUES (%s, %s, %s)
        """,
        (
            datetime.utcnow(),
            app_rating,
            app_comment,
        ),
    )

    conn.commit()
    cur.close()
    conn.close()
