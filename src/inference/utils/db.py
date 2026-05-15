import os
import json
import psycopg2
from datetime import datetime
from typing import Optional
from psycopg2.extras import Json # stores Python dict as JSONB
from dotenv import load_dotenv
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
            run_id TEXT NOT NULL,
            device_type TEXT NOT NULL,
            predictions JSONB NOT NULL,
            top_k INTEGER NOT NULL,
            llm_status TEXT DEFAULT NULL,
            llm_backend TEXT DEFAULT NULL,
            llm_answer TEXT DEFAULT NULL,
            llm_timestamp TIMESTAMP DEFAULT NULL
        )
    """)

    # Lightweight compatibility migration for existing volumes created by older schema versions.
    cur.execute("ALTER TABLE clip_predict ADD COLUMN IF NOT EXISTS run_id TEXT;")
    cur.execute("ALTER TABLE clip_predict ADD COLUMN IF NOT EXISTS top_k INTEGER;")
    cur.execute("ALTER TABLE clip_predict ADD COLUMN IF NOT EXISTS llm_status TEXT;")
    cur.execute("ALTER TABLE clip_predict ADD COLUMN IF NOT EXISTS llm_backend TEXT;")
    cur.execute("ALTER TABLE clip_predict ADD COLUMN IF NOT EXISTS llm_answer TEXT;")
    cur.execute("ALTER TABLE clip_predict ADD COLUMN IF NOT EXISTS llm_timestamp TIMESTAMP;")
    cur.execute("UPDATE clip_predict SET run_id = COALESCE(run_id, 'unknown') WHERE run_id IS NULL;")
    cur.execute("UPDATE clip_predict SET top_k = COALESCE(top_k, 5) WHERE top_k IS NULL;")
    cur.execute("ALTER TABLE clip_predict ALTER COLUMN run_id SET NOT NULL;")
    cur.execute("ALTER TABLE clip_predict ALTER COLUMN top_k SET NOT NULL;")

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
    run_id: str,
    device_type: str,
    predictions: dict,
    top_k: int,
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO clip_predict (
            timestamp,
            image_url,
            run_id,
            device_type,
            predictions,
            top_k
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (
            datetime.utcnow(),
            image_url,
            run_id,
            device_type,
            Json(predictions),
            top_k,
        ),
    )
    row = cur.fetchone()

    conn.commit()
    cur.close()
    conn.close()
    return int(row[0]) if row else None


def update_prediction(
    prediction_id: int,
    llm_backend: str,
    llm_answer: dict,
    llm_status: str = "completed",
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE clip_predict
        SET
            llm_backend = %s,
            llm_answer = %s,
            llm_status = %s,
            llm_timestamp = %s
        WHERE id = %s;
        """,
        (
            llm_backend,
            json.dumps(llm_answer, ensure_ascii=False),
            llm_status,
            datetime.utcnow(),
            prediction_id,
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
