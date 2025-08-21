# 📦 database_cloud.py
from dotenv import load_dotenv
import os
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()  # keep if you also use a .env

DB_CONFIG = {
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres.pmpxrcaqntppejmdupby"),
    "password": os.getenv("PGPASSWORD", "booooooooook123!"),  # consider rotating & using env vars
    "host": os.getenv("PGHOST", "aws-0-eu-north-1.pooler.supabase.com"),
    "port": os.getenv("PGPORT", "6543"),
}

def get_connection():
    return psycopg2.connect(
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        sslmode="require",  # good practice with Supabase
    )

def ensure_schema():
    """
    Create the table if missing and backfill columns added later (like call_number).
    This avoids crashes when the table was created before a new column was introduced.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1) Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS books (
                    id SERIAL PRIMARY KEY,
                    isbn TEXT,
                    isbn10 TEXT,
                    isbn13 TEXT,
                    title TEXT,
                    authors TEXT,
                    publisher TEXT,
                    year TEXT,
                    edition TEXT,
                    series TEXT,
                    genre TEXT,
                    language TEXT,
                    additional_text TEXT,
                    -- table-level constraint; the table might already exist without call_number
                    UNIQUE (isbn, title, authors)
                );
            """)
            # 2) Make sure the new column exists even if table pre-existed
            cur.execute("""ALTER TABLE books ADD COLUMN IF NOT EXISTS call_number TEXT;""")
            # (Optional) helpful indexes for lookup speed
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_books_isbns ON books (isbn, isbn10, isbn13);""")
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_books_title_authors ON books (title, authors);""")
        conn.commit()

def create_table():
    # kept for backward compatibility; call ensure_schema() instead
    ensure_schema()

def insert_book(metadata):
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO books (
                    isbn, isbn10, isbn13, title, authors, publisher, year, edition,
                    series, genre, language, additional_text, call_number
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (isbn, title, authors) DO NOTHING;
            """, (
                metadata.get("isbn"),
                metadata.get("isbn10"),
                metadata.get("isbn13"),
                metadata.get("title"),
                ', '.join(metadata.get("authors", [])) if metadata.get("authors") else metadata.get("authors"),
                metadata.get("publisher"),
                metadata.get("year"),
                metadata.get("edition"),
                metadata.get("series"),
                metadata.get("genre"),
                metadata.get("language"),
                metadata.get("additional_text"),
                metadata.get("call_number"),
            ))
        conn.commit()

def search_book(isbn=None, title=None, authors=None):
    ensure_schema()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if isbn:
                cur.execute("""
                    SELECT * FROM books
                    WHERE isbn = %s OR isbn10 = %s OR isbn13 = %s
                    LIMIT 1
                """, (isbn, isbn, isbn))
            elif title and authors:
                # authors param may be list or string; normalize to stored format
                authors_str = ', '.join(authors) if isinstance(authors, (list, tuple)) else authors
                cur.execute("""
                    SELECT * FROM books
                    WHERE title = %s AND authors = %s
                    LIMIT 1
                """, (title, authors_str))
            else:
                return None
            return cur.fetchone()
