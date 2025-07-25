# 📦 database_cloud.py
from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()  # load from .env


DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "booooooooook123!",
    "host": "db.xmdjjsftfilghvgkzqxr.supabase.co",  # <--- MUST be this
    "port": "5432"
}


def get_connection():
    return psycopg2.connect(
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"]
    )

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
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
            UNIQUE (isbn, title, authors)
        );
    ''')
    conn.commit()
    conn.close()

def insert_book(metadata):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO books (
            isbn, isbn10, isbn13, title, authors, publisher, year, edition,
            series, genre, language, additional_text
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (isbn, title, authors) DO NOTHING;
    ''', (
        metadata.get("isbn"),
        metadata.get("isbn10"),
        metadata.get("isbn13"),
        metadata.get("title"),
        ', '.join(metadata.get("authors", [])) if metadata.get("authors") else None,
        metadata.get("publisher"),
        metadata.get("year"),
        metadata.get("edition"),
        metadata.get("series"),
        metadata.get("genre"),
        metadata.get("language"),
        metadata.get("additional_text")
    ))
    conn.commit()
    conn.close()

def search_book(isbn=None, title=None, authors=None):
    conn = get_connection()
    cursor = conn.cursor()
    result = None

    if isbn:
        cursor.execute('''
            SELECT * FROM books WHERE isbn = %s OR isbn10 = %s OR isbn13 = %s
        ''', (isbn, isbn, isbn))
        result = cursor.fetchone()
    elif title and authors:
        cursor.execute('''
            SELECT * FROM books WHERE title = %s AND authors = %s
        ''', (title, ', '.join(authors)))
        result = cursor.fetchone()

    conn.close()
    return result
