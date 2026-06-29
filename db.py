#!/usr/bin/env python3
"""
SQLite layer: sqlite-vec (vector search) + FTS5 (keyword/BM25 search).
All state lives in a single file (see config.DB_PATH).
"""
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import sqlite_vec

from config import DB_PATH, EMBEDDING_DIM


def get_connection() -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Load the sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read performance
    return conn


def init_db():
    """
    Create tables if they don't exist.
    Schema:
      chunks       — source of truth (file_path, chunk text, embedding blob, mtime)
      chunks_fts   — FTS5 virtual table mirroring chunks.text for keyword search
      vec_chunks   — sqlite-vec virtual table for vector similarity search
    """
    conn = get_connection()

    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path   TEXT    NOT NULL,
            chunk_index INTEGER NOT NULL,
            text        TEXT    NOT NULL,
            embedding   BLOB,
            file_mtime  REAL,
            UNIQUE(file_path, chunk_index)
        );

        -- FTS5 for keyword / BM25 search
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='porter ascii'
        );

        -- sqlite-vec for cosine / L2 vector search
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding FLOAT[{EMBEDDING_DIM}]
        );

        -- Keep FTS5 in sync with chunks via triggers
        CREATE TRIGGER IF NOT EXISTS chunks_ai
            AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad
            AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END;

        CREATE TRIGGER IF NOT EXISTS chunks_au
            AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
    """)

    conn.commit()
    conn.close()


def get_stats() -> dict:
    """Return counts of indexed chunks and files."""
    conn = get_connection()
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    files  = conn.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()[0]
    conn.close()
    return {"chunks": chunks, "files": files}


if __name__ == "__main__":
    init_db()
    stats = get_stats()
    print(f"DB initialised — {stats['chunks']} chunks across {stats['files']} files")
    print(f"Path: {DB_PATH}")
