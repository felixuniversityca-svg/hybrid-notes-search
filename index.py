#!/usr/bin/env python3
"""
Incremental indexer: chunks every Markdown file under NOTES_DIR, embeds each
chunk, and stores it in SQLite. Only re-indexes files whose mtime has changed.

Usage:
    python index.py              # incremental (skip unchanged files)
    python index.py --full       # wipe the index and re-index everything
    python index.py --stats      # show index stats and exit

Point it at any folder of Markdown:
    NOTES_DIR=~/my-notes python index.py
"""
import sys
import time
import numpy as np
from pathlib import Path

from config import NOTES_DIR
from db import get_connection, init_db, get_stats
from embeddings import embed, ensure_model_available

# Folders to skip entirely
SKIP_DIRS = {".git", ".obsidian", ".trash"}

WORDS_PER_CHUNK = 300    # ~400 tokens
OVERLAP_WORDS   = 40     # overlap between adjacent chunks


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks, stripping YAML frontmatter."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:].strip()

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + WORDS_PER_CHUNK, len(words))
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) > 50:  # skip tiny fragments
            chunks.append(chunk)
        start += WORDS_PER_CHUNK - OVERLAP_WORDS

    return chunks


def get_note_files() -> list[Path]:
    """Walk NOTES_DIR and return all .md files that should be indexed."""
    files = []
    for path in NOTES_DIR.rglob("*.md"):
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        files.append(path)
    return sorted(files)


def index_file(conn, path: Path) -> int:
    """Index a single file. Returns number of chunks written (0 = skipped/unchanged)."""
    mtime = path.stat().st_mtime

    existing = conn.execute(
        "SELECT file_mtime FROM chunks WHERE file_path = ? LIMIT 1",
        (str(path),)
    ).fetchone()

    if existing and existing["file_mtime"] == mtime:
        return 0  # unchanged

    # Delete old vectors first: capture the ids while the chunks rows still
    # exist, then drop the chunks (a trigger keeps FTS5 in sync). Deleting
    # chunks first would leave the vec_chunks rows orphaned forever.
    old_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM chunks WHERE file_path = ?", (str(path),))]
    if old_ids:
        conn.executemany("DELETE FROM vec_chunks WHERE rowid = ?",
                         [(i,) for i in old_ids])
    conn.execute("DELETE FROM chunks WHERE file_path = ?", (str(path),))

    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)

    if not chunks:
        conn.commit()
        return 0

    embeddings = embed(chunks)

    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        emb_bytes = np.array(emb, dtype=np.float32).tobytes()
        rows.append((str(path), i, chunk, emb_bytes, mtime))

    conn.executemany(
        "INSERT OR REPLACE INTO chunks (file_path, chunk_index, text, embedding, file_mtime) "
        "VALUES (?, ?, ?, ?, ?)",
        rows
    )

    # Sync embeddings to vec_chunks after chunks are committed
    conn.commit()
    for row in conn.execute(
        "SELECT id, embedding FROM chunks WHERE file_path = ?", (str(path),)
    ):
        conn.execute(
            "INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (row["id"], row["embedding"])
        )

    conn.commit()
    return len(chunks)


def main():
    if "--stats" in sys.argv:
        stats = get_stats()
        print(f"Index stats: {stats['chunks']} chunks across {stats['files']} files")
        return

    if not NOTES_DIR.exists():
        print(f"Notes folder not found: {NOTES_DIR}\n"
              f"Set NOTES_DIR to a folder of Markdown files, or create ./notes.")
        sys.exit(1)

    full_reindex = "--full" in sys.argv

    # Fail fast with a clear message if the model isn't cached and offline.
    try:
        ensure_model_available()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)

    init_db()
    conn = get_connection()

    if full_reindex:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM vec_chunks")
        conn.commit()
        print("Full re-index: cleared existing index.")

    files = get_note_files()
    print(f"Scanning {len(files)} Markdown files under {NOTES_DIR}...")

    start_time = time.time()
    indexed, skipped, total_chunks = 0, 0, 0

    for i, path in enumerate(files):
        try:
            n = index_file(conn, path)
            if n > 0:
                indexed += 1
                total_chunks += n
            else:
                skipped += 1
        except Exception as e:
            # Skip an unreadable/problematic file but report it loudly.
            print(f"  Error indexing {path.name}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(files)} files processed...")

    elapsed = time.time() - start_time
    conn.close()

    print(f"\nDone in {elapsed:.1f}s")
    print(f"   Indexed: {indexed} files ({total_chunks} chunks)")
    print(f"   Skipped: {skipped} files (unchanged)")
    stats = get_stats()
    print(f"   Total in DB: {stats['chunks']} chunks across {stats['files']} files")


if __name__ == "__main__":
    main()
