#!/usr/bin/env python3
"""
Hybrid search over a folder of Markdown notes:
    0.7 x vector (cosine, sqlite-vec) + 0.3 x keyword (BM25, FTS5)
merged with Reciprocal Rank Fusion (RRF).

Usage:
    python search.py "reciprocal rank fusion"
    python search.py "vector database" --top-k 8
    python search.py "embeddings" --path-prefix subfolder
    python search.py "hybrid search" --json     # machine-readable output
"""
import os
import re
import json
import sqlite3
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

from config import NOTES_DIR
from db import get_connection
from embeddings import embed_one

VECTOR_WEIGHT  = 0.7
KEYWORD_WEIGHT = 0.3
RRF_K = 60          # standard RRF constant: higher = gentler rank penalty
FANOUT = 3          # over-fetch factor per index before fusion trims to top_k


def _dir_prefix(path_prefix: str) -> str:
    """Absolute path prefix with a trailing separator, so 'sub' does not match 'subway.md'."""
    return str(NOTES_DIR / path_prefix) + os.sep


def _fts_match_query(query: str) -> Optional[str]:
    """
    Turn a raw query into a safe FTS5 MATCH expression.

    A raw query is unsafe: in FTS5 '-' is the NOT operator (so "Black-Scholes"
    means "Black NOT Scholes") and bare tokens are implicitly ANDed, so one
    missing word drops the row. We split into word tokens, quote each so it is
    treated literally, and join with OR. BM25 then ranks by how many (and how
    rare) the matched tokens are. Returns None if the query has no word tokens.
    """
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def vector_search(conn, query_embedding: np.ndarray, top_k: int,
                  path_prefix: Optional[str] = None) -> list:
    """Find top chunks by cosine similarity using sqlite-vec."""
    query_bytes = query_embedding.astype(np.float32).tobytes()

    # sqlite-vec applies K as a hard KNN limit, so over-fetch and filter after.
    results = conn.execute("""
        SELECT c.id, c.file_path, c.text, v.distance
        FROM vec_chunks v
        JOIN chunks c ON v.rowid = c.id
        WHERE v.embedding MATCH ?
          AND K = ?
        ORDER BY v.distance ASC
    """, (query_bytes, top_k * FANOUT)).fetchall()

    if path_prefix:
        boundary = _dir_prefix(path_prefix)
        results = [r for r in results if r["file_path"].startswith(boundary)]

    return results[:top_k * 2]


def keyword_search(conn, query: str, top_k: int,
                   path_prefix: Optional[str] = None) -> list:
    """Find top chunks by BM25 keyword matching using FTS5 (porter stemming)."""
    match = _fts_match_query(query)
    if not match:
        return []

    try:
        if path_prefix:
            # Escape LIKE wildcards in the prefix so a literal % or _ is not a pattern.
            like = (_dir_prefix(path_prefix)
                    .replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
            results = conn.execute("""
                SELECT c.id, c.file_path, c.text, f.rank
                FROM chunks_fts f
                JOIN chunks c ON f.rowid = c.id
                WHERE chunks_fts MATCH ?
                  AND c.file_path LIKE ? ESCAPE '\\'
                ORDER BY f.rank
                LIMIT ?
            """, (match, like, top_k * 2)).fetchall()
        else:
            results = conn.execute("""
                SELECT c.id, c.file_path, c.text, f.rank
                FROM chunks_fts f
                JOIN chunks c ON f.rowid = c.id
                WHERE chunks_fts MATCH ?
                ORDER BY f.rank
                LIMIT ?
            """, (match, top_k * 2)).fetchall()
    except sqlite3.OperationalError:
        # Belt and suspenders: any residual FTS5 parse error degrades to vector-only.
        results = []

    return results


def rrf_merge(vector_results: list, keyword_results: list,
              top_k: int) -> list[dict]:
    """
    Reciprocal Rank Fusion: items appearing high in both ranked lists score best.
    Formula: score = sum(weight / (K + rank)). Scores are small and relative,
    they rank results against each other and are not similarity values.
    """
    scores: dict[int, dict] = {}

    for rank, row in enumerate(vector_results):
        rid = row["id"]
        if rid not in scores:
            scores[rid] = {"file_path": row["file_path"],
                           "text": row["text"], "score": 0.0}
        scores[rid]["score"] += VECTOR_WEIGHT / (RRF_K + rank + 1)

    for rank, row in enumerate(keyword_results):
        rid = row["id"]
        if rid not in scores:
            scores[rid] = {"file_path": row["file_path"],
                           "text": row["text"], "score": 0.0}
        scores[rid]["score"] += KEYWORD_WEIGHT / (RRF_K + rank + 1)

    merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


def search(query: str, top_k: int = 5, path_prefix: Optional[str] = None) -> list[dict]:
    """Main search entrypoint: returns a list of result dicts."""
    conn = get_connection()
    try:
        query_embedding = embed_one(query)
        vec_results = vector_search(conn, query_embedding, top_k, path_prefix)
        kw_results  = keyword_search(conn, query, top_k, path_prefix)
        merged = rrf_merge(vec_results, kw_results, top_k)
    finally:
        conn.close()
    return merged


def format_path(file_path: str) -> str:
    """Show the path relative to NOTES_DIR for readability."""
    try:
        return str(Path(file_path).relative_to(NOTES_DIR))
    except ValueError:
        return file_path


def main():
    parser = argparse.ArgumentParser(description="Hybrid (vector + BM25) search over Markdown notes")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--path-prefix", default=None,
                        help="Restrict search to files under this subfolder of NOTES_DIR")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON (for programmatic use)")
    args = parser.parse_args()

    query = " ".join(args.query)
    results = search(query, top_k=args.top_k, path_prefix=args.path_prefix)

    if args.json:
        print(json.dumps([
            {"file": format_path(r["file_path"]),
             "score": round(r["score"], 6),
             "text": r["text"][:500]}
            for r in results
        ], indent=2))
        return

    if not results:
        print(f"No results found for: '{query}'")
        return

    prefix_note = f" (in {args.path_prefix})" if args.path_prefix else ""
    print(f"\nTop {len(results)} results for '{query}'{prefix_note}\n")
    print("-" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.4f}")
        print(f"    {format_path(r['file_path'])}")
        snippet = r["text"][:280].replace("\n", " ")
        print(f"    {snippet}...")

    print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
