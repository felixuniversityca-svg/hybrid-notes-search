#!/usr/bin/env python3
"""
Retrieval eval: does hybrid actually beat vector-only and keyword-only?

Indexes a small labeled corpus (eval/corpus/), runs a fixed set of queries
through each retrieval method, and reports Hit@1, Hit@3, and MRR. The queries
are chosen to be honest about the tradeoff: some are paraphrases with no shared
words (vector should win), some are exact technical terms (keyword should win).
Hybrid should match or beat the better of the two on the aggregate.

Run:
    python eval.py
"""
import os
from pathlib import Path

HERE = Path(__file__).parent
# Point config at the eval corpus and a throwaway DB before importing the modules.
os.environ["NOTES_DIR"] = str(HERE / "eval" / "corpus")
os.environ["NOTES_DB"] = str(HERE / "eval" / "eval.db")

from db import get_connection, init_db          # noqa: E402
from embeddings import embed_one, ensure_model_available  # noqa: E402
from index import index_file                    # noqa: E402
from search import vector_search, keyword_search, rrf_merge  # noqa: E402

# (query, the single relevant file in the corpus)
QUERIES = [
    ("how does money grow if I never touch it",   "compound-interest.md"),  # paraphrase
    ("Black-Scholes option pricing",              "black-scholes.md"),      # exact term
    ("what is the powerhouse of the cell",        "mitochondria.md"),       # paraphrase (word absent)
    ("SYN ACK three-way handshake",               "tcp-handshake.md"),      # exact terms
    ("brewing coffee with a plunger pot",         "french-press.md"),       # mixed
    ("merging several ranked lists into one",     "reciprocal-rank-fusion.md"),  # mixed
    # Adversarial: exact identifier the embeddings blur against a near-twin doc.
    ("E0433",                                     "rust-e0433.md"),         # keyword should save it
    # Adversarial: paraphrase whose literal words also fit a wrong doc (solar cell).
    ("what part of a cell produces its energy",   "mitochondria.md"),       # vector should save it
]

TOP_K = 20  # large enough to rank the whole small corpus


def build_index():
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM vec_chunks")
    conn.commit()
    for path in sorted(Path(os.environ["NOTES_DIR"]).glob("*.md")):
        index_file(conn, path)
    return conn


def ranked_files(rows) -> list[str]:
    """Collapse a chunk-level ranking to a de-duplicated file-level ranking."""
    order = []
    for r in rows:
        name = Path(r["file_path"]).name
        if name not in order:
            order.append(name)
    return order


def rank_of(files: list[str], target: str):
    """1-based rank of target, or None if absent."""
    return files.index(target) + 1 if target in files else None


def score(ranks: list) -> dict:
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hit3 = sum(1 for r in ranks if r is not None and r <= 3)
    mrr = sum((1.0 / r) for r in ranks if r is not None) / n
    return {"hit1": hit1, "hit3": hit3, "mrr": mrr, "n": n}


def main():
    ensure_model_available()
    conn = build_index()

    methods = {"vector-only": [], "keyword-only": [], "hybrid": []}
    print("Per-query rank of the relevant note (lower is better, X = not found):\n")
    print(f"  {'query':<42} {'vec':>4} {'kw':>4} {'hyb':>4}")
    print("  " + "-" * 58)

    for query, target in QUERIES:
        emb = embed_one(query)
        vrows = vector_search(conn, emb, TOP_K)
        krows = keyword_search(conn, query, TOP_K)
        hrows = rrf_merge(vrows, krows, TOP_K)

        rv = rank_of(ranked_files(vrows), target)
        rk = rank_of(ranked_files(krows), target)
        rh = rank_of(ranked_files(hrows), target)
        methods["vector-only"].append(rv)
        methods["keyword-only"].append(rk)
        methods["hybrid"].append(rh)

        fmt = lambda r: str(r) if r is not None else "X"
        print(f"  {query[:42]:<42} {fmt(rv):>4} {fmt(rk):>4} {fmt(rh):>4}")

    conn.close()

    print(f"\nAggregate over {len(QUERIES)} queries:\n")
    print(f"  {'method':<14} {'Hit@1':>6} {'Hit@3':>6} {'MRR':>6}")
    print("  " + "-" * 36)
    for name, ranks in methods.items():
        s = score(ranks)
        print(f"  {name:<14} {s['hit1']}/{s['n']:<4} {s['hit3']}/{s['n']:<4} {s['mrr']:>6.3f}")


if __name__ == "__main__":
    main()
