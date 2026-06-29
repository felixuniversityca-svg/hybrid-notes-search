#!/usr/bin/env python3
"""
Unit checks for the Reciprocal Rank Fusion merge. Pure logic, no DB or model
needed. Run:  python test_search.py
"""
from search import rrf_merge, VECTOR_WEIGHT, KEYWORD_WEIGHT


def row(i: int) -> dict:
    """A minimal stand-in for a sqlite Row (rrf_merge only reads these keys)."""
    return {"id": i, "file_path": f"{i}.md", "text": f"doc {i}"}


def test_agreement_ranks_first():
    # doc 1 sits atop both lists; 2 is vector-only, 3 is keyword-only.
    merged = rrf_merge([row(1), row(2)], [row(1), row(3)], top_k=3)
    assert merged[0]["file_path"] == "1.md", [r["file_path"] for r in merged]


def test_vector_weight_dominates():
    # Same rank in each list, but the vector weight is higher, so it wins.
    assert VECTOR_WEIGHT > KEYWORD_WEIGHT
    merged = rrf_merge([row(2)], [row(3)], top_k=2)
    assert merged[0]["file_path"] == "2.md"


def test_earlier_rank_beats_later():
    # Within one list, rank 0 must outscore rank 1.
    merged = rrf_merge([row(1), row(2)], [], top_k=2)
    assert [r["file_path"] for r in merged] == ["1.md", "2.md"]


def test_top_k_trims():
    merged = rrf_merge([row(i) for i in range(10)], [], top_k=3)
    assert len(merged) == 3


def test_empty_inputs():
    assert rrf_merge([], [], top_k=5) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
