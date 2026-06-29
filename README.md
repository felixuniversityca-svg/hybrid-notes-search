# hybrid-notes-search

Hybrid semantic + keyword search over a folder of Markdown notes. Fully local, no API keys, runs offline after a one-time 23 MB model download.

It combines two search methods and fuses them with Reciprocal Rank Fusion:

- **0.7 vector search** (cosine similarity, [sqlite-vec](https://github.com/asg017/sqlite-vec)) for meaning and paraphrases
- **0.3 keyword search** (BM25, SQLite FTS5) for exact terms, names, and rare words

So a search for "how do I combine two ranked lists" finds a note about Reciprocal Rank Fusion even though the note never uses the word in the query.

## Demo

```
$ python search.py "how do I combine two ranked lists"

Top 3 results for 'how do I combine two ranked lists'
------------------------------------------------------------
[1] score=0.0115
    reciprocal-rank-fusion.md
    # Reciprocal Rank Fusion  Reciprocal Rank Fusion (RRF) is a simple, robust way
    to combine several ranked result lists into one...

[2] score=0.0113
    vector-vs-keyword-search.md
    # Vector search versus keyword search  Keyword search (BM25 over an inverted
    index, like SQLite FTS5) matches the exact terms a user types...
```

The query shares no words with the top result. That is the vector half doing its job; the keyword half pulls in exact-term matches the embeddings would miss.

## Quickstart

Runs out of the box on the sample notes in `notes/`:

```bash
git clone https://github.com/felixuniversityca-svg/hybrid-notes-search
cd hybrid-notes-search
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python index.py                      # build the index (downloads the model on first run)
python search.py "reciprocal rank fusion"
```

## Use it on your own notes

Point `NOTES_DIR` at any folder of Markdown files:

```bash
NOTES_DIR=~/my-notes python index.py
NOTES_DIR=~/my-notes python search.py "what did I decide about pricing"
```

Re-running `index.py` is incremental: it only re-embeds files whose modification time changed, so keeping the index fresh is cheap. Use `python index.py --full` to rebuild from scratch and `python index.py --stats` to see counts.

`search.py` flags: `--top-k N` (number of results), `--path-prefix SUB` (restrict to a subfolder), `--json` (machine-readable output for piping into other tools).

## How it works

1. **Index** (`index.py`): every `.md` file is split into overlapping ~300-word chunks (40-word overlap), YAML frontmatter stripped. Each chunk is embedded and stored in SQLite.
2. **Store** (`db.py`): one SQLite file holds three tables, kept in sync by triggers: `chunks` (text + embedding blob + mtime), `chunks_fts` (FTS5 for BM25), `vec_chunks` (sqlite-vec for cosine).
3. **Search** (`search.py`): the query is embedded once, then run through both the vector and keyword indexes. The two ranked lists are merged with RRF, `score = sum(weight / (k + rank))` with `k = 60`, so a chunk ranked highly by both methods wins.
4. **Embed** (`embeddings.py`): `all-MiniLM-L6-v2` (384-dim) via FastEmbed, ONNX on CPU, cached at `~/.cache/fastembed`.

## Tech

Python, SQLite (FTS5 + [sqlite-vec](https://github.com/asg017/sqlite-vec)), [FastEmbed](https://github.com/qdrant/fastembed) with `all-MiniLM-L6-v2`. No external services, no API keys, no GPU.

## Why

Cloud vector databases are overkill for a personal note collection. A single SQLite file plus a 23 MB local model gives you private, instant, free semantic search over thousands of notes. This is a clean extraction of the search layer I use daily over my own notes.

## License

MIT
