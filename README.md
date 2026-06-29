# hybrid-notes-search

Hybrid semantic + keyword search over a folder of Markdown notes. Fully local, no API keys, runs offline after a one-time 23 MB model download.

It combines two search methods and fuses them with Reciprocal Rank Fusion:

- **0.7 vector search** (cosine similarity, [sqlite-vec](https://github.com/asg017/sqlite-vec)) for meaning and paraphrases
- **0.3 keyword search** (BM25, SQLite FTS5) for exact terms, names, and rare words

So a note that never contains the words you typed can still be the top result, because the vector half matches on meaning.

![Demo: searching notes by meaning](demo.gif)

## Demo

```
$ python search.py "running models on my laptop without the cloud"

Top 3 results for 'running models on my laptop without the cloud'
------------------------------------------------------------
[1] score=0.0164
    local-embeddings.md
    # Running embeddings locally  You do not need an API to embed text.
    all-MiniLM-L6-v2 is a small sentence embedding model ... runs on CPU ...

[2] score=0.0161
    reciprocal-rank-fusion.md
    # Reciprocal Rank Fusion  ... a simple way to combine several ranked
    result lists into one ...

[3] score=0.0159
    vector-vs-keyword-search.md
    # Vector search versus keyword search  ...
```

The top note never uses the words "laptop" or "cloud", yet it ranks first because the meaning matches. The keyword half complements this by pulling in exact-term hits that embeddings blur (codes, names, jargon).

## Requirements

Python 3.10+ built with SQLite loadable-extension support (`sqlite-vec` needs it). Most Linux distro Pythons and Homebrew's macOS Python work; the macOS **system** Python does not. On macOS, `brew install python` and use that interpreter. Verify any interpreter with:

```bash
python3 -c "import sqlite3; print(hasattr(sqlite3.connect(':memory:'), 'enable_load_extension'))"
# must print True
```

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
3. **Search** (`search.py`): the query is embedded once, then run through both indexes. The keyword query is sanitized into OR-of-quoted-tokens so FTS5 operator characters (a `-` means NOT) stay literal and a single missing word does not drop the row. The two ranked lists are merged with RRF, `score = sum(weight / (k + rank))` with `k = 60`, so a chunk ranked highly by both methods wins.
4. **Embed** (`embeddings.py`): `all-MiniLM-L6-v2` (384-dim) via FastEmbed, ONNX on CPU, cached at `~/.cache/fastembed`.

Vector search always returns the nearest chunks, so even an off-topic query yields its closest matches rather than nothing.

## Evaluation

`eval.py` is a small reproducible benchmark: it indexes a labeled corpus (`eval/corpus/`) and scores each retrieval method on a fixed query set. The queries are chosen to be honest about the tradeoff, some are paraphrases with no shared words (vector should win), some are exact technical terms (keyword should win).

```
method          Hit@1   Hit@3    MRR
vector-only      8/8     8/8     1.000
keyword-only     6/8     8/8     0.875
hybrid           8/8     8/8     1.000
```

Read it honestly: hybrid matches the better component on every query and never does worse. Keyword-only trails because it misses paraphrases whose words are absent from the note; dense vectors handle those. On this small, clean corpus vector retrieval is already strong, so hybrid ties it here rather than beating it. The value of fusion is robustness: it inherits the strengths of both without you knowing in advance which a query needs, and the keyword half is what saves exact identifiers and rare terms on larger, messier corpora. Reproduce with `python eval.py`.

## Tests

```bash
python test_search.py     # unit checks for the RRF merge, no DB or model needed
```

## Tech

Python, SQLite (FTS5 + [sqlite-vec](https://github.com/asg017/sqlite-vec)), [FastEmbed](https://github.com/qdrant/fastembed) with `all-MiniLM-L6-v2`. No external services, no API keys, no GPU.

## Why

Cloud vector databases are overkill for a personal note collection. A single SQLite file plus a 23 MB local model gives you private, instant, free semantic search over thousands of notes. This is a clean extraction of the search layer I use daily over my own notes.

## License

MIT
