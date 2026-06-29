# Running embeddings locally

You do not need an API to embed text. all-MiniLM-L6-v2 is a small sentence
embedding model that produces 384-dimensional vectors and runs on CPU. Through
FastEmbed it ships as a quantized ONNX model of about 23 MB, downloaded once and
cached on disk, after which everything works offline.

Storing those vectors in SQLite via the sqlite-vec extension keeps the whole
stack in a single file with no separate database server. For a personal note
collection of thousands of documents this is fast enough to feel instant, fully
private, and free to run.
