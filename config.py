"""Shared configuration. Override either path with an environment variable."""
import os
from pathlib import Path

# Folder of Markdown notes to index and search. Override with NOTES_DIR.
NOTES_DIR = Path(os.environ.get("NOTES_DIR", "notes")).expanduser()

# Where the SQLite index lives. Override with NOTES_DB.
DB_PATH = Path(os.environ.get("NOTES_DB", "data/index.db")).expanduser()

# all-MiniLM-L6-v2 produces 384-dim vectors.
EMBEDDING_DIM = 384
