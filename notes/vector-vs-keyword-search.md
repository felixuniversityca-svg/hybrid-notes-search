# Vector search versus keyword search

Keyword search (BM25 over an inverted index, like SQLite FTS5) matches the exact
terms a user types. It is precise for names, codes, and rare words, but it misses
synonyms and paraphrases: a search for "car" will not find a note that only says
"automobile".

Vector search embeds text into a dense vector and ranks by cosine similarity, so
it captures meaning and finds paraphrases. Its weakness is the mirror image:
it can drift away from exact terms and rare identifiers.

Hybrid search runs both and merges the results, keeping the precision of keywords
and the recall of semantics. A typical weighting leans toward the vector side
(for example 0.7 vector, 0.3 keyword) and fuses with Reciprocal Rank Fusion.
