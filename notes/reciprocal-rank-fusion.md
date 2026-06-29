# Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) is a simple, robust way to combine several ranked
result lists into one. Each item gets a score of 1 / (k + rank) summed across
every list it appears in, where rank is its position in that list and k is a
constant (60 is the common default). Items that rank highly in more than one
list rise to the top.

RRF needs no score calibration. You can fuse a cosine-similarity list and a
BM25 keyword list even though their raw scores are on completely different
scales, because RRF only looks at rank position, not the underlying score. That
is why it is a popular choice for hybrid search.
