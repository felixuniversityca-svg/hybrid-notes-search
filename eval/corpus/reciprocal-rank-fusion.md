# Reciprocal Rank Fusion

Reciprocal Rank Fusion is a simple way to merge several ranked lists into a single
ordering. Each item scores 1 / (k + position) summed over every list it appears
in, where k is a constant such as 60. Items near the top of more than one list
rise to the top of the merged result. Because it reads only positions and never
the underlying scores, it can blend lists whose scores live on completely
different scales without any calibration.
