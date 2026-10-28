"""vecdex: approximate nearest-neighbor search over HNSW graphs, written in C++.

    import vecdex
    index = vecdex.Index("l2", dim=128)
    ids = index.add_items(vectors)                   # float32 array of shape (n, 128)
    ids, distances = index.knn_query(queries, k=10)  # both of shape (len(queries), 10)

help(vecdex.Index) documents the parameters and the input rules.
"""

from vecdex._core import Index, __version__

__all__ = ["Index", "__version__"]
