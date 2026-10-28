"""Retrievers: top-k search by cosine similarity, behind one interface.

FaissFlatRetriever is exact and serves as the reference. FaissHnswRetriever and VecdexRetriever
are approximate HNSW indexes built with the same parameters, so swapping one for the other
changes nothing else in the demo.
"""

from abc import ABC, abstractmethod

import faiss
import numpy as np


class Retriever(ABC):
    """Top-k search over unit-length float32 embeddings, each stored under a caller's int64 id.

    Subclasses wrap an index that numbers its vectors itself (_add, _search); this class checks
    the input and translates the index's numbers into the caller's ids.
    """

    name: str

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids = np.empty(0, dtype=np.int64)  # the caller's id for each of the index's numbers

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        embeddings = self._checked(embeddings)
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape != (len(embeddings),):
            raise ValueError(f"{len(embeddings)} embeddings need as many ids, got {ids.shape}")
        numbers = np.asarray(self._add(embeddings), dtype=np.int64)
        size = max(len(self._ids), int(numbers.max(initial=-1)) + 1)
        self._ids = np.concatenate([self._ids, np.full(size - len(self._ids), -1, np.int64)])
        self._ids[numbers] = ids

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The k best (ids, cosine similarities) per query, best first, as (queries, k) arrays.

        A missing result (fewer than k stored) is id -1 with similarity -inf.
        """
        queries = self._checked(queries)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if len(self._ids) == 0:
            raise ValueError("nothing to search: add embeddings first")
        numbers, scores = self._search(queries, k)
        numbers = np.asarray(numbers, dtype=np.int64)
        found = (numbers >= 0) & (numbers < len(self._ids))
        ids = np.where(found, self._ids[np.where(found, numbers, 0)], -1)
        return ids, np.where(found, scores, -np.inf).astype(np.float32)

    @abstractmethod
    def _add(self, embeddings: np.ndarray) -> np.ndarray:
        """Stores the embeddings; returns the number the index gave each one."""

    @abstractmethod
    def _search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """The index's numbers and cosine similarities of the k best matches per query."""

    def _checked(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected an (n, {self.dim}) array, got shape {vectors.shape}")
        # Inner product equals cosine similarity only between unit vectors.
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
            raise ValueError("embeddings must have unit length")
        return vectors


class _FaissRetriever(Retriever):
    def __init__(self, index: faiss.Index) -> None:
        super().__init__(index.d)
        self._index = index

    def _add(self, embeddings: np.ndarray) -> np.ndarray:
        start = self._index.ntotal
        self._index.add(embeddings)
        return np.arange(start, self._index.ntotal)  # FAISS numbers vectors in insertion order

    def _search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        similarities, numbers = self._index.search(queries, k)  # -1 marks a missing result
        return numbers, similarities


class FaissFlatRetriever(_FaissRetriever):
    """Exact search: compares each query with every stored embedding."""

    name = "faiss-flat"

    def __init__(self, dim: int) -> None:
        super().__init__(faiss.IndexFlatIP(dim))


class FaissHnswRetriever(_FaissRetriever):
    """FAISS's HNSW index (IndexHNSWFlat) on inner product."""

    name = "faiss-hnsw"

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, ef: int = 64) -> None:
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef
        super().__init__(index)


class VecdexRetriever(Retriever):
    """vecdex's HNSW index through its Python bindings, on cosine distance."""

    name = "vecdex"

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, ef: int = 64) -> None:
        try:
            import vecdex  # optional: the demo runs without it
        except ImportError as error:
            raise ImportError(
                f"VecdexRetriever needs the vecdex Python bindings, which aren't installed "
                f"({error}). Install them with `pip install .` from the repository root."
            ) from error
        super().__init__(dim)
        self._index = vecdex.Index(space="cosine", dim=dim, M=M, ef_construction=ef_construction)
        self._ef = ef

    def _add(self, embeddings: np.ndarray) -> np.ndarray:
        return self._index.add_items(embeddings)

    def _search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        numbers, distances = self._index.knn_query(queries, k=k, ef=self._ef)
        return numbers, 1.0 - distances  # vecdex's cosine distance is 1 - cos
