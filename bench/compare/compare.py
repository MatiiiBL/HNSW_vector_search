"""Recall-vs-QPS sweeps of FAISS, hnswlib and vecdex HNSW indexes, run from Python.

    python bench/compare/compare.py <dir> <name> [--libs faiss-hnsw,hnswlib,vecdex]
        [--metric l2|ip] [--M 16] [--ef-construction 200] [--k 10]
        [--ef 10,16,32,64,128,256,512] [--repeats 5] [--max-base N] [--max-queries N]
        [--out-dir DIR]

Reads <dir>/<name>_base.fvecs, <name>_query.fvecs and <name>_groundtruth.ivecs, like vecdex_eval,
and writes one CSV per library, <out-dir>/<name>-<lib>.csv, with vecdex_eval's columns, so
scripts/plot_recall_qps.py plots these results and the C++ ones alike. Rows also go to stdout.

Every library gets the same build parameters and one thread, for building and for searching.
Each timed pass is one batched call over all queries, so Python's per-call overhead is amortized
equally. QPS is the median of `repeats` timed passes, after one untimed warmup pass.

--metric ip expects unit-length vectors (the GloVe files from scripts/download_glove100.sh),
whose inner product is their cosine similarity. --max-base keeps only the first N base vectors
and recomputes the ground truth exactly, for quick smoke runs.
"""

import argparse
import ctypes
import statistics
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIBS = ("faiss-hnsw", "hnswlib", "vecdex")
COLUMNS = "lib,dataset,n,dim,M,ef_construction,build_s,memory_mb,k,ef,recall,recall_dist,qps"
SEED = 42  # what vecdex_eval passes to hnswlib, and vecdex's default
TOLERANCE = 1e-5  # threshold recall's relative tolerance, as in vecdex::threshold_recall_at_k


# ---------------------------------------------------------------------------------------------
# The indexes under test


class Engine(ABC):
    """One library's HNSW index. build() and search() are the calls being timed."""

    @abstractmethod
    def build(self, base: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int, ef: int) -> np.ndarray:
        """Ids of up to k neighbors per query, closest first, as a (queries, k) array."""

    @abstractmethod
    def memory_bytes(self) -> int | None:
        """Size of the built index, or None if the library can't report it."""


# Each engine imports its library in __init__, so a missing library skips only its own sweep.


class FaissHnsw(Engine):
    def __init__(self, dim: int, metric: str, M: int, ef_construction: int) -> None:
        import faiss

        faiss.omp_set_num_threads(1)  # FAISS runs both add() and search() on OpenMP threads
        self._faiss = faiss
        faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        # Like hnswlib and vecdex: M links per node on the upper layers, 2M on layer 0. FAISS
        # seeds its level generator itself; there's no parameter for it.
        self._index = faiss.IndexHNSWFlat(dim, M, faiss_metric)
        self._index.hnsw.efConstruction = ef_construction

    def build(self, base: np.ndarray) -> None:
        self._index.add(base)

    def search(self, queries: np.ndarray, k: int, ef: int) -> np.ndarray:
        self._index.hnsw.efSearch = ef
        _, ids = self._index.search(queries, k)
        return ids

    def memory_bytes(self) -> int:
        return self._faiss.serialize_index(self._index).size


class Hnswlib(Engine):
    def __init__(self, dim: int, metric: str, M: int, ef_construction: int) -> None:
        import hnswlib

        self._index = hnswlib.Index(space=metric, dim=dim)  # "l2" and "ip" are its names too
        self._M = M
        self._ef_construction = ef_construction

    def build(self, base: np.ndarray) -> None:
        self._index.init_index(
            max_elements=len(base),
            M=self._M,
            ef_construction=self._ef_construction,
            random_seed=SEED,
        )
        # Labels are row numbers, so they match the ground truth's ids.
        self._index.add_items(base, np.arange(len(base)), num_threads=1)

    def search(self, queries: np.ndarray, k: int, ef: int) -> np.ndarray:
        self._index.set_ef(ef)
        ids, _ = self._index.knn_query(queries, k=k, num_threads=1)
        return ids

    def memory_bytes(self) -> int:
        return self._index.index_file_size()


class Vecdex(Engine):
    def __init__(self, dim: int, metric: str, M: int, ef_construction: int) -> None:
        try:
            import vecdex
        except ImportError as error:
            raise ImportError(
                f"the vecdex Python bindings aren't installed ({error}); "
                "install them with `pip install .` from the repository root"
            ) from error
        self._index = vecdex.Index(space=metric, dim=dim, M=M, ef_construction=ef_construction)

    def build(self, base: np.ndarray) -> None:
        self._index.add_items(base)  # ids are assigned in row order, starting at 0

    def search(self, queries: np.ndarray, k: int, ef: int) -> np.ndarray:
        ids, _ = self._index.knn_query(queries, k=k, ef=ef)
        return ids

    def memory_bytes(self) -> int | None:
        # HnswIndex::memory_bytes() exists in C++; the bindings may not expose it.
        report = getattr(self._index, "memory_bytes", None)
        return None if report is None else int(report())


ENGINES = {"faiss-hnsw": FaissHnsw, "hnswlib": Hnswlib, "vecdex": Vecdex}


# ---------------------------------------------------------------------------------------------
# Datasets and recall


@dataclass(frozen=True)
class Dataset:
    base: np.ndarray  # (n, dim) float32
    queries: np.ndarray  # (queries, dim) float32
    truth: np.ndarray  # (queries, k) ids of the true nearest neighbors, closest first


def read_vecs(path: Path, dtype: str) -> np.ndarray:
    """Reads a TEXMEX .fvecs/.ivecs file: each row is an int32 dimension, then dim 4-byte values."""
    words = np.fromfile(path, dtype="<i4")
    dim = int(words[0]) if words.size else 0
    if dim <= 0 or words.size % (dim + 1) != 0:
        raise ValueError(f"{path}: not an .fvecs/.ivecs file (bad dimension or size)")
    records = words.reshape(-1, dim + 1)
    if np.any(records[:, 0] != dim):
        raise ValueError(f"{path}: records have different dimensions")
    return np.ascontiguousarray(records[:, 1:]).view(dtype)


def distances(queries: np.ndarray, base: np.ndarray, ids: np.ndarray, metric: str) -> np.ndarray:
    """Float64 distance from each query to the base vectors named in its row of `ids`.

    Ids outside the base (a library's marker for a missing result) get an infinite distance.
    """
    valid = (ids >= 0) & (ids < len(base))
    vectors = base[np.where(valid, ids, 0)].astype(np.float64)
    q = queries.astype(np.float64)[:, np.newaxis, :]
    if metric == "l2":
        d = ((vectors - q) ** 2).sum(axis=2)
    else:
        d = 1.0 - (vectors * q).sum(axis=2)  # the 1 - <a, b> convention of vecdex and hnswlib
    return np.where(valid, d, np.inf)


def exact_neighbors(base: np.ndarray, queries: np.ndarray, k: int, metric: str) -> np.ndarray:
    """The k nearest base rows of each query by brute force, in float64."""
    x = base.astype(np.float64)
    norms = (x**2).sum(axis=1)
    truth = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), 256):  # bounds the (queries, n) distance block
        dots = queries[start : start + 256].astype(np.float64) @ x.T
        # Rank-equivalent to the metric: |q|^2 and the constant 1 don't change the order.
        d = norms - 2.0 * dots if metric == "l2" else -dots
        top = np.argpartition(d, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, top, axis=1), axis=1, kind="stable")
        truth[start : start + 256] = np.take_along_axis(top, order, axis=1)
    return truth


def id_recall(found: np.ndarray, truth: np.ndarray) -> float:
    """Mean recall@k by id: the share of each row's k true ids that appear in its k found ids."""
    hits = (truth[:, :, np.newaxis] == found[:, np.newaxis, :]).any(axis=2).sum(axis=1)
    return float(hits.mean()) / truth.shape[1]


def threshold_recall(found_distance: np.ndarray, kth_distance: np.ndarray) -> float:
    """Mean distance-threshold recall@k (ann-benchmarks): a result is a hit if it's no farther
    than the k-th true neighbor, plus a relative tolerance for rounding."""
    threshold = kth_distance + TOLERANCE * np.maximum(np.abs(kth_distance), 1.0)
    hits = (found_distance <= threshold[:, np.newaxis]).sum(axis=1)
    return float(hits.mean()) / found_distance.shape[1]


# ---------------------------------------------------------------------------------------------
# Command line and sweep


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def int_list(text: str) -> list[int]:
    return [positive_int(item) for item in text.split(",")]


def lib_list(text: str) -> list[str]:
    libs = text.split(",")
    for lib in libs:
        if lib not in ENGINES:
            raise argparse.ArgumentTypeError(f"unknown library {lib}; choose from {LIBS}")
    return libs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path)
    parser.add_argument("name")
    parser.add_argument("--libs", type=lib_list, default=list(LIBS))
    parser.add_argument("--metric", choices=("l2", "ip"), default="l2")
    parser.add_argument("--M", type=positive_int, default=16)
    parser.add_argument("--ef-construction", type=positive_int, default=200)
    parser.add_argument("--k", type=positive_int, default=10)
    parser.add_argument("--ef", type=int_list, default=[10, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--max-base", type=positive_int)
    parser.add_argument("--max-queries", type=positive_int)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    return parser.parse_args()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def prefer_performance_cores() -> None:
    """Asks macOS to keep this thread on a performance core, as vecdex_eval does (D-012)."""
    if sys.platform != "darwin":
        return
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    qos_class_user_interactive = 0x21
    if libsystem.pthread_set_qos_class_self_np(qos_class_user_interactive, 0) != 0:
        log("warning: could not raise the thread's QoS class")


def load(args: argparse.Namespace) -> Dataset:
    """The dataset after the --max-base and --max-queries cuts."""
    prefix = args.dir / args.name
    base = read_vecs(Path(f"{prefix}_base.fvecs"), "<f4")
    queries = read_vecs(Path(f"{prefix}_query.fvecs"), "<f4")[: args.max_queries]
    if queries.shape[1] != base.shape[1]:
        raise ValueError("queries and base vectors have different dimensions")
    if args.k > len(base):
        raise ValueError(f"k = {args.k} exceeds the {len(base)} base vectors")
    if args.metric == "ip":
        norms = np.linalg.norm(base, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("--metric ip needs unit-length base vectors (cosine similarity)")

    if args.max_base is not None and args.max_base < len(base):
        base = base[: args.max_base].copy()  # copy, so the full array can be freed
        log(f"computing exact ground truth for the first {len(base)} base vectors")
        return Dataset(base, queries, exact_neighbors(base, queries, args.k, args.metric))

    truth = read_vecs(Path(f"{prefix}_groundtruth.ivecs"), "<i4")[: len(queries), : args.k]
    if truth.shape != (len(queries), args.k):
        raise ValueError(f"ground truth is {truth.shape}, need {len(queries)} x {args.k}")
    if truth.min() < 0 or truth.max() >= len(base):
        raise ValueError("ground truth names vectors that aren't in the base set")
    return Dataset(base, queries, truth)


def sweep(lib: str, engine: Engine, data: Dataset, args: argparse.Namespace) -> list[str]:
    """Builds the index, then runs every ef; returns one CSV row per ef."""
    base, queries, truth = data.base, data.queries, data.truth
    start = time.perf_counter()
    engine.build(base)
    build_s = time.perf_counter() - start
    memory = engine.memory_bytes()
    memory_mb = "nan" if memory is None else f"{memory / 1e6:.0f}"
    log(f"{lib}: built in {build_s:.1f} s, index {memory_mb} MB")

    # The k-th true neighbor's distance, recomputed from the vectors like every found distance
    # below: the libraries report distances in different units (squared L2, similarity, 1 - ip).
    kth_distance = distances(queries, base, truth[:, -1:], args.metric)[:, 0]
    rows = []
    for ef in args.ef:
        engine.search(queries, args.k, ef)  # warmup: caches, branch predictors, scratch pools
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            found = engine.search(queries, args.k, ef)
            times.append(time.perf_counter() - start)
        qps = len(queries) / statistics.median(times)

        found = np.asarray(found, dtype=np.int64)
        recall = id_recall(found, truth)
        recall_dist = threshold_recall(distances(queries, base, found, args.metric), kth_distance)
        row = (
            f"{lib},{args.name},{len(base)},{base.shape[1]},{args.M},{args.ef_construction},"
            f"{build_s:.1f},{memory_mb},{args.k},{ef},{recall:.4f},{recall_dist:.4f},{qps:.0f}"
        )
        print(row, flush=True)
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    prefer_performance_cores()
    data = load(args)
    dim = data.base.shape[1]
    log(
        f"loaded {len(data.base)} x {dim} base vectors, {len(data.queries)} queries; "
        f"metric {args.metric}, M {args.M}, efConstruction {args.ef_construction}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(COLUMNS, flush=True)
    finished = 0
    for lib in args.libs:
        try:
            engine = ENGINES[lib](dim, args.metric, args.M, args.ef_construction)
        except ImportError as error:
            log(f"{lib}: skipped, {error}")
            continue
        rows = sweep(lib, engine, data, args)
        out = args.out_dir / f"{args.name}-{lib}.csv"
        out.write_text("\n".join([COLUMNS, *rows]) + "\n")
        log(f"{lib}: wrote {out}")
        finished += 1
        del engine  # free this index before the next library builds its own
    return 0 if finished else 1


if __name__ == "__main__":
    sys.exit(main())
