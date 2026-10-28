"""vecdex.Index and threads: add_items and knn_query release the GIL, the index's reader-writer
lock keeps concurrent calls safe and their results unchanged, and num_threads spreads one call
over several threads."""

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import vecdex

DIM = 16
SPACES = ["l2", "ip", "cosine"]


def random_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


def integer_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    """integer_matrix of tests/hnsw_parallel_test.cpp, bit for bit: components in [-128, 127]
    from the top 8 bits of std::mt19937, which NumPy's legacy RandomState is, seeded the same
    way. Squared distances between them are exact under any kernel."""
    raw = np.random.RandomState(seed).randint(0, 2**32, size=n * dim, dtype=np.uint32)
    return ((raw >> 24).astype(np.int32) - 128).astype(np.float32).reshape(n, dim)


def exact_neighbors(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """The ids of each query's k nearest base vectors under squared L2, ties by smaller id like
    BruteForceIndex. Integer arithmetic, so exact for integer_vectors."""
    b, q = base.astype(np.int64), queries.astype(np.int64)
    distances = (q * q).sum(axis=1)[:, None] - 2 * q @ b.T + (b * b).sum(axis=1)[None, :]
    return np.argsort(distances, axis=1, kind="stable")[:, :k]


def recall(ids: np.ndarray, truth: np.ndarray) -> float:
    """recall@k as src/recall.cpp defines it, averaged over the queries."""
    return float((ids[:, :, None] == truth[:, None, :]).any(axis=2).mean())


def build(base: np.ndarray, num_threads: int = 1) -> vecdex.Index:
    index = vecdex.Index("l2", DIM, M=8, ef_construction=40)
    index.add_items(base, num_threads=num_threads)
    return index


def scale_to(seconds: float, trial: Callable[[], object]) -> int:
    """How many times larger than `trial` a workload must be to take about `seconds`."""
    start = time.perf_counter()
    trial()
    return max(1, math.ceil(seconds / max(time.perf_counter() - start, 1e-6)))


def python_ticks_during(call: Callable[[], object]) -> int:
    """Runs call() while another Python thread ticks about once a millisecond, and counts the
    ticks in the middle half of the call. A call that held the GIL would allow none there: the
    other thread can only run Python code while it holds the GIL, and the quarter-call margins
    exceed the 5 ms switch interval in which it might run just before or after the call."""
    ticks: list[float] = []
    stop = threading.Event()

    def tick() -> None:
        last = 0.0
        while not stop.is_set():
            now = time.perf_counter()
            if now - last >= 1e-3:
                ticks.append(now)
                last = now

    worker = threading.Thread(target=tick)
    worker.start()
    try:
        start = time.perf_counter()
        call()
        end = time.perf_counter()
    finally:
        stop.set()
        worker.join()
    quarter = (end - start) / 4
    return sum(start + quarter < t < end - quarter for t in ticks)


# --- The GIL ------------------------------------------------------------------------------


@pytest.mark.parametrize("num_threads", [1, 4])
def test_knn_query_releases_the_gil(num_threads):
    index = build(random_vectors(2000))
    queries = random_vectors(100, seed=1)

    def query(data: np.ndarray) -> object:
        return index.knn_query(data, k=10, ef=100, num_threads=num_threads)

    batch = np.tile(queries, (scale_to(0.2, lambda: query(queries)), 1))
    assert python_ticks_during(lambda: query(batch)) > 0


@pytest.mark.parametrize("num_threads", [1, 4])
def test_add_items_releases_the_gil(num_threads):
    trial = random_vectors(500)
    repeats = scale_to(0.2, lambda: build(trial, num_threads))
    batch = random_vectors(500 * repeats, seed=1)
    assert python_ticks_during(lambda: build(batch, num_threads)) > 0


# --- The reader-writer lock -----------------------------------------------------------------


def test_concurrent_queries_match_sequential_ones():
    index = build(random_vectors(3000))
    queries = random_vectors(800, seed=1)
    expected_ids, expected_distances = index.knn_query(queries, k=10, ef=50)

    with ThreadPoolExecutor(max_workers=8) as pool:
        # Batches of rows (contiguous views) and single 1-D rows, all at once.
        batches = list(pool.map(lambda q: index.knn_query(q, k=10, ef=50),
                                np.array_split(queries, 16)))
        singles = list(pool.map(lambda q: index.knn_query(q, k=10, ef=50), queries))

    for results in (batches, singles):
        np.testing.assert_array_equal(np.concatenate([ids for ids, _ in results]), expected_ids)
        np.testing.assert_array_equal(
            np.concatenate([distances for _, distances in results]), expected_distances
        )


def searches_during(index: vecdex.Index, queries: np.ndarray, add_all: Callable[[], None]) -> int:
    """Runs add_all() while three threads query the index, and returns how many searches ran.
    Every answer must come from some consistent state of the index, which only grows."""
    adding = threading.Event()
    adding.set()

    def add_then_stop() -> None:
        try:
            add_all()
        finally:
            adding.clear()

    def search_until_done() -> int:
        searches = 0
        while adding.is_set():
            ids, _ = index.knn_query(queries, k=5)
            assert ids.min() >= 0 and ids.max() < len(index)
            searches += 1
        return searches

    with ThreadPoolExecutor(max_workers=4) as pool:
        searchers = [pool.submit(search_until_done) for _ in range(3)]
        pool.submit(add_then_stop).result()
        return sum(searcher.result() for searcher in searchers)  # re-raises any failure


def test_queries_during_adds_are_safe_and_leave_the_build_unchanged():
    initial = random_vectors(500)
    batches = [random_vectors(100, seed=seed) for seed in range(1, 21)]
    queries = random_vectors(50, seed=99)
    index = build(initial)

    def add_all() -> None:
        for batch in batches:
            index.add_items(batch)

    assert searches_during(index, queries, add_all) > 0, "no search overlapped the adds"

    # add_items calls ran one at a time and in order, so the graph must be exactly the one a
    # single-threaded build makes from the same vectors and seed.
    reference = build(np.concatenate([initial, *batches]))
    assert len(index) == len(reference) == 2500
    for got, want in zip(index.knn_query(queries, k=10), reference.knn_query(queries, k=10)):
        np.testing.assert_array_equal(got, want)


def test_queries_during_parallel_adds_are_safe():
    index = build(random_vectors(500))

    def add_all() -> None:
        for seed in range(1, 21):
            index.add_items(random_vectors(100, seed=seed), num_threads=4)

    assert searches_during(index, random_vectors(50, seed=99), add_all) > 0
    assert len(index) == 2500


# --- num_threads ----------------------------------------------------------------------------


def test_parallel_add_recall_is_within_0_005_of_the_serial_build():
    # The data, parameters and bar of the C++ test HnswParallel.MatchesSerialRecallOnRandomData,
    # so its calibration carries over: the serial build is the same graph (recall@10 0.9606 at
    # ef 16 in both languages), and over 100 builds from Python per thread count, the worst was
    # 0.0002 below it on 2 threads, 0.0004 on 4 and 0.0011 on 10 (Apple M4, 2026-10-28). A
    # parallel build may also be better; the bar is one-sided.
    base, queries = integer_vectors(2000, 16, seed=1), integer_vectors(1000, 16, seed=2)
    truth = exact_neighbors(base, queries, k=10)

    def recall_of_build(num_threads: int) -> float:
        index = vecdex.Index("l2", 16, M=16, ef_construction=64, seed=42)
        ids = index.add_items(base, num_threads=num_threads)
        np.testing.assert_array_equal(ids, np.arange(len(base)))
        found, _ = index.knn_query(queries, k=10, ef=16)
        return recall(found, truth)

    serial = recall_of_build(1)
    for num_threads in (2, 4, 0):  # 0: one per hardware thread
        assert recall_of_build(num_threads) >= serial - 0.005, f"num_threads={num_threads}"


@pytest.mark.parametrize("space", SPACES)
def test_parallel_queries_return_exactly_the_sequential_results(space):
    index = vecdex.Index(space, DIM, M=8, ef_construction=40)
    index.add_items(random_vectors(3000))
    tiny = vecdex.Index(space, DIM)
    tiny.add_items(random_vectors(5))
    queries = random_vectors(500, seed=1)

    # Many rows; one 1-D row (one row for the threads to share); none; and k > len, which pads.
    for target, data, k in ((index, queries, 10), (index, queries[0], 10),
                            (index, queries[:0], 10), (tiny, queries, 8)):
        expected = target.knn_query(data, k=k, ef=50)
        for num_threads in (2, 4, 0):
            got = target.knn_query(data, k=k, ef=50, num_threads=num_threads)
            for array, want in zip(got, expected):
                np.testing.assert_array_equal(array, want)
                assert array.shape == want.shape and array.dtype == want.dtype


def test_negative_num_threads_raise_value_error():
    index = build(random_vectors(10))
    message = "num_threads must be positive, or 0 for one per hardware thread, got -1"
    with pytest.raises(ValueError, match=message):
        index.add_items(random_vectors(5, seed=1), num_threads=-1)
    assert len(index) == 10
    with pytest.raises(ValueError, match=message):
        index.knn_query(random_vectors(2, seed=1), num_threads=-1)


def test_num_threads_is_keyword_only():
    # hnswlib's positional arguments differ (add_items(data, ids), knn_query(data, k,
    # num_threads)), so a positional thread count would mean something else after a port.
    index = build(random_vectors(10))
    with pytest.raises(TypeError):
        index.add_items(random_vectors(5, seed=1), 2)
    with pytest.raises(TypeError):
        index.knn_query(random_vectors(2, seed=1), 1, None, 2)
    assert len(index) == 10


def test_non_finite_values_raise_value_error_with_several_threads():
    index = build(random_vectors(1000))
    batch = random_vectors(500, seed=1)
    batch[300, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        index.add_items(batch, num_threads=4)  # checked in full before any thread starts
    assert len(index) == 1000
    with pytest.raises(ValueError, match="NaN or inf"):
        index.knn_query(batch, num_threads=4)  # checked on the thread that searches row 300
