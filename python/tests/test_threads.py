"""vecdex.Index from several Python threads: add_items and knn_query release the GIL, and the
index's reader-writer lock keeps concurrent calls safe and their results unchanged."""

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import vecdex

DIM = 16


def random_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


def build(base: np.ndarray) -> vecdex.Index:
    index = vecdex.Index("l2", DIM, M=8, ef_construction=40)
    index.add_items(base)
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


def test_knn_query_releases_the_gil():
    index = build(random_vectors(2000))
    queries = random_vectors(100, seed=1)
    repeats = scale_to(0.2, lambda: index.knn_query(queries, k=10, ef=100))
    batch = np.tile(queries, (repeats, 1))
    assert python_ticks_during(lambda: index.knn_query(batch, k=10, ef=100)) > 0


def test_add_items_releases_the_gil():
    trial = random_vectors(500)
    repeats = scale_to(0.2, lambda: build(trial))
    batch = random_vectors(500 * repeats, seed=1)
    assert python_ticks_during(lambda: build(batch)) > 0


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


def test_queries_during_adds_are_safe_and_leave_the_build_unchanged():
    initial = random_vectors(500)
    batches = [random_vectors(100, seed=seed) for seed in range(1, 21)]
    queries = random_vectors(50, seed=99)
    index = build(initial)
    adding = threading.Event()
    adding.set()

    def add_all() -> None:
        try:
            for batch in batches:
                index.add_items(batch)
        finally:
            adding.clear()

    def search_until_done() -> int:
        searches = 0
        while adding.is_set():
            ids, _ = index.knn_query(queries, k=5)
            # Every answer comes from some consistent state of the index, which only grows.
            assert ids.min() >= 0 and ids.max() < len(index)
            searches += 1
        return searches

    with ThreadPoolExecutor(max_workers=4) as pool:
        searchers = [pool.submit(search_until_done) for _ in range(3)]
        pool.submit(add_all).result()
        searches = sum(searcher.result() for searcher in searchers)  # re-raises any failure
    assert searches > 0, "no search overlapped the adds"

    # add_items calls ran one at a time and in order, so the graph must be exactly the one a
    # single-threaded build makes from the same vectors and seed.
    reference = build(np.concatenate([initial, *batches]))
    assert len(index) == len(reference) == 2500
    for got, want in zip(index.knn_query(queries, k=10), reference.knn_query(queries, k=10)):
        np.testing.assert_array_equal(got, want)
