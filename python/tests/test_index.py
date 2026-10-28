"""vecdex.Index from Python: results against NumPy brute force, the input rules, and the result
conventions (shapes, dtypes, padding, ownership)."""

import gc

import numpy as np
import pytest

import vecdex

DIM = 16
SPACES = ["l2", "ip", "cosine"]


def random_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


def distance_matrix(base: np.ndarray, queries: np.ndarray, space: str) -> np.ndarray:
    """All query-to-base distances in float64, with the index's conventions: squared L2,
    1 - <q, x>, and 1 - cos(q, x)."""
    b = base.astype(np.float64)
    q = queries.astype(np.float64)
    if space == "cosine":
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    if space == "l2":
        return (q * q).sum(axis=1)[:, None] - 2.0 * q @ b.T + (b * b).sum(axis=1)[None, :]
    return 1.0 - q @ b.T


def build(space: str, base: np.ndarray, **params) -> vecdex.Index:
    index = vecdex.Index(space, base.shape[1], **params)
    index.add_items(base)
    return index


@pytest.mark.parametrize("space", SPACES)
def test_recall_and_distances_match_brute_force(space):
    base, queries, k = random_vectors(2000), random_vectors(100, seed=1), 10
    index = build(space, base, M=16, ef_construction=100)
    ids, distances = index.knn_query(queries, k=k, ef=64)

    exact = distance_matrix(base, queries, space)
    truth = np.argsort(exact, axis=1)[:, :k]
    recall = np.mean([len(set(found) & set(true)) / k for found, true in zip(ids, truth)])
    # Measured 0.999 (l2), 1.000 (ip) and 1.000 (cosine) on arm64 (about 0.89 at ef = 10). The
    # margin absorbs other kernels' summation orders; a wrong metric would score far lower.
    assert recall >= 0.98, f"{space}: recall@{k} = {recall:.3f}"
    # Whatever was found, its distance must be the true one under the space's convention.
    np.testing.assert_allclose(
        distances, np.take_along_axis(exact, ids, axis=1), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("space", SPACES)
def test_exact_when_ef_covers_every_node(space):
    # With ef >= n the search can't stop early, so on a connected graph HNSW is exact. Checked
    # by distance rather than by id, so float ties broken differently than NumPy's float64
    # don't count as misses (the threshold recall of D-012).
    base, queries, k = random_vectors(500), random_vectors(50, seed=1), 10
    index = build(space, base, M=8, ef_construction=50)
    ids, _ = index.knn_query(queries, k=k, ef=len(base))

    exact = distance_matrix(base, queries, space)
    kth = np.sort(exact, axis=1)[:, k - 1 : k]
    found = np.take_along_axis(exact, ids, axis=1)
    assert np.all(found <= kth + 1e-5 * np.maximum(np.abs(kth), 1.0))


def test_add_items_returns_consecutive_int64_ids():
    index = vecdex.Index("l2", DIM)
    first = index.add_items(random_vectors(10))
    second = index.add_items(random_vectors(5, seed=1))
    assert first.dtype == np.int64 and first.shape == (10,)
    np.testing.assert_array_equal(first, np.arange(10))
    np.testing.assert_array_equal(second, np.arange(10, 15))
    assert len(index) == 15


def test_knn_query_returns_int64_ids_and_float32_distances_of_shape_n_by_k():
    index = build("l2", random_vectors(200))
    ids, distances = index.knn_query(random_vectors(7, seed=1), k=5)
    assert ids.dtype == np.int64 and distances.dtype == np.float32
    assert ids.shape == distances.shape == (7, 5)
    assert ids.flags.c_contiguous and distances.flags.c_contiguous
    assert np.all(np.diff(distances, axis=1) >= 0)  # each row closest first


def test_one_dimensional_input_is_one_vector():
    base = random_vectors(50)
    index = vecdex.Index("l2", DIM)
    np.testing.assert_array_equal(index.add_items(base[0]), [0])
    ids, distances = index.knn_query(base[0], k=1)
    assert ids.shape == distances.shape == (1, 1)
    assert ids[0, 0] == 0 and distances[0, 0] == 0.0


def test_vector_finds_itself():
    base = random_vectors(300)
    index = build("l2", base)
    ids, distances = index.knn_query(base, k=1, ef=100)
    np.testing.assert_array_equal(ids[:, 0], np.arange(len(base)))
    np.testing.assert_array_equal(distances[:, 0], 0.0)


def test_fewer_than_k_results_are_padded_with_minus_one_and_inf():
    base = random_vectors(5)
    index = build("l2", base)
    ids, distances = index.knn_query(base[:2], k=8)
    assert sorted(ids[0, :5]) == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(ids[:, 5:], -1)
    np.testing.assert_array_equal(distances[:, 5:], np.inf)
    assert np.all(np.isfinite(distances[:, :5]))


def test_empty_index():
    index = vecdex.Index("cosine", DIM)
    assert len(index) == 0
    ids, distances = index.knn_query(random_vectors(3), k=4)
    np.testing.assert_array_equal(ids, np.full((3, 4), -1))
    np.testing.assert_array_equal(distances, np.full((3, 4), np.inf, dtype=np.float32))

    added = index.add_items(np.empty((0, DIM), dtype=np.float32))
    assert added.shape == (0,) and len(index) == 0
    ids, distances = index.knn_query(np.empty((0, DIM), dtype=np.float32), k=4)
    assert ids.shape == distances.shape == (0, 4)


def test_ef_defaults_to_64_and_knn_query_uses_it_unless_given_one():
    queries = random_vectors(50, seed=1)
    index = build("l2", random_vectors(1000), M=4, ef_construction=8)  # sparse: ef matters
    assert index.ef == 64
    index.set_ef(200)
    assert index.ef == 200
    index.ef = 10
    assert index.ef == 10
    narrow, _ = index.knn_query(queries, k=10)
    np.testing.assert_array_equal(narrow, index.knn_query(queries, k=10, ef=10)[0])
    wide, _ = index.knn_query(queries, k=10, ef=400)
    assert not np.array_equal(narrow, wide)  # so ef really reached the search


def test_same_seed_gives_the_same_results():
    base, queries = random_vectors(1000), random_vectors(50, seed=1)
    a = build("l2", base, M=8, ef_construction=40, seed=7)
    b = build("l2", base, M=8, ef_construction=40, seed=7)
    for x, y in zip(a.knn_query(queries, k=10, ef=16), b.knn_query(queries, k=10, ef=16)):
        np.testing.assert_array_equal(x, y)


def test_properties_and_repr():
    index = vecdex.Index("ip", 8, M=12, ef_construction=50)
    index.add_items(random_vectors(3, dim=8))
    assert (index.space, index.dim, index.M, index.ef_construction) == ("ip", 8, 12, 50)
    assert index.memory_bytes() >= 3 * 8 * 4  # at least the vectors themselves
    assert repr(index) == "<vecdex.Index space='ip' dim=8 M=12 ef_construction=50 ef=64 len=3>"
    assert isinstance(vecdex.__version__, str) and vecdex.__version__


# --- The input rules: float32, C-contiguous, aligned; never converted silently -------------


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int32, ">f4"])
def test_other_dtypes_are_rejected_with_type_error(dtype):
    index = vecdex.Index("l2", DIM)
    data = random_vectors(4).astype(dtype)
    with pytest.raises(TypeError, match=r"must be float32.*astype"):
        index.add_items(data)
    with pytest.raises(TypeError, match="must be float32"):
        index.knn_query(data)
    assert len(index) == 0


def test_objects_without_the_buffer_protocol_are_rejected_with_type_error():
    index = vecdex.Index("l2", 2)
    with pytest.raises(TypeError):
        index.add_items([[1.0, 2.0]])
    with pytest.raises(TypeError):
        index.knn_query([[1.0, 2.0]])


@pytest.mark.parametrize(
    "make",
    [
        lambda x: np.asfortranarray(x),
        lambda x: x[::2],  # every other row
        lambda x: np.repeat(x, 2, axis=1)[:, ::2],  # every other column of a wider array
    ],
    ids=["fortran-order", "row-stride", "column-stride"],
)
def test_non_contiguous_arrays_are_rejected(make):
    index = vecdex.Index("l2", DIM)
    data = make(random_vectors(8))
    assert data.shape[1] == DIM and not data.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        index.add_items(data)
    with pytest.raises(ValueError, match="C-contiguous"):
        index.knn_query(data)
    index.add_items(np.ascontiguousarray(data))  # the explicit copy the message suggests
    assert len(index) == len(data)


def test_unaligned_arrays_are_rejected():
    raw = bytearray(4 * DIM + 1)
    data = np.frombuffer(raw, dtype=np.float32, count=DIM, offset=1)  # one byte off
    assert not data.flags.aligned
    index = vecdex.Index("l2", DIM)
    with pytest.raises(ValueError, match="aligned"):
        index.add_items(data)
    index.add_items(data.copy())
    assert len(index) == 1


@pytest.mark.parametrize("shape", [(), (2, 2, DIM)], ids=["0-d", "3-d"])
def test_wrong_number_of_dimensions_is_rejected(shape):
    index = vecdex.Index("l2", DIM)
    data = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        index.add_items(data)
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        index.knn_query(data)


def test_dimension_mismatch_raises_value_error():
    index = build("l2", random_vectors(10))
    for data in (random_vectors(3, dim=DIM + 1), np.empty((0, DIM - 1), dtype=np.float32)):
        with pytest.raises(ValueError, match="dimension"):
            index.add_items(data)
        with pytest.raises(ValueError, match="dimension"):
            index.knn_query(data)
    assert len(index) == 10


def test_read_only_arrays_views_and_other_buffers_are_read_in_place():
    base = random_vectors(100)
    base.setflags(write=False)
    index = vecdex.Index("l2", DIM)
    index.add_items(base[10:20])  # a contiguous view into a read-only array
    np.testing.assert_array_equal(index.add_items(memoryview(base[:5])), np.arange(10, 15))
    ids, _ = index.knn_query(base[10:12], k=1)
    np.testing.assert_array_equal(ids[:, 0], [0, 1])


@pytest.mark.parametrize("space", SPACES)
def test_inputs_are_never_modified(space):
    # Cosine normalizes what it stores and what it searches for: always copies, never the
    # caller's arrays.
    base, queries = random_vectors(50), random_vectors(5, seed=1)
    base_before, queries_before = base.copy(), queries.copy()
    index = build(space, base)
    index.knn_query(queries, k=3)
    np.testing.assert_array_equal(base, base_before)
    np.testing.assert_array_equal(queries, queries_before)


# --- Errors: the C++ exceptions arrive as ValueError, and a rejected call changes nothing ---


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_values_raise_value_error_and_add_nothing(bad):
    index = build("l2", random_vectors(10))
    batch = random_vectors(5, seed=1)
    batch[4, 3] = bad  # the last row: the four before it must not be added either
    with pytest.raises(ValueError, match="NaN or inf"):
        index.add_items(batch)
    assert len(index) == 10
    with pytest.raises(ValueError, match="NaN or inf"):
        index.knn_query(batch)


def test_cosine_rejects_zero_vectors():
    index = vecdex.Index("cosine", DIM)
    with pytest.raises(ValueError, match="zero"):
        index.add_items(np.zeros((1, DIM), dtype=np.float32))
    index.add_items(random_vectors(3))
    with pytest.raises(ValueError, match="zero"):
        index.knn_query(np.zeros(DIM, dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"space": "euclidean"}, "space must be"),
        ({"dim": 0}, "dim must be positive"),
        ({"dim": -3}, "dim must be positive"),
        ({"M": 1}, "M must be at least 2"),  # from HnswIndex: std::invalid_argument
        ({"ef_construction": 0}, "ef_construction must be positive"),
    ],
)
def test_bad_constructor_arguments_raise_value_error(kwargs, message):
    args = {"space": "l2", "dim": DIM} | kwargs
    with pytest.raises(ValueError, match=message):
        vecdex.Index(**args)


def test_negative_seed_is_a_type_error():
    with pytest.raises(TypeError):
        vecdex.Index("l2", DIM, seed=-1)


def test_bad_k_and_ef_raise_value_error():
    index = build("l2", random_vectors(10))
    queries = random_vectors(2, seed=1)
    for kwargs in ({"k": 0}, {"k": -1}, {"ef": 0}):
        with pytest.raises(ValueError, match="must be positive"):
            index.knn_query(queries, **kwargs)
    with pytest.raises(ValueError, match="ef must be positive"):
        index.set_ef(0)
    with pytest.raises(ValueError, match="ef must be positive"):
        index.ef = -5
    assert index.ef == 64


# --- Lifetimes: results are NumPy-owned memory, never views into the index ---------------


def test_results_stay_valid_after_the_index_is_gone():
    base = random_vectors(100)
    index = build("l2", base)
    added = index.add_items(random_vectors(3, seed=2))
    ids, distances = index.knn_query(base[:4], k=3)
    expected = (added.copy(), ids.copy(), distances.copy())
    for array in (added, ids, distances):
        assert array.flags.owndata and array.base is None
    del index
    gc.collect()
    for array, copy in zip((added, ids, distances), expected):
        np.testing.assert_array_equal(array, copy)
