"""vecdex.Index.save and vecdex.Index.load: round trips give identical results, the loaded index
keeps building as the original would, and files that can't be written or read raise
RuntimeError naming the path."""

import os
import pathlib
import re

import numpy as np
import pytest

import vecdex

DIM = 16
SPACES = ["l2", "ip", "cosine"]


def random_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


def build(space: str = "l2", n: int = 1000) -> vecdex.Index:
    index = vecdex.Index(space, DIM, M=8, ef_construction=40, seed=7)
    index.add_items(random_vectors(n))
    return index


def assert_same_results(a: vecdex.Index, b: vecdex.Index, queries: np.ndarray) -> None:
    for got, want in zip(a.knn_query(queries, k=10, ef=50), b.knn_query(queries, k=10, ef=50)):
        np.testing.assert_array_equal(got, want)


def mentions(path: pathlib.Path) -> str:
    """A pytest.raises match for an error message that names `path`."""
    return re.escape(str(path))


# --- Round trips --------------------------------------------------------------------------


@pytest.mark.parametrize("space", SPACES)
def test_round_trip_gives_identical_results(space, tmp_path):
    index = build(space, n=2000)
    path = tmp_path / "index.vdx"
    index.save(path)
    loaded = vecdex.Index.load(path)
    assert isinstance(loaded, vecdex.Index)
    assert (loaded.space, loaded.dim, loaded.M, loaded.ef_construction, len(loaded)) == (
        space, DIM, 8, 40, 2000)
    assert loaded.memory_bytes() == index.memory_bytes()
    # Same ids and bit-identical distances: the same stored vectors and graph, searched by the
    # same code.
    assert_same_results(loaded, index, random_vectors(200, seed=1))


def test_empty_index_round_trips(tmp_path):
    path = tmp_path / "empty.vdx"
    vecdex.Index("cosine", DIM).save(path)
    loaded = vecdex.Index.load(path)
    assert (loaded.space, len(loaded)) == ("cosine", 0)
    ids, _ = loaded.knn_query(random_vectors(1), k=2)
    np.testing.assert_array_equal(ids, [[-1, -1]])


def test_ef_is_not_saved_and_starts_at_the_default(tmp_path):
    index = build()
    index.ef = 200
    path = tmp_path / "index.vdx"
    index.save(path)
    assert vecdex.Index.load(path).ef == 64


def test_adding_to_a_loaded_index_builds_what_adding_to_the_original_does(tmp_path):
    # load() restores the level generator, so the same seed keeps drawing the same levels.
    index = build()
    path = tmp_path / "index.vdx"
    index.save(path)
    loaded = vecdex.Index.load(path)
    more = random_vectors(500, seed=2)
    np.testing.assert_array_equal(loaded.add_items(more), np.arange(1000, 1500))
    index.add_items(more)
    assert_same_results(loaded, index, random_vectors(200, seed=3))


def test_saving_again_replaces_the_file(tmp_path):
    index = build()
    path = tmp_path / "index.vdx"
    index.save(path)
    index.add_items(random_vectors(10, seed=2))
    index.save(path)
    assert len(vecdex.Index.load(path)) == 1010
    assert [p.name for p in tmp_path.iterdir()] == ["index.vdx"]  # no temporary file left


@pytest.mark.parametrize("as_path", [str, pathlib.Path], ids=["str", "pathlib.Path"])
def test_paths_may_be_str_or_path_like(as_path, tmp_path):
    index = build()
    path = as_path(tmp_path / "index.vdx")
    index.save(path)
    assert len(vecdex.Index.load(path)) == 1000


def test_paths_of_other_types_are_a_type_error():
    with pytest.raises(TypeError):
        vecdex.Index.load(42)


# --- Errors -------------------------------------------------------------------------------


def test_loading_a_missing_file_raises_runtime_error(tmp_path):
    path = tmp_path / "missing.vdx"
    with pytest.raises(RuntimeError, match=mentions(path)):
        vecdex.Index.load(path)


@pytest.mark.parametrize("keep", [0.5, 0.99], ids=["half", "all but the end"])
def test_loading_a_truncated_file_raises_runtime_error(keep, tmp_path):
    path = tmp_path / "index.vdx"
    build().save(path)
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * keep)])
    with pytest.raises(RuntimeError, match=mentions(path) + ".*truncated"):
        vecdex.Index.load(path)


def test_loading_a_corrupted_file_raises_runtime_error(tmp_path):
    path = tmp_path / "index.vdx"
    build().save(path)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01  # one bit, somewhere in the graph or the vectors
    path.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match=mentions(path) + ".*checksum"):
        vecdex.Index.load(path)


@pytest.mark.parametrize(
    "content", [b"", b"not an index\n" * 100, np.random.default_rng(0).bytes(4096)],
    ids=["empty", "text", "random"],
)
def test_loading_another_kind_of_file_raises_runtime_error(content, tmp_path):
    path = tmp_path / "index.vdx"
    path.write_bytes(content)
    with pytest.raises(RuntimeError, match=mentions(path)):
        vecdex.Index.load(path)


def test_saving_into_a_missing_directory_raises_runtime_error(tmp_path):
    path = tmp_path / "missing" / "index.vdx"
    with pytest.raises(RuntimeError, match=mentions(path)):
        build().save(path)
    assert not path.parent.exists()


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="needs POSIX permissions that apply (root may write anywhere)")
def test_saving_into_a_read_only_directory_raises_runtime_error(tmp_path):
    directory = tmp_path / "read-only"
    directory.mkdir()
    directory.chmod(0o500)
    try:
        with pytest.raises(RuntimeError, match=mentions(directory / "index.vdx")):
            build().save(directory / "index.vdx")
        assert list(directory.iterdir()) == []
    finally:
        directory.chmod(0o700)  # so pytest can clean up
