# vecdex for Python

NumPy bindings for vecdex, an approximate nearest-neighbor (HNSW) index written in C++20. The
API follows hnswlib's.

## Install

From the repository root, in a virtual environment (needs CMake ≥ 3.25 and a C++20 compiler):

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install .                    # builds the extension; pybind11 comes with the build
pip install ".[test]" && pytest  # the same, plus the test suite
```

## Quick start

```python
import numpy as np
import vecdex

vectors = np.random.default_rng(0).standard_normal((10_000, 128), dtype=np.float32)

index = vecdex.Index("cosine", dim=128, M=16, ef_construction=200)
ids = index.add_items(vectors, num_threads=4)           # int64: 0, 1, ..., 9999
labels, distances = index.knn_query(vectors[:5], k=10)  # both of shape (5, 10)
```

## API

| | |
|---|---|
| `Index(space, dim, M=16, ef_construction=200, seed=42)` | `space` is `"l2"`, `"ip"` or `"cosine"` |
| `add_items(data, *, num_threads=1) -> ids` | adds vectors; ids are consecutive from `len(index)` |
| `knn_query(data, k=1, ef=None, *, num_threads=1) -> (ids, distances)` | `ef=None` uses `index.ef` |
| `ef`, `set_ef(ef)` | search breadth, the recall-versus-speed dial (default 64) |
| `len(index)`, `dim`, `space`, `M`, `ef_construction` | |
| `memory_bytes()` | bytes held by vectors and graph links |

## Conventions

- **Input is float32, C-contiguous and aligned,** shaped `(n, dim)`, or `(dim,)` for one
  vector. It is read in place through the buffer protocol (NumPy arrays and their contiguous
  views, `memoryview`, ...), with no copy: `add_items` needs no memory beyond what the index
  itself grows by.
- **Nothing is converted silently.** Any other dtype, float64 included, raises `TypeError`; a
  non-contiguous or unaligned array raises `ValueError`. The error names the one-line fix
  (`data.astype(np.float32)`, `np.ascontiguousarray(data)`), so a copy only happens where you
  can see it.
- **A rejected batch adds nothing:** `add_items` checks every row before it stores any.
- **Results** are new NumPy arrays: ids `int64`, distances `float32`, shape `(n, k)`, each row
  closest first. They own their memory and stay valid after the index is gone.
- **Fewer than k neighbors** (when `k > len(index)`): the row ends with id `-1` and distance
  `inf`, as in FAISS.
- **Distances:** `"l2"` is the *squared* Euclidean distance, `"ip"` is `1 - <q, x>`, and
  `"cosine"` is `1 - cos(q, x)`. Cosine normalizes copies of your vectors, never your arrays.
- **Errors:** `ValueError` for bad arguments and values (dimension mismatch, NaN or inf, a zero
  vector under cosine, `k < 1`, `num_threads < 0`), `TypeError` for wrong types.

## Threads

- **One call on several threads:** `num_threads` spreads `add_items` or `knn_query` over that
  many threads; `0` means one per hardware thread. It is keyword-only, and 1 by default, so a
  call only takes more cores when asked to.
  - `knn_query` returns the same results for any `num_threads`.
  - `add_items` with `num_threads=1` builds the same graph as adding the vectors one at a time,
    on every run (with the same `seed`). More threads build an equally good graph faster, but it
    depends on how the threads interleave, so it can differ from run to run.
- **Several Python threads on one index:** `add_items` and `knn_query` release the GIL.
  Queries from several threads run in parallel; an `add_items` waits for running queries and
  runs alone. Don't modify an array while a call that reads it is running.

Saving and loading will follow.

## Development

An editable install that rebuilds the extension when it is imported after a C++ change:

```sh
pip install scikit-build-core pybind11==3.1.0 numpy pytest
pip install --no-build-isolation -Cbuild-dir=build/python -Ceditable.rebuild=true -e .
pytest
```

`python/bench/binding_overhead.py` compares `knn_query` throughput, one query per call and
batched, with the C++ harness `vecdex_eval` on the same data:

```sh
python python/bench/binding_overhead.py data/siftsmall siftsmall --cpp build/release/bench/vecdex_eval
```
