# Cross-library comparison: FAISS, hnswlib and vecdex from Python

[`compare.py`](compare.py) builds an HNSW index with each library and sweeps efSearch, printing
recall@10 and single-thread queries per second. It's the Python counterpart of the C++ harness
`vecdex_eval`: the same command-line shape, datasets and CSV columns, so
`scripts/plot_recall_qps.py` plots either. In Python every library is reached the same way, a
pip-installed extension module, and FAISS can join in.

## Setup

```sh
python3 -m venv bench/compare/.venv   # Python 3.13; .venv/ is gitignored
bench/compare/.venv/bin/pip install -r bench/compare/requirements.txt
scripts/download_sift1m.sh                                         # data/sift/
PYTHON=bench/compare/.venv/bin/python scripts/download_glove100.sh  # data/glove-100/
```

The vecdex rows need its Python bindings: `bench/compare/.venv/bin/pip install .` from the
repository root. Without them, vecdex is skipped with a message and the other libraries still
run.

## Usage

```sh
py=bench/compare/.venv/bin/python
$py bench/compare/compare.py data/sift sift --out-dir results/                      # SIFT1M, L2
$py bench/compare/compare.py data/glove-100 glove-100 --metric ip --out-dir results/ # GloVe, cosine
$py scripts/plot_recall_qps.py sift.png results/sift-*.csv --title "SIFT1M, M=16, efC=200"

# Smoke run: the first 10K base vectors and 100 queries, with exact ground truth for the subset.
$py bench/compare/compare.py data/glove-100 glove-100 --metric ip --max-base 10000 \
    --max-queries 100 --out-dir /tmp/smoke
```

Options: `--libs faiss-hnsw,hnswlib,vecdex`, `--M 16`, `--ef-construction 200`, `--k 10`,
`--ef 10,16,32,64,128,256,512`, `--repeats 5`. Each library writes
`<out-dir>/<name>-<lib>.csv` with the columns
`lib,dataset,n,dim,M,ef_construction,build_s,memory_mb,k,ef,recall,recall_dist,qps`. The rows
also go to stdout and the progress to stderr.

## Method

- **Same parameters everywhere:** M=16 (2M links on layer 0), efConstruction=200, and the same
  efSearch list. hnswlib gets seed 42, vecdex's default. FAISS has no seed parameter.
- **One thread, batched queries:** `faiss.omp_set_num_threads(1)`, and `num_threads=1` for
  hnswlib's build and search. vecdex's `add_items` and `knn_query` run on the calling thread. Each timed pass is
  one call over all queries, so Python's per-call overhead is spread across 10,000 queries
  equally for every library. QPS is the median of 5 passes after one untimed warmup pass.
  Like `vecdex_eval`, the script asks macOS for a performance core (QoS class).
- **Recall@10, two ways**, matching `src/recall.cpp`:
  - `recall`: the share of the true 10 nearest ids that come back.
  - `recall_dist`: the share of returned results no farther than the 10th true neighbor, plus
    a tolerance of 1e-5 × max(|d₁₀|, 1). Ties then count as hits (ann-benchmarks' definition).
    The libraries report distances in different units (FAISS: squared L2 or similarity,
    hnswlib and vecdex: squared L2 or 1 − ip), so the script recomputes every distance, found
    and true, from the vectors in float64.
- **Memory** is the built index's size as the library reports it: FAISS's serialized index,
  hnswlib's `index_file_size()` and vecdex's `memory_bytes()`. All three hold the full float32
  vectors plus the links.
- **Build time** is the wall time of the single `add` call.

## Datasets

| Name | Base × dim | Queries | Metric | Ground truth |
|---|---|---|---|---|
| `sift` | 1,000,000 × 128 | 10,000 | L2 | TEXMEX, 100 per query |
| `glove-100` | 1,183,514 × 100 | 10,000 | cosine (`--metric ip` on unit vectors) | ann-benchmarks, 100 per query |

`scripts/download_glove100.sh` downloads `glove-100-angular.hdf5` from ann-benchmarks (485 MB,
SHA-256 pinned) and `scripts/convert_ann_hdf5.py` turns it into `.fvecs`/`.ivecs`. The vectors
are scaled to unit length. ann-benchmarks computed the ground truth with the angular distance
1 − cos on the original vectors. Between unit vectors, squared L2 is 2 − 2 cos, so L2, inner
product and cosine all rank neighbors the same way and that ground truth holds for each. Exact
search on the converted vectors, under inner product or L2, reproduces it: recall@10 = 1.0 on
the first 100 queries. So `vecdex_eval data/glove-100 glove-100` works too, with L2.

## Caveats

- **hnswlib on arm64 has no SIMD.** Its distance code is SSE/AVX only (`#if defined(__SSE__)`),
  and pip builds it from source, so on Apple silicon it runs a scalar loop. D-012 describes the
  same trap in the C++ harness, which therefore also runs hnswlib with vecdex's kernel. Read
  hnswlib's Mac numbers as "hnswlib as a Mac user gets it", not as its graph code's speed.
- FAISS and vecdex do use NEON on arm64 (`faiss.get_compile_options()` lists `ARM_NEON`).
- Python adds one call per batch and the conversion of results to NumPy arrays: noise next to
  10,000 searches, but the absolute QPS isn't comparable with `vecdex_eval`, which calls search
  once per query in C++.
