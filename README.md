# vecdex

[![CI](https://github.com/MatiiiBL/HNSW_vector_search/actions/workflows/ci.yml/badge.svg)](https://github.com/MatiiiBL/HNSW_vector_search/actions/workflows/ci.yml)

Approximate nearest-neighbor (ANN) vector search in C++20, built around
[HNSW](https://arxiv.org/abs/1603.09320) graphs. The finished library will have Python bindings,
serve as the retriever in a small RAG demo, and be benchmarked against hnswlib and FAISS.

**Status: work in progress.** Done so far:
- an HNSW index;
- SIMD distance kernels: NEON, and AVX2 selected at runtime;
- an exact brute-force baseline to measure it against.

| Milestone | State |
|---|---|
| Exact brute-force k-NN, `.fvecs`/`.ivecs` I/O, recall@k | ✅ Done |
| SIMD distance kernels (NEON, AVX2), runtime dispatch, CI | ✅ Done |
| HNSW build and search | ✅ Done |
| Benchmark harness, profiling, comparison with hnswlib | Planned |
| Save/load to disk | Planned |
| Multithreaded construction | Planned |
| Python bindings (pybind11) | Planned |
| RAG demo; final benchmarks against hnswlib and FAISS | Planned |

## Build and test

Requires CMake ≥ 3.25, Ninja, and a C++20 compiler. Developed with Apple clang 21 on arm64.
GoogleTest is downloaded automatically.

```sh
cmake --workflow --preset asan     # Debug + AddressSanitizer + UBSan: the default dev loop
cmake --workflow --preset debug    # Debug, no sanitizers
cmake --workflow --preset release  # -O3
```

Each workflow configures, builds, and runs the tests, in `build/<preset>/`. The code builds
with `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror`.

## Dataset

The end-to-end tests use [siftsmall](http://corpus-texmex.irisa.fr/) (10,000 base vectors and
100 queries, 128 dimensions, with ground truth). It isn't committed. Download it with:

```sh
scripts/download_siftsmall.sh   # 5 MB, into data/siftsmall/, verifies the SHA-256
scripts/download_sift1m.sh      # 160 MB (580 MB unpacked), into data/sift/, for benchmarks
```

Without siftsmall, those tests are skipped rather than failed.

## Example

```cpp
#include <vecdex/vecdex.hpp>

auto base = vecdex::read_fvecs("data/siftsmall/siftsmall_base.fvecs");

vecdex::HnswIndex index(base.dim(), vecdex::Metric::L2, {.M = 16, .ef_construction = 200});
index.add(base);

// The 10 approximate nearest neighbors, closest first, as {squared distance, id}.
// ef (here 64) is the recall-versus-speed dial: higher is more accurate and slower.
std::vector<vecdex::Neighbor> hits = index.search(base.row(0), 10, 64);

// Exact search, for ground truth: vecdex::BruteForceIndex exact(base.dim()); ...
// Other metrics: Metric::InnerProduct (1 - <a,b>) or Metric::Cosine (1 - cos).
```

## Performance so far

Measured on an Apple M4 (single thread, Release build, median of 5 runs):

| | Scalar | NEON | Speedup |
|---|---|---|---|
| L2² distance, dim 128 | 28.7 ns | 5.7 ns | 5.0× |
| L2² distance, dim 768 | 310.9 ns | 37.3 ns | 8.3× |
| Exact search on siftsmall (10K × 128, k = 10) | 3,237 queries/s | 13,975 queries/s | 4.3× |

**HNSW on SIFT1M** (1M × 128; M=16, efConstruction=200; single thread; recall@10 against the
published ground truth):

| ef | recall@10 | queries/s | vs. exact search (130 queries/s) |
|---|---|---|---|
| 32 | 0.903 | 15,683 | 121× |
| 64 | 0.963 | 9,031 | 69× |
| 128 | 0.989 | 5,065 | 39× |
| 512 | 0.999 | 1,558 | 12× |

Build: 246 s single-threaded; index size 674 MB. These numbers aren't yet compared with hnswlib
or FAISS on the same machine; that's the next milestone. Reproduce them with
`build/release/bench/vecdex_eval data/sift sift`.

To be fair to the compiler: once it's allowed to reorder floating-point additions (the effect of
`-ffast-math`), its auto-vectorized loop comes within 1–6% of the hand-written NEON kernels. The
hand-written kernels give that speed by default, without `-ffast-math`, and they let x86-64
builds switch to AVX2 at runtime.

Run the benchmarks yourself with `cmake --workflow --preset release`, then
`build/release/bench/vecdex_bench`. Setting `VECDEX_ISA=scalar` forces the scalar kernels for
comparison.

## Design notes so far

- **Contiguous storage.** All vectors live in one row-major buffer, not one allocation per
  vector, so a scan reads memory sequentially.
- **Squared L2.** `sqrt` is monotonic, so ranking by squared distance gives the same order
  without computing it.
- **SIMD kernels with runtime dispatch.** There are scalar, NEON, and AVX2+FMA kernels, each
  using four independent accumulators so the FMA units stay busy. NEON is always present on
  ARM64, so it's chosen at compile time. On x86-64 the AVX2 code is compiled per function
  (`[[gnu::target]]`) and enabled only if the CPU supports it, so one binary runs everywhere.
- **Cosine through normalization.** Vectors are normalized once, when stored, so a cosine
  query is just an inner product.
- **HNSW** (Malkov & Yashunin, 2018), following the paper's algorithms with hnswlib's choices
  where the paper leaves room:
  - The *heuristic* neighbor selection, which keeps links diverse on clustered data.
  - Layer-0 links stored in one flat array of fixed-size blocks, instead of one allocation per
    node.
  - An epoch-stamped visited list, borrowed from a pool, so concurrent searches are safe and
    starting a search is O(1).
  - A portable level generator, so a seed builds the same graph on every platform.
  - A graph inspector that checks the invariants (degree bounds, no self-loops or duplicate
    links) and connectivity.
- **Top-k with a bounded max-heap:** O(n log k) time and O(k) memory per query. Ties are broken
  by smaller id, so results are deterministic.
- **Input is validated at the API boundary.** Dimension mismatches, `k == 0`, and NaN/infinite
  values throw `std::invalid_argument`, and a rejected call leaves the index unchanged. File
  readers derive sizes from the file length rather than trusting headers, so a corrupted file
  can't trigger a huge allocation.
- **Sanitizer-checked, on two architectures.** CI runs the tests:
  - on Linux x86-64 with GCC and Clang, under ASan and UBSan;
  - under Valgrind memcheck;
  - on macOS arm64.

  The suite includes tests that prove the sanitizers are actually active, and a check that the
  SIMD path really ran.

## License

MIT
