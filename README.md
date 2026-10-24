# vecdex

[![CI](https://github.com/MatiiiBL/HNSW_vector_search/actions/workflows/ci.yml/badge.svg)](https://github.com/MatiiiBL/HNSW_vector_search/actions/workflows/ci.yml)

Approximate nearest-neighbor (ANN) vector search in C++20, built around
[HNSW](https://arxiv.org/abs/1603.09320) graphs. The finished library will have Python bindings,
serve as the retriever in a small RAG demo, and be benchmarked against hnswlib and FAISS.

**Status: work in progress.** Done so far:
- the exact brute-force baseline, which every approximate index will be measured against;
- SIMD distance kernels: NEON, and AVX2 selected at runtime.

| Milestone | State |
|---|---|
| Exact brute-force k-NN, `.fvecs`/`.ivecs` I/O, recall@k | ✅ Done |
| SIMD distance kernels (NEON, AVX2), runtime dispatch, CI | ✅ Done |
| HNSW build and search | Planned |
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
scripts/download_siftsmall.sh   # downloads into data/siftsmall/ and verifies the SHA-256
```

Without it, those tests are skipped rather than failed.

## Example

```cpp
#include <vecdex/vecdex.hpp>

auto base = vecdex::read_fvecs("data/siftsmall/siftsmall_base.fvecs");
vecdex::BruteForceIndex index(base.dim());
index.add(base);

// The 10 nearest stored vectors, closest first, as {squared distance, id}.
std::vector<vecdex::Neighbor> hits = index.search(base.row(0), 10);

// Other metrics: Metric::InnerProduct (1 - <a,b>) or Metric::Cosine (1 - cos).
vecdex::BruteForceIndex by_angle(base.dim(), vecdex::Metric::Cosine);
```

## Performance so far

Measured on an Apple M4 (single thread, Release build, median of 5 runs):

| | Scalar | NEON | Speedup |
|---|---|---|---|
| L2² distance, dim 128 | 28.7 ns | 5.7 ns | 5.0× |
| L2² distance, dim 768 | 310.9 ns | 37.3 ns | 8.3× |
| Exact search on siftsmall (10K × 128, k = 10) | 3,237 queries/s | 13,975 queries/s | 4.3× |

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
