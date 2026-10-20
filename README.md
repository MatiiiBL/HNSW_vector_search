# vecdex

Approximate nearest-neighbor (ANN) vector search in C++20, built around
[HNSW](https://arxiv.org/abs/1603.09320) graphs. The finished library will have Python bindings,
serve as the retriever in a small RAG demo, and be benchmarked against hnswlib and FAISS.

**Status: work in progress.** The exact brute-force baseline is done: every approximate index
will be measured against it.

| Milestone | State |
|---|---|
| Exact brute-force k-NN, `.fvecs`/`.ivecs` I/O, recall@k | ✅ Done |
| SIMD distance kernels (NEON, AVX2) and CI | Planned |
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
mkdir -p data
curl -o data/siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf data/siftsmall.tar.gz -C data
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
```

## Design notes so far

- **Contiguous storage.** All vectors live in one row-major buffer, not one allocation per
  vector, so a scan reads memory sequentially.
- **Squared L2.** `sqrt` is monotonic, so ranking by squared distance gives the same order
  without computing it.
- **Top-k with a bounded max-heap:** O(n log k) time and O(k) memory per query. Ties are broken
  by smaller id, so results are deterministic.
- **Input is validated at the API boundary.** Dimension mismatches, `k == 0`, and NaN/infinite
  values throw `std::invalid_argument`, and a rejected call leaves the index unchanged. File
  readers derive sizes from the file length rather than trusting headers, so a corrupted file
  can't trigger a huge allocation.
- **Sanitizer-checked.** The test suite runs under ASan and UBSan, and it includes tests that
  prove the sanitizers are actually active.

## License

MIT
