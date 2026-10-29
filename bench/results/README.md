# Benchmark results

The raw output behind the numbers in the top-level README. Every file is a CSV with the columns
`lib,dataset,n,dim,M,ef_construction,build_s,memory_mb,k,ef,recall,recall_dist,qps` (plus
`threads` for the C++ harness): one row per efSearch value. `recall` is recall@10 by id;
`recall_dist` is by distance threshold (ann-benchmarks' definition) and is the one plotted.

## Where each file came from

All runs are SIFT1M (1M × 128, L2, 10,000 queries), M = 16, efConstruction = 200, k = 10, on a
13" MacBook Air M4 (4P + 6E cores, 16 GB, macOS 27, Apple clang 21, Release `-O3`).

| Files | Tool | Code |
|---|---|---|
| `sift1m/cpp-vecdex.csv`, `cpp-hnswlib*.csv` | `bench/hnsw_eval.cpp` (`vecdex_eval --lib ...`) | `2ba3349` |
| `sift1m/cpp-vecdex-m3-baseline.csv` | `vecdex_eval` | `86bdfe7` (before the M4 optimizations) |
| `sift1m/cpp-vecdex-pooled-scratch.csv` | `vecdex_eval` | `a9aabd4` |
| `sift1m/cpp-vecdex-prefetch.csv` | `vecdex_eval` | `3051c93` |
| `sift1m/cpp-vecdex-threads-N.csv` | `vecdex_eval --threads N --ef 64,128` | `2ba3349` |
| `sift1m/sift-{faiss-hnsw,hnswlib,vecdex}.csv` | `bench/compare/compare.py` (Python, batched, 1 thread) | `2ba3349`, faiss-cpu 1.15.1, hnswlib 0.8.0 (pip) |

Plots: `sift-python.png` (FAISS, hnswlib, vecdex from Python), `sift1m-cpp.png` (the C++
harness), `sift1m-optimizations.png` (vecdex before and after each M4 optimization). They're made
by `scripts/plot_recall_qps.py`.

## Conditions

- **One benchmark at a time,** plugged in, with no builds or other workloads running. The
  load average was logged before every run.
- **The optimization runs** (baseline, pooled scratch, prefetch, final) ran on 2026-10-28 between
  00:26 and 01:08. Each version ran twice, in the order A B C D D C B A so slow thermal drift on
  the fanless chip cancels out; the two runs of each version agree within 2%. Only the first run
  of each is in this folder.
- **The library comparisons, thread scaling and Python runs** ran on 2026-10-29 between 14:39 and
  15:40. The lid was closed but the Mac was plugged in, with `caffeinate` holding it awake
  (display off). A control run in that window gave vecdex 17,935 QPS at ef = 64 against 17,972 in
  the night run (0.2% apart), so the display-off state didn't throttle anything.
- **Runs the machine slept through were thrown out,** checked against `pmset -g log`. That
  includes a first attempt at GloVe-100, which isn't here yet.

## Reproducing

```sh
cmake --workflow --preset release
build/release/bench/vecdex_eval data/sift sift                                  # vecdex
build/release/bench/vecdex_eval data/sift sift --lib hnswlib-vecdex-kernel      # same kernel
build/release/bench/vecdex_eval data/sift sift --lib hnswlib                    # stock
build/release/bench/vecdex_eval data/sift sift --threads 4 --ef 64,128

python3 -m venv bench/compare/.venv
bench/compare/.venv/bin/pip install -r bench/compare/requirements.txt && bench/compare/.venv/bin/pip install .
for lib in faiss-hnsw hnswlib vecdex; do
    bench/compare/.venv/bin/python bench/compare/compare.py data/sift sift --libs $lib --out-dir out
done
```
