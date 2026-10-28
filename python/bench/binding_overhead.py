"""Measure what the Python binding costs: vecdex.Index.knn_query throughput with one query per
call and with every query in one call, next to vecdex_eval (the C++ harness) on the same
dataset and build parameters.

    python python/bench/binding_overhead.py DIR NAME [--cpp build/release/bench/vecdex_eval]
        [--M 16] [--ef-construction 200] [--k 10] [--ef 10,16,32,64,128,256,512] [--repeats 5]

Reads DIR/NAME_base.fvecs, NAME_query.fvecs and NAME_groundtruth.ivecs, and times like
vecdex_eval: one thread, one untimed warmup pass, then the median of --repeats timed passes
over all queries. stdout gets CSV rows with vecdex_eval's columns (lib vecdex-python-batch and
vecdex-python-single, plus vecdex_eval's own rows with --cpp), which scripts/plot_recall_qps.py
can plot; stderr gets a table of the overhead per query.
"""

import argparse
import csv
import ctypes
import io
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import vecdex

COLUMNS = "lib,dataset,n,dim,M,ef_construction,build_s,memory_mb,k,ef,recall,recall_dist,qps"


def read_vecs(path: Path, dtype: type) -> np.ndarray:
    """A TEXMEX .fvecs/.ivecs file: every row is an int32 dimension d, then d 4-byte values."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0]) if raw.size else 0
    if dim <= 0 or raw.size % (dim + 1) != 0:
        raise SystemExit(f"{path}: not a .fvecs/.ivecs file")
    rows = raw.reshape(-1, dim + 1)
    if np.any(rows[:, 0] != dim):
        raise SystemExit(f"{path}: rows have different dimensions")
    return np.ascontiguousarray(rows[:, 1:]).view(dtype)  # drop the dimension column


def prefer_performance_cores() -> None:
    """Like vecdex_eval: ask macOS for the user-interactive QoS class, which steers this thread,
    and the C++ code it calls, to a performance core."""
    if sys.platform != "darwin":
        return
    qos_class_user_interactive = 0x21
    if ctypes.CDLL(None).pthread_set_qos_class_self_np(qos_class_user_interactive, 0) != 0:
        print("warning: could not raise the thread's QoS class", file=sys.stderr)


def median_seconds(run: Callable[[], object], repeats: int) -> float:
    """vecdex_eval's timing: an untimed warmup pass, then the median of `repeats` passes."""
    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def recalls(ids, distances, truth, kth_distance, k: int) -> tuple[float, float]:
    """Id-based and distance-threshold recall@k, as defined in src/recall.cpp."""
    by_id = np.mean([np.isin(true[:k], found).sum() / k for found, true in zip(ids, truth)])
    threshold = kth_distance + 1e-5 * np.maximum(np.abs(kth_distance), 1.0)
    by_distance = np.mean((distances <= threshold[:, None]).sum(axis=1) / k)
    return float(by_id), float(by_distance)


def run_cpp(binary: Path, args: argparse.Namespace) -> dict[int, dict[str, str]]:
    """vecdex_eval's CSV rows for the same sweep, by ef. Its log goes to our stderr."""
    command = [str(binary), str(args.dir), args.name, "--M", str(args.M),
               "--ef-construction", str(args.ef_construction), "--k", str(args.k),
               "--ef", ",".join(map(str, args.ef)), "--repeats", str(args.repeats)]
    print("running", " ".join(command), file=sys.stderr)
    output = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True).stdout
    return {int(row["ef"]): row for row in csv.DictReader(io.StringIO(output))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path)
    parser.add_argument("name")
    parser.add_argument("--cpp", type=Path, help="vecdex_eval's path, to run it afterwards")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", type=lambda s: [int(x) for x in s.split(",")],
                        default=[10, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefer_performance_cores()
    prefix = args.dir / args.name
    base = read_vecs(Path(f"{prefix}_base.fvecs"), np.float32)
    queries = read_vecs(Path(f"{prefix}_query.fvecs"), np.float32)
    truth = read_vecs(Path(f"{prefix}_groundtruth.ivecs"), np.int32)
    k = args.k
    if truth.shape[0] < len(queries) or truth.shape[1] < k:
        raise SystemExit(f"ground truth is {truth.shape}, need at least ({len(queries)}, {k})")
    # The k-th true neighbor's squared distance, for threshold recall (exact on SIFT's integers).
    kth_neighbor = base[truth[: len(queries), k - 1]].astype(np.float64)
    kth_distance = ((queries.astype(np.float64) - kth_neighbor) ** 2).sum(axis=1)
    print(f"vecdex {vecdex.__version__}, numpy {np.__version__}, Python "
          f"{sys.version.split()[0]}; {len(base)} x {base.shape[1]} base vectors, "
          f"{len(queries)} queries", file=sys.stderr)

    index = vecdex.Index("l2", base.shape[1], M=args.M, ef_construction=args.ef_construction)
    start = time.perf_counter()
    index.add_items(base)
    build_s = time.perf_counter() - start
    # The base vectors plus the index: add_items reads the batch in place. ru_maxrss is in bytes
    # on macOS and in KiB on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 1e6 if sys.platform == "darwin" else peak * 1024 / 1e6
    print(f"built in {build_s:.1f} s; peak RSS {peak_mb:.0f} MB", file=sys.stderr)

    print(COLUMNS)
    shared = [args.name, len(base), base.shape[1], args.M, args.ef_construction,
              f"{build_s:.1f}", f"{index.memory_bytes() / 1e6:.0f}", k]
    rows = []  # (ef, recall, batch q/s, single q/s)
    for ef in args.ef:
        ids, distances = index.knn_query(queries, k=k, ef=ef)
        for i, query in enumerate(queries):
            single_ids, single_distances = index.knn_query(query, k=k, ef=ef)
            if not (np.array_equal(single_ids[0], ids[i])
                    and np.array_equal(single_distances[0], distances[i])):
                raise SystemExit(f"ef {ef}: query {i} differs between batch and single calls")
        recall, recall_dist = recalls(ids, distances, truth, kth_distance, k)

        # Both close over this iteration's ef and run before the next one.
        def batch() -> None:
            index.knn_query(queries, k=k, ef=ef)

        def single() -> None:
            for query in queries:
                index.knn_query(query, k=k, ef=ef)

        qps = {}
        for mode, run in (("batch", batch), ("single", single)):
            qps[mode] = len(queries) / median_seconds(run, args.repeats)
            print(f"vecdex-python-{mode}", *shared, ef, f"{recall:.4f}", f"{recall_dist:.4f}",
                  f"{qps[mode]:.0f}", sep=",", flush=True)
        rows.append((ef, recall, qps["batch"], qps["single"]))

    cpp = run_cpp(args.cpp, args) if args.cpp else {}
    for ef in args.ef:
        if ef in cpp:
            print(",".join(cpp[ef][column] for column in COLUMNS.split(",")))

    # Microseconds per query above C++'s: what crossing the binding costs.
    header = "\n  ef  recall   batch q/s  single q/s"
    print(header + ("    C++ q/s  batch +us/q  single +us/q" if cpp else ""), file=sys.stderr)
    for ef, recall, batch_qps, single_qps in rows:
        line = f"{ef:4d}  {recall:.4f}  {batch_qps:10.0f}  {single_qps:10.0f}"
        if ef in cpp:
            cpp_qps = float(cpp[ef]["qps"])
            line += (f"  {cpp_qps:9.0f}  {1e6 / batch_qps - 1e6 / cpp_qps:11.2f}"
                     f"  {1e6 / single_qps - 1e6 / cpp_qps:12.2f}")
            if f"{recall:.4f}" != cpp[ef]["recall"]:
                line += "  (recall differs from C++'s: a different graph?)"
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
