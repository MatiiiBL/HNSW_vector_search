"""Plot recall-vs-QPS curves from vecdex_eval CSV output.

    python scripts/plot_recall_qps.py OUT.png [LABEL=]RESULTS.csv ... [--title TEXT]

Each CSV becomes one line, labeled LABEL or, without one, by its `lib` column. Recall is the
distance-threshold recall (`recall_dist`), as in ann-benchmarks. Up and to the right is better.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to a file; no display needed
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)


def read_series(spec: str) -> tuple[str, list[float], list[float]]:
    label, sep, path = spec.partition("=")
    if not sep:
        label, path = "", spec
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    rows.sort(key=lambda row: int(row["ef"]))
    recall = [float(row["recall_dist"]) for row in rows]
    qps = [float(row["qps"]) for row in rows]
    return label or rows[0]["lib"], recall, qps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("series", nargs="+", help="[LABEL=]path/to/results.csv")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for spec in args.series:
        label, recall, qps = read_series(spec)
        ax.plot(recall, qps, marker="o", markersize=4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (distance threshold)")
    ax.set_ylabel("queries per second, 1 thread (log scale)")
    ax.set_title(args.title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
