"""Convert an ann-benchmarks HDF5 dataset to TEXMEX .fvecs/.ivecs files.

    python scripts/convert_ann_hdf5.py IN.hdf5 OUT_DIR NAME

Writes OUT_DIR/NAME_base.fvecs (the HDF5's `train` set), NAME_query.fvecs (`test`) and
NAME_groundtruth.ivecs (`neighbors`), the layout that vecdex_eval and bench/compare read, then
reads each file back and checks it against the HDF5.

Angular datasets such as GloVe get unit-length vectors. Between unit vectors, squared L2 is
2 - 2 cos, a decreasing function of the cosine similarity, so L2, inner product and cosine all
rank neighbors alike and the published ground truth holds for each of them; the L2-only
vecdex_eval can run the dataset. The HDF5's `distances` aren't converted: they're angular
distances, and both harnesses recompute the k-th true distance in their own metric instead.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def write_vecs(path: Path, rows: np.ndarray) -> None:
    """Writes 4-byte values as a TEXMEX file: each row is an int32 dimension, then the row."""
    records = np.empty((len(rows), rows.shape[1] + 1), dtype="<i4")
    records[:, 0] = rows.shape[1]
    records[:, 1:] = rows.view("<i4")  # the bytes of each float, unchanged
    records.tofile(path)


def read_vecs(path: Path, dtype: str) -> np.ndarray:
    """Reads a TEXMEX file back; the same checks as bench/compare/compare.py's reader."""
    words = np.fromfile(path, dtype="<i4")
    dim = int(words[0]) if words.size else 0
    if dim <= 0 or words.size % (dim + 1) != 0:
        raise ValueError(f"{path}: not an .fvecs/.ivecs file (bad dimension or size)")
    records = words.reshape(-1, dim + 1)
    if np.any(records[:, 0] != dim):
        raise ValueError(f"{path}: records have different dimensions")
    return np.ascontiguousarray(records[:, 1:]).view(dtype)


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError(f"{np.sum(norms == 0)} zero vectors have no direction to keep")
    return (vectors / norms).astype("<f4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("name")
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        metric = f.attrs["distance"]
        base = np.asarray(f["train"], dtype="<f4")
        queries = np.asarray(f["test"], dtype="<f4")
        truth = np.asarray(f["neighbors"], dtype="<i4")
    if metric not in ("angular", "euclidean"):
        raise SystemExit(f"{args.hdf5}: unsupported distance {metric!r}")
    if truth.shape[0] != queries.shape[0] or truth.min() < 0 or truth.max() >= len(base):
        raise SystemExit(f"{args.hdf5}: the neighbors don't match the train and test sets")
    if metric == "angular":
        base, queries = unit_rows(base), unit_rows(queries)

    outputs = {
        args.out_dir / f"{args.name}_base.fvecs": base,
        args.out_dir / f"{args.name}_query.fvecs": queries,
        args.out_dir / f"{args.name}_groundtruth.ivecs": truth,
    }
    # Written under temporary names and renamed only once all three check out, so a failed
    # run never leaves a complete-looking dataset behind.
    for path, rows in outputs.items():
        part = path.with_name(path.name + ".part")
        write_vecs(part, rows)
        if not np.array_equal(read_vecs(part, rows.dtype.str), rows):
            raise SystemExit(f"{part}: reads back differently from what was written")
        print(f"{path.name}: {rows.shape[0]} x {rows.shape[1]} ({metric}), read back intact")
    for path in outputs:
        path.with_name(path.name + ".part").rename(path)


if __name__ == "__main__":
    main()
