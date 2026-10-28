#!/usr/bin/env bash
# Downloads GloVe-100-angular from ann-benchmarks (1,183,514 base vectors, 10K queries, dim 100;
# 485 MB) into data/glove-100/, checks its SHA-256, and converts it to glove-100_base.fvecs,
# glove-100_query.fvecs and glove-100_groundtruth.ivecs with unit-length vectors (see
# scripts/convert_ann_hdf5.py for why). The conversion needs numpy and h5py: run it with
# PYTHON=bench/compare/.venv/bin/python, or any Python that has them.
# Safe to re-run: does nothing if the converted files are already there.
set -euo pipefail

cd "$(dirname "$0")/.."

readonly dir="data/glove-100"
if [[ -f $dir/glove-100_base.fvecs && -f $dir/glove-100_query.fvecs &&
      -f $dir/glove-100_groundtruth.ivecs ]]; then
    echo "GloVe-100 is already in $dir/"
    exit 0
fi

readonly url="https://ann-benchmarks.com/glove-100-angular.hdf5"
readonly sha256="544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4"
readonly hdf5="$dir/glove-100-angular.hdf5"

mkdir -p "$dir"
# The HDF5 survives a failed conversion, so a re-run needn't download it again. It only gets
# its final name once its checksum matches.
if [[ ! -f $hdf5 ]]; then
    curl --fail --show-error --retry 3 --retry-delay 5 --output "$hdf5.part" "$url"
    echo "$sha256  $hdf5.part" | shasum --algorithm 256 --check --quiet
    mv "$hdf5.part" "$hdf5"
fi
"${PYTHON:-python3}" scripts/convert_ann_hdf5.py "$hdf5" "$dir" glove-100
rm "$hdf5"
echo "GloVe-100 is ready in $dir/"
