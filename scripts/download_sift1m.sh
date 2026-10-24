#!/usr/bin/env bash
# Downloads SIFT1M (1M base vectors, 10K queries, dim 128; about 160 MB, 580 MB unpacked) into
# data/sift/ and checks its SHA-256. Safe to re-run: does nothing if the dataset is already there.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f data/sift/sift_base.fvecs ]]; then
    echo "SIFT1M is already in data/sift/"
    exit 0
fi

readonly url="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
readonly sha256="92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a"
readonly archive="data/sift.tar.gz"

mkdir -p data
curl --fail --show-error --retry 3 --retry-delay 5 --output "$archive" "$url"
echo "$sha256  $archive" | shasum --algorithm 256 --check --quiet
tar -xzf "$archive" -C data
rm "$archive"
echo "SIFT1M is ready in data/sift/"
