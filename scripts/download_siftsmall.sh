#!/usr/bin/env bash
# Downloads the siftsmall dataset (5 MB) into data/siftsmall/ and checks its SHA-256.
# Safe to re-run: does nothing if the dataset is already there.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f data/siftsmall/siftsmall_base.fvecs ]]; then
    echo "siftsmall is already in data/siftsmall/"
    exit 0
fi

readonly url="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
readonly sha256="b8f1e59b20319ac44279d5251706909dd3a5b8ca5ce2a11ddb1e73902252770e"
readonly archive="data/siftsmall.tar.gz"

mkdir -p data
curl --fail --silent --show-error --retry 3 --retry-delay 5 --output "$archive" "$url"
echo "$sha256  $archive" | shasum --algorithm 256 --check --quiet
tar -xzf "$archive" -C data
rm "$archive"
echo "siftsmall is ready in data/siftsmall/"
