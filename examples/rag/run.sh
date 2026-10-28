#!/usr/bin/env bash
# The RAG demo in one command, from a clean clone: creates examples/rag/.venv, installs the
# pinned requirements, downloads the corpus (SQuAD v1.1 dev, 4.9 MB, SHA-256 checked) into
# data/squad/, embeds it (cached after the first run), and runs the retrieval demo.
# Arguments go to demo.py. Set PYTHON to build the venv with another interpreter than python3.
set -euo pipefail

cd "$(dirname "$0")/../.."

readonly venv="examples/rag/.venv"
if [[ ! -x $venv/bin/python ]]; then
    "${PYTHON:-python3}" -m venv "$venv"
fi
readonly python="$venv/bin/python"
"$python" -m pip install --quiet --disable-pip-version-check \
    --requirement examples/rag/requirements.txt

# Once the repository has Python bindings, install them too, so the demo also runs vecdex.
# After changing the C++ code, refresh them with: examples/rag/.venv/bin/pip install .
if [[ -f pyproject.toml ]] && ! "$python" -c "import vecdex" 2>/dev/null; then
    "$python" -m pip install --quiet --disable-pip-version-check .
fi

readonly corpus="data/squad/dev-v1.1.json"
if [[ ! -f $corpus ]]; then
    mkdir -p "$(dirname "$corpus")"
    curl --fail --silent --show-error --retry 3 --retry-delay 5 --output "$corpus.part" \
        https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
    echo "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9  $corpus.part" |
        shasum --algorithm 256 --check --quiet
    mv "$corpus.part" "$corpus"
fi

# Two processes: PyTorch (embedding) and FAISS (retrieval) each bundle an OpenMP runtime.
"$python" examples/rag/embed.py
"$python" examples/rag/demo.py "$@"
