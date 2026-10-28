# RAG demo: question answering over Wikipedia, retrieval only

A retrieval-augmented generation (RAG) pipeline answers a question in two steps: *retrieve* the
passages most similar to it, then have a language model *generate* an answer from them. This
demo builds the retrieval half, which is the part vecdex does. It embeds Wikipedia passages and
questions with a sentence-embedding model, indexes the passages in three interchangeable
retrievers, and checks the approximate ones against exact search. The top passages it prints are
the context a generator would receive. Generation itself is left out, so the demo needs no API
keys, accounts or paid services.

```sh
examples/rag/run.sh                # from the repository root of a clean clone
examples/rag/run.sh --ef 16 --k 5  # arguments go to demo.py (see demo.py --help)
```

The first run creates `examples/rag/.venv` (Python 3.13, about 1.1 GB, mostly PyTorch) and
installs the pinned `requirements.txt`. It downloads the corpus (4.9 MB, SHA-256 checked) into
`data/squad/` and the embedding model (91 MB) into the Hugging Face cache, then embeds everything
in about a minute on an M4 (on its GPU, through PyTorch's MPS backend). Later runs reuse the
cached embeddings and take a few seconds. Set `PYTHON` to build the venv with another
interpreter. The Hugging Face library suggests setting `HF_TOKEN`; the model is public and the
demo doesn't need one.

## What it reports

```
retriever     recall@10 vs exact   right passage at #1   in top 10
faiss-flat                1.0000                0.6229      0.9167
faiss-hnsw                0.9985                0.6226      0.9156
vecdex                    0.9980                0.6224      0.9149
```

- **recall@10 vs exact:** the share of exact search's top 10 that an approximate retriever also
  returns, averaged over all 10,570 questions. This checks the index, not the embeddings.
- **right passage at #1 / in top 10:** how often the passage a question was written about comes
  back first, or at all. This measures the whole retrieval pipeline (mostly the embedding model).
  Approximate search barely moves it: 62.24% at #1 with vecdex and 62.26% with FAISS's HNSW,
  against 62.29% with exact search.

Without the vecdex bindings, its row reads `skipped:` and says how to install them.

It then prints a few questions with their top 3 passages. They include failures: SQuAD questions
such as "how many permanent objects are located there?" only make sense next to their paragraph.

## How it works

| File | Role |
|---|---|
| [`run.sh`](run.sh) | One command: venv, requirements, corpus download, the two steps below |
| [`corpus.py`](corpus.py) | Loads SQuAD: one passage per paragraph, each question linked to its passage |
| [`embed.py`](embed.py) | Step 1: embeds passages and questions, cached in `data/squad/*.npz` |
| [`retrievers.py`](retrievers.py) | The `Retriever` interface and its three implementations |
| [`demo.py`](demo.py) | Step 2: indexes the passages in each retriever, asks every question, reports |

- **Embeddings:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (Apache 2.0) at a pinned revision. It makes 384-dimensional vectors, normalized to unit length,
  so cosine similarity is a plain inner product. Each passage is embedded with its article title.
  178 of the 2,067 passages exceed the model's 256-token limit and are truncated.
- **Retrievers** (`add(embeddings, ids)`, `search(queries, k) -> (ids, similarities)`):
  - `FaissFlatRetriever`: FAISS `IndexFlatIP`, exact. The reference.
  - `FaissHnswRetriever`: FAISS `IndexHNSWFlat` on inner product.
  - `VecdexRetriever`: vecdex's HNSW index through its Python bindings, `space="cosine"`.

  Both HNSW retrievers use M=16, efConstruction=200 and efSearch=64, and FAISS runs on one
  thread, like vecdex.
- **Two processes:** PyTorch (embedding) and FAISS (retrieval) each bundle their own OpenMP
  runtime. On macOS a process that loads both aborts with `OMP: Error #15`, so `embed.py` and
  `demo.py` run separately and share the vectors through the cache file.

## Plugging in vecdex

`VecdexRetriever` is written against the bindings' API:
`vecdex.Index(space="cosine", dim=384, M=16, ef_construction=200)`, `add_items(array) -> ids`,
`knn_query(queries, k=10, ef=64) -> (ids, distances)`, with cosine distance = 1 − cos. Until the
bindings are installed, constructing it raises an `ImportError` that says so, and the demo runs
the FAISS retrievers only. When the repository has the Python package (`pyproject.toml`),
`run.sh` builds the bindings and installs them into the venv on its first run. After changing
the C++ code, refresh them with `examples/rag/.venv/bin/pip install .`

## Corpus

[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) development set (Rajpurkar et al.,
2016), licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/): 2,067
paragraphs from 48 Wikipedia articles, with 10,570 crowd-written questions about them. The file
is `dev-v1.1.json`, 4,854,279 bytes, SHA-256 `95aa6a52…66972c9`. It isn't committed; `run.sh`
downloads it.
