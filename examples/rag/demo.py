"""Step 2 of the RAG demo: retrieve passages for questions, and check approximate against exact.

    python examples/rag/demo.py [--k 10] [--M 16] [--ef-construction 200] [--ef 64]
                                [--examples 3]

Indexes the passage embeddings from embed.py in every retriever and asks each one all the
questions. Reports how much of each approximate retriever's top k matches exact search
(recall@k), how often the passage a question was written about comes back, and a few example
questions with their top passages: the context a generator would receive.
"""

import argparse
import textwrap

import faiss
import numpy as np

from corpus import EMBEDDINGS_PATH, Corpus, load_corpus
from retrievers import FaissFlatRetriever, FaissHnswRetriever, Retriever, VecdexRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef", type=int, default=64)
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


def load_embeddings(corpus: Corpus) -> tuple[np.ndarray, np.ndarray, str]:
    """Passage and question embeddings, and the model that made them."""
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit(f"{EMBEDDINGS_PATH} is missing: run examples/rag/embed.py first")
    with np.load(EMBEDDINGS_PATH) as cached:
        if cached["texts_sha256"] != corpus.fingerprint():
            raise SystemExit(f"{EMBEDDINGS_PATH} is out of date: run examples/rag/embed.py")
        return cached["passages"], cached["questions"], str(cached["model"])


def overlap(found: np.ndarray, exact: np.ndarray) -> float:
    """Mean recall@k of `found` against exact search: the share of exact top-k ids it returns."""
    hits = (exact[:, :, np.newaxis] == found[:, np.newaxis, :]).any(axis=2).sum(axis=1)
    return float(hits.mean()) / exact.shape[1]


def show_example(corpus: Corpus, question: int, ids: np.ndarray, scores: np.ndarray) -> None:
    asked = corpus.questions[question]
    print(f"\nQ: {asked.text}  [asked about: {corpus.passages[asked.passage].title}]")
    for rank, (passage, score) in enumerate(zip(ids[:3], scores[:3], strict=True), start=1):
        p = corpus.passages[passage]
        mark = "  <- the passage it was asked about" if passage == asked.passage else ""
        print(f"  {rank}. {score:.3f}  {p.title}{mark}")
        print(textwrap.indent(textwrap.shorten(p.text, width=150, placeholder=" ..."), "     "))


def main() -> None:
    args = parse_args()
    faiss.omp_set_num_threads(1)  # every retriever on one thread, like vecdex
    corpus = load_corpus()
    passages, questions, model = load_embeddings(corpus)
    print(
        f"corpus: SQuAD v1.1 dev, {len(corpus.passages)} Wikipedia passages, "
        f"{len(corpus.questions)} questions as queries\n"
        f"embeddings: {model}, {passages.shape[1]} dimensions, unit length\n"
        f"HNSW: M={args.M}, efConstruction={args.ef_construction}, efSearch={args.ef}\n"
    )

    def answer(retriever: Retriever) -> tuple[np.ndarray, np.ndarray]:
        retriever.add(passages, np.arange(len(corpus.passages)))
        return retriever.search(questions, args.k)

    exact, scores = answer(FaissFlatRetriever(passages.shape[1]))
    gold = np.array([q.passage for q in corpus.questions])

    def report(name: str, ids: np.ndarray) -> None:
        at_1 = np.mean(ids[:, 0] == gold)
        in_k = np.mean((ids == gold[:, np.newaxis]).any(axis=1))
        print(f"{name:<12}{overlap(ids, exact):>20.4f}{at_1:>22.4f}{in_k:>12.4f}")

    print(
        f"{'retriever':<12}{f'recall@{args.k} vs exact':>20}{'right passage at #1':>22}"
        f"{f'in top {args.k}':>12}"
    )
    report("faiss-flat", exact)
    showcase = ("faiss-flat", exact, scores)
    hnsw = {"M": args.M, "ef_construction": args.ef_construction, "ef": args.ef}
    for retriever_class in (FaissHnswRetriever, VecdexRetriever):
        try:
            retriever = retriever_class(passages.shape[1], **hnsw)
        except ImportError as error:
            print(f"{retriever_class.name:<12}skipped: {error}")
            continue
        ids, scores = answer(retriever)
        report(retriever.name, ids)
        showcase = (retriever.name, ids, scores)

    name, ids, scores = showcase
    print(f"\nExample questions, top 3 passages from {name} (cosine similarity):")
    picks = np.random.default_rng(seed=0).choice(len(gold), args.examples, replace=False)
    for question in picks:
        show_example(corpus, question, ids[question], scores[question])


if __name__ == "__main__":
    main()
