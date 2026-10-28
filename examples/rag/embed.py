"""Step 1 of the RAG demo: embed every passage and question, and cache the vectors on disk.

    python examples/rag/embed.py

Uses all-MiniLM-L6-v2 (384 dimensions) with unit-length output, so cosine similarity is a plain
inner product. Writes data/squad/embeddings-all-MiniLM-L6-v2.npz, or does nothing if that file
already matches the corpus and the model.

Embedding and retrieval run as separate processes: PyTorch and FAISS each bundle their own
OpenMP runtime, and on macOS a process that loads both aborts (OMP: Error #15).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from corpus import EMBEDDINGS_PATH, load_corpus

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # a pinned commit: reproducible vectors


def main() -> None:
    corpus = load_corpus()
    fingerprint = corpus.fingerprint()
    model_id = f"{MODEL}@{REVISION}"
    if EMBEDDINGS_PATH.exists():
        with np.load(EMBEDDINGS_PATH) as cached:
            if cached["texts_sha256"] == fingerprint and cached["model"] == model_id:
                print(f"embeddings: cached in {EMBEDDINGS_PATH}")
                return

    model = SentenceTransformer(MODEL, revision=REVISION)
    passage_texts = corpus.passage_texts()
    # verbose=False: the count below replaces the tokenizer's warning about one long passage.
    lengths = [len(ids) for ids in model.tokenizer(passage_texts, verbose=False)["input_ids"]]
    truncated = sum(length > model.max_seq_length for length in lengths)
    print(
        f"embeddings: {len(passage_texts)} passages and {len(corpus.questions)} questions with "
        f"{MODEL} on {model.device}; {truncated} passages exceed the model's "
        f"{model.max_seq_length} tokens and are truncated"
    )

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = EMBEDDINGS_PATH.with_name(EMBEDDINGS_PATH.stem + ".part.npz")
    np.savez(
        partial,
        passages=encode(passage_texts),
        questions=encode(corpus.question_texts()),
        texts_sha256=fingerprint,
        model=model_id,
    )
    partial.rename(EMBEDDINGS_PATH)  # an interrupted run leaves no half-written cache behind
    print(f"embeddings: wrote {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
