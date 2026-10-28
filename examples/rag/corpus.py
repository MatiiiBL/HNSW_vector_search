"""The demo's corpus: the development set of SQuAD v1.1, Wikipedia paragraphs with questions.

Every paragraph is one passage, and every question records the passage it was written about,
which tells the demo whether retrieval found the right one.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "squad"
CORPUS_PATH = DATA_DIR / "dev-v1.1.json"  # downloaded by run.sh
EMBEDDINGS_PATH = DATA_DIR / "embeddings-all-MiniLM-L6-v2.npz"  # written by embed.py


@dataclass(frozen=True)
class Passage:
    title: str  # the Wikipedia article
    text: str


@dataclass(frozen=True)
class Question:
    text: str
    passage: int  # index of the passage it was asked about


@dataclass(frozen=True)
class Corpus:
    passages: list[Passage]
    questions: list[Question]

    def passage_texts(self) -> list[str]:
        # The title goes in too: a paragraph often doesn't name its own subject.
        return [f"{p.title}. {p.text}" for p in self.passages]

    def question_texts(self) -> list[str]:
        return [q.text for q in self.questions]

    def fingerprint(self) -> str:
        """SHA-256 of every text that gets embedded, to tell whether cached embeddings match."""
        digest = hashlib.sha256()
        for text in self.passage_texts() + self.question_texts():
            digest.update(text.encode())
            digest.update(b"\0")
        return digest.hexdigest()


def load_corpus(path: Path = CORPUS_PATH) -> Corpus:
    with path.open(encoding="utf-8") as f:
        articles = json.load(f)["data"]
    passages: list[Passage] = []
    questions: list[Question] = []
    for article in articles:
        title = article["title"].replace("_", " ")
        for paragraph in article["paragraphs"]:
            passages.append(Passage(title, paragraph["context"]))
            index = len(passages) - 1
            questions.extend(Question(qa["question"], index) for qa in paragraph["qas"])
    return Corpus(passages, questions)
