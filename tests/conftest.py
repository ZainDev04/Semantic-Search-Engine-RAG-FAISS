"""
Shared fixtures. Everything here runs without torch or a model download, so
the whole suite finishes in a few seconds on CPU.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def corpus() -> list[str]:
    # Small, hand-made corpus with obvious keyword and topic structure.
    return [
        "Fix memory leak in the dataloader when num_workers is greater than zero",
        "Add streaming support for parquet files",
        "Fix CastError when loading a dataset with mixed integer and float columns",
        "Bump version to 2.14.0",
        "Support push_to_hub for IterableDataset",
        "Fix streaming datasets that are not reset correctly between epochs",
    ]


@pytest.fixture
def records(corpus) -> list[dict]:
    return [
        {
            "id": f"{i:010x}",
            "title": text.split(" when ")[0].split(" for ")[0],
            "body": text,
            "text": text,
            "author": "test",
            "date": "2026-01-01",
        }
        for i, text in enumerate(corpus)
    ]


class FakeSearcher:
    """Returns a fixed ranking, so fusion and evaluation logic can be tested
    independently of any real retriever."""

    def __init__(self, ranking: list[int], texts: list[str], name: str = "fake"):
        self.ranking = ranking
        self.texts = texts
        self.name = name

    def search(self, query: str, k: int = 5) -> list[dict]:
        return [
            {"rank": rank + 1, "score": 1.0 / (rank + 1), "index": idx, "text": self.texts[idx]}
            for rank, idx in enumerate(self.ranking[:k])
        ]


@pytest.fixture
def fake_searcher_cls():
    return FakeSearcher
