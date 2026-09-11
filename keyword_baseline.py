"""
Plain BM25 keyword search, implemented from scratch with numpy only.

BM25 is short enough to write directly, and having it in the repo makes
the baseline auditable. This is the classic lexical method that dense
retrieval gets compared against: it only matches literal tokens, weighted
by how rare a token is across the corpus and how often it repeats inside
a document, with a length normalisation term.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.name = "BM25 keyword"
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(t) for t in texts]
        self.doc_lens = np.array([len(d) for d in self.tokenized])
        self.avg_doc_len = self.doc_lens.mean() if len(self.doc_lens) else 0.0

        self.doc_freqs: list[Counter] = [Counter(d) for d in self.tokenized]

        df = Counter()
        for d in self.tokenized:
            for term in set(d):
                df[term] += 1
        n_docs = len(texts)
        self.idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str, doc_index: int) -> float:
        q_terms = tokenize(query)
        freqs = self.doc_freqs[doc_index]
        dl = self.doc_lens[doc_index]
        score = 0.0
        for term in q_terms:
            if term not in freqs:
                continue
            f = freqs[term]
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_doc_len or 1))
            score += idf * (f * (self.k1 + 1)) / (denom or 1)
        return score

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = np.array([self.score(query, i) for i in range(len(self.texts))])
        idxs = np.argsort(-scores)[:k]
        return [
            {
                "rank": rank + 1,
                "score": float(scores[i]),
                "text": self.texts[i],
                "index": int(i),
            }
            for rank, i in enumerate(idxs)
        ]
