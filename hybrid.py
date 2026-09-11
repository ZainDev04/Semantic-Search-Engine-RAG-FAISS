"""
Hybrid retrieval: combine a keyword searcher and a dense searcher with
reciprocal rank fusion (RRF).

RRF ignores the raw scores (which live on different scales for BM25 and
cosine similarity) and only uses each document's rank in each list:

    fused(d) = sum over searchers of 1 / (c + rank_of_d)

with c = 60 as in Cormack, Clarke and Buettcher (2009). A document that
ranks well in both lists rises to the top; one that ranks well in only
one list still gets credit. This is the usual first thing to try when
keyword and dense search each win on different queries.
"""

from __future__ import annotations


class HybridSearch:
    def __init__(self, searchers: list, c: int = 60, depth: int = 50):
        self.searchers = searchers
        self.c = c
        self.depth = depth
        self.name = "hybrid RRF (" + " + ".join(s.name.split(" (")[0] for s in searchers) + ")"

    def search(self, query: str, k: int = 5) -> list[dict]:
        fused: dict[int, float] = {}
        texts: dict[int, str] = {}
        for searcher in self.searchers:
            for r in searcher.search(query, k=self.depth):
                fused[r["index"]] = fused.get(r["index"], 0.0) + 1.0 / (self.c + r["rank"])
                texts[r["index"]] = r["text"]
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            {"rank": rank + 1, "score": score, "index": idx, "text": texts[idx]}
            for rank, (idx, score) in enumerate(ranked)
        ]
