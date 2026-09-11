"""
Command-line search over the commit dataset.

    python search.py "fix memory leak in dataloader"
    python search.py "fix memory leak in dataloader" --method bm25
    python search.py "fix memory leak in dataloader" --method hybrid --k 10

Methods: dense (default), bm25, lsa, hybrid (bm25 + dense, rank fusion).
"""

import argparse
import json
from pathlib import Path

from keyword_baseline import BM25

DATA_PATH = Path(__file__).parent / "data" / "commits.jsonl"


def build(method: str, texts: list[str]):
    if method == "bm25":
        return BM25(texts)
    if method == "lsa":
        from embedder import LSAIndex
        return LSAIndex(texts)
    from embedder import DenseIndex
    dense = DenseIndex(texts)
    if method == "dense":
        return dense
    from hybrid import HybridSearch
    return HybridSearch([BM25(texts), dense])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--method", choices=["dense", "bm25", "lsa", "hybrid"], default="dense")
    args = parser.parse_args()

    with DATA_PATH.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    texts = [r["text"] for r in records]

    searcher = build(args.method, texts)
    print(f"Method: {searcher.name}\n")
    for r in searcher.search(args.query, k=args.k):
        rec = records[r["index"]]
        print(f"{r['rank']}. [{r['score']:.4f}] {rec['title']}  ({rec['id']}, {rec['date']})")


if __name__ == "__main__":
    main()
