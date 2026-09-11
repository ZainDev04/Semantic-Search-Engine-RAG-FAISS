"""
Measures search quality without inventing relevance labels.

There is no human-labelled relevance data for this corpus, and making
labels up would defeat the point. Instead the script runs a
self-retrieval test grounded in the data itself. Every commit has two
related texts: a short title and a longer body. The bodies are indexed.
For a random sample of commits with a non-empty body, the TITLE is used
as the query and the search has to bring back that commit's own BODY.

Title and body describe the same change, usually in different words
(the title is a one-line summary, the body carries the detail), so
recovering the body from the title is a fair test of whether a method
matches meaning and not only surface tokens. It is still a proxy for
real user queries, and the README says so.

Metrics: Recall@5, Recall@10, Mean Reciprocal Rank (MRR).

The script also splits the queries by lexical overlap (what share of the
title's tokens literally appear in the body) and reports MRR per bucket.
That shows where each method earns its keep: keyword search should do
well when overlap is high, dense retrieval should hold up better when
the title and body use different words.

Usage:
    python evaluate.py                      # all methods, default model
    python evaluate.py --model BAAI/bge-small-en-v1.5
    python evaluate.py --skip-dense         # BM25 + LSA only, no torch
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from keyword_baseline import BM25, tokenize

DATA_PATH = Path(__file__).parent / "data" / "commits.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
SAMPLE_SIZE = 200
SEED = 0


def load_records() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def lexical_overlap(title: str, body: str) -> float:
    """Share of distinct title tokens (3+ chars) that also occur in the body."""
    t = {tok for tok in tokenize(title) if len(tok) >= 3}
    if not t:
        return 0.0
    b = set(tokenize(body))
    return len(t & b) / len(t)


def evaluate(searcher, sample: list[int], titles: list[str]) -> dict:
    per_query = []
    t0 = time.perf_counter()
    for idx in sample:
        ranked = [r["index"] for r in searcher.search(titles[idx], k=10)]
        rank = ranked.index(idx) + 1 if idx in ranked else None
        per_query.append(rank)
    elapsed = time.perf_counter() - t0
    n = len(sample)
    return {
        "n_queries": n,
        "recall_at_5": sum(1 for r in per_query if r and r <= 5) / n,
        "recall_at_10": sum(1 for r in per_query if r and r <= 10) / n,
        "mrr": sum(1.0 / r for r in per_query if r) / n,
        "ms_per_query": 1000 * elapsed / n,
        "ranks": per_query,
    }


def bucket_mrr(ranks: list, overlaps: list[float], threshold: float) -> dict:
    out = {}
    for label, keep in (("low_overlap", lambda o: o < threshold), ("high_overlap", lambda o: o >= threshold)):
        rs = [r for r, o in zip(ranks, overlaps) if keep(o)]
        out[label] = {
            "n": len(rs),
            "mrr": sum(1.0 / r for r in rs if r) / len(rs) if rs else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="sentence-transformers model name")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    records = load_records()
    bodies = [r["body"] if r["body"].strip() else r["title"] for r in records]
    titles = [r["title"] for r in records]

    # A body needs some substance to be recoverable at all; very short ones are noise.
    eligible = [i for i, r in enumerate(records) if len(r["body"].split()) >= 5]
    random.seed(SEED)
    sample = random.sample(eligible, min(SAMPLE_SIZE, len(eligible)))
    overlaps = [lexical_overlap(titles[i], bodies[i]) for i in sample]
    threshold = sorted(overlaps)[len(overlaps) // 2]  # median split

    print(f"Corpus: {len(records)} commits, {len(eligible)} with a body of 5+ words")
    print(f"Eval sample: {len(sample)} title->body queries, seed {SEED}")
    print(f"Median title/body lexical overlap: {threshold:.2f}")
    print()

    from embedder import LSAIndex, DEFAULT_MODEL

    searchers = [BM25(bodies), LSAIndex(bodies)]
    if not args.skip_dense:
        from embedder import DenseIndex
        from hybrid import HybridSearch

        dense = DenseIndex(bodies, model_name=args.model or DEFAULT_MODEL)
        searchers.append(dense)
        searchers.append(HybridSearch([searchers[0], dense]))

    results = {}
    for s in searchers:
        results[s.name] = evaluate(s, sample, titles)
        results[s.name]["by_overlap"] = bucket_mrr(results[s.name]["ranks"], overlaps, threshold)

    print("Task: given a commit's TITLE, recover that commit's BODY from the index")
    print(f"{'method':<40}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'ms/q':>8}   {'MRR low-ovl':>11}{'MRR high-ovl':>13}")
    for name, r in results.items():
        lo, hi = r["by_overlap"]["low_overlap"], r["by_overlap"]["high_overlap"]
        print(
            f"{name:<40}{r['recall_at_5']:>7.3f}{r['recall_at_10']:>7.3f}{r['mrr']:>7.3f}"
            f"{r['ms_per_query']:>8.1f}   {lo['mrr']:>11.3f}{hi['mrr']:>13.3f}"
        )
    lo_n = results[searchers[0].name]["by_overlap"]["low_overlap"]["n"]
    print(f"\n(low-overlap bucket: {lo_n} queries with overlap < {threshold:.2f}; high: {len(sample) - lo_n})")

    model_slug = (args.model or DEFAULT_MODEL).split("/")[-1] if not args.skip_dense else "no-dense"
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{model_slug}.json"
    out = {
        "dense_model": None if args.skip_dense else (args.model or DEFAULT_MODEL),
        "corpus_size": len(records),
        "sample_size": len(sample),
        "seed": SEED,
        "overlap_threshold": threshold,
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "ranks"} for k, v in results.items()},
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved raw results to {out_path}")


if __name__ == "__main__":
    main()
