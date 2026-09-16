"""
Measures the RAG pipeline on a small hand-written question set.

data/questions.jsonl holds 22 questions I wrote by reading commit bodies,
each tagged with the commit that answers it. The questions are paraphrased
(they do not copy the commit title), so this is closer to a real query than
the title->body test in evaluate.py.

For every retriever (bm25, dense, hybrid) and every question, the pipeline
retrieves top-k commits, generates an answer, and records:

    retrieval_hit        the expected commit was in the top-k context
    cited_expected       the answer cites the expected commit's id
    any_citation         the answer cites at least one id
    citation_precision   share of cited ids that were actually retrieved
                         (anything below 1.0 is a fabricated citation)
    abstained            the model said the commits do not cover it

The first number is the retriever's job; the rest are the generator's.
Keeping them separate is the point: it shows which half of the system a
failure belongs to.

    python evaluate_rag.py                       # local model, all retrievers
    python evaluate_rag.py --retrievers bm25
    python evaluate_rag.py --backend llamacpp    # llama-server must be running
    python evaluate_rag.py --backend claude      # needs ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag import answer, build_generator, build_retriever, load_records

QUESTIONS_PATH = Path(__file__).parent / "data" / "questions.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
ABSTAIN_MARKER = "do not cover"


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    cited_rows = [r for r in rows if r["cited_ids"]]
    precision = [
        1 - len(r["unsupported_citations"]) / len(r["cited_ids"]) for r in cited_rows
    ]
    return {
        "n": n,
        "retrieval_hit": sum(r["retrieval_hit"] for r in rows) / n,
        "cited_expected": sum(r["cited_expected"] for r in rows) / n,
        "any_citation": len(cited_rows) / n,
        "citation_precision": sum(precision) / len(precision) if precision else None,
        "abstained": sum(r["abstained"] for r in rows) / n,
        "mean_generate_seconds": sum(r["generate_seconds"] for r in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--backend", choices=["local", "llamacpp", "claude", "extractive"], default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    records = load_records()
    questions = load_questions()
    generator = build_generator(args.backend, args.model)
    print(f"Generator: {generator.name}\nQuestions: {len(questions)}, k={args.k}\n")

    all_results = {}
    for method in args.retrievers:
        retriever = build_retriever(method, records)
        rows = []
        for q in questions:
            r = answer(q["question"], retriever, generator, records, k=args.k)
            r["id"] = q["id"]
            r["expected"] = q["expected"]
            r["retrieval_hit"] = q["expected"] in r["retrieved_ids"]
            r["cited_expected"] = q["expected"] in r["cited_ids"]
            r["abstained"] = ABSTAIN_MARKER in r["answer"].lower()
            rows.append(r)
            flag = "hit" if r["retrieval_hit"] else "MISS"
            cite = "cited" if r["cited_expected"] else "-"
            print(f"  {method:<7}{q['id']}  retrieval {flag:<5} {cite:<6} {r['generate_seconds']:.0f}s")
        all_results[method] = {"summary": summarise(rows), "rows": rows}
        print()

    print(f"{'retriever':<10}{'retr.hit':>9}{'cited exp':>10}{'any cite':>9}{'cite prec':>10}{'abstain':>8}{'s/answer':>9}")
    for method, res in all_results.items():
        s = res["summary"]
        prec = f"{s['citation_precision']:.2f}" if s["citation_precision"] is not None else "n/a"
        print(
            f"{method:<10}{s['retrieval_hit']:>9.2f}{s['cited_expected']:>10.2f}"
            f"{s['any_citation']:>9.2f}{prec:>10}{s['abstained']:>8.2f}{s['mean_generate_seconds']:>9.1f}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    slug = generator.name.split(" (")[0] if generator.name.startswith("extractive") else generator.name.split("(")[1].rstrip(")")
    out_path = RESULTS_DIR / f"rag_{slug}.json"
    out_path.write_text(
        json.dumps({"generator": generator.name, "k": args.k, "retrievers": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
