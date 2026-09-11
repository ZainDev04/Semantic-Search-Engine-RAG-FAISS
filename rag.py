"""
Retrieval-augmented generation over the commit corpus.

    question -> retriever (bm25 | dense | hybrid) -> top-k commits
             -> language model writes an answer citing commit ids
             -> citations are checked against what was actually retrieved

Three generation backends:

    extractive  no model at all: returns the top-1 commit's title and body,
                cited. Sets the floor a generator has to beat. If a language
                model cannot outperform "print the best search result", it is
                not adding anything.

    local   Qwen2.5-0.5B-Instruct through `transformers`, on CPU. No API key,
            no cost, reproducible by anyone who clones the repo. Small enough
            for an 8 GB machine; pass --model to use a larger one.
    claude  Anthropic API (claude-opus-5). Needs ANTHROPIC_API_KEY. Better
            answers; not used for the committed numbers because the results
            in this repo must be reproducible without a paid key.

Citations are the part that gets measured. The model is told to cite commit
ids in square brackets, e.g. [c6fc5cdbf0]. After generation the answer is
scanned for ids, and each one is checked against the retrieved set. A cited
id that was not retrieved is a hallucination by construction: the model had
no way to know it.

    python rag.py "why did to_csv turn integers into floats"
    python rag.py "..." --retriever bm25 --k 5
    python rag.py "..." --backend claude
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from keyword_baseline import BM25

DATA_PATH = Path(__file__).parent / "data" / "commits.jsonl"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
BODY_CHARS = 1200  # per commit, keeps the prompt inside a small model's window
ID_PATTERN = re.compile(r"\b[0-9a-f]{10}\b")

SYSTEM_PROMPT = """You answer questions about a software project's commit history.

You are given a numbered list of commits. Each has an id, a title and a body.
Answer the question using only what those commits say. Cite every commit you
rely on by putting its id in square brackets, like [c6fc5cdbf0], right after
the claim it supports. If none of the commits answer the question, say
"The retrieved commits do not cover this." and cite nothing. Keep the answer
under 120 words. Do not invent commit ids.

Example of the expected format (the id below is illustrative, never cite it):
The loader dropped the encoding option because of a version check bug [0123456789]."""


def load_records() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_retriever(method: str, records: list[dict]):
    texts = [r["text"] for r in records]
    if method == "bm25":
        return BM25(texts)
    from embedder import DenseIndex

    dense = DenseIndex(texts)
    if method == "dense":
        return dense
    from hybrid import HybridSearch

    return HybridSearch([BM25(texts), dense])


def format_context(hits: list[dict], records: list[dict]) -> str:
    parts = []
    for n, h in enumerate(hits, 1):
        r = records[h["index"]]
        body = r["body"][:BODY_CHARS]
        parts.append(f"[{n}] id={r['id']}\ntitle: {r['title']}\nbody: {body}")
    return "\n\n".join(parts)


class ExtractiveGenerator:
    """Returns the top-ranked commit verbatim, cited. No model involved."""

    name = "extractive (top-1 commit, no model)"

    def generate(self, system: str, user: str, hits: list[dict] | None = None, records=None) -> str:
        r = records[hits[0]["index"]]
        return f"{r['title']}. {r['body'][:400]} [{r['id']}]"


class LocalGenerator:
    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, max_new_tokens: int = 200):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"local ({model_name.split('/')[-1]})"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, system: str, user: str) -> str:
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ClaudeGenerator:
    def __init__(self, model_name: str = DEFAULT_CLAUDE_MODEL):
        import anthropic

        self.name = f"claude ({model_name})"
        self.model_name = model_name
        self.client = anthropic.Anthropic()

    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=system,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()


def build_generator(backend: str, model_name: str | None):
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend == "claude":
        return ClaudeGenerator(model_name or DEFAULT_CLAUDE_MODEL)
    return LocalGenerator(model_name or DEFAULT_LOCAL_MODEL)


def answer(question: str, retriever, generator, records: list[dict], k: int = 5) -> dict:
    t0 = time.perf_counter()
    hits = retriever.search(question, k=k)
    t_retrieve = time.perf_counter() - t0
    retrieved_ids = [records[h["index"]]["id"] for h in hits]

    user = f"Commits:\n\n{format_context(hits, records)}\n\nQuestion: {question}"
    t0 = time.perf_counter()
    if isinstance(generator, ExtractiveGenerator):
        text = generator.generate(SYSTEM_PROMPT, user, hits=hits, records=records)
    else:
        text = generator.generate(SYSTEM_PROMPT, user)
    t_generate = time.perf_counter() - t0

    cited = list(dict.fromkeys(ID_PATTERN.findall(text)))
    return {
        "question": question,
        "retrieved_ids": retrieved_ids,
        "answer": text,
        "cited_ids": cited,
        "unsupported_citations": [c for c in cited if c not in retrieved_ids],
        "retrieve_seconds": t_retrieve,
        "generate_seconds": t_generate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--backend", choices=["local", "claude", "extractive"], default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    records = load_records()
    retriever = build_retriever(args.retriever, records)
    generator = build_generator(args.backend, args.model)

    result = answer(args.question, retriever, generator, records, k=args.k)
    print(f"Retriever: {retriever.name}\nGenerator: {generator.name}\n")
    print("Retrieved:")
    for rid in result["retrieved_ids"]:
        rec = next(r for r in records if r["id"] == rid)
        print(f"  {rid}  {rec['title']}")
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Cited: {result['cited_ids'] or 'none'}")
    if result["unsupported_citations"]:
        print(f"WARNING, cited ids that were not retrieved: {result['unsupported_citations']}")
    print(f"retrieval {result['retrieve_seconds']:.2f}s, generation {result['generate_seconds']:.1f}s")


if __name__ == "__main__":
    main()
