# Semantic search over commit history

Search a git repository's commit messages by meaning instead of exact words, and measure whether that beats plain keyword search. On this corpus it doesn't, and this README explains why rather than hiding it.

The corpus is 2,500 unique commit messages from [huggingface/datasets](https://github.com/huggingface/datasets). Four retrieval methods are compared on the same label-free task:

| method | what it is |
|---|---|
| BM25 | classic keyword search, written from scratch in `keyword_baseline.py` |
| LSA | TF-IDF + truncated SVD, the pre-transformer "semantic" method |
| dense | sentence-transformer embeddings in a FAISS index (three models tried) |
| hybrid | BM25 + dense, merged with reciprocal rank fusion |

## Results

Task: for 200 randomly sampled commits, use the commit's one-line **title** as the query and check whether the search returns that commit's **body** from an index of all 2,500 bodies. Recall@k is the share of queries where the right body is in the top k; MRR is the mean of 1/rank.

Queries are also split into two halves of 100: titles whose every word (3+ characters) appears in the body, and titles where at least one word does not. The second half is where a method has to match meaning and not only tokens.

| method | R@5 | R@10 | MRR | MRR, partial overlap | MRR, full overlap |
|---|---|---|---|---|---|
| BM25 keyword | **0.835** | 0.860 | **0.727** | **0.535** | **0.919** |
| LSA (200d) | 0.505 | 0.585 | 0.361 | 0.259 | 0.464 |
| dense, all-MiniLM-L6-v2 (22M) | 0.720 | 0.805 | 0.600 | 0.407 | 0.794 |
| dense, bge-small-en-v1.5 (33M) | 0.725 | 0.780 | 0.653 | 0.437 | 0.869 |
| dense, all-mpnet-base-v2 (110M) | 0.725 | 0.780 | 0.584 | 0.398 | 0.771 |
| hybrid, BM25 + MiniLM | 0.830 | **0.865** | 0.720 | **0.535** | 0.905 |
| hybrid, BM25 + bge-small | 0.775 | 0.850 | 0.701 | 0.500 | 0.901 |
| hybrid, BM25 + mpnet | 0.795 | 0.855 | 0.705 | 0.525 | 0.885 |

Raw output, including per-query latency, is in `results/`, one file per dense model. The random seed is fixed (0), so the sample is reproducible.

### What the numbers say

**Keyword search wins this task, and that is a property of the task.** A commit's title and body are written by the same person about the same change, minutes apart. Half the sampled titles have every word present in the body, and the median overlap is 1.0. Titles reuse exact identifiers (`CastError`, `data_dir`, `push_to_hub`) that BM25 matches literally and that a general-purpose embedding model treats as low-information tokens. This is close to the best case for lexical search. A benchmark built from real user queries, which paraphrase and use different vocabulary, would look different; I did not have one for this corpus.

**Dense retrieval alone is clearly behind, even on the partial-overlap half.** The best dense model (bge-small) reaches 0.653 MRR against BM25's 0.727, and 0.437 against 0.535 on the harder half. So the embedding models are not rescuing the queries where keywords fail; they are losing on those too.

**Bigger is not better.** bge-small (33M parameters) is the best dense model, and all-mpnet-base-v2 (110M) lands below MiniLM (22M) while being four to five times slower per query. Training data and objective decided this, not size. None of the three was trained on code or commit text.

**Hybrid matches BM25 but does not beat it.** Rank fusion with MiniLM ties BM25 on the partial-overlap half and edges it on Recall@10, at two to three times the latency. If the goal were to ship one method for this corpus, plain BM25 would be the defensible pick; the hybrid is what I would choose if the query distribution were less extractive than this benchmark.

**LSA is the floor.** Cheapest method by far (about 4 ms/query, no model download), and it shows how much sentence transformers improved on the previous generation of "semantic" search.

## Caveats

- The evaluation is a proxy. Recovering a body from its own title is not the same as answering real user queries, and it favors exact-match methods by construction. There are no human relevance judgments.
- One random sample of 200 queries. Differences of a point or two are within noise; the BM25 vs dense gap is not.
- 2,500 documents is small. Latencies are for an exact (brute-force) FAISS index on CPU and say nothing about scaling.

## Data cleaning

Three things in the raw `git log` output distorted the evaluation for every method and were removed in `fetch_data.py`:

- the `(#1234)` PR-number suffix GitHub appends to squash-merged titles (present on about 84% of commits, never in the body, never something a person would search for);
- git trailer lines (`Co-authored-by:`, `Signed-off-by:`) in bodies, which left some bodies with no actual content;
- duplicate commits: `git log --all` walks every branch, so cherry-picks and release-branch commits appeared under several hashes. Before deduplication 12% of the corpus was a repeat of another document, and the "correct" body could lose rank 1 to its own twin.

Commits whose cleaned body is shorter than 5 words are excluded from the query sample (they remain in the index). That leaves 1,465 eligible commits, from which the 200 are drawn.

## Running it

```
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python search.py "handle missing files when loading a dataset"          # dense (MiniLM)
python search.py "handle missing files when loading a dataset" --method bm25
python search.py "handle missing files when loading a dataset" --method hybrid --k 10

python evaluate.py                                                       # BM25, LSA, MiniLM, hybrid
python evaluate.py --model BAAI/bge-small-en-v1.5
python evaluate.py --skip-dense                                          # BM25 + LSA only, no torch
```

`data/commits.jsonl` is committed, so nothing needs to be cloned to run the above. To rebuild it from a fresh metadata-only clone of the source repo:

```
python fetch_data.py
```

## Layout

```
fetch_data.py        clone huggingface/datasets (metadata only), extract, clean, dedupe
keyword_baseline.py  BM25 from scratch
embedder.py          DenseIndex (sentence-transformers + FAISS) and LSAIndex (scikit-learn)
hybrid.py            reciprocal rank fusion of any two searchers
search.py            command-line search
evaluate.py          self-retrieval evaluation with overlap split; writes results/<model>.json
data/commits.jsonl   2,500 cleaned commit records: id, title, body, text, author, date
results/*.json       raw evaluation output per dense model
```
