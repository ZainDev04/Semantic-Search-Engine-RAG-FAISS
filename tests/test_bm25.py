import math

import pytest

from keyword_baseline import BM25, tokenize


def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert tokenize("Fix CastError in load_dataset(v2.1)!") == [
        "fix", "casterror", "in", "load", "dataset", "v2", "1",
    ]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_exact_keyword_query_ranks_matching_doc_first(corpus):
    bm25 = BM25(corpus)
    hits = bm25.search("memory leak dataloader", k=3)
    assert hits[0]["index"] == 0
    assert hits[0]["rank"] == 1
    assert hits[0]["text"] == corpus[0]


def test_search_returns_k_results_in_descending_score(corpus):
    bm25 = BM25(corpus)
    hits = bm25.search("streaming", k=4)
    assert len(hits) == 4
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert [h["rank"] for h in hits] == [1, 2, 3, 4]


def test_absent_term_scores_zero(corpus):
    bm25 = BM25(corpus)
    for i in range(len(corpus)):
        assert bm25.score("kubernetes", i) == 0.0


def test_rarer_term_carries_more_idf(corpus):
    bm25 = BM25(corpus)
    # "fix" appears in three documents, "parquet" in one.
    assert bm25.idf["parquet"] > bm25.idf["fix"]


def test_idf_matches_textbook_formula(corpus):
    bm25 = BM25(corpus)
    n = len(corpus)
    df_fix = sum(1 for d in corpus if "fix" in tokenize(d))
    expected = math.log(1 + (n - df_fix + 0.5) / (df_fix + 0.5))
    assert bm25.idf["fix"] == pytest.approx(expected)


def test_term_frequency_saturates():
    # Repeating a term raises the score, but with diminishing returns (k1).
    docs = ["cache", "cache cache", "cache cache cache cache cache cache cache cache"]
    bm25 = BM25(docs, b=0.0)  # disable length normalisation to isolate saturation
    s1, s2, s8 = (bm25.score("cache", i) for i in range(3))
    assert s1 < s2 < s8
    assert (s2 - s1) > (s8 - s2) / 6  # each extra repeat adds less than the first did


def test_length_normalisation_prefers_shorter_doc_with_same_match():
    docs = ["parquet streaming", "parquet streaming " + "filler " * 40]
    bm25 = BM25(docs)
    assert bm25.score("parquet", 0) > bm25.score("parquet", 1)


def test_empty_corpus_does_not_crash():
    bm25 = BM25([])
    assert bm25.search("anything") == []
