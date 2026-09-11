import pytest

from evaluate import bucket_mrr, evaluate, lexical_overlap


def test_lexical_overlap_ignores_short_tokens_and_case():
    # "in" and "a" are shorter than 3 chars and do not count.
    assert lexical_overlap("Fix Leak in a loader", "the LOADER had a leak; fix it") == 1.0


def test_lexical_overlap_partial_and_empty():
    assert lexical_overlap("alpha beta gamma", "alpha only") == pytest.approx(1 / 3)
    assert lexical_overlap("", "anything") == 0.0
    assert lexical_overlap("a b", "a b") == 0.0  # no token is 3+ chars


def test_evaluate_metrics_from_known_ranks(corpus, fake_searcher_cls):
    # Query 0 -> its own doc at rank 1; query 1 -> rank 3; query 2 -> not found.
    class Oracle:
        name = "oracle"

        def search(self, query, k=10):
            ranking = {corpus[0]: [0, 1, 2], corpus[1]: [5, 4, 1], corpus[2]: [0, 1, 3]}[query]
            return fake_searcher_cls(ranking, corpus).search(query, k)

    r = evaluate(Oracle(), sample=[0, 1, 2], titles=corpus)
    assert r["n_queries"] == 3
    assert r["ranks"] == [1, 3, None]
    assert r["recall_at_5"] == pytest.approx(2 / 3)
    assert r["recall_at_10"] == pytest.approx(2 / 3)
    assert r["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert r["ms_per_query"] >= 0


def test_bucket_mrr_splits_on_threshold():
    ranks = [1, None, 2, 4]
    overlaps = [0.2, 0.4, 0.9, 1.0]
    out = bucket_mrr(ranks, overlaps, threshold=0.5)
    assert out["low_overlap"]["n"] == 2
    assert out["low_overlap"]["mrr"] == pytest.approx((1 + 0) / 2)
    assert out["high_overlap"]["n"] == 2
    assert out["high_overlap"]["mrr"] == pytest.approx((1 / 2 + 1 / 4) / 2)


def test_bucket_mrr_empty_bucket_is_none():
    out = bucket_mrr([1, 2], [0.9, 0.8], threshold=0.5)
    assert out["low_overlap"] == {"n": 0, "mrr": None}
