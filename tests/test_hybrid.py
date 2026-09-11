from hybrid import HybridSearch


def test_rrf_rewards_documents_ranked_well_by_both(corpus, fake_searcher_cls):
    a = fake_searcher_cls([0, 1, 2, 3], corpus, name="a")
    b = fake_searcher_cls([1, 0, 5, 4], corpus, name="b")
    hybrid = HybridSearch([a, b], c=60)
    hits = hybrid.search("q", k=6)

    # Docs 0 and 1 are in both lists at ranks (1,2) and (2,1): tied and on top.
    assert {hits[0]["index"], hits[1]["index"]} == {0, 1}
    assert hits[0]["score"] == hits[1]["score"]
    # Everything ranked by only one searcher comes after.
    assert {h["index"] for h in hits[2:]} == {2, 3, 4, 5}


def test_rrf_score_formula(corpus, fake_searcher_cls):
    a = fake_searcher_cls([0], corpus)
    b = fake_searcher_cls([0], corpus)
    hits = HybridSearch([a, b], c=60).search("q", k=1)
    assert hits[0]["score"] == 2 * (1.0 / (60 + 1))


def test_hybrid_ignores_raw_score_scale(corpus, fake_searcher_cls):
    # Two searchers whose raw scores live on totally different scales must
    # still contribute equally: only ranks are used.
    class BigScores(fake_searcher_cls):
        def search(self, query, k=5):
            return [dict(h, score=h["score"] * 1e6) for h in super().search(query, k)]

    a = fake_searcher_cls([0, 1], corpus)
    b = BigScores([1, 0], corpus)
    hits = HybridSearch([a, b]).search("q", k=2)
    assert hits[0]["score"] == hits[1]["score"]


def test_hybrid_respects_k_and_output_shape(corpus, fake_searcher_cls):
    a = fake_searcher_cls([0, 1, 2, 3, 4, 5], corpus)
    hits = HybridSearch([a]).search("q", k=3)
    assert len(hits) == 3
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert all(set(h) == {"rank", "score", "index", "text"} for h in hits)


def test_hybrid_name_lists_components(corpus, fake_searcher_cls):
    a = fake_searcher_cls([], corpus, name="BM25 keyword")
    b = fake_searcher_cls([], corpus, name="dense (all-MiniLM-L6-v2)")
    assert HybridSearch([a, b]).name == "hybrid RRF (BM25 keyword + dense)"
