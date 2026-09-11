from embedder import LSAIndex, _format


def test_lsa_returns_normalised_vectors_and_ranked_hits(corpus):
    index = LSAIndex(corpus * 3, n_components=4)  # repeat so min_df=2 keeps terms
    norms = (index.embeddings ** 2).sum(axis=1)
    assert all(abs(n - 1.0) < 1e-5 or n == 0.0 for n in norms)
    hits = index.search("streaming parquet", k=3)
    assert len(hits) == 3
    assert [h["rank"] for h in hits] == [1, 2, 3]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_lsa_caps_components_to_corpus_size(corpus):
    # Asking for more components than the matrix allows must not crash.
    index = LSAIndex(corpus * 2, n_components=500)
    assert index.embeddings.shape[0] == len(corpus) * 2
    assert index.embeddings.shape[1] < 500


def test_format_drops_faiss_padding_indices():
    # FAISS pads with -1 when it cannot fill k results; those must be dropped.
    hits = _format([0.9, 0.5, 0.0], [2, 0, -1], ["a", "b", "c"])
    assert [h["index"] for h in hits] == [2, 0]
    assert hits[0]["text"] == "c"
