"""
Dense retrieval backends. Each one turns text into vectors and answers
nearest-neighbour queries over them.

    DenseIndex  transformer sentence embeddings (default all-MiniLM-L6-v2)
                stored in an exact FAISS inner-product index.
    LSAIndex    TF-IDF + truncated SVD (latent semantic analysis), the
                pre-transformer way to do "semantic" search. Included as a
                baseline so the gap between old and new methods is measured
                rather than assumed. Needs only scikit-learn.

Every backend exposes the same interface:

    index = SomeIndex(texts)
    index.search(query, k) -> [{"rank", "score", "index", "text"}, ...]
    index.name              -> short label used in result tables
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseIndex:
    def __init__(self, texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.texts = texts
        self.model_name = model_name
        self.name = f"dense ({model_name.split('/')[-1]})"
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embeddings, dtype="float32")
        # Vectors are unit-normalised, so inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype="float32")

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores, idxs = self.index.search(self.encode_query(query), k)
        return _format(scores[0], idxs[0], self.texts)


class LSAIndex:
    def __init__(self, texts: list[str], n_components: int = 200):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        self.texts = texts
        self.name = f"LSA (tf-idf + SVD, {n_components}d)"
        self._normalize = normalize
        self.vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", min_df=2)
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(n_components, min(tfidf.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=0)
        self.embeddings = normalize(self.svd.fit_transform(tfidf)).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        reduced = self.svd.transform(self.vectorizer.transform([query]))
        return self._normalize(reduced).astype("float32")

    def search(self, query: str, k: int = 5) -> list[dict]:
        sims = self.embeddings @ self.encode_query(query)[0]
        idxs = np.argsort(-sims)[:k]
        return _format(sims[idxs], idxs, self.texts)


def _format(scores, idxs, texts) -> list[dict]:
    return [
        {"rank": rank + 1, "score": float(s), "index": int(i), "text": texts[int(i)]}
        for rank, (s, i) in enumerate(zip(scores, idxs))
        if int(i) >= 0
    ]
