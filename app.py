"""
Browser demo of the search and RAG pipeline.

    streamlit run app.py

Type a question, pick a retriever, and see the ranked commits. Optionally
turn on generation to get an answer with citations and see the citation
check run: every cited id is marked as retrieved (green) or fabricated (red).

Indexes and generators are built once per process and cached. The BM25 and
dense indexes are cached separately and the hybrid retriever is assembled
from them, so the embedding model loads once (about 20 s on CPU with the
embedding cache warm) no matter how often you switch retrievers.
"""

from __future__ import annotations

import os

import streamlit as st

from keyword_baseline import BM25
from rag import SYSTEM_PROMPT, answer, build_generator, load_records

st.set_page_config(page_title="Commit search + RAG", page_icon="🔎", layout="wide")

EXAMPLES = [
    "Why did resuming a streaming dataset after a filter call skip rows that had not been emitted yet?",
    "Why did streaming reads of a private Lance dataset return 401 errors?",
    "Why did exporting a dataset to CSV or JSON turn integer columns into floats?",
    "Was there a security problem where extracting an archive could write files through a symlink?",
]


@st.cache_resource(show_spinner="Loading corpus...")
def records() -> list[dict]:
    return load_records()


@st.cache_resource(show_spinner="Building BM25 index...")
def bm25_index():
    return BM25([r["text"] for r in records()])


@st.cache_resource(show_spinner="Loading embedding model and building dense index (once per session)...")
def dense_index():
    from embedder import DenseIndex

    return DenseIndex([r["text"] for r in records()])


# The two indexes are cached separately so that switching between dense and
# hybrid does not load the embedding model a second time. Fusing them is free.
def retriever(method: str):
    if method == "bm25":
        return bm25_index()
    if method == "dense":
        return dense_index()
    from hybrid import HybridSearch

    return HybridSearch([bm25_index(), dense_index()])


@st.cache_resource(show_spinner="Loading generator...")
def generator(backend: str, model: str | None):
    return build_generator(backend, model or None)


# Shareable links: ?q=<question>&answer=1 pre-fills the question and turns
# generation on. Only applied once, before the widgets are created.
if "query" not in st.session_state:
    st.session_state["query"] = st.query_params.get("q", "")
    st.session_state["generate"] = st.query_params.get("answer", "0") == "1"

# ---- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("Retrieval")
    method = st.radio(
        "Retriever",
        ["hybrid", "dense", "bm25"],
        captions=["BM25 + dense, rank fusion", "all-MiniLM-L6-v2 in FAISS", "keyword, from scratch"],
    )
    k = st.slider("Top k", 1, 10, 5)

    st.header("Generation")
    generate = st.toggle("Answer the question", key="generate")
    backend = st.selectbox(
        "Backend",
        ["extractive", "claude", "local"],
        format_func=lambda b: {
            "extractive": "extractive (top-1 commit, no model)",
            "claude": "claude-opus-5 (needs ANTHROPIC_API_KEY)",
            "local": "Qwen2.5-0.5B on CPU (~20 s)",
        }[b],
        disabled=not generate,
    )
    model_override = st.text_input("Model override", "", disabled=not generate, placeholder="optional")
    if backend == "claude" and generate and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set; the claude backend will fail.")

    with st.expander("System prompt"):
        st.code(SYSTEM_PROMPT, language=None)

# ---- main --------------------------------------------------------------------

st.title("Semantic search and RAG over commit history")
st.caption(
    "2,500 commits from huggingface/datasets. Search by meaning, then have a model answer with "
    "citations that are checked against what was actually retrieved."
)

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["query"] = ex

query = st.text_input("Question", key="query", placeholder="e.g. why did to_csv turn integer columns into floats")

if not query:
    st.stop()

recs = records()
ret = retriever(method)
by_id = {r["id"]: r for r in recs}

if generate:
    gen = generator(backend, model_override.strip())
    with st.spinner(f"Retrieving with {ret.name}, generating with {gen.name}..."):
        result = answer(query, ret, gen, recs, k=k)
    hits = ret.search(query, k=k)  # same call answer() made; cheap
    left, right = st.columns([3, 2])
else:
    gen, result = None, None
    with st.spinner(f"Retrieving with {ret.name}..."):
        hits = ret.search(query, k=k)
    left, right = st.container(), None

with left:
    st.subheader(f"Retrieved · {ret.name}")
    for h in hits:
        r = recs[h["index"]]
        label = f"**{h['rank']}.** {r['title']}  \n`{r['id']}` · {r['date']} · score {h['score']:.4f}"
        if result and r["id"] in result["cited_ids"]:
            label += " · :green[cited]"
        with st.expander(label, expanded=h["rank"] == 1):
            st.text(r["body"] or "(no body)")

if result:
    with right:
        st.subheader(f"Answer · {gen.name}")
        st.write(result["answer"])
        st.caption(f"retrieval {result['retrieve_seconds']:.2f}s · generation {result['generate_seconds']:.1f}s")
        st.markdown("**Citation check**")
        if not result["cited_ids"]:
            st.info("The answer cites nothing.")
        for cid in result["cited_ids"]:
            if cid in result["unsupported_citations"]:
                st.error(f"`{cid}` is not in the retrieved set: fabricated.")
            else:
                st.success(f"`{cid}` {by_id[cid]['title']}")
