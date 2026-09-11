import pytest

from evaluate_rag import summarise
from rag import BODY_CHARS, ID_PATTERN, ExtractiveGenerator, answer, format_context


class ScriptedGenerator:
    """Returns a canned answer so the citation checker can be tested alone."""

    name = "scripted"

    def __init__(self, text: str):
        self.text = text

    def generate(self, system: str, user: str) -> str:
        return self.text


def test_id_pattern_matches_only_10_hex_chars():
    text = "see [0123456789] and [abcdef0123], not [12345] or [0123456789ab] or [ABCDEF0123]"
    assert ID_PATTERN.findall(text) == ["0123456789", "abcdef0123"]


def test_extractive_generator_cites_top_hit(records, corpus, fake_searcher_cls):
    retriever = fake_searcher_cls([2, 0, 1], corpus)
    result = answer("q", retriever, ExtractiveGenerator(), records, k=3)
    assert result["retrieved_ids"] == [records[2]["id"], records[0]["id"], records[1]["id"]]
    assert result["cited_ids"] == [records[2]["id"]]
    assert result["unsupported_citations"] == []
    assert records[2]["title"] in result["answer"]


def test_fabricated_citation_is_flagged(records, corpus, fake_searcher_cls):
    retriever = fake_searcher_cls([0, 1], corpus)
    fake_id = "ffffffffff"
    gen = ScriptedGenerator(f"The fix was in [{records[0]['id']}] and also [{fake_id}].")
    result = answer("q", retriever, gen, records, k=2)
    assert result["cited_ids"] == [records[0]["id"], fake_id]
    assert result["unsupported_citations"] == [fake_id]


def test_repeated_citation_is_counted_once(records, corpus, fake_searcher_cls):
    retriever = fake_searcher_cls([0], corpus)
    rid = records[0]["id"]
    result = answer("q", retriever, ScriptedGenerator(f"[{rid}] then again [{rid}]"), records, k=1)
    assert result["cited_ids"] == [rid]


def test_format_context_numbers_hits_and_truncates_bodies(records, corpus, fake_searcher_cls):
    long_records = [dict(r, body="x" * (BODY_CHARS + 500)) for r in records]
    hits = fake_searcher_cls([1, 0], corpus).search("q", k=2)
    ctx = format_context(hits, long_records)
    assert ctx.startswith(f"[1] id={records[1]['id']}")
    assert f"[2] id={records[0]['id']}" in ctx
    assert "x" * BODY_CHARS in ctx
    assert "x" * (BODY_CHARS + 1) not in ctx


def test_summarise_aggregates_rows():
    rows = [
        # cited the expected id, fully supported
        {"retrieval_hit": True, "cited_expected": True, "cited_ids": ["a"],
         "unsupported_citations": [], "abstained": False, "generate_seconds": 1.0},
        # cited two ids, one fabricated -> precision 0.5
        {"retrieval_hit": True, "cited_expected": False, "cited_ids": ["b", "z"],
         "unsupported_citations": ["z"], "abstained": False, "generate_seconds": 3.0},
        # abstained, cited nothing -> excluded from precision
        {"retrieval_hit": False, "cited_expected": False, "cited_ids": [],
         "unsupported_citations": [], "abstained": True, "generate_seconds": 2.0},
    ]
    s = summarise(rows)
    assert s["n"] == 3
    assert s["retrieval_hit"] == pytest.approx(2 / 3)
    assert s["cited_expected"] == pytest.approx(1 / 3)
    assert s["any_citation"] == pytest.approx(2 / 3)
    assert s["citation_precision"] == pytest.approx((1.0 + 0.5) / 2)
    assert s["abstained"] == pytest.approx(1 / 3)
    assert s["mean_generate_seconds"] == pytest.approx(2.0)


def test_summarise_precision_is_none_when_nothing_cited():
    rows = [{"retrieval_hit": True, "cited_expected": False, "cited_ids": [],
             "unsupported_citations": [], "abstained": True, "generate_seconds": 0.0}]
    assert summarise(rows)["citation_precision"] is None
