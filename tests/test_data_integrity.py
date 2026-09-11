"""Sanity checks on the committed dataset, so a bad regeneration is caught."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def commits():
    with (ROOT / "data" / "commits.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def questions():
    with (ROOT / "data" / "questions.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_corpus_has_expected_size_and_fields(commits):
    assert len(commits) == 2500
    for r in commits:
        assert set(r) == {"id", "title", "body", "text", "author", "date"}
        assert len(r["id"]) == 10
        assert r["title"]


def test_corpus_ids_and_texts_are_unique(commits):
    assert len({r["id"] for r in commits}) == len(commits)
    assert len({r["text"] for r in commits}) == len(commits)


def test_corpus_was_cleaned(commits):
    for r in commits:
        assert not r["title"].rstrip().endswith(")") or "(#" not in r["title"][-12:]
        assert "co-authored-by:" not in r["body"].lower()
        assert "signed-off-by:" not in r["body"].lower()


def test_every_question_points_at_a_real_commit(commits, questions):
    ids = {r["id"] for r in commits}
    assert len(questions) == 22
    for q in questions:
        assert q["expected"] in ids, q["id"]
        assert q["question"].strip()
