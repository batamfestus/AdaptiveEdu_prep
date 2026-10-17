"""Tests for core/knowledge.py (the study-notes search). Run with:  pytest -q"""
import json
from pathlib import Path

import pytest

from core.config import BASE_DIR
from core.knowledge import Chunk, KnowledgeBase, KnowledgeError, chunk_markdown, coverage, load_chunks
from tests.fakes import hash_embed

NOTES = """<!-- a comment for humans -->
topic: Current Electricity

## Apply Ohm's law
Voltage equals current times resistance, V = IR. Resistance is measured in ohms.

## Analyse series and parallel circuits
Resistors in parallel: the combined resistance is smaller than any single branch.
"""


# ------------------------------------------------------------------ chunking
def test_markdown_is_split_into_one_chunk_per_competency():
    chunks = chunk_markdown(NOTES, "ce.md")
    assert [c.competency for c in chunks] == ["Apply Ohm's law", "Analyse series and parallel circuits"]
    assert chunks[0].id == "current-electricity/apply-ohm-s-law"
    assert all(c.topic == "Current Electricity" and c.source == "ce.md" for c in chunks)
    assert "comment" not in " ".join(c.text for c in chunks)


def test_long_sections_are_split_at_paragraph_breaks():
    paragraph = "word " * 150            # about 750 characters
    text = f"topic: Motion\n\n## Big\n{paragraph}\n\n{paragraph}\n\n{paragraph}"
    chunks = chunk_markdown(text, "m.md")
    assert len(chunks) == 3 and chunks[0].id == "motion/big" and chunks[1].id == "motion/big#2"
    assert all(len(c.text) <= 1200 for c in chunks)


def test_malformed_files_are_rejected_clearly():
    with pytest.raises(KnowledgeError, match="topic"):
        chunk_markdown("## Heading\nbody", "bad.md")
    with pytest.raises(KnowledgeError, match="no '## competency'"):
        chunk_markdown("topic: Motion\n\njust text", "bad.md")


def test_load_chunks_skips_readme_and_rejects_duplicates(tmp_path):
    (tmp_path / "README.md").write_text("# ignore me", encoding="utf-8")
    (tmp_path / "a.md").write_text(NOTES, encoding="utf-8")
    assert len(load_chunks(tmp_path)) == 2
    (tmp_path / "b.md").write_text(NOTES, encoding="utf-8")
    with pytest.raises(KnowledgeError, match="same topic and competency"):
        load_chunks(tmp_path)


def test_shipped_notes_match_the_syllabus_exactly():
    """Catches typos: a note whose heading doesn't match a syllabus competency would never be found."""
    seed = json.loads(Path(BASE_DIR, "data", "syllabus_seed.json").read_text(encoding="utf-8"))
    syllabus = {t["name"]: [c["name"] for c in t["competencies"]] for t in seed["topics"]}
    covered, total, missing, orphans = coverage(load_chunks(BASE_DIR / "data" / "knowledge"), syllabus)
    assert (covered, total) == (18, 18) and not missing and not orphans


def test_coverage_reports_gaps_and_orphans():
    chunks = [Chunk("a", "Motion", "Interpret motion graphs", "x", "m.md"), Chunk("b", "Motion", "Typo comptency", "x", "m.md")]
    syllabus = {"Motion": ["Interpret motion graphs", "Distinguish scalars and vectors"]}
    covered, total, missing, orphans = coverage(chunks, syllabus)
    assert (covered, total) == (1, 2)
    assert missing == ["Motion > Distinguish scalars and vectors"] and orphans == ["Motion > Typo comptency"]


# ------------------------------------------------------------------ searching (both engines)
@pytest.fixture(scope="module")
def real_chunks():
    return load_chunks(BASE_DIR / "data" / "knowledge")


def test_keyword_search_finds_the_right_note(real_chunks):
    kb = KnowledgeBase(real_chunks, use_chroma=False)
    assert kb.mode == "keyword"
    hits = kb.search("combined resistance of resistors in parallel", topic="Current Electricity")
    assert hits[0].chunk.competency == "Analyse series and parallel circuits"
    assert all(h.chunk.topic == "Current Electricity" for h in hits)
    assert 0 < hits[0].score <= 1


def test_topic_filter_keeps_other_topics_out(real_chunks):
    kb = KnowledgeBase(real_chunks, use_chroma=False)
    hits = kb.search("energy and work done by a force", topic="Motion")
    assert all(h.chunk.topic == "Motion" for h in hits)


def test_competency_notes_come_first(real_chunks):
    kb = KnowledgeBase(real_chunks, use_chroma=False)
    hits = kb.search("resistance current voltage", topic="Current Electricity",
                     competency="Calculate electrical energy and power", k=3)
    assert hits[0].chunk.competency == "Calculate electrical energy and power"
    assert len({h.chunk.id for h in hits}) == len(hits)   # no duplicates


def test_empty_and_unmatched_queries(real_chunks):
    kb = KnowledgeBase(real_chunks, use_chroma=False)
    assert kb.search("   ") == []
    assert kb.search("zzzzqqqq", topic="Motion") == []


def test_chroma_search_with_supplied_embeddings(real_chunks):
    pytest.importorskip("chromadb")
    kb = KnowledgeBase(real_chunks, embed_fn=hash_embed)
    assert kb.mode == "chroma" and not kb.fallback_reason
    hits = kb.search("combined resistance of resistors in parallel", topic="Current Electricity", k=2)
    assert hits[0].chunk.competency == "Analyse series and parallel circuits"
    assert all(h.chunk.topic == "Current Electricity" for h in hits)
    first = kb.search("current voltage", topic="Current Electricity", competency="Apply Ohm's law", k=3)
    assert first[0].chunk.competency == "Apply Ohm's law"


def test_two_knowledge_bases_do_not_share_an_index(real_chunks):
    pytest.importorskip("chromadb")
    small = KnowledgeBase(real_chunks[:2], embed_fn=hash_embed)
    big = KnowledgeBase(real_chunks, embed_fn=hash_embed)
    assert len(small.search("energy", k=10)) <= 2 and len(big.search("energy", k=10)) > 2


def test_falls_back_to_keywords_when_chroma_cannot_start(real_chunks):
    def broken_embedder(texts):
        raise RuntimeError("model download blocked")
    kb = KnowledgeBase(real_chunks, embed_fn=broken_embedder)
    assert kb.mode == "keyword" and "model download blocked" in kb.fallback_reason
    assert kb.search("half-life of a radioactive isotope")  # still works


def test_falls_back_when_a_search_fails_midway(real_chunks, monkeypatch):
    pytest.importorskip("chromadb")
    kb = KnowledgeBase(real_chunks, embed_fn=hash_embed)
    monkeypatch.setattr(kb, "_search_chroma", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index corrupt")))
    hits = kb.search("half-life of a radioactive isotope", topic="Electromagnetism and Modern Physics")
    assert hits and kb.mode == "keyword" and "index corrupt" in kb.fallback_reason
