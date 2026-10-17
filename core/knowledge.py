"""
core/knowledge.py
-----------------
The study-notes knowledge base: the "R" in RAG (Retrieval-Augmented Generation).

HOW RAG WORKS HERE (explain this in an interview):
  1. Your notes (data/knowledge/*.md) are split into small CHUNKS, one per competency.
  2. Each chunk is turned into an EMBEDDING: a list of numbers that captures its
     meaning, so "resistors in parallel" lands near "combined resistance".
  3. When a student asks something, the question is embedded the same way and we find
     the nearest chunks (vector search), optionally restricted by METADATA
     (only chunks about this topic).
  4. Those chunks are handed to Claude with the instruction "answer using ONLY these
     notes". The answer is GROUNDED in notes you wrote and checked, not in whatever
     the model half-remembers.

The index is DERIVED DATA: it is rebuilt from the Markdown files whenever the app starts
(Streamlit Cloud has no permanent disk), so the files are the single source of truth.

GRACEFUL DEGRADATION: if Chroma can't start (not installed, a database-version problem on
some hosts), we fall back to simple keyword matching instead of crashing. The tutor gets
a bit less clever, but it keeps working. `KnowledgeBase.mode` says which one is active.
"""
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MAX_CHUNK_CHARS = 1200   # a note longer than this is split at paragraph breaks
_STOPWORDS = frozenset(
    "a an the of to in on at is are was were be by for from and or as with that this it its "
    "which what how why when does do did can will if then than into not no but so".split()
)


class KnowledgeError(Exception):
    """A notes file is malformed."""


@dataclass(frozen=True)
class Chunk:
    id: str            # e.g. "current-electricity/apply-ohms-law"
    topic: str         # must match a syllabus topic name
    competency: str    # must match a syllabus competency name
    text: str
    source: str        # the file it came from


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float       # higher = more relevant (0 to 1)


# ============================================================ loading the notes
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Turn one Markdown notes file into chunks.

    Format:   topic: <exact syllabus topic name>      (first non-comment line)
              ## <exact competency name>              (starts a new chunk)
              ...the notes...
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # HTML comments are for humans
    lines = text.strip().splitlines()
    topic = None
    for line in lines:
        if line.strip():
            match = re.match(r"topic:\s*(.+)", line.strip(), flags=re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
            break
    if not topic:
        raise KnowledgeError(f"{source}: the first line must be 'topic: <topic name>'.")

    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for competency, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        # Split an over-long section at paragraph breaks so each chunk stays focused.
        pieces, current = [], ""
        for paragraph in re.split(r"\n\s*\n", body):
            if current and len(current) + len(paragraph) > MAX_CHUNK_CHARS:
                pieces.append(current)
                current = paragraph
            else:
                current = f"{current}\n\n{paragraph}".strip()
        pieces.append(current)
        for number, piece in enumerate(pieces, start=1):
            chunk_id = f"{_slug(topic)}/{_slug(competency)}" + (f"#{number}" if number > 1 else "")
            chunks.append(Chunk(chunk_id, topic, competency, piece, source))
    if not chunks:
        raise KnowledgeError(f"{source}: no '## competency' sections with text were found.")
    return chunks


def load_chunks(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise KnowledgeError("Two notes have the same topic and competency. Each must be unique.")
    return chunks


def coverage(chunks: list[Chunk], syllabus: dict[str, list[str]]) -> tuple[int, int, list[str], list[str]]:
    """How much of the syllabus do the notes cover?
    Returns (covered, total, missing "topic > competency" names, notes that match nothing)."""
    have = {(c.topic.lower(), c.competency.lower()) for c in chunks}
    wanted = [(t, c) for t, comps in syllabus.items() for c in comps]
    missing = [f"{t} > {c}" for t, c in wanted if (t.lower(), c.lower()) not in have]
    known = {(t.lower(), c.lower()) for t, c in wanted}
    orphans = sorted({f"{c.topic} > {c.competency}" for c in chunks if (c.topic.lower(), c.competency.lower()) not in known})
    return len(wanted) - len(missing), len(wanted), missing, orphans


# ============================================================ searching
def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


class KnowledgeBase:
    """Search the notes. `mode` is "chroma" (vector search) or "keyword" (fallback)."""

    def __init__(self, chunks: list[Chunk], *, use_chroma: bool = True,
                 embed_fn: Callable[[list[str]], list[list[float]]] | None = None):
        """`embed_fn` lets tests (or a different embedding model) supply the embeddings.
        Leave it as None to use Chroma's built-in local model (a small download on first use)."""
        self.chunks = {c.id: c for c in chunks}
        self._embed_fn = embed_fn
        self.mode = "keyword"
        self.fallback_reason = ""
        self._collection = None

        if use_chroma:
            try:
                self._build_chroma(chunks)
                self.mode = "chroma"
            except Exception as e:  # not installed, SQLite too old, model download blocked...
                self._collection = None
                self.fallback_reason = f"{type(e).__name__}: {e}"[:200]

        # The keyword index is cheap, so we always build it (it is also the safety net).
        self._doc_tokens = {c.id: Counter(_tokens(f"{c.competency} {c.text}")) for c in chunks}
        n = max(len(chunks), 1)
        document_frequency = Counter(t for counts in self._doc_tokens.values() for t in counts)
        self._idf = {t: math.log(1 + n / df) for t, df in document_frequency.items()}

    # ---- Chroma ----
    def _build_chroma(self, chunks: list[Chunk]) -> None:
        import chromadb

        client = chromadb.EphemeralClient()  # in memory: rebuilt at every start, nothing to persist
        # A unique name, because EphemeralClient instances in one process share state.
        self._collection = client.get_or_create_collection(
            f"notes_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
        documents = [f"{c.competency}. {c.text}" for c in chunks]
        kwargs = dict(
            ids=[c.id for c in chunks], documents=documents,
            metadatas=[{"topic": c.topic, "competency": c.competency} for c in chunks],
        )
        if self._embed_fn:
            kwargs["embeddings"] = self._embed_fn(documents)
        self._collection.add(**kwargs)

    def _search_chroma(self, query: str, where: dict | None, k: int) -> list[Hit]:
        kwargs = dict(n_results=min(k, max(len(self.chunks), 1)), where=where)
        if self._embed_fn:
            kwargs["query_embeddings"] = self._embed_fn([query])
        else:
            kwargs["query_texts"] = [query]
        result = self._collection.query(**kwargs)
        return [
            Hit(self.chunks[cid], max(0.0, 1.0 - float(distance)))   # cosine distance -> similarity
            for cid, distance in zip(result["ids"][0], result["distances"][0])
        ]

    # ---- keyword fallback ----
    def _search_keyword(self, query: str, topic: str | None, competency: str | None, k: int) -> list[Hit]:
        wanted = Counter(_tokens(query))
        scored = []
        for cid, counts in self._doc_tokens.items():
            chunk = self.chunks[cid]
            if topic and chunk.topic != topic:
                continue
            if competency and chunk.competency != competency:
                continue
            score = sum(self._idf.get(t, 0) * (1 + math.log(counts[t])) for t in wanted if counts.get(t))
            if score > 0:
                scored.append((score, cid))
        scored.sort(reverse=True)
        top = scored[0][0] if scored else 1.0
        return [Hit(self.chunks[cid], score / top) for score, cid in scored[:k]]

    # ---- the public search ----
    def search(self, query: str, *, topic: str | None = None, competency: str | None = None, k: int = 3) -> list[Hit]:
        """Best notes for a query. If a competency is given its notes come FIRST, then other
        notes from the same topic fill the remaining places."""
        query = (query or "").strip()
        if not query:
            return []
        hits: list[Hit] = []
        stages = []
        if topic and competency:
            stages.append((topic, competency))
        stages.append((topic, None))
        for stage_topic, stage_competency in stages:
            if len(hits) >= k:
                break
            for hit in self._search_stage(query, stage_topic, stage_competency, k):
                if all(hit.chunk.id != seen.chunk.id for seen in hits) and len(hits) < k:
                    hits.append(hit)
        return hits

    def _search_stage(self, query: str, topic: str | None, competency: str | None, k: int) -> list[Hit]:
        if self.mode == "chroma":
            try:
                clauses = [{"topic": topic}] if topic else []
                if competency:
                    clauses.append({"competency": competency})
                where = None if not clauses else (clauses[0] if len(clauses) == 1 else {"$and": clauses})
                return self._search_chroma(query, where, k)
            except Exception as e:  # a failure at search time also degrades gracefully
                self.mode, self.fallback_reason = "keyword", f"{type(e).__name__}: {e}"[:200]
        return self._search_keyword(query, topic, competency, k)
