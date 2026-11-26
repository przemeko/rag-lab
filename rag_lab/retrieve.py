"""Retrieve top-k chunks: dense (MiniLM), BM25 (FTS5), or hybrid (RRF)."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from rag_lab.paths import (
    CANDIDATE_N,
    DB_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    RRF_K,
)

Method = Literal["dense", "bm25", "hybrid"]

# Same character class as FTS5 tokenchars '._-' plus alphanumerics.
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]+")
# Multi-segment dotted names (spark.sql.shuffle.partitions, etc.).
_DOTTED_ID_RE = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9._-]+)+")
# Wider BM25 pool so stopword-heavy OR queries still surface rare identifiers.
_BM25_POOL = 100


@dataclass
class RetrievedChunk:
    id: int
    file: str
    section: str
    text: str
    score: float


@lru_cache(maxsize=1)
def _model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _embeddings() -> np.ndarray:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing {EMBEDDINGS_PATH}; run `make embed` first.")
    return np.load(EMBEDDINGS_PATH)


def _load_chunks_by_id(conn: sqlite3.Connection | None = None) -> dict[int, tuple[str, str, str]]:
    own = conn is None
    if own:
        conn = sqlite3.connect(DB_PATH)
    assert conn is not None
    try:
        rows = conn.execute("SELECT id, file, section, text FROM chunks ORDER BY id").fetchall()
    finally:
        if own:
            conn.close()
    return {row[0]: (row[1], row[2], row[3]) for row in rows}


def escape_fts_query(question: str) -> str:
    """Build a safe FTS5 MATCH string: quoted tokens joined with OR."""
    tokens = _FTS_TOKEN_RE.findall(question)
    if not tokens:
        return '""'
    parts: list[str] = []
    for tok in tokens:
        escaped = tok.replace('"', '""')
        parts.append(f'"{escaped}"')
    return " OR ".join(parts)


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fuse ranked id lists with RRF. Returns (id, score) sorted by score desc.

    Ranks are 1-based. A document missing from a list contributes nothing
    (not rank 0).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _candidate_n(k: int) -> int:
    return max(CANDIDATE_N, k)


def _definitional_boost(text: str, dotted: list[str]) -> int:
    """Prefer HTML/markdown table rows that define a dotted config name."""
    if not dotted:
        return 0
    boost = 0
    for term in dotted:
        if f"<code>{term}</code></td>" in text:
            boost = max(boost, 2)
        elif f"<code>{term}</code>" in text or f"`{term}`" in text:
            boost = max(boost, 1)
    return boost


def retrieve_dense(question: str, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """Return the top-k chunks by cosine similarity."""
    embeddings = _embeddings()
    chunks = _load_chunks_by_id()
    ids = sorted(chunks.keys())
    if len(ids) != embeddings.shape[0]:
        raise RuntimeError(
            f"Chunk count ({len(ids)}) != embedding rows ({embeddings.shape[0]}); "
            "re-run ingest and embed."
        )

    query = _model().encode([question], normalize_embeddings=True)
    query_vec = np.asarray(query, dtype=np.float32)[0]
    scores = embeddings @ query_vec  # cosine: both sides L2-normalized

    top_idx = np.argsort(-scores)[:k]
    results: list[RetrievedChunk] = []
    for idx in top_idx:
        chunk_id = ids[int(idx)]
        file, section, text = chunks[chunk_id]
        results.append(
            RetrievedChunk(
                id=chunk_id,
                file=file,
                section=section,
                text=text,
                score=float(scores[int(idx)]),
            )
        )
    return results


def retrieve_bm25(
    question: str,
    k: int = DEFAULT_TOP_K,
    *,
    db_path: sqlite3.Connection | str | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k chunks by FTS5 bm25 (lower raw score is better)."""
    match = escape_fts_query(question)
    dotted = _DOTTED_ID_RE.findall(question)
    pool = max(_BM25_POOL, k)

    own_conn = False
    if isinstance(db_path, sqlite3.Connection):
        conn = db_path
    else:
        path = db_path if db_path is not None else str(DB_PATH)
        conn = sqlite3.connect(path)
        own_conn = True
    try:
        rows = conn.execute(
            """
            SELECT c.id, c.file, c.section, c.text, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (match, pool),
        ).fetchall()
    finally:
        if own_conn:
            conn.close()

    # BM25 length-normalization prefers short cross-references over the HTML
    # table row that defines a property. When the question names a dotted id,
    # promote definitional code/table mentions, then fall back to bm25 order.
    rows = sorted(
        rows,
        key=lambda r: (-_definitional_boost(r[3], dotted), r[4]),
    )[:k]

    # FTS5 bm25() is lower-is-better; keep the raw value as the chunk score.
    return [
        RetrievedChunk(
            id=int(row[0]),
            file=row[1],
            section=row[2],
            text=row[3],
            score=float(row[4]),
        )
        for row in rows
    ]


def retrieve_hybrid(question: str, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """Fuse dense and BM25 rankings with RRF; return top-k."""
    n = _candidate_n(k)
    dense = retrieve_dense(question, k=n)
    bm25 = retrieve_bm25(question, k=n)
    fused = reciprocal_rank_fusion(
        [[c.id for c in dense], [c.id for c in bm25]],
        k=RRF_K,
    )
    chunks = _load_chunks_by_id()
    results: list[RetrievedChunk] = []
    for chunk_id, score in fused[:k]:
        file, section, text = chunks[chunk_id]
        results.append(
            RetrievedChunk(
                id=chunk_id,
                file=file,
                section=section,
                text=text,
                score=float(score),
            )
        )
    return results


def retrieve(
    question: str,
    k: int = DEFAULT_TOP_K,
    method: Method = "hybrid",
) -> list[RetrievedChunk]:
    """Return top-k chunks using dense, bm25, or hybrid (RRF) retrieval."""
    if method == "dense":
        return retrieve_dense(question, k=k)
    if method == "bm25":
        return retrieve_bm25(question, k=k)
    if method == "hybrid":
        return retrieve_hybrid(question, k=k)
    raise ValueError(f"Unknown retrieval method: {method!r}")
