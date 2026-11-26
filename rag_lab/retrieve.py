"""Retrieve top-k chunks: dense, BM25, hybrid (RRF), or hybrid + cross-encoder rerank."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from rag_lab.paths import (
    CANDIDATE_N,
    CROSS_ENCODER_MODEL,
    DB_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    RRF_K,
)

Method = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]

# Scorer: CrossEncoder-like (.predict) or callable(pairs) -> scores.
RerankScorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]] | Any

# Same character class as FTS5 tokenchars '._-' plus alphanumerics.
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]+")
# Dotted identifier: a token with '.' between word characters.
_DOTTED_IN_TOKEN_RE = re.compile(r"[A-Za-z0-9]\.[A-Za-z0-9]")


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


def _torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _cross_encoder() -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CROSS_ENCODER_MODEL, device=_torch_device())


def cross_encoder_device() -> str:
    """Device used by the cross-encoder (loads the model on first call)."""
    model = _cross_encoder()
    device = getattr(model, "device", None)
    if device is not None:
        text = str(device)
        if text.startswith("cuda"):
            return "cuda"
        if text.startswith("mps"):
            return "mps"
        return "cpu"
    return _torch_device()


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


def _quote_fts_token(tok: str) -> str:
    return '"' + tok.replace('"', '""') + '"'


def escape_fts_query(question: str) -> str:
    """Build a safe FTS5 MATCH string.

    Tokens are quoted. Without dotted identifiers, join with OR.
    If any token is a dotted identifier ('.' between word characters), require
    every such identifier and OR the remaining tokens:
    ``"id" AND ("tok1" OR "tok2" ...)``.
    """
    tokens = _FTS_TOKEN_RE.findall(question)
    if not tokens:
        return '""'

    identifiers = [t for t in tokens if _DOTTED_IN_TOKEN_RE.search(t)]
    others = [t for t in tokens if not _DOTTED_IN_TOKEN_RE.search(t)]

    if not identifiers:
        return " OR ".join(_quote_fts_token(t) for t in tokens)

    required = " AND ".join(_quote_fts_token(t) for t in identifiers)
    if not others:
        return required
    optional = " OR ".join(_quote_fts_token(t) for t in others)
    return f"{required} AND ({optional})"


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


def _score_pairs(
    scorer: RerankScorer,
    pairs: Sequence[tuple[str, str]],
) -> Sequence[float]:
    if hasattr(scorer, "predict"):
        return scorer.predict(pairs)
    return scorer(pairs)


def rerank_chunks(
    question: str,
    candidates: list[RetrievedChunk],
    k: int,
    *,
    scorer: RerankScorer | None = None,
) -> list[RetrievedChunk]:
    """Rerank candidates by cross-encoder score; return top-k.

    Scores pairs ``(question, chunk.text)``. Tied scores keep the incoming
    hybrid order (stable sort). Does not add or drop ids before the k cut.
    """
    if not candidates:
        return []
    if scorer is None:
        scorer = _cross_encoder()

    pairs = [(question, c.text) for c in candidates]
    raw_scores = _score_pairs(scorer, pairs)
    scored = [
        RetrievedChunk(
            id=c.id,
            file=c.file,
            section=c.section,
            text=c.text,
            score=float(raw_scores[i]),
        )
        for i, c in enumerate(candidates)
    ]
    # Stable: equal scores keep incoming candidate order.
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]


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
            (match, k),
        ).fetchall()
    finally:
        if own_conn:
            conn.close()

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


def retrieve_hybrid_rerank(
    question: str,
    k: int = DEFAULT_TOP_K,
    *,
    scorer: RerankScorer | None = None,
) -> list[RetrievedChunk]:
    """Hybrid top-N candidates, then cross-encoder rerank to top-k."""
    n = _candidate_n(k)
    candidates = retrieve_hybrid(question, k=n)
    return rerank_chunks(question, candidates, k=k, scorer=scorer)


def retrieve(
    question: str,
    k: int = DEFAULT_TOP_K,
    method: Method = "hybrid_rerank",
) -> list[RetrievedChunk]:
    """Return top-k chunks using dense, bm25, hybrid, or hybrid_rerank."""
    if method == "dense":
        return retrieve_dense(question, k=k)
    if method == "bm25":
        return retrieve_bm25(question, k=k)
    if method == "hybrid":
        return retrieve_hybrid(question, k=k)
    if method == "hybrid_rerank":
        return retrieve_hybrid_rerank(question, k=k)
    raise ValueError(f"Unknown retrieval method: {method!r}")
