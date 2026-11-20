"""Retrieve top-k chunks by cosine similarity over local embeddings."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from rag_lab.paths import DB_PATH, DEFAULT_TOP_K, EMBEDDING_MODEL, EMBEDDINGS_PATH


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


def _load_chunks_by_id() -> dict[int, tuple[str, str, str]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT id, file, section, text FROM chunks ORDER BY id").fetchall()
    finally:
        conn.close()
    return {row[0]: (row[1], row[2], row[3]) for row in rows}


def retrieve(question: str, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """Return the top-k chunks most similar to the question."""
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
