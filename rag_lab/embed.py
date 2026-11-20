"""Embed all chunks with sentence-transformers and write a .npy matrix."""

from __future__ import annotations

import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_lab.paths import DB_PATH, EMBEDDING_MODEL, EMBEDDINGS_PATH


def load_chunk_texts(db_path=DB_PATH) -> tuple[list[int], list[str]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    finally:
        conn.close()
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return ids, texts


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def main() -> None:
    ids, texts = load_chunk_texts()
    if not texts:
        raise SystemExit(f"No chunks in {DB_PATH}; run `make ingest` first.")

    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")
    matrix = embed_texts(texts)
    assert matrix.shape[0] == len(ids)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, matrix)
    print(f"Wrote {matrix.shape} embeddings -> {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
