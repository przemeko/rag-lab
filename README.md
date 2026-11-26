# rag-lab

Learning project: retrieve-and-generate answers over the Apache Spark
documentation. Lesson 3 adds **cross-encoder reranking** on top of lesson-2
hybrid (dense MiniLM + BM25 via RRF). Lesson 1 dense and lesson 2 hybrid
snapshots stay in `eval/results/baseline.json` and `eval/results/hybrid.json`.

Later lessons build on this pipeline and must be compared with `make eval`.

## Requirements

- Python 3.12+ (managed via [uv](https://docs.astral.sh/uv/))
- Git
- Optional: [Ollama](https://ollama.com/) with a local model (default `llama3.2`) for `answer`

## Setup

```bash
make install
```

## Pipeline

```bash
make ingest   # download Spark docs, chunk, write data/chunks.sqlite (+ FTS5 index)
make embed    # embed chunks -> data/embeddings.npy (skip if unchanged)
make eval     # dense / bm25 / hybrid / hybrid_rerank -> eval/results/rerank.json
```

Ask a question (retrieval always; generation needs Ollama). Default method is
`hybrid_rerank`; choose with `--method dense|bm25|hybrid|hybrid_rerank`:

```bash
uv run python -m rag_lab.answer "What does spark.sql.adaptive.skewJoin.enabled do?"
uv run python -m rag_lab.answer --method hybrid "What is spark.sql.shuffle.partitions?"
```

## Retrieval

| Method | Behavior |
|--------|----------|
| `dense` | MiniLM embeddings, cosine similarity, top-k |
| `bm25` | SQLite FTS5 `bm25()` ascending over chunk text, top-k |
| `hybrid` | Top N=20 from dense and from BM25, fused with RRF (`1/(60+rank)`), then top-k |
| `hybrid_rerank` (default) | Hybrid top N=20, then `cross-encoder/ms-marco-MiniLM-L-6-v2` scores `(question, chunk.text)` and returns top-k |

FTS5 uses `unicode61` with `tokenchars '._-'` so names like
`spark.sql.shuffle.partitions` stay one token. MATCH queries quote tokens; if
the question contains a dotted identifier, that token is required
(`"id" AND ("tok1" OR …)`). Otherwise tokens are joined with OR.

## Checks

```bash
make check    # ruff + pytest (no model downloads, no Ollama)
```

## Notebooks

```bash
make notebook   # Jupyter Notebook with cwd notebooks/ (gitignored)
```

## Layout

| Path | Role |
|------|------|
| `rag_lab/ingest.py` | Download docs, chunk by `#`/`##`/`###`, SQLite + FTS5 |
| `rag_lab/embed.py` | sentence-transformers embeddings |
| `rag_lab/retrieve.py` | dense / BM25 / hybrid (RRF) / hybrid_rerank top-k |
| `rag_lab/answer.py` | retrieve + Ollama (`--method`) |
| `rag_lab/evaluate.py` | recall@5 / MRR@5 for all four methods + latency |
| `eval/questions.yaml` | fixed evaluation set |
| `eval/results/baseline.json` | lesson-1 dense snapshot (do not overwrite) |
| `eval/results/hybrid.json` | lesson-2 comparison snapshot (do not overwrite) |
| `eval/results/rerank.json` | lesson-3 comparison output from `make eval` |
| `data/` | gitignored docs, SQLite, embeddings |
| `notebooks/` | gitignored local Jupyter notebooks |

Pinned Apache Spark commit: see `SPARK_COMMIT` in `rag_lab/paths.py`
(currently the `v3.5.4` tag).

## Evaluation notes

- Reported **MRR** is **MRR@5**: retrieval only returns top-5, so ranks beyond 5 are never observed.
- `make eval` prints dense / bm25 / hybrid / hybrid_rerank side by side, mean
  latency for hybrid vs hybrid_rerank (with device), and writes `rerank.json` only.
  `baseline.json` and `hybrid.json` are frozen lesson snapshots.
- `all-MiniLM-L6-v2` truncates at **256 tokens** (~1000 characters). Chunks may be up to
  2000 characters (`MAX_CHUNK_CHARS`), so the tail of a long chunk is silently dropped
  from the embedding.
- The cross-encoder truncates each `(question, chunk.text)` pair at **512 tokens**.
  Long chunks may lose their tail; use text as stored (no wrapping or cropping).

## Constraints

Plain Python — no LangChain, LlamaIndex, or vector DB. Storage is SQLite
(chunks + FTS5) and a `.npy` file for embeddings.
