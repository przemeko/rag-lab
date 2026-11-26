# rag-lab

Learning project: retrieve-and-generate answers over the Apache Spark
documentation. Lesson 2 adds **hybrid search** (BM25 via SQLite FTS5 + dense
MiniLM, fused with Reciprocal Rank Fusion). Lesson 1's dense-only baseline
remains in `eval/results/baseline.json` as the comparison snapshot.

Later lessons (reranking, …) build on this hybrid retriever and must be
compared with `make eval`.

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
make eval     # dense vs bm25 vs hybrid -> eval/results/hybrid.json
```

Ask a question (retrieval always; generation needs Ollama). Default method is
`hybrid`; choose with `--method dense|bm25|hybrid`:

```bash
uv run python -m rag_lab.answer "What does spark.sql.adaptive.skewJoin.enabled do?"
uv run python -m rag_lab.answer --method bm25 "What is spark.sql.shuffle.partitions?"
```

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
| `rag_lab/retrieve.py` | dense / BM25 / hybrid (RRF) top-k |
| `rag_lab/answer.py` | retrieve + Ollama (`--method`) |
| `rag_lab/evaluate.py` | recall@5 / MRR@5 for all three methods |
| `eval/questions.yaml` | fixed evaluation set |
| `eval/results/baseline.json` | lesson-1 dense snapshot (do not overwrite) |
| `eval/results/hybrid.json` | lesson-2 comparison output from `make eval` |
| `data/` | gitignored docs, SQLite, embeddings |
| `notebooks/` | gitignored local Jupyter notebooks |

Pinned Apache Spark commit: see `SPARK_COMMIT` in `rag_lab/paths.py`
(currently the `v3.5.4` tag).

## Evaluation notes

- Reported **MRR** is **MRR@5**: retrieval only returns top-5, so ranks beyond 5 are never observed.
- `make eval` prints dense / bm25 / hybrid side by side and writes `hybrid.json` only.
  `baseline.json` is the frozen lesson-1 dense result.
- `all-MiniLM-L6-v2` truncates at **256 tokens** (~1000 characters). Chunks may be up to
  2000 characters (`MAX_CHUNK_CHARS`), so the tail of a long chunk is silently dropped
  from the embedding. Worth revisiting when comparing methods.
- FTS5 tokenizer keeps `.` and `-` inside tokens (`unicode61` + `tokenchars '._-'`) so
  config names like `spark.sql.shuffle.partitions` match as a single term.

## Constraints

Plain Python — no LangChain, LlamaIndex, or vector DB. Storage is SQLite
(chunks + FTS5) and a `.npy` file for embeddings.
