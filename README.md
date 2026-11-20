# rag-lab

Learning project: retrieve-and-generate answers over the Apache Spark
documentation. Lesson 1 is a **naive RAG baseline** (heading chunks +
dense embeddings + cosine top-k + local Ollama).

Later lessons add techniques (hybrid search, reranking, …) and must beat
this baseline on `make eval`.

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
make ingest   # download Spark docs (pinned commit), chunk, write data/chunks.sqlite
make embed    # embed chunks -> data/embeddings.npy
make eval     # recall@5 + MRR@5 on eval/questions.yaml -> eval/results/baseline.json
```

Ask a question (retrieval always; generation needs Ollama):

```bash
uv run python -m rag_lab.answer "What does spark.sql.adaptive.skewJoin.enabled do?"
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
| `rag_lab/ingest.py` | Download docs, chunk by `#`/`##`/`###`, SQLite |
| `rag_lab/embed.py` | sentence-transformers embeddings |
| `rag_lab/retrieve.py` | cosine top-k |
| `rag_lab/answer.py` | retrieve + Ollama |
| `rag_lab/evaluate.py` | recall@5 / MRR@5 |
| `eval/questions.yaml` | fixed evaluation set |
| `data/` | gitignored docs, SQLite, embeddings |
| `notebooks/` | gitignored local Jupyter notebooks |

Pinned Apache Spark commit: see `SPARK_COMMIT` in `rag_lab/paths.py`
(currently the `v3.5.4` tag).

## Evaluation notes

- Reported **MRR** is **MRR@5**: retrieval only returns top-5, so ranks beyond 5 are never observed.
- `all-MiniLM-L6-v2` truncates at **256 tokens** (~1000 characters). Chunks may be up to
  2000 characters (`MAX_CHUNK_CHARS`), so the tail of a long chunk is silently dropped
  from the embedding. Worth revisiting when comparing methods.

## Constraints

Plain Python — no LangChain, LlamaIndex, or vector DB. Storage is SQLite
for chunks and a `.npy` file for embeddings.
