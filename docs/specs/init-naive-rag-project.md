# Spec: rag-lab, lesson 1 — naive RAG baseline over the Apache Spark docs

## Goal
Build the simplest working RAG pipeline over the Apache Spark documentation
and measure its retrieval quality on a fixed evaluation set. This is the
baseline that later lessons (hybrid search, reranking) must beat.

## Scope
In scope:
- Ingest the Markdown files from `docs/` of apache/spark (pinned commit).
- Chunk by Markdown headings.
- Embed chunks with sentence-transformers and store them locally.
- Retrieve top-k chunks by cosine similarity.
- Answer a question with a local LLM via Ollama, using retrieved chunks.
- Evaluate retrieval with recall@5 and MRR on `eval/questions.yaml`.

Out of scope (later lessons): BM25, hybrid search, reranking, contextual
retrieval, answer-quality (faithfulness) evaluation, any web UI.

## Constraints
- Plain Python 3.12, no LangChain / LlamaIndex.
- Dependencies: `sentence-transformers`, `numpy`, `pyyaml`, `requests`
  (for Ollama HTTP API), `pytest`. Nothing else without a reason in the PR.
- Storage: SQLite (stdlib) for chunks + a `.npy` file for embeddings.
  No vector database.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Runs locally on a MacBook; no API keys, no network calls except
  downloading the docs, the model, and talking to local Ollama.
- Code, comments, README and PR description in English.
- Simplest readable code; no abstractions "for later lessons".

## Repository layout
rag_lab/
  ingest.py     # download docs at a pinned commit, chunk, write SQLite
  embed.py      # embed all chunks, write embeddings .npy
  retrieve.py   # retrieve(question, k) -> list of chunks with scores
  answer.py     # CLI: question -> retrieved chunks -> Ollama answer
  evaluate.py   # CLI: recall@5 and MRR over eval/questions.yaml
eval/
  questions.yaml
tests/
data/           # gitignored: docs checkout, SQLite, embeddings
Makefile        # make install, make ingest, make embed, make eval, make check
README.md

## Chunking
- One chunk = one section: text from a `##` or `###` heading up to the
  next heading of the same or higher level.
- Each chunk stores: `id`, `file` (e.g. `sql-performance-tuning.md`),
  `section` (heading text), `text`.
- Strip Jekyll front matter and Liquid tags (`{% ... %}`).
- Sections longer than ~2000 characters are split into consecutive pieces
  with the same `file` and `section`.

## Evaluation set
`eval/questions.yaml` holds the 10 questions below. A question counts as
a hit at rank r if the chunk at rank r matches ANY expected `file` +
`section` pair (section match = expected section text is contained in
the chunk's heading, case-insensitive).

- id: q1
  question: What is the default value of spark.sql.shuffle.partitions and when should I change it?
  expected: [{file: sql-performance-tuning.md, section: Tuning Partitions}]
- id: q2
  question: What does spark.sql.files.maxPartitionBytes control?
  expected: [{file: sql-performance-tuning.md, section: Tuning Partitions}]
- id: q3
  question: What does spark.sql.adaptive.skewJoin.enabled do?
  expected: [{file: sql-performance-tuning.md, section: Optimizing Skew Join}]
- id: q4
  question: What is the difference between repartition() and coalesce()?
  expected: [{file: rdd-programming-guide.md, section: Transformations},
             {file: rdd-programming-guide.md, section: Shuffle operations}]
- id: q5
  question: What does spark.history.fs.logDirectory configure in the history server?
  expected: [{file: monitoring.md, section: Viewing After the Fact}]
- id: q6
  question: My job writes thousands of tiny output files. How can I reduce their number?
  expected: [{file: sql-performance-tuning.md, section: Coalesce Hints},
             {file: sql-performance-tuning.md, section: Coalescing Post Shuffle Partitions}]
- id: q7
  question: One task in a stage runs much longer than all the others. Why, and how can Spark handle it automatically?
  expected: [{file: sql-performance-tuning.md, section: Optimizing Skew Join},
             {file: sql-performance-tuning.md, section: Splitting skewed shuffle partitions}]
- id: q8
  question: When is it worth keeping a DataFrame in memory between actions, and what does it cost?
  expected: [{file: sql-performance-tuning.md, section: Caching Data},
             {file: rdd-programming-guide.md, section: RDD Persistence}]
- id: q9
  question: Where does the driver run when I submit a job to a cluster, and how do I choose?
  expected: [{file: submitting-applications.md, section: Launching Applications with spark-submit}]
- id: q10
  question: Why do workers fail to connect to the master in a standalone cluster?
  expected: [{file: spark-standalone.md, section: Cluster Launch Scripts}]

Before opening the PR, verify that every expected file + section exists
in the ingested chunks; if a section name is wrong, fix it in the YAML
and say so in the PR description. Do NOT change the questions.

## Output of `make eval`
A table with one row per question (id, rank of first hit or "-",
top-1 file + section), then overall recall@5 and MRR. Also write
the same result to `eval/results/baseline.json` and commit it.

## Tests (`make check` = pytest + ruff)
- Chunking: a small Markdown fixture produces the expected chunks
  (front matter stripped, nested headings, long-section split).
- Metrics: recall@5 and MRR on hand-made rankings, including
  "no hit" and "multiple expected sections".
- Tests must not download models or call Ollama.

## Verification before opening the PR
1. `make install && make ingest && make embed && make eval` runs cleanly
   from a fresh clone.
2. `make check` passes; CI runs `make check` on every PR.
3. `python -m rag_lab.answer "What does spark.sql.adaptive.skewJoin.enabled do?"`
   prints retrieved chunks (file + section) and an answer, if Ollama runs.
   If Ollama is unavailable in your environment, say so in the PR.
4. The PR description includes the `make eval` table and the pinned
   apache/spark commit.

## Definition of done
One PR that leaves a working pipeline, a committed baseline result,
and a README explaining how to run each step.