# Spec: rag-lab, lesson 2 — hybrid search (BM25 + dense via RRF)

## Goal
Add keyword search (BM25 via SQLite FTS5) over the existing `chunks` table and
fuse it with the current dense retriever using Reciprocal Rank Fusion (RRF).
Measure dense, BM25, and hybrid side by side on the fixed eval set. This is
the retrieval upgrade that later lessons (reranking) will sit on top of.

## Scope
In scope:
- Build an FTS5 index from the existing `chunks` table (same SQLite file).
- BM25 retrieve + RRF fusion with the existing MiniLM + NumPy dense retrieve.
- A `method` switch on retrieval: `dense` | `bm25` | `hybrid`.
- Evaluation that reports all three methods and writes `eval/results/hybrid.json`.
- Unit tests for RRF, FTS query escaping, and FTS indexing (in-memory).

Out of scope (later lessons or explicit non-goals):
- Cross-encoder reranking (lesson 3).
- Changes to chunking, `eval/questions.yaml`, or `eval/results/baseline.json`.
- Score-weight tuning, a second BM25 library (`rank_bm25`), or a vector DB.

Standing rules (plain Python, no LangChain / vector DB, tests without model
downloads or Ollama, English, `make check`) are in `AGENTS.md`. Follow them;
do not repeat or weaken them.

## Current code to extend
- `rag_lab/ingest.py` — `write_chunks` drops and recreates `data/chunks.sqlite`
  with table `chunks (id INTEGER PRIMARY KEY, file, section, text)`. Create the
  FTS5 virtual table here (or in a helper it calls) after the rows are inserted.
  Re-run `make ingest`. Do not change `chunk_markdown` or other chunking.
- `rag_lab/retrieve.py` — `retrieve(question, k=DEFAULT_TOP_K)` is dense-only
  (`embeddings @ query_vec`, `RetrievedChunk`). Add BM25, RRF, and a `method`
  argument. Keep the `RetrievedChunk` shape.
- `rag_lab/evaluate.py` — `run_eval` / `main` always write
  `eval/results/baseline.json`. That must stop. `make eval` (`uv run python -m
  rag_lab.evaluate`) should print the comparison table and write
  `eval/results/hybrid.json` only.
- `rag_lab/answer.py` — today calls `retrieve(args.question, k=args.k)`. Pass
  the method through (CLI flag, default `hybrid`).
- `rag_lab/embed.py` and `data/embeddings.npy` — unchanged. Re-embedding is
  not required if chunk order/ids stay the same after re-ingest.
- `rag_lab/paths.py` — `DB_PATH`, `DEFAULT_TOP_K` (5), `RESULTS_DIR`. Add
  constants for RRF and candidate-list size if useful; no config framework.

## BM25 via FTS5
Python's stdlib `sqlite3` already includes FTS5 on current CPython (no new
dependency). If FTS5 is missing, fail with a clear error; do not add
`rank_bm25` unless FTS5 is genuinely unavailable and you explain it in the PR.

Tokenizer: `unicode61` with `tokenchars '._-'`, so identifiers such as
`spark.sql.shuffle.partitions` and hyphenated config names stay a single
token. The default `unicode61` (and `porter`) split on `.` and `-`, which
throws away the exact-term signal this lesson is meant to recover.

Query escaping: never pass a raw user question to `MATCH`. Punctuation in
questions (dots in config names, `()`, quotes) breaks FTS5 syntax (e.g.
`syntax error near "."`). Keep it simple: extract tokens with a regex such as
`[A-Za-z0-9._-]+` (same character class as `tokenchars`), escape any `"` by
doubling, wrap each token in double quotes, and join with `OR`.

FTS5's `bm25()` is lower-is-better. Rank with `ORDER BY bm25(...)` ascending.
Map `rowid` back to `chunks.id`. Index `text` (the chunk body already starts
with the heading, so section titles are included).

## RRF
For each question, take the top N results from dense and from BM25 (N = 20,
or `max(20, k)` if the caller asks for k > 20). Fuse by

    score(d) = Σ 1 / (60 + rank_i(d))

with 1-based ranks. A document that appears in only one list still gets that
list's term. Sort by descending RRF score and return the top k as
`RetrievedChunk` (the `score` field is the RRF score for hybrid, the BM25
value for `bm25`, cosine for `dense`). Do not introduce learned or hand-tuned
mixture weights.

## Evaluation
`eval/questions.yaml` is fixed. Hit rule is unchanged (`is_hit` /
`first_hit_rank` in `rag_lab/evaluate.py`: file match + expected section
contained in the chunk heading, case-insensitive). Metrics are still
recall@5 and MRR@5 (`k=5`); ranks beyond 5 are not observed.

`make eval` must:
- Run all three methods (`dense`, `bm25`, `hybrid`) on every question.
- Print one comparison table: per question, the first-hit rank (or `-`) under
  each method, plus the hybrid top-1 `file / section`; then recall@5 and
  MRR@5 for each method.
- Write the same data to `eval/results/hybrid.json` and commit it.
- Leave `eval/results/baseline.json` unchanged.

Committed dense baseline (do not regenerate or overwrite this file):

| id  | rank | notes |
|-----|------|-------|
| q1  | -    | miss  |
| q2  | 1    |       |
| q3  | 1    |       |
| q4  | 4    |       |
| q5  | 2    |       |
| q6  | -    | miss  |
| q7  | -    | miss  |
| q8  | 4    |       |
| q9  | 1    |       |
| q10 | 5    |       |

recall@5 = 0.70, MRR@5 = 0.42. Baseline hits in the top-5 are q2, q3, q4, q5,
q8, q9, q10.

## Tests (`make check` = ruff + pytest)
Add tests that do not download models, touch `data/`, or call Ollama:
- RRF: hand-made ranked id lists; check 1-based ranks, k=60, a doc in both
  lists beats a doc in one, and a doc missing from a list is not treated as
  rank 0.
- FTS query escaping: questions with dots, parentheses, and quotes produce a
  safe MATCH string (quoted OR tokens) and do not raise a syntax error.
- FTS index: a tiny in-memory `chunks` table (a few short English texts,
  including one dotted identifier) can be indexed and searched; the identifier
  query returns the matching chunk first; `bm25()` sort order is lower-is-better.

Existing chunking and metrics tests stay green.

## Verification before opening the PR
1. `make ingest` rebuilds `data/chunks.sqlite` with the FTS index.
   `make embed` only if embeddings are missing or chunk ids changed.
2. `make eval` prints the three-method table, writes `eval/results/hybrid.json`,
   and does not modify `eval/results/baseline.json`.
3. `make check` passes; CI still only runs `make check`.
4. `uv run python -m rag_lab.answer "What is the default value of spark.sql.shuffle.partitions and when should I change it?"`
   shows the hybrid (or `--method`) ranking. If Ollama is unavailable, say so;
   retrieval-only is enough.
5. Update the README: hybrid search, how to choose `method`, and that
   `make eval` now writes `hybrid.json` (baseline stays as the lesson-1 snapshot).

## Definition of done
One PR, independently testable, that leaves a working pipeline. Success:
- q1's first hit is in the hybrid top-5.
- No baseline top-5 hit (q2, q3, q4, q5, q8, q9, q10) drops out of the
  hybrid top-5. q6 and q7 may still miss; fixing them is not required.
- `make check` passes.
- The PR description includes the comparison table and a short note on
  questions where the methods disagree (especially q1).
