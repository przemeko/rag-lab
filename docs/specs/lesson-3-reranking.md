# Spec: rag-lab, lesson 3 — cross-encoder reranking

## Goal
Rerank the hybrid (dense + BM25 via RRF) candidate list with a cross-encoder
so that a document already in the top-20, but not the top-5, can rise. Measure
dense, BM25, hybrid, and hybrid+rerank side by side, including mean retrieval
latency. This sits on top of the lesson-2 retriever; it must not change
candidate generation.

## Scope
In scope:
- Rerank the hybrid top-N (N = `CANDIDATE_N` = 20, fixed) with
  `sentence-transformers` `CrossEncoder` model
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- A new retrieval method `hybrid_rerank` on the existing `method` switch.
  It becomes the default for `retrieve()` and `rag_lab/answer.py`.
- Evaluation that reports all four methods, mean per-question latency for
  `hybrid` vs `hybrid_rerank`, and writes `eval/results/rerank.json`.
- Unit tests of rerank ordering with an injected fake scorer (no model
  download).

Out of scope:
- Chunking changes, new eval questions, or edits to
  `eval/questions.yaml`, `eval/results/baseline.json`, or
  `eval/results/hybrid.json`.
- Mixing the cross-encoder score with the RRF score, learned or hand-tuned
  weights, ColBERT, or LLM-as-judge.
- Trying to recover q6 or q7: they are not in the hybrid top-20, and a
  reranker cannot add documents that were never retrieved.

Standing rules (plain Python, no LangChain / vector DB, tests without model
downloads or Ollama, English, `make check`) are in `AGENTS.md`. Follow them;
do not repeat or weaken them.

## Lesson from PR #3 (binding)
No corpus-specific heuristics: no rules based on HTML or Markdown structure,
file names, section headings, or the eval questions. No tuning on the eval
set beyond what this spec states. No score mixing between the reranker and
RRF (the cross-encoder score alone reorders the hybrid candidates).

If a success criterion is not met, report it honestly in the PR description
with the actual ranks and a short explanation. Do not add tweaks to chase
the number. An honest miss is acceptable DoD for this lesson; a tweaked win
is not.

## Current code to extend
- `rag_lab/retrieve.py` — `retrieve(question, k=DEFAULT_TOP_K, method="hybrid")`
  dispatches to `retrieve_dense`, `retrieve_bm25`, and `retrieve_hybrid`.
  `retrieve_hybrid` already fuses top `max(CANDIDATE_N, k)` from each side
  with RRF (`RRF_K` = 60) and returns `RetrievedChunk`. Add
  `hybrid_rerank`: call `retrieve_hybrid(..., k=max(CANDIDATE_N, k))`,
  rerank those candidates, return the top k. Keep the `RetrievedChunk`
  shape; `score` is the cross-encoder score (higher is better).
- `rag_lab/paths.py` — `RETRIEVAL_METHODS = ("dense", "bm25", "hybrid")`,
  `CANDIDATE_N = 20`, `DEFAULT_TOP_K = 5`. Append `"hybrid_rerank"` to the
  tuple (it drives the eval loop and the `answer.py` `--method` choices).
  Add a model-name constant if useful; no config framework.
- `rag_lab/evaluate.py` — `run_eval` loops `RETRIEVAL_METHODS`, but
  `print_table` / comparison rows are hard-coded to three rank columns and
  hybrid top-1, and `main()` always writes `eval/results/hybrid.json`.
  That must stop. `make eval` (`uv run python -m rag_lab.evaluate`) should
  print the four-method table plus latencies and write
  `eval/results/rerank.json` only. Leave `baseline.json` and `hybrid.json`
  untouched.
- `rag_lab/answer.py` — `--method` already uses `RETRIEVAL_METHODS`; change
  the default from `hybrid` to `hybrid_rerank`.
- `rag_lab/ingest.py`, `rag_lab/embed.py`, chunking, and the FTS index —
  unchanged. Re-ingest / re-embed is not required.

## Cross-encoder rerank
Use `sentence_transformers.CrossEncoder`. `sentence-transformers` is already
a project dependency (see `pyproject.toml`); do not add a new package. The
model downloads at runtime on first `make eval` / `answer` use, the same
way MiniLM does. Tests and CI (`make check`) must not download it.

For each question, score the hybrid candidates as pairs
`(question, chunk.text)` and sort by descending score. Use `chunk.text` as
stored: the heading is already at the start of a section's first piece; do
not prepend `file` or `section`, and do not invent other wrappers.

This model truncates each pair at **512 tokens**. Long chunks may lose
their tail; split pieces of a long section may also start mid-text with no
heading. Use the text as stored. Do not change chunking, window, or crop
the text to dodge truncation.

N = 20 is constant (`CANDIDATE_N`). Eval and `answer` still return top-k
with `k = DEFAULT_TOP_K` (5). If the caller asks for k > 20, rerank
`max(CANDIDATE_N, k)` hybrid candidates and cut to k. Rerank only reorders
the candidate list; it never adds or drops ids before the k cut. On a tied
score, keep a deterministic order (stable with respect to the incoming
hybrid order, or by chunk id — pick one and test it).

Do not blend the cross-encoder score with the RRF score.

## Evaluation
`eval/questions.yaml` is fixed. Hit rule is unchanged (`is_hit` /
`first_hit_rank` in `rag_lab/evaluate.py`: file match + expected section
contained in the chunk heading, case-insensitive). Metrics are still
recall@5 and MRR@5 (`k=5`); ranks beyond 5 are not observed in the default
table. For a miss that might sit just outside the top-5, inspect the
uncut reranked list so the PR can report the actual first-hit rank.

`make eval` must:
- Run all four methods (`dense`, `bm25`, `hybrid`, `hybrid_rerank`) on
  every question.
- Print one comparison table: per question, the first-hit rank (or `-`)
  under each method, plus the `hybrid_rerank` top-1 `file / section`; then
  recall@5 and MRR@5 for each method.
- Report mean per-question wall time of `retrieve(..., k=5, method=...)`
  for `hybrid` and `hybrid_rerank`. Exclude one-time model load (warmup
  call is fine). State the device (`cpu`, `mps`, or `cuda`). Put the means
  and the device in the printed table and in `rerank.json`.
- Write the same data to `eval/results/rerank.json` and commit it.
- Leave `eval/results/baseline.json` and `eval/results/hybrid.json`
  unchanged.

Committed lesson-2 hybrid snapshot (`eval/results/hybrid.json`; do not
regenerate or overwrite this file). Ranks are first-hit @5; q1's first
hit is rank 6 if you retrieve the hybrid top-20, so the table shows `-`.

| id  | dense | bm25 | hybrid | notes |
|-----|-------|------|--------|-------|
| q1  | -     | 4    | - (6)  | miss @5; in hybrid top-20 |
| q2  | 1     | -    | 1      |       |
| q3  | 1     | -    | 1      |       |
| q4  | 4     | -    | 4      |       |
| q5  | 2     | 1    | 2      |       |
| q6  | -     | -    | -      | miss; not in hybrid top-20 |
| q7  | -     | -    | -      | miss; not in hybrid top-20 |
| q8  | 4     | -    | 3      |       |
| q9  | 1     | -    | 4      |       |
| q10 | 5     | 1    | 3      |       |

| method | recall@5 | MRR@5 |
|--------|----------|-------|
| dense  | 0.70     | 0.42  |
| bm25   | 0.30     | 0.225 |
| hybrid | 0.70     | 0.367 |

Baseline top-5 hits (lesson 1, still the no-regression set): q2, q3, q4,
q5, q8, q9, q10.

## Tests (`make check` = ruff + pytest)
Add tests that do not download models, touch `data/`, or call Ollama.
Inject the scorer (a callable, or a fake `CrossEncoder`-like object, passed
into the rerank helper) so the real model is never loaded in tests:
- Ordering: a hand-made candidate list comes back sorted by the fake
  scores, highest first; `RetrievedChunk.score` is the scorer's value.
- Rerank only reorders: every returned id was in the input; no new chunk
  is invented.
- The k cut is applied after reordering (k=2 of 4 candidates, etc.).
- Tied scores use the documented deterministic order.

Existing chunking, metrics, and hybrid tests stay green.

## Verification before opening the PR
1. No need to re-run `make ingest` or `make embed` unless those artifacts
   are missing.
2. `make eval` prints the four-method table and latencies, writes
   `eval/results/rerank.json`, and does not modify `baseline.json` or
   `hybrid.json`.
3. `make check` passes; CI still only runs `make check`.
4. `uv run python -m rag_lab.answer "What is the default value of spark.sql.shuffle.partitions and when should I change it?"`
   shows the `hybrid_rerank` (or `--method`) ranking. If Ollama is
   unavailable, say so; retrieval-only is enough.
5. Update the README: the new default method, how to choose `method`,
   that `make eval` now writes `rerank.json` (baseline and hybrid stay as
   snapshots), the 512-token truncation note, and latency.

## Definition of done
One PR, independently testable, that leaves a working pipeline.

Target success:
- q1's first hit is in the `hybrid_rerank` top-5.
- recall@5 >= 0.70, and no baseline top-5 hit (q2, q3, q4, q5, q8, q9,
  q10) drops out of the `hybrid_rerank` top-5. q6 and q7 may still miss;
  do not try to fix them.
- MRR@5 > the committed hybrid value 0.367.
- `make check` passes.
- The PR description includes the comparison table, mean latencies with
  device, and a short note on questions where the methods disagree
  (especially q1).

If q1 does not enter the top-5, or MRR@5 does not beat 0.367, do not add
tweaks. Report the actual first-hit ranks (including beyond 5 if you
inspect the uncut list) and a short explanation. That honest report, plus
the implementation, tests, `rerank.json`, and no regressions on the
baseline hits, fulfills DoD for this lesson.
