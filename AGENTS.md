# AGENTS.md

## Project
rag-lab: a learning project for RAG over the Apache Spark docs.
Each lesson adds one retrieval technique and must be compared
against the previous result with `make eval`.

## Rules
- Plain Python 3.12. No LangChain, LlamaIndex, or vector databases
  unless the spec explicitly asks for them.
- Simplest readable code. 
- Everything (code, comments, docs, PR descriptions) in English.
- Never commit secrets, downloaded docs, models, SQLite files,
  or embeddings (`data/` is gitignored).
- use uv for managing project

## Evaluation
- `eval/questions.yaml` is fixed. Never change questions to improve
  a score. Fixing a wrong expected section is allowed only with an
  explanation in the PR.
- Every PR that changes retrieval includes the `make eval` table
  and commits its result to `eval/results/<name>.json`.
- Tests must not download models or call Ollama.

## Before opening a PR
- `make check` passes.
- You ran the changed pipeline steps yourself and report the output.
- One PR = one working, independently testable change.