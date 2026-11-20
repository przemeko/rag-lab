.PHONY: install ingest embed eval check

install:
	uv sync --group dev

ingest:
	uv run python -m rag_lab.ingest

embed:
	uv run python -m rag_lab.embed

eval:
	uv run python -m rag_lab.evaluate

check:
	uv run ruff check .
	uv run pytest
