.PHONY: install ingest embed eval check notebook

install:
	uv sync --group dev --group notebook

ingest:
	uv run python -m rag_lab.ingest

embed:
	uv run python -m rag_lab.embed

eval:
	uv run python -m rag_lab.evaluate

check:
	uv run ruff check .
	uv run pytest

notebook:
	uv run jupyter notebook --notebook-dir=notebooks
