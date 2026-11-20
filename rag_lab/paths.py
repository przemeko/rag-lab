"""Shared paths and constants for the naive RAG pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "spark-docs"
DB_PATH = DATA_DIR / "chunks.sqlite"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
RESULTS_DIR = ROOT / "eval" / "results"

# Apache Spark docs pinned to the v3.5.4 release tag.
SPARK_COMMIT = "a6f220d951742f4074b37772485ee0ec7a774e7d"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 2000
DEFAULT_TOP_K = 5
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
