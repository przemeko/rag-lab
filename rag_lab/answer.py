"""CLI: retrieve chunks for a question and answer with local Ollama."""

from __future__ import annotations

import argparse
import sys

import requests

from rag_lab.paths import DEFAULT_TOP_K, OLLAMA_MODEL, OLLAMA_URL, RETRIEVAL_METHODS
from rag_lab.retrieve import retrieve


def build_prompt(question: str, contexts: list[tuple[str, str, str]]) -> str:
    blocks = []
    for file, section, text in contexts:
        blocks.append(f"### {file} / {section}\n{text}")
    context = "\n\n".join(blocks)
    return (
        "You answer questions about Apache Spark using only the context below. "
        "If the context is insufficient, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def ask_ollama(prompt: str, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL) -> str:
    response = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a Spark docs question with RAG.")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument(
        "--method",
        choices=RETRIEVAL_METHODS,
        default="hybrid",
        help="Retrieval method (default: hybrid)",
    )
    args = parser.parse_args(argv)

    hits = retrieve(args.question, k=args.k, method=args.method)
    print(f"Retrieved chunks ({args.method}):", flush=True)
    for i, hit in enumerate(hits, start=1):
        print(f"  {i}. [{hit.score:.3f}] {hit.file} / {hit.section}", flush=True)

    prompt = build_prompt(args.question, [(h.file, h.section, h.text) for h in hits])
    try:
        answer = ask_ollama(prompt)
    except requests.RequestException as exc:
        print(f"\nOllama unavailable ({exc}). Showing retrieval only.", file=sys.stderr)
        return 0

    print("\nAnswer:")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
