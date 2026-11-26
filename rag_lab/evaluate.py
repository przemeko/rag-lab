"""Evaluate dense, BM25, and hybrid retrieval on eval/questions.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rag_lab.paths import DEFAULT_TOP_K, QUESTIONS_PATH, RESULTS_DIR, RETRIEVAL_METHODS
from rag_lab.retrieve import Method, RetrievedChunk


def is_hit(chunk: RetrievedChunk, expected: list[dict]) -> bool:
    """True if chunk matches any expected file + section pair."""
    heading = chunk.section.casefold()
    for item in expected:
        if chunk.file != item["file"]:
            continue
        if item["section"].casefold() in heading:
            return True
    return False


def first_hit_rank(ranking: list[RetrievedChunk], expected: list[dict]) -> int | None:
    for rank, chunk in enumerate(ranking, start=1):
        if is_hit(chunk, expected):
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int = DEFAULT_TOP_K) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    total = 0.0
    for rank in ranks:
        if rank is not None:
            total += 1.0 / rank
    return total / len(ranks)


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["questions"]


def _method_summary(
    questions: list[dict],
    method: Method,
    k: int,
) -> dict:
    from rag_lab.retrieve import retrieve

    rows: list[dict] = []
    ranks: list[int | None] = []

    for q in questions:
        ranking = retrieve(q["question"], k=k, method=method)
        rank = first_hit_rank(ranking, q["expected"])
        ranks.append(rank)
        top = ranking[0] if ranking else None
        rows.append(
            {
                "id": q["id"],
                "rank": rank,
                "top1_file": top.file if top else None,
                "top1_section": top.section if top else None,
            }
        )

    return {
        "method": method,
        "k": k,
        "recall_at_k": recall_at_k(ranks, k=k),
        "mrr": mean_reciprocal_rank(ranks),
        "questions": rows,
    }


def run_eval(k: int = DEFAULT_TOP_K) -> dict:
    """Run dense, bm25, and hybrid; return a comparison summary."""
    questions = load_questions()
    by_method: dict[str, dict] = {}
    for method in RETRIEVAL_METHODS:
        by_method[method] = _method_summary(questions, method, k)  # type: ignore[arg-type]

    comparison_rows: list[dict] = []
    for i, q in enumerate(questions):
        hybrid_row = by_method["hybrid"]["questions"][i]
        comparison_rows.append(
            {
                "id": q["id"],
                "dense_rank": by_method["dense"]["questions"][i]["rank"],
                "bm25_rank": by_method["bm25"]["questions"][i]["rank"],
                "hybrid_rank": hybrid_row["rank"],
                "hybrid_top1_file": hybrid_row["top1_file"],
                "hybrid_top1_section": hybrid_row["top1_section"],
            }
        )

    return {
        "k": k,
        "methods": by_method,
        "questions": comparison_rows,
    }


def _rank_cell(rank: int | None) -> str:
    return str(rank) if rank is not None else "-"


def print_table(summary: dict) -> None:
    print(
        f"{'id':<6} {'dense':<6} {'bm25':<6} {'hybrid':<7} hybrid top-1"
    )
    print("-" * 88)
    for row in summary["questions"]:
        top = (
            f"{row['hybrid_top1_file']} / {row['hybrid_top1_section']}"
            if row["hybrid_top1_file"]
            else "-"
        )
        print(
            f"{row['id']:<6} "
            f"{_rank_cell(row['dense_rank']):<6} "
            f"{_rank_cell(row['bm25_rank']):<6} "
            f"{_rank_cell(row['hybrid_rank']):<7} "
            f"{top}"
        )
    print("-" * 88)
    k = summary["k"]
    for method in RETRIEVAL_METHODS:
        m = summary["methods"][method]
        print(
            f"{method:<7} recall@{k}: {m['recall_at_k']:.3f}  "
            f"MRR@{k}: {m['mrr']:.3f}"
        )


def main() -> None:
    summary = run_eval()
    print_table(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "hybrid.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
