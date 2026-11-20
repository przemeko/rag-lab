"""Evaluate retrieval with recall@5 and MRR on eval/questions.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rag_lab.paths import DEFAULT_TOP_K, QUESTIONS_PATH, RESULTS_DIR
from rag_lab.retrieve import RetrievedChunk


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


def run_eval(k: int = DEFAULT_TOP_K) -> dict:
    from rag_lab.retrieve import retrieve

    questions = load_questions()
    rows: list[dict] = []
    ranks: list[int | None] = []

    for q in questions:
        ranking = retrieve(q["question"], k=k)
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

    summary = {
        "k": k,
        "recall_at_k": recall_at_k(ranks, k=k),
        "mrr": mean_reciprocal_rank(ranks),
        "questions": rows,
    }
    return summary


def print_table(summary: dict) -> None:
    print(f"{'id':<6} {'rank':<6} top-1")
    print("-" * 72)
    for row in summary["questions"]:
        rank_s = str(row["rank"]) if row["rank"] is not None else "-"
        top = (
            f"{row['top1_file']} / {row['top1_section']}"
            if row["top1_file"]
            else "-"
        )
        print(f"{row['id']:<6} {rank_s:<6} {top}")
    print("-" * 72)
    print(f"recall@{summary['k']}: {summary['recall_at_k']:.3f}")
    print(f"MRR@{summary['k']}:     {summary['mrr']:.3f}")


def main() -> None:
    summary = run_eval()
    print_table(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "baseline.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
