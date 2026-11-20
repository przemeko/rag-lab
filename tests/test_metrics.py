"""Metric unit tests on hand-made rankings — no network, no models."""

from rag_lab.evaluate import first_hit_rank, is_hit, mean_reciprocal_rank, recall_at_k
from rag_lab.retrieve import RetrievedChunk


def _chunk(file: str, section: str, chunk_id: int = 1) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, file=file, section=section, text="...", score=0.9)


def test_is_hit_case_insensitive_section_substring():
    chunk = _chunk("sql-performance-tuning.md", "Optimizing Skew Join (AQE)")
    expected = [{"file": "sql-performance-tuning.md", "section": "Optimizing Skew Join"}]
    assert is_hit(chunk, expected)


def test_is_hit_requires_matching_file():
    chunk = _chunk("other.md", "Tuning Partitions")
    expected = [{"file": "sql-performance-tuning.md", "section": "Tuning Partitions"}]
    assert not is_hit(chunk, expected)


def test_first_hit_rank_and_no_hit():
    ranking = [
        _chunk("a.md", "Wrong", 1),
        _chunk("b.md", "Target Section", 2),
    ]
    expected = [{"file": "b.md", "section": "Target"}]
    assert first_hit_rank(ranking, expected) == 2
    assert first_hit_rank(ranking, [{"file": "z.md", "section": "Nope"}]) is None


def test_multiple_expected_sections():
    ranking = [
        _chunk("rdd-programming-guide.md", "Shuffle operations", 1),
    ]
    expected = [
        {"file": "rdd-programming-guide.md", "section": "Transformations"},
        {"file": "rdd-programming-guide.md", "section": "Shuffle operations"},
    ]
    assert first_hit_rank(ranking, expected) == 1


def test_recall_at_5_and_mrr():
    ranks = [1, 3, None, 5, 10]
    assert recall_at_k(ranks, k=5) == 0.6  # 1,3,5 hit; None and 10 miss
    # MRR = (1 + 1/3 + 0 + 1/5 + 1/10) / 5
    expected_mrr = (1.0 + 1 / 3 + 0.0 + 0.2 + 0.1) / 5
    assert abs(mean_reciprocal_rank(ranks) - expected_mrr) < 1e-9
