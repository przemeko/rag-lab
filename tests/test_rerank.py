"""Cross-encoder rerank unit tests — fake scorer only; no model download."""

from __future__ import annotations

from rag_lab.retrieve import RetrievedChunk, rerank_chunks


def _chunk(chunk_id: int, text: str = "body", score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        file=f"f{chunk_id}.md",
        section=f"S{chunk_id}",
        text=text,
        score=score,
    )


def test_rerank_orders_by_fake_scores():
    candidates = [_chunk(1, "a"), _chunk(2, "b"), _chunk(3, "c")]
    # Map text -> score so order is independent of input order.
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}

    def scorer(pairs):
        return [scores[text] for _, text in pairs]

    out = rerank_chunks("q", candidates, k=3, scorer=scorer)
    assert [c.id for c in out] == [2, 3, 1]
    assert [c.score for c in out] == [0.9, 0.5, 0.1]


def test_rerank_only_reorders_input_ids():
    candidates = [_chunk(10), _chunk(20), _chunk(30)]

    def scorer(pairs):
        return [0.2, 0.8, 0.5]

    out = rerank_chunks("q", candidates, k=3, scorer=scorer)
    assert {c.id for c in out} == {10, 20, 30}
    assert len(out) == 3


def test_rerank_applies_k_after_reorder():
    candidates = [_chunk(1), _chunk(2), _chunk(3), _chunk(4)]

    def scorer(pairs):
        # Prefer later ids so reorder is visible before the cut.
        return [0.1, 0.2, 0.9, 0.8]

    out = rerank_chunks("q", candidates, k=2, scorer=scorer)
    assert [c.id for c in out] == [3, 4]


def test_rerank_ties_keep_incoming_order():
    candidates = [_chunk(1, "a"), _chunk(2, "b"), _chunk(3, "c")]

    def scorer(pairs):
        return [0.5, 0.5, 0.5]

    out = rerank_chunks("q", candidates, k=3, scorer=scorer)
    assert [c.id for c in out] == [1, 2, 3]


def test_rerank_accepts_predict_object():
    candidates = [_chunk(1, "x"), _chunk(2, "y")]

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.3, 0.7]

    out = rerank_chunks("q", candidates, k=2, scorer=FakeCrossEncoder())
    assert [c.id for c in out] == [2, 1]
    assert out[0].score == 0.7
