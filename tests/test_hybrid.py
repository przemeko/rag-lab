"""Hybrid search unit tests — RRF, FTS escaping, FTS index. No models/network."""

from __future__ import annotations

import sqlite3

import pytest

from rag_lab.ingest import build_fts_index
from rag_lab.retrieve import escape_fts_query, reciprocal_rank_fusion, retrieve_bm25


def test_rrf_one_based_ranks_and_k60():
    # Single list: score(id) = 1/(60+rank)
    fused = reciprocal_rank_fusion([[10, 20, 30]], k=60)
    assert fused == [
        (10, 1.0 / 61),
        (20, 1.0 / 62),
        (30, 1.0 / 63),
    ]


def test_rrf_doc_in_both_lists_beats_doc_in_one():
    # 1 at rank 1 in both; 2 at rank 2 in only one list.
    fused = reciprocal_rank_fusion([[1, 2], [1, 3]], k=60)
    scores = dict(fused)
    assert scores[1] == pytest.approx(1.0 / 61 + 1.0 / 61)
    assert scores[2] == pytest.approx(1.0 / 62)
    assert scores[3] == pytest.approx(1.0 / 62)
    assert fused[0][0] == 1


def test_rrf_missing_from_list_is_not_rank_zero():
    # If missing docs were treated as rank 0, 2 would get 1/60 from list 2.
    fused = reciprocal_rank_fusion([[1, 2], [1]], k=60)
    scores = dict(fused)
    assert 2 in scores
    assert scores[2] == pytest.approx(1.0 / 62)
    assert scores[2] != pytest.approx(1.0 / 62 + 1.0 / 60)


def test_escape_fts_query_dots_parens_quotes():
    q = 'What is spark.sql.shuffle.partitions (and "default")?'
    match = escape_fts_query(q)
    assert match == (
        '"What" OR "is" OR "spark.sql.shuffle.partitions" OR "and" OR "default"'
    )
    # Each token is double-quoted; internal " doubled.
    assert escape_fts_query('say "hi"') == '"say" OR "hi"'


def test_escape_fts_query_empty():
    assert escape_fts_query("???") == '""'


def test_definitional_table_row_outranks_cross_reference():
    """HTML property-definition rows beat short cross-references for dotted ids."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            file TEXT NOT NULL,
            section TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO chunks (file, section, text) VALUES (?, ?, ?)",
        [
            (
                "xref.md",
                "AQE",
                "If unset, equals <code>spark.sql.shuffle.partitions</code> for coalescing.",
            ),
            (
                "tune.md",
                "Other Configuration Options",
                "<td><code>spark.sql.shuffle.partitions</code></td><td>200</td>",
            ),
            (
                "list.md",
                "Notes",
                "- `spark.sql.shuffle.partitions`",
            ),
        ],
    )
    build_fts_index(conn)
    conn.commit()

    hits = retrieve_bm25(
        "What is the default of spark.sql.shuffle.partitions?",
        k=3,
        db_path=conn,
    )
    assert hits[0].file == "tune.md"
    assert "200" in hits[0].text


def test_fts_index_dotted_identifier_ranks_first():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            file TEXT NOT NULL,
            section TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    rows = [
        ("a.md", "Other", "General shuffle discussion without the key."),
        (
            "b.md",
            "Config",
            "Set spark.sql.shuffle.partitions to control post-shuffle parallelism.",
        ),
        ("c.md", "Tuning", "Tune memory and disk for large joins."),
    ]
    conn.executemany(
        "INSERT INTO chunks (file, section, text) VALUES (?, ?, ?)",
        rows,
    )
    build_fts_index(conn)
    conn.commit()

    # Safe MATCH must not raise on dotted config names.
    hits = retrieve_bm25(
        "What is spark.sql.shuffle.partitions?",
        k=3,
        db_path=conn,
    )
    assert hits, "expected at least one BM25 hit"
    assert "spark.sql.shuffle.partitions" in hits[0].text

    # bm25() sort order is lower-is-better (ascending raw scores).
    raw = [
        row[0]
        for row in conn.execute(
            """
            SELECT bm25(chunks_fts) AS rank
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            """,
            (escape_fts_query("spark.sql.shuffle.partitions"),),
        )
    ]
    assert raw == sorted(raw)
    assert hits[0].score == pytest.approx(raw[0])
