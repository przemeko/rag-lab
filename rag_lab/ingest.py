"""Download Spark docs at a pinned commit, chunk by heading, write SQLite."""

from __future__ import annotations

import io
import re
import sqlite3
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rag_lab.paths import DATA_DIR, DB_PATH, DOCS_DIR, MAX_CHUNK_CHARS, SPARK_COMMIT

FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
LIQUID_TAG_RE = re.compile(r"\{%.*?%\}", re.DOTALL)
# Spark guides use #, ##, and ### for real sections (not only ## / ###).
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
# Code blocks where "#" is a comment, not a Markdown heading.
HIGHLIGHT_BLOCK_RE = re.compile(
    r"\{%\s*highlight\b.*?%\}.*?\{%\s*endhighlight\s*%\}",
    re.DOTALL | re.IGNORECASE,
)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class Chunk:
    file: str
    section: str
    text: str


def strip_front_matter(text: str) -> str:
    return FRONT_MATTER_RE.sub("", text, count=1)


def strip_liquid_tags(text: str) -> str:
    return LIQUID_TAG_RE.sub("", text)


def strip_jekyll(text: str) -> str:
    """Remove Jekyll front matter and Liquid tags."""
    return strip_liquid_tags(strip_front_matter(text))


def _mask_code_blocks(text: str) -> str:
    """Blank out code blocks (same length) so "#" comments are not seen as headings."""
    chars = list(text)
    for pattern in (HIGHLIGHT_BLOCK_RE, FENCED_CODE_RE):
        for match in pattern.finditer(text):
            for i in range(match.start(), match.end()):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


def _split_long(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into pieces of at most max_chars, preferring paragraph breaks."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        break_at = window.rfind("\n\n")
        if break_at < max_chars // 4:
            break_at = window.rfind("\n")
        if break_at < max_chars // 4:
            break_at = max_chars
        piece = remaining[:break_at].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[break_at:].lstrip()
    if remaining.strip():
        pieces.append(remaining.strip())
    return pieces


def chunk_markdown(filename: str, source: str, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split Markdown into # / ## / ### sections, then cap long sections."""
    # Keep highlight/fence delimiters until headings are found, then strip Liquid tags.
    text = strip_front_matter(source)
    matches = list(HEADING_RE.finditer(_mask_code_blocks(text)))
    if not matches:
        return []

    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        section = match.group(2).strip()
        start = match.start()
        end = len(text)
        for later in matches[i + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        body = strip_liquid_tags(text[start:end]).strip()
        for piece in _split_long(body, max_chars=max_chars):
            chunks.append(Chunk(file=filename, section=section, text=piece))
    return chunks


def download_docs(docs_dir: Path = DOCS_DIR, commit: str = SPARK_COMMIT) -> Path:
    """Download the Spark source tarball and extract docs/*.md into docs_dir."""
    if docs_dir.is_dir() and any(docs_dir.glob("*.md")):
        print(f"Docs already present at {docs_dir}")
        return docs_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://codeload.github.com/apache/spark/tar.gz/{commit}"
    print(f"Downloading apache/spark docs at {commit[:12]}...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Paths look like spark-<sha>/docs/foo.md
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[1] == "docs" and parts[2].endswith(".md"):
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                (docs_dir / parts[2]).write_bytes(extracted.read())

    n = len(list(docs_dir.glob("*.md")))
    print(f"Extracted {n} Markdown files -> {docs_dir}")
    return docs_dir


def _require_fts5(conn: sqlite3.Connection) -> None:
    """Fail clearly if this SQLite build has no FTS5."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts5_probe")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "SQLite FTS5 is required for BM25 hybrid search but is unavailable "
            f"in this Python build ({exc})."
        ) from exc


def build_fts_index(conn: sqlite3.Connection) -> None:
    """Create chunks_fts over chunks.text (unicode61 + dotted/hyphen identifiers)."""
    _require_fts5(conn)
    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize="unicode61 tokenchars '._-'"
        )
        """
    )
    conn.execute("INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks")


def write_chunks(chunks: list[Chunk], db_path: Path = DB_PATH) -> int:
    """Replace the chunks table with the given rows. Returns row count."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
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
            [(c.file, c.section, c.text) for c in chunks],
        )
        build_fts_index(conn)
        conn.commit()
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()


def load_all_markdown(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Chunk every top-level .md file under docs_dir (non-recursive for simplicity)."""
    chunks: list[Chunk] = []
    # Spark keeps the guide pages at docs/*.md; skip _layouts etc.
    for path in sorted(docs_dir.glob("*.md")):
        source = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(chunk_markdown(path.name, source))
    return chunks


def main() -> None:
    docs_dir = download_docs()
    chunks = load_all_markdown(docs_dir)
    n = write_chunks(chunks)
    print(f"Wrote {n} chunks from {docs_dir} -> {DB_PATH}")


if __name__ == "__main__":
    main()
