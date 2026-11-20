"""Chunking unit tests — no network, no models."""

from rag_lab.ingest import chunk_markdown, strip_jekyll

FIXTURE = """\
---
layout: global
title: Example
---

# Title

Intro before any section.

## First Section

Hello world.
{% highlight python %}
print("x")
{% endhighlight %}

### Nested

Nested body.

## Second Section

Short.

## Long Section

""" + ("word " * 600)


def test_strip_jekyll_removes_front_matter_and_liquid():
    cleaned = strip_jekyll(FIXTURE)
    assert "layout: global" not in cleaned
    assert "{%" not in cleaned
    assert "Hello world." in cleaned


def test_chunk_markdown_nested_headings():
    chunks = chunk_markdown("example.md", FIXTURE, max_chars=2000)
    sections = [c.section for c in chunks]
    assert "Title" in sections
    assert "First Section" in sections
    assert "Nested" in sections
    assert "Second Section" in sections

    first = next(c for c in chunks if c.section == "First Section")
    assert "Hello world." in first.text
    # Nested ### is lower level, so it stays inside the ## parent section.
    assert "Nested body." in first.text

    nested = next(c for c in chunks if c.section == "Nested")
    assert "Nested body." in nested.text
    assert nested.file == "example.md"


def test_long_section_is_split():
    chunks = chunk_markdown("example.md", FIXTURE, max_chars=2000)
    long_pieces = [c for c in chunks if c.section == "Long Section"]
    assert len(long_pieces) >= 2
    assert all(len(c.text) <= 2000 for c in long_pieces)
    assert all(c.file == "example.md" for c in long_pieces)
    assert all(c.section == "Long Section" for c in long_pieces)


def test_hash_comments_inside_highlight_are_not_headings():
    source = """\
## Caching

Before the example.

{% highlight python %}
# Wrong: Don't do this!!
print(rdd.map(lambda x: hdfs.put(x)))
{% endhighlight %}

## After

Done.
"""
    chunks = chunk_markdown("rdd-programming-guide.md", source, max_chars=2000)
    sections = [c.section for c in chunks]
    assert sections == ["Caching", "After"]
    assert "Wrong: Don't do this!!" not in sections
    caching = next(c for c in chunks if c.section == "Caching")
    assert "# Wrong: Don't do this!!" in caching.text
    assert "print(rdd.map" in caching.text
