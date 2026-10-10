"""Tests for Markdown heading-based chunker."""
from pathlib import Path
from src.chunker import MarkdownChunker
from src.knowledge_loader import KnowledgeDocument


def test_markdown_section_chunking():
    """Verify that level-two headings create distinct semantic chunks with metadata."""
    sample_content = """# Rose Care

## Overview
Rose is a flowering plant.

## Watering
Water soil when top layer dries.

## Sunlight
Requires direct sunlight.
"""
    doc = KnowledgeDocument(
        file_path=Path("knowledge/plants/rose-plant.md"),
        relative_path="plants/rose-plant.md",
        metadata={"title": "Rose Plant Care", "status": "draft", "verified": False},
        content=sample_content,
        title="Rose Plant Care",
        doc_type="Plant",
        status="draft",
        verified=False,
        tags=["rose", "flowering"]
    )

    chunker = MarkdownChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 3
    headings = [c.section_heading for c in chunks]
    assert "Overview" in headings
    assert "Watering" in headings
    assert "Sunlight" in headings

    watering_chunk = next(c for c in chunks if c.section_heading == "Watering")
    assert "Water soil when top layer dries." in watering_chunk.content
    assert watering_chunk.plant_name == "Rose Plant"
    assert watering_chunk.source_file == "plants/rose-plant.md"
    assert "rose" in watering_chunk.tags


def test_chunker_handles_empty_or_headingless_doc():
    """Verify documents without ## headings still produce a fallback chunk."""
    doc = KnowledgeDocument(
        file_path=Path("knowledge/plants/simple.md"),
        relative_path="plants/simple.md",
        metadata={},
        content="Just plain text without any markdown headings.",
        title="Simple Plant",
        doc_type="Plant"
    )

    chunker = MarkdownChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert "Just plain text" in chunks[0].content
