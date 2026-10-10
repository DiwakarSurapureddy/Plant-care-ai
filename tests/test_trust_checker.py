"""Tests for TrustChecker freshness and verification rules."""
from datetime import date
from pathlib import Path
from src.chunker import KnowledgeChunk
from src.trust_checker import TrustChecker


def test_draft_status_warning():
    """Verify notes with status 'draft' produce a warning."""
    chunk = KnowledgeChunk(
        chunk_id="test#1",
        doc_title="Test Plant",
        plant_name="Test Plant",
        section_heading="Watering",
        content="Water occasionally.",
        source_file="plants/test.md",
        file_path=Path("plants/test.md"),
        doc_type="Plant",
        status="draft",
        verified=False,
        metadata={"status": "draft", "verified": False}
    )

    checker = TrustChecker()
    warnings = checker.check_chunk(chunk)

    assert any(w.severity == "warning" and "draft" in w.message.lower() for w in warnings)


def test_stale_note_warning():
    """Verify notes whose stale_after date has passed trigger a stale warning."""
    chunk = KnowledgeChunk(
        chunk_id="test#2",
        doc_title="Old Plant Note",
        plant_name="Old Plant",
        section_heading="Soil",
        content="Old soil mix recommendation.",
        source_file="plants/old.md",
        file_path=Path("plants/old.md"),
        doc_type="Plant",
        status="active",
        verified=True,
        metadata={"stale_after": "2020-01-01"}
    )

    checker = TrustChecker(current_date=date(2026, 10, 10))
    warnings = checker.check_chunk(chunk)

    assert any("expired" in w.message.lower() or "stale" in w.message.lower() for w in warnings)


def test_price_unverified_caution():
    """Verify price section with unverified status yields specific price caution."""
    chunk = KnowledgeChunk(
        chunk_id="test#3",
        doc_title="Money Plant",
        plant_name="Money Plant",
        section_heading="Price Information",
        content="Nursery price: Not available yet.",
        source_file="plants/money-plant.md",
        file_path=Path("plants/money-plant.md"),
        doc_type="Plant",
        status="draft",
        verified=False,
        metadata={"status": "draft", "verified": False}
    )

    checker = TrustChecker()
    warnings = checker.check_chunk(chunk)

    assert any(w.severity == "caution" and "price" in w.message.lower() for w in warnings)


def test_malformed_date_does_not_crash():
    """Verify malformed stale_after date strings are handled safely."""
    chunk = KnowledgeChunk(
        chunk_id="test#4",
        doc_title="Weird Date Plant",
        plant_name="Weird Date Plant",
        section_heading="Overview",
        content="Some content",
        source_file="plants/weird.md",
        file_path=Path("plants/weird.md"),
        doc_type="Plant",
        status="active",
        verified=True,
        metadata={"stale_after": "not-a-real-date"}
    )

    checker = TrustChecker()
    # Should not raise exception
    warnings = checker.check_chunk(chunk)
    assert isinstance(warnings, list)
