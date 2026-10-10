"""Tests for KnowledgeLoader and YAML frontmatter parsing."""
import pytest
from pathlib import Path
from src.knowledge_loader import KnowledgeLoader
from src.config import KNOWLEDGE_DIR


def test_load_existing_five_plant_notes():
    """Verify that all five initial plant notes are discovered and parsed."""
    loader = KnowledgeLoader(KNOWLEDGE_DIR)
    docs, errors = loader.load_all()

    assert len(errors) == 0, f"Encountered unexpected parsing errors: {errors}"
    assert len(docs) >= 5, f"Expected at least 5 documents, found {len(docs)}"

    doc_paths = [d.file_path.name for d in docs]
    expected_notes = [
        "money-plant.md",
        "rose-plant.md",
        "hibiscus-plant.md",
        "tulsi-plant.md",
        "aloe-vera.md"
    ]
    for note in expected_notes:
        assert note in doc_paths, f"Missing expected note {note}"


def test_frontmatter_metadata_extraction():
    """Verify frontmatter values like type, title, status, tags."""
    loader = KnowledgeLoader(KNOWLEDGE_DIR)
    docs, _ = loader.load_all()

    money_plant_doc = next((d for d in docs if d.file_path.name == "money-plant.md"), None)
    assert money_plant_doc is not None
    assert money_plant_doc.doc_type == "Plant"
    assert "Money Plant" in money_plant_doc.title
    assert money_plant_doc.status == "draft"
    assert "indoor-plant" in money_plant_doc.tags
    assert money_plant_doc.verified is False


def test_invalid_frontmatter_handling(tmp_path):
    """Verify loader handles broken YAML without crashing."""
    broken_file = tmp_path / "broken.md"
    broken_file.write_text("---\ntitle: [unclosed list\n---\n# Broken", encoding="utf-8")

    loader = KnowledgeLoader(tmp_path)
    docs, errors = loader.load_all()

    assert len(errors) == 1
    assert "Invalid YAML" in errors[0][1]
