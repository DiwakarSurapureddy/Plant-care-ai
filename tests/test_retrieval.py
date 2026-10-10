"""Tests for keyword search, semantic search, hybrid retriever, and fallback behavior."""
import pytest
from src.chunker import MarkdownChunker
from src.config import KNOWLEDGE_DIR
from src.knowledge_loader import KnowledgeLoader
from src.keyword_search import KeywordSearchEngine
from src.semantic_search import SemanticSearchEngine
from src.retriever import HybridRetriever


@pytest.fixture(scope="module")
def loaded_chunks():
    loader = KnowledgeLoader(KNOWLEDGE_DIR)
    docs, _ = loader.load_all()
    chunker = MarkdownChunker()
    return chunker.chunk_all(docs)


def test_keyword_retrieval_money_plant(loaded_chunks):
    """Verify keyword retrieval correctly matches Money Plant watering queries."""
    kw_engine = KeywordSearchEngine(loaded_chunks)
    results = kw_engine.search("How do I water my Money Plant?", top_k=3)

    assert len(results) > 0
    top_chunk = results[0].chunk
    assert "Money Plant" in top_chunk.plant_name or "money-plant" in top_chunk.source_file
    assert top_chunk.source_file.endswith("money-plant.md")


def test_keyword_retrieval_rose_sunlight(loaded_chunks):
    """Verify keyword retrieval surfaces sunlight section for Rose."""
    kw_engine = KeywordSearchEngine(loaded_chunks)
    results = kw_engine.search("sunlight requirement for Rose", top_k=3)

    assert len(results) > 0
    top_chunk = results[0].chunk
    assert "Rose" in top_chunk.plant_name or "rose" in top_chunk.source_file


def test_unknown_question_retrieval(loaded_chunks):
    """Verify unrelated questions return empty or below threshold results."""
    kw_engine = KeywordSearchEngine(loaded_chunks)
    results = kw_engine.search("quantum mechanics in astrophysics", top_k=3)
    assert len(results) == 0


def test_semantic_retrieval(loaded_chunks):
    """Verify semantic search returns relevant chunks when model is available."""
    semantic_engine = SemanticSearchEngine(loaded_chunks)
    if not semantic_engine.is_available:
        pytest.skip("SentenceTransformers not available in environment")

    results = semantic_engine.search("How often should I hydrate indoor vines?", top_k=3)
    assert len(results) > 0
    # Money Plant or watering guide should be among top results
    assert any("money" in r.chunk.source_file.lower() or "watering" in r.chunk.source_file.lower() for r in results)


def test_retriever_hybrid_and_attribution(loaded_chunks):
    """Verify HybridRetriever combines results and preserves source file attribution."""
    retriever = HybridRetriever(loaded_chunks)
    resp = retriever.retrieve("Tulsi propagation and cuttings", mode="hybrid", top_k=3)

    assert len(resp.evidence) > 0
    assert any("tulsi" in ev.chunk.source_file.lower() for ev in resp.evidence)
    for ev in resp.evidence:
        assert ev.chunk.source_file != ""
        assert ev.chunk.section_heading != ""


def test_retriever_fallback_behavior(loaded_chunks):
    """Verify graceful fallback to keyword search when semantic engine is disabled."""
    fake_semantic_engine = SemanticSearchEngine(loaded_chunks)
    # Simulate unavailable semantic engine
    fake_semantic_engine._is_available = False
    fake_semantic_engine._availability_error = "Mocked offline error"

    retriever = HybridRetriever(loaded_chunks, semantic_engine=fake_semantic_engine)
    resp = retriever.retrieve("Aloe Vera succulent care", mode="semantic", top_k=3)

    assert resp.fallback_used is True
    assert "fallback" in resp.actual_mode.lower()
    assert len(resp.evidence) > 0
    assert any("aloe-vera" in ev.chunk.source_file.lower() for ev in resp.evidence)
