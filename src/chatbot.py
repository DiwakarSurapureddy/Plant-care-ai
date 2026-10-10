"""Chatbot orchestrator integrating Knowledge, Retriever, Trust Checker, and Ollama."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.answer_generator import AnswerGenerator, GenerationResult
from src.chunker import KnowledgeChunk, MarkdownChunker
from src.config import KNOWLEDGE_DIR, RETRIEVAL_TOP_K
from src.knowledge_loader import KnowledgeDocument, KnowledgeLoader
from src.ollama_client import OllamaClient, OllamaStatus
from src.retriever import HybridRetriever, RetrievalResponse
from src.trust_checker import TrustChecker, TrustWarning


@dataclass
class ChatResponse:
    """Consolidated response bundle consumed by the Gradio interface."""
    answer: str
    source_documents: List[str]
    trust_warnings: List[TrustWarning]
    related_plants: List[str]
    retrieval_mode_used: str
    retrieval_fallback: bool
    retrieval_diagnostic: Optional[str]
    ollama_status: OllamaStatus
    raw_evidence_count: int


class PlantCareChatbot:
    """High-level orchestrator for the Plant Care AI application."""

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.knowledge_dir = Path(knowledge_dir)
        self.loader = KnowledgeLoader(self.knowledge_dir)
        self.chunker = MarkdownChunker()
        self.trust_checker = TrustChecker()
        self.ollama_client = OllamaClient()
        self.answer_generator = AnswerGenerator(self.ollama_client)

        self.documents: List[KnowledgeDocument] = []
        self.chunks: List[KnowledgeChunk] = []
        self.load_errors: List[Tuple[Path, str]] = []
        self.retriever: Optional[HybridRetriever] = None

        self.rebuild_index()

    def rebuild_index(self) -> Dict[str, Any]:
        """Re-scan knowledge directory and rebuild retrieval indices."""
        self.documents, self.load_errors = self.loader.load_all()
        self.chunks = self.chunker.chunk_all(self.documents)

        if self.retriever is None:
            self.retriever = HybridRetriever(self.chunks)
        else:
            self.retriever.update_chunks(self.chunks)

        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Return knowledge base summary statistics."""
        plant_names = sorted(list(set(doc.title for doc in self.documents)))
        return {
            "total_files": len(self.documents),
            "total_chunks": len(self.chunks),
            "plants": plant_names,
            "load_errors": len(self.load_errors),
            "load_error_details": [f"{p.name}: {err}" for p, err in self.load_errors]
        }

    def get_ollama_status(self) -> OllamaStatus:
        """Inspect Ollama service and configured model."""
        return self.ollama_client.check_status()

    def extract_related_plants(self, chunks: List[KnowledgeChunk]) -> List[str]:
        """Find mentioned related plants from retrieved chunks."""
        related: List[str] = []
        for c in chunks:
            if "related" in c.section_heading.lower():
                for line in c.content.splitlines():
                    clean = line.strip().lstrip("-*• ").strip()
                    if clean and clean not in related:
                        related.append(clean)
        return related

    def chat(
        self,
        user_query: str,
        retrieval_mode: str = "hybrid",
        top_k: int = RETRIEVAL_TOP_K
    ) -> ChatResponse:
        """Handle a user question end-to-end."""
        ollama_status = self.get_ollama_status()

        if not user_query.strip():
            return ChatResponse(
                answer="Please enter a plant-care question or select one of the examples below.",
                source_documents=[],
                trust_warnings=[],
                related_plants=[],
                retrieval_mode_used="none",
                retrieval_fallback=False,
                retrieval_diagnostic=None,
                ollama_status=ollama_status,
                raw_evidence_count=0
            )

        # 1. Retrieve knowledge
        retrieval_resp: RetrievalResponse = self.retriever.retrieve(
            query=user_query,
            mode=retrieval_mode,
            top_k=top_k
        )

        retrieved_chunks = [ev.chunk for ev in retrieval_resp.evidence]

        # 2. Trust and Freshness Checks
        trust_warnings = self.trust_checker.check_all(retrieved_chunks)

        # 3. Related plants extraction
        related_plants = self.extract_related_plants(retrieved_chunks)

        # 4. Generate grounded answer
        gen_result: GenerationResult = self.answer_generator.generate_answer(
            query=user_query,
            evidence=retrieval_resp.evidence
        )

        return ChatResponse(
            answer=gen_result.answer,
            source_documents=gen_result.source_documents,
            trust_warnings=trust_warnings,
            related_plants=related_plants,
            retrieval_mode_used=retrieval_resp.actual_mode,
            retrieval_fallback=retrieval_resp.fallback_used,
            retrieval_diagnostic=retrieval_resp.warning_message or gen_result.diagnostic_message,
            ollama_status=ollama_status,
            raw_evidence_count=len(retrieval_resp.evidence)
        )
