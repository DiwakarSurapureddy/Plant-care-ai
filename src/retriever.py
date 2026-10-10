"""Unified Retriever supporting Keyword, Semantic, and Hybrid strategies."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re

from src.chunker import KnowledgeChunk
from src.keyword_search import KeywordSearchEngine, SearchResult
from src.semantic_search import SemanticResult, SemanticSearchEngine


@dataclass
class RetrievedEvidence:
    """Consolidated search item with score and attribution."""
    chunk: KnowledgeChunk
    score: float
    source_mode: str
    reasons: List[str] = field(default_factory=list)


@dataclass
class RetrievalResponse:
    """Overall result of a retrieval operation."""
    evidence: List[RetrievedEvidence]
    requested_mode: str
    actual_mode: str
    fallback_used: bool = False
    warning_message: Optional[str] = None


class HybridRetriever:
    """Unified retriever combining KeywordSearchEngine and SemanticSearchEngine."""

    def __init__(
        self,
        chunks: List[KnowledgeChunk],
        semantic_engine: Optional[SemanticSearchEngine] = None,
        keyword_engine: Optional[KeywordSearchEngine] = None
    ):
        self.chunks = list(chunks)
        self.keyword_engine = keyword_engine or KeywordSearchEngine(self.chunks)
        self.semantic_engine = semantic_engine or SemanticSearchEngine(self.chunks)

    def update_chunks(self, chunks: List[KnowledgeChunk]):
        """Update both search engines with new chunks."""
        self.chunks = list(chunks)
        self.keyword_engine.set_chunks(self.chunks)
        self.semantic_engine.set_chunks(self.chunks)

    def _balance_multi_entity(
        self,
        ranked_evidence: List[RetrievedEvidence],
        query: str,
        top_k: int
    ) -> List[RetrievedEvidence]:
        """Ensure multi-plant comparison queries include evidence from each mentioned plant."""
        query_lower = query.lower()
        plant_to_core = {}
        for c in self.chunks:
            if c.plant_name and c.doc_type == "Plant":
                cores = [w for w in c.plant_name.lower().split() if w not in ("plant", "care", "guide")]
                if cores:
                    plant_to_core[c.plant_name] = cores

        mentioned_plants = []
        for plant, cores in plant_to_core.items():
            if any(re.search(rf"\b{re.escape(core)}\b", query_lower) for core in cores):
                if plant not in mentioned_plants:
                    mentioned_plants.append(plant)

        if len(mentioned_plants) > 1:
            selected: List[RetrievedEvidence] = []
            selected_ids = set()

            # Ensure evidence from each mentioned plant
            for plant in mentioned_plants:
                best = next(
                    (e for e in ranked_evidence if e.chunk.plant_name == plant and e.chunk.chunk_id not in selected_ids),
                    None
                )
                if best:
                    selected.append(best)
                    selected_ids.add(best.chunk.chunk_id)

            # Fill remaining budget
            for e in ranked_evidence:
                if len(selected) >= top_k:
                    break
                if e.chunk.chunk_id not in selected_ids:
                    selected.append(e)
                    selected_ids.add(e.chunk.chunk_id)

            return selected[:top_k]

        return ranked_evidence[:top_k]

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 4
    ) -> RetrievalResponse:
        """
        Retrieve chunks using specified mode: 'keyword', 'semantic', or 'hybrid'.
        Falls back to keyword if semantic fails or dependencies missing.
        """
        mode = mode.lower().strip()
        if mode not in ("keyword", "semantic", "hybrid"):
            mode = "hybrid"

        search_k = max(top_k * 2, 8)

        # Keyword mode
        if mode == "keyword":
            kw_results = self.keyword_engine.search(query, top_k=search_k)
            evidence = [
                RetrievedEvidence(
                    chunk=res.chunk,
                    score=res.score,
                    source_mode="keyword",
                    reasons=res.match_reasons
                )
                for res in kw_results
            ]
            balanced = self._balance_multi_entity(evidence, query, top_k)
            return RetrievalResponse(
                evidence=balanced,
                requested_mode=mode,
                actual_mode="keyword"
            )

        # Check semantic availability
        can_use_semantic = self.semantic_engine.is_available
        if not can_use_semantic:
            warning_msg = (
                f"Semantic search unavailable ({self.semantic_engine.availability_error}). "
                "Fell back to keyword search."
            )
            kw_results = self.keyword_engine.search(query, top_k=search_k)
            evidence = [
                RetrievedEvidence(
                    chunk=res.chunk,
                    score=res.score,
                    source_mode="keyword_fallback",
                    reasons=res.match_reasons
                )
                for res in kw_results
            ]
            balanced = self._balance_multi_entity(evidence, query, top_k)
            return RetrievalResponse(
                evidence=balanced,
                requested_mode=mode,
                actual_mode="keyword (fallback)",
                fallback_used=True,
                warning_message=warning_msg
            )

        # Execute semantic search
        try:
            semantic_results = self.semantic_engine.search(query, top_k=search_k)
        except Exception as e:
            warning_msg = f"Semantic search failed ({e}). Fell back to keyword search."
            kw_results = self.keyword_engine.search(query, top_k=search_k)
            evidence = [
                RetrievedEvidence(
                    chunk=res.chunk,
                    score=res.score,
                    source_mode="keyword_fallback",
                    reasons=res.match_reasons
                )
                for res in kw_results
            ]
            balanced = self._balance_multi_entity(evidence, query, top_k)
            return RetrievalResponse(
                evidence=balanced,
                requested_mode=mode,
                actual_mode="keyword (fallback)",
                fallback_used=True,
                warning_message=warning_msg
            )

        if mode == "semantic":
            evidence = [
                RetrievedEvidence(
                    chunk=res.chunk,
                    score=res.score,
                    source_mode="semantic",
                    reasons=[f"Vector similarity: {res.similarity:.3f}"]
                )
                for res in semantic_results
            ]
            balanced = self._balance_multi_entity(evidence, query, top_k)
            return RetrievalResponse(
                evidence=balanced,
                requested_mode=mode,
                actual_mode="semantic"
            )

        # Hybrid mode
        kw_results = self.keyword_engine.search(query, top_k=search_k)
        combined: Dict[str, RetrievedEvidence] = {}

        for rank, s_res in enumerate(semantic_results):
            cid = s_res.chunk.chunk_id
            combined[cid] = RetrievedEvidence(
                chunk=s_res.chunk,
                score=float(s_res.score) * 1.5,
                source_mode="hybrid",
                reasons=[f"Semantic score: {s_res.score:.3f}"]
            )

        for rank, k_res in enumerate(kw_results):
            cid = k_res.chunk.chunk_id
            norm_kw_score = min(k_res.score / 5.0, 1.5)
            if cid in combined:
                combined[cid].score += norm_kw_score
                combined[cid].reasons.extend(k_res.match_reasons)
            else:
                combined[cid] = RetrievedEvidence(
                    chunk=k_res.chunk,
                    score=norm_kw_score,
                    source_mode="hybrid",
                    reasons=k_res.match_reasons
                )

        ranked = sorted(combined.values(), key=lambda x: x.score, reverse=True)
        balanced = self._balance_multi_entity(ranked, query, top_k)

        return RetrievalResponse(
            evidence=balanced,
            requested_mode="hybrid",
            actual_mode="hybrid",
            fallback_used=False
        )
