"""Offline Keyword Search Engine for Knowledge Chunks."""
from dataclasses import dataclass
from typing import List, Tuple
import re
import string

from src.chunker import KnowledgeChunk


@dataclass
class SearchResult:
    """Represents a scored chunk retrieved by search."""
    chunk: KnowledgeChunk
    score: float
    match_reasons: List[str]


class KeywordSearchEngine:
    """TF/Weighted keyword search over structured KnowledgeChunk objects."""

    STOP_WORDS = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "by", "is", "are", "was", "were", "it", "its", "that", "this", "do", "does",
        "how", "what", "which", "can", "i", "my", "me", "you", "your", "plant", "care"
    }

    def __init__(self, chunks: List[KnowledgeChunk]):
        self.chunks = list(chunks)

    def set_chunks(self, chunks: List[KnowledgeChunk]):
        """Update index with new chunks."""
        self.chunks = list(chunks)

    def _tokenize(self, text: str) -> List[str]:
        """Normalize, strip punctuation, and tokenize string."""
        clean = text.lower()
        # Remove punctuation
        clean = clean.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        tokens = [tok for tok in clean.split() if len(tok) > 1 and tok not in self.STOP_WORDS]
        return tokens

    def search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Search chunks by keyword overlap with heuristic ranking weights."""
        query_tokens = self._tokenize(query)
        raw_query_lower = query.lower().strip()

        if not query_tokens and not raw_query_lower:
            return []

        results: List[Tuple[float, List[str], KnowledgeChunk]] = []

        for chunk in self.chunks:
            score = 0.0
            reasons = []

            # 1. Exact plant core name match
            plant_lower = chunk.plant_name.lower()
            core_plant_tokens = [w for w in plant_lower.split() if w not in ("plant", "care", "guide")]
            if core_plant_tokens and any(w in raw_query_lower for w in core_plant_tokens):
                score += 6.0
                reasons.append(f"Plant name match ('{chunk.plant_name}')")

            # 2. Section heading match
            heading_lower = chunk.section_heading.lower()
            if heading_lower and (heading_lower in raw_query_lower or any(tok in heading_lower for tok in query_tokens)):
                score += 3.5
                reasons.append(f"Section heading match ('{chunk.section_heading}')")

            # 3. Tag matches
            tag_matches = [t for t in chunk.tags if t.lower() in raw_query_lower or any(tok in t.lower() for tok in query_tokens)]
            if tag_matches:
                score += 2.5 * len(tag_matches)
                reasons.append(f"Tag match ({', '.join(tag_matches)})")

            # 4. Content token frequency
            chunk_content_lower = chunk.content.lower()
            token_hits = 0
            for token in query_tokens:
                pattern = r"\b" + re.escape(token) + r"\b"
                matches = len(re.findall(pattern, chunk_content_lower))
                if matches > 0:
                    token_hits += 1
                    score += min(matches * 0.8, 3.0)

            if token_hits > 0:
                reasons.append(f"Matched {token_hits} keyword(s) in content")

            # If user asks about comparison or multiple plants
            doc_title_lower = chunk.doc_title.lower()
            if any(tok in doc_title_lower for tok in query_tokens):
                score += 1.5

            if score > 0.5:
                results.append((score, reasons, chunk))

        # Sort descending by score
        results.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(chunk=chunk, score=round(score, 3), match_reasons=reasons)
            for score, reasons, chunk in results[:top_k]
        ]
