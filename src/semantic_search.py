"""Semantic Search Engine with SentenceTransformers, lazy loading, caching, and fallback."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import pickle
import numpy as np

from src.chunker import KnowledgeChunk
from src.config import EMBEDDINGS_CACHE_FILE, EMBEDDING_MODEL_NAME, SEMANTIC_SIMILARITY_THRESHOLD


@dataclass
class SemanticResult:
    """Represents a chunk retrieved via semantic embedding similarity."""
    chunk: KnowledgeChunk
    score: float
    similarity: float


class SemanticSearchEngine:
    """Semantic vector search over KnowledgeChunks with lazy model loading and disk caching."""

    def __init__(
        self,
        chunks: List[KnowledgeChunk],
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_file: Path = EMBEDDINGS_CACHE_FILE,
        similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
    ):
        self.chunks = list(chunks)
        self.model_name = model_name
        self.cache_file = Path(cache_file)
        self.similarity_threshold = similarity_threshold

        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._is_available: Optional[bool] = None
        self._availability_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        """Check if sentence-transformers can be imported and initialized."""
        if self._is_available is None:
            try:
                import sentence_transformers  # noqa: F401
                self._is_available = True
            except ImportError as e:
                self._is_available = False
                self._availability_error = f"SentenceTransformers not installed: {e}"
            except Exception as e:
                self._is_available = False
                self._availability_error = f"Embedding environment error: {e}"
        return self._is_available

    @property
    def availability_error(self) -> Optional[str]:
        return self._availability_error

    def _get_model(self):
        """Lazily load SentenceTransformer model."""
        if self._model is None:
            if not self.is_available:
                raise RuntimeError(self._availability_error or "Semantic search unavailable")
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                self._is_available = False
                self._availability_error = (
                    f"Failed to load embedding model '{self.model_name}'. "
                    f"First-time download requires internet connectivity: {e}"
                )
                raise RuntimeError(self._availability_error)
        return self._model

    def _compute_chunks_hash(self) -> str:
        """Compute MD5 hash across chunk texts and IDs to detect knowledge base changes."""
        hasher = hashlib.md5()
        for chunk in self.chunks:
            hasher.update(chunk.chunk_id.encode("utf-8"))
            hasher.update(chunk.searchable_text.encode("utf-8"))
        return hasher.hexdigest()

    def build_index(self, force_rebuild: bool = False):
        """Build or load cached embeddings for all current chunks."""
        if not self.chunks or not self.is_available:
            return

        current_hash = self._compute_chunks_hash()

        # Check cache if not forcing rebuild
        if not force_rebuild and self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                if (
                    cache_data.get("model_name") == self.model_name and
                    cache_data.get("chunks_hash") == current_hash and
                    len(cache_data.get("chunk_ids", [])) == len(self.chunks)
                ):
                    self._embeddings = cache_data["embeddings"]
                    return
            except Exception:
                pass  # Corrupted cache or mismatch; recompute

        model = self._get_model()
        texts = [chunk.searchable_text for chunk in self.chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings = embeddings

        # Save to cache
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                pickle.dump({
                    "model_name": self.model_name,
                    "chunks_hash": current_hash,
                    "chunk_ids": [c.chunk_id for c in self.chunks],
                    "embeddings": self._embeddings
                }, f)
        except Exception:
            pass

    def search(self, query: str, top_k: int = 4) -> List[SemanticResult]:
        """Vector similarity search using normalized cosine similarity."""
        if not self.chunks or not query.strip():
            return []

        if not self.is_available:
            return []

        if self._embeddings is None or len(self._embeddings) != len(self.chunks):
            self.build_index()

        if self._embeddings is None:
            return []

        model = self._get_model()
        query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Cosine similarity for normalized vectors is dot product
        similarities = np.dot(self._embeddings, query_vector)

        ranked_indices = np.argsort(similarities)[::-1]
        results: List[SemanticResult] = []

        for idx in ranked_indices:
            sim = float(similarities[idx])
            if sim < self.similarity_threshold:
                continue
            results.append(SemanticResult(
                chunk=self.chunks[idx],
                score=round(sim, 4),
                similarity=round(sim, 4)
            ))
            if len(results) >= top_k:
                break

        return results

    def set_chunks(self, chunks: List[KnowledgeChunk]):
        """Update chunks and reset embedding index."""
        self.chunks = list(chunks)
        self._embeddings = None
