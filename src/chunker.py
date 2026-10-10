"""Chunking logic for structured Markdown documents based on headings."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import re

from src.knowledge_loader import KnowledgeDocument


@dataclass
class KnowledgeChunk:
    """Represents a discrete semantic chunk of a knowledge document."""
    chunk_id: str
    doc_title: str
    plant_name: str
    section_heading: str
    content: str
    source_file: str
    file_path: Path
    doc_type: str
    status: str
    verified: bool
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        """Returns synthesized text optimal for embedding and keyword search."""
        header_context = f"{self.doc_title} - {self.section_heading}" if self.section_heading else self.doc_title
        tags_context = f"Tags: {', '.join(self.tags)}" if self.tags else ""
        return f"{header_context}\n{tags_context}\n{self.content}".strip()


class MarkdownChunker:
    """Splits Markdown documents into section chunks based primarily on level-two headings (##)."""

    def __init__(self, fallback_max_chars: int = 1200):
        self.fallback_max_chars = fallback_max_chars

    def chunk_document(self, doc: KnowledgeDocument) -> List[KnowledgeChunk]:
        """Split a KnowledgeDocument into KnowledgeChunk instances."""
        chunks: List[KnowledgeChunk] = []
        lines = doc.content.splitlines()

        # Extract plant name from title or filename
        plant_name = doc.title
        if " Care" in plant_name:
            plant_name = plant_name.replace(" Care", "").strip()

        current_heading = "Overview"
        current_lines: List[str] = []
        section_idx = 0

        heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$")

        for line in lines:
            match = heading_pattern.match(line.strip())
            if match:
                level = len(match.group(1))
                heading_text = match.group(2).strip()

                # Treat # (main doc title) and ## (major sections) as section boundaries
                if level in (1, 2):
                    # Flush previous section
                    content_str = "\n".join(current_lines).strip()
                    if content_str:
                        chunk_id = f"{doc.relative_path}#{section_idx}_{current_heading.lower().replace(' ', '_')}"
                        chunks.append(KnowledgeChunk(
                            chunk_id=chunk_id,
                            doc_title=doc.title,
                            plant_name=plant_name,
                            section_heading=current_heading,
                            content=content_str,
                            source_file=doc.relative_path,
                            file_path=doc.file_path,
                            doc_type=doc.doc_type,
                            status=doc.status,
                            verified=doc.verified,
                            tags=list(doc.tags),
                            metadata=doc.metadata
                        ))
                        section_idx += 1
                        current_lines = []

                    current_heading = heading_text
                    continue

            current_lines.append(line)

        # Flush trailing section
        content_str = "\n".join(current_lines).strip()
        if content_str:
            chunk_id = f"{doc.relative_path}#{section_idx}_{current_heading.lower().replace(' ', '_')}"
            chunks.append(KnowledgeChunk(
                chunk_id=chunk_id,
                doc_title=doc.title,
                plant_name=plant_name,
                section_heading=current_heading,
                content=content_str,
                source_file=doc.relative_path,
                file_path=doc.file_path,
                doc_type=doc.doc_type,
                status=doc.status,
                verified=doc.verified,
                tags=list(doc.tags),
                metadata=doc.metadata
            ))

        return chunks

    def chunk_all(self, docs: List[KnowledgeDocument]) -> List[KnowledgeChunk]:
        """Chunk all provided documents."""
        all_chunks: List[KnowledgeChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
