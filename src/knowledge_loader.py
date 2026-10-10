"""Knowledge Loader for OKF-style Markdown files with YAML frontmatter."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass
class KnowledgeDocument:
    """Represents a parsed Markdown document from the knowledge base."""
    file_path: Path
    relative_path: str
    metadata: Dict[str, Any]
    content: str
    title: str = ""
    doc_type: str = "Plant"
    status: str = "draft"
    verified: bool = False
    tags: List[str] = field(default_factory=list)
    parse_error: Optional[str] = None


class KnowledgeLoader:
    """Discovers, parses, and loads OKF-style Markdown notes."""

    IGNORED_FILENAMES = {"index.md", "log.md", "readme.md"}

    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = Path(knowledge_dir)

    def parse_file(self, file_path: Path) -> KnowledgeDocument:
        """Parse a single Markdown file with optional YAML frontmatter."""
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            rel = str(file_path.relative_to(self.knowledge_dir)) if self.knowledge_dir in file_path.parents else file_path.name
            return KnowledgeDocument(
                file_path=file_path,
                relative_path=rel,
                metadata={},
                content="",
                parse_error=f"Failed to read file: {e}"
            )

        metadata: Dict[str, Any] = {}
        body = raw_text

        # Check for YAML frontmatter between --- and ---
        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                frontmatter_str = parts[1]
                body = parts[2].strip()
                try:
                    parsed_yaml = yaml.safe_load(frontmatter_str)
                    if isinstance(parsed_yaml, dict):
                        metadata = parsed_yaml
                    elif parsed_yaml is not None:
                        metadata = {"_raw_frontmatter": parsed_yaml}
                except Exception as e:
                    rel = str(file_path.relative_to(self.knowledge_dir)) if self.knowledge_dir in file_path.parents else file_path.name
                    return KnowledgeDocument(
                        file_path=file_path,
                        relative_path=rel,
                        metadata={},
                        content=body,
                        parse_error=f"Invalid YAML frontmatter: {e}"
                    )

        # Fallback title from markdown heading if not in metadata
        title = str(metadata.get("title", "")).strip()
        if not title:
            for line in body.splitlines():
                line_str = line.strip()
                if line_str.startswith("# "):
                    title = line_str[2:].strip()
                    break
        if not title:
            title = file_path.stem.replace("-", " ").replace("_", " ").title()

        doc_type = str(metadata.get("type", "Plant")).strip()
        status = str(metadata.get("status", "draft")).strip()
        verified_raw = metadata.get("verified", False)
        verified = bool(verified_raw) if isinstance(verified_raw, bool) else (str(verified_raw).lower() in ("true", "1", "yes"))

        tags_raw = metadata.get("tags", [])
        if isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if t]
        elif isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        else:
            tags = []

        try:
            relative_path = str(file_path.relative_to(self.knowledge_dir)).replace("\\", "/")
        except ValueError:
            relative_path = file_path.name

        return KnowledgeDocument(
            file_path=file_path,
            relative_path=relative_path,
            metadata=metadata,
            content=body,
            title=title,
            doc_type=doc_type,
            status=status,
            verified=verified,
            tags=tags
        )

    def load_all(self) -> Tuple[List[KnowledgeDocument], List[Tuple[Path, str]]]:
        """
        Recursively discover and parse eligible markdown files.
        Returns (valid_documents, errors_list).
        """
        documents: List[KnowledgeDocument] = []
        errors: List[Tuple[Path, str]] = []

        if not self.knowledge_dir.exists():
            return documents, [(self.knowledge_dir, "Directory does not exist")]

        for file_path in sorted(self.knowledge_dir.rglob("*.md")):
            if file_path.name.lower() in self.IGNORED_FILENAMES:
                continue

            doc = self.parse_file(file_path)
            if doc.parse_error:
                errors.append((file_path, doc.parse_error))
            else:
                documents.append(doc)

        return documents, errors
