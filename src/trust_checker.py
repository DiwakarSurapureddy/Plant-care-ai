"""Trust and freshness checker for knowledge notes and retrieved chunks."""
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import re

from src.chunker import KnowledgeChunk


@dataclass
class TrustWarning:
    """Represents a trust, verification, or freshness advisory."""
    source_file: str
    chunk_heading: str
    severity: str  # "warning", "caution", "info"
    message: str


class TrustChecker:
    """Evaluates retrieved chunks for draft status, lack of verification, and staleness."""

    def __init__(self, current_date: Optional[date] = None):
        self.current_date = current_date or date.today()

    def _parse_date(self, date_val: Any) -> Optional[date]:
        """Safely parse date objects, ISO strings, or malformed values."""
        if isinstance(date_val, (datetime, date)):
            return date_val if isinstance(date_val, date) else date_val.date()
        if not isinstance(date_val, str) or not date_val.strip():
            return None

        clean_val = date_val.strip()
        # Common ISO formats
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(clean_val, fmt).date()
            except ValueError:
                continue
        return None

    def check_chunk(self, chunk: KnowledgeChunk) -> List[TrustWarning]:
        """Evaluate a single KnowledgeChunk for trust issues."""
        warnings: List[TrustWarning] = []
        meta = chunk.metadata or {}

        # 1. Draft or Deprecated status
        status_val = str(chunk.status or meta.get("status", "draft")).lower().strip()
        if status_val == "draft":
            warnings.append(TrustWarning(
                source_file=chunk.source_file,
                chunk_heading=chunk.section_heading,
                severity="warning",
                message=f"Note is in 'draft' status; care details are preliminary and not formally finalized."
            ))
        elif status_val in ("deprecated", "archived"):
            warnings.append(TrustWarning(
                source_file=chunk.source_file,
                chunk_heading=chunk.section_heading,
                severity="warning",
                message=f"Note has been marked as '{status_val}'. Information may be obsolete."
            ))

        # 2. Verification status
        is_verified = bool(chunk.verified or meta.get("verified", False))
        if not is_verified:
            # Special emphasis if discussing prices or specific medical/nursery claims
            is_price_section = "price" in chunk.section_heading.lower() or "price" in chunk.content.lower()
            if is_price_section:
                warnings.append(TrustWarning(
                    source_file=chunk.source_file,
                    chunk_heading=chunk.section_heading,
                    severity="caution",
                    message="Price data is unverified or marked unavailable. Real nursery price requires verified local source."
                ))
            else:
                warnings.append(TrustWarning(
                    source_file=chunk.source_file,
                    chunk_heading=chunk.section_heading,
                    severity="info",
                    message="Note is unverified (verified: false or absent). Treat guidance as practical draft advice."
                ))

        # 3. Staleness check
        stale_after_raw = meta.get("stale_after")
        if stale_after_raw:
            stale_date = self._parse_date(stale_after_raw)
            if stale_date and stale_date < self.current_date:
                warnings.append(TrustWarning(
                    source_file=chunk.source_file,
                    chunk_heading=chunk.section_heading,
                    severity="warning",
                    message=f"Note expired on {stale_date.isoformat()} (stale_after passed). Requires review."
                ))

        return warnings

    def check_all(self, chunks: List[KnowledgeChunk]) -> List[TrustWarning]:
        """Check a list of retrieved chunks, deduplicating warnings per file/message."""
        all_warnings: List[TrustWarning] = []
        seen = set()

        for chunk in chunks:
            for w in self.check_chunk(chunk):
                key = (w.source_file, w.severity, w.message)
                if key not in seen:
                    seen.add(key)
                    all_warnings.append(w)

        return all_warnings
