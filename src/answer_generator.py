"""Grounded Answer Generator enforcing strict attribution to retrieved knowledge."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.chunker import KnowledgeChunk
from src.ollama_client import OllamaClient
from src.retriever import RetrievedEvidence


SYSTEM_PROMPT = """You are Plant Care AI, an expert botanical and nursery assistant.
Your answers MUST be strictly grounded in the provided Knowledge Base Context.

CRITICAL RULES:
1. Primary Evidence: Use ONLY the provided knowledge chunks as your factual source. Do NOT make up facts.
2. Prices & Availability: Do NOT invent nursery prices, plant sizes, stock availability, or nursery addresses. If the retrieved context states "Not available yet" or does not specify price, explicitly state that verified local nursery pricing is not available.
3. Unknown Information: If the context does not contain the answer, explicitly state: "I could not find reliable information about this in the current plant knowledge base. Please add a relevant verified knowledge note."
4. Comparisons: For comparison queries, compare only the plants present in the retrieved evidence.
5. Related Plants: Mention related plants only if they appear in the retrieved context or are clearly labeled as general suggestions.
6. Clarity: Provide structured, practical answers using Markdown bullet points and headings.
7. Ambiguity: Ask a clarifying question if the user's inquiry or plant symptom is ambiguous.
8. No Live Search: Never claim you searched live web pages or databases. All knowledge is from local Markdown files.
"""


@dataclass
class GenerationResult:
    """The generated answer with metadata, sources, and diagnostic info."""
    answer: str
    evidence_used: List[RetrievedEvidence]
    is_grounded: bool
    source_documents: List[str]
    ollama_used: bool
    diagnostic_message: Optional[str] = None


class AnswerGenerator:
    """Combines retrieved evidence with the local Ollama LLM to produce grounded answers."""

    NO_EVIDENCE_MESSAGE = (
        "I could not find reliable information about this in the current plant knowledge base. "
        "Please add a relevant verified knowledge note in the `knowledge/` directory."
    )

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client

    def _build_context_prompt(self, evidence: List[RetrievedEvidence], user_query: str) -> str:
        """Format retrieved knowledge chunks into an explicit evidence context block."""
        context_blocks = []
        for i, item in enumerate(evidence, 1):
            c = item.chunk
            status_tag = f"[{c.status.upper()}]" if c.status else ""
            verified_tag = "[VERIFIED]" if c.verified else "[UNVERIFIED]"
            context_blocks.append(
                f"--- EVIDENCE CHUNK {i} ---\n"
                f"Source File: {c.source_file}\n"
                f"Plant / Subject: {c.plant_name}\n"
                f"Section: {c.section_heading}\n"
                f"Status: {status_tag} {verified_tag}\n"
                f"Content:\n{c.content}\n"
            )

        context_str = "\n".join(context_blocks)
        prompt = (
            f"KNOWLEDGE BASE CONTEXT:\n{context_str}\n"
            f"USER QUESTION: {user_query}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Answer the user's question directly and concisely based ONLY on the evidence above.\n"
            f"- Cite the relevant plant and section details where appropriate.\n"
            f"- If price is asked and marked unavailable, clearly say nursery pricing is not available yet.\n"
            f"- Format with bullet points where helpful."
        )
        return prompt

    def generate_answer(
        self,
        query: str,
        evidence: List[RetrievedEvidence]
    ) -> GenerationResult:
        """Generate a grounded answer using retrieved evidence and Ollama."""
        # Check if any evidence was retrieved
        if not evidence:
            return GenerationResult(
                answer=self.NO_EVIDENCE_MESSAGE,
                evidence_used=[],
                is_grounded=False,
                source_documents=[],
                ollama_used=False,
                diagnostic_message="No matching chunks found above the retrieval threshold."
            )

        sources = list(dict.fromkeys([f"{e.chunk.source_file} (## {e.chunk.section_heading})" for e in evidence]))

        # Format context for Ollama
        prompt = self._build_context_prompt(evidence, query)

        # Attempt generation via Ollama
        success, response_or_err = self.ollama_client.generate(prompt, system_prompt=SYSTEM_PROMPT)

        if success:
            return GenerationResult(
                answer=response_or_err,
                evidence_used=evidence,
                is_grounded=True,
                source_documents=sources,
                ollama_used=True,
                diagnostic_message=None
            )
        else:
            # If Ollama service or model is unavailable, synthesize direct grounded fallback
            # from retrieved knowledge chunks so the user still gets full benefit of the knowledge base!
            direct_synthesis = self._synthesize_direct_extract(evidence, query, response_or_err)
            return GenerationResult(
                answer=direct_synthesis,
                evidence_used=evidence,
                is_grounded=True,
                source_documents=sources,
                ollama_used=False,
                diagnostic_message=response_or_err
            )

    def _synthesize_direct_extract(
        self,
        evidence: List[RetrievedEvidence],
        query: str,
        ollama_error: str
    ) -> str:
        """Direct grounded extraction fallback when Ollama is unavailable."""
        lines = [
            "> **Notice**: Local Ollama model generation was bypassed because: "
            f"*{ollama_error}*.\n",
            "> Displaying direct extracts from the matched knowledge base below:\n"
        ]

        # Group by plant
        by_plant = {}
        for ev in evidence:
            plant = ev.chunk.plant_name
            by_plant.setdefault(plant, []).append(ev)

        for plant, items in by_plant.items():
            lines.append(f"### {plant}")
            for item in items:
                c = item.chunk
                lines.append(f"**{c.section_heading}** (`{c.source_file}`):")
                lines.append(f"{c.content}\n")

        return "\n".join(lines)
