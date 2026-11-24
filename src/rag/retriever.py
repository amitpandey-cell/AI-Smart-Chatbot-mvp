"""RAG Retriever and context assembler.

Finds relevant chunks, applies relevance thresholds, and formats untrusted context
blocks accompanied by structured citations.
"""

from typing import Any, Dict, List, Optional, Tuple
from src.config import get_settings
from src.database.repositories import DocumentRepository
from src.safety.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """Retrieves document chunks and constructs grounded context blocks."""

    def __init__(self, doc_repo: Optional[DocumentRepository] = None):
        self.settings = get_settings()
        self.doc_repo = doc_repo or DocumentRepository()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Retrieve top matching chunks, filter by score, and format context.

        Returns:
            Tuple of (formatted_context_string, list_of_citation_dicts)
        """
        k = top_k or self.settings.RAG_TOP_K
        threshold = min_score if min_score is not None else self.settings.RAG_MIN_SCORE

        matched_chunks = self.doc_repo.search_chunks(query, top_k=k)
        if not matched_chunks:
            return "", []

        valid_chunks: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        context_parts: List[str] = []

        for chunk in matched_chunks:
            score = chunk.get("score", 1.0)
            if score < threshold:
                continue

            valid_chunks.append(chunk)
            citation = {
                "document_id": chunk["document_id"],
                "filename": chunk.get("filename", "Unknown Document"),
                "chunk_index": chunk["chunk_index"],
                "score": float(score),
                "excerpt": chunk["chunk_text"][:250] + ("..." if len(chunk["chunk_text"]) > 250 else ""),
            }
            citations.append(citation)

            context_parts.append(
                f"[Source: {citation['filename']} | Chunk {citation['chunk_index']} | Score: {citation['score']}]\n"
                f"{chunk['chunk_text']}"
            )

        if not valid_chunks:
            return "", []

        formatted_context = "\n\n".join(context_parts)
        return formatted_context, citations
