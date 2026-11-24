"""Database repositories implementing parameterized queries for documents, chunks, and chat history.

Supports natural language search with token-overlap fallback and ensures SQL injection prevention.
"""

from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional
import uuid

from src.database.connection import DatabaseManager, get_db_manager
from src.safety.logger import get_logger

logger = get_logger(__name__)


class DocumentRepository:
    """Repository handling document and chunk persistence."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or get_db_manager()

    def create_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        chunk_count: int = 0,
    ) -> Dict[str, Any]:
        """Insert document record."""
        sql = """
            INSERT INTO documents (id, filename, file_type, file_size_bytes, chunk_count)
            VALUES (%s, %s, %s, %s, %s)
        """
        self.db.execute(sql, (doc_id, filename, file_type, file_size_bytes, chunk_count))
        return self.get_document(doc_id) or {}

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single document metadata."""
        sql = "SELECT id, filename, file_type, file_size_bytes, chunk_count, created_at FROM documents WHERE id = %s"
        return self.db.fetch_one(sql, (doc_id,))

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents ordered by creation time descending."""
        sql = "SELECT id, filename, file_type, file_size_bytes, chunk_count, created_at FROM documents ORDER BY created_at DESC"
        return self.db.fetch_all(sql)

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and cascading chunks."""
        # Clean chunks first in case foreign key cascade is disabled in some configurations
        self.db.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        count = self.db.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        return count > 0

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Insert a batch of document chunks."""
        inserted = 0
        for chunk in chunks:
            sql = """
                INSERT INTO document_chunks (id, document_id, chunk_index, chunk_text, char_length)
                VALUES (%s, %s, %s, %s, %s)
            """
            self.db.execute(
                sql,
                (
                    chunk["id"],
                    chunk["document_id"],
                    chunk["chunk_index"],
                    chunk["chunk_text"],
                    chunk.get("char_length", len(chunk["chunk_text"])),
                ),
            )
            inserted += 1
        return inserted

    def search_chunks(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search chunks using fulltext MATCH or token overlap scoring."""
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        tokens = [t.lower() for t in re.findall(r"\b[A-Za-z0-9_-]{3,}\b", clean_query)]

        # If running MySQL with fulltext
        if self.db.db_type == "mysql":
            try:
                sql = """
                    SELECT c.id, c.document_id, c.chunk_index, c.chunk_text, d.filename,
                           MATCH(c.chunk_text) AGAINST(%s IN NATURAL LANGUAGE MODE) AS score
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE MATCH(c.chunk_text) AGAINST(%s IN NATURAL LANGUAGE MODE)
                    ORDER BY score DESC
                    LIMIT %s
                """
                results = self.db.fetch_all(sql, (clean_query, clean_query, top_k))
                if results:
                    return results
            except Exception as exc:
                logger.warning("MySQL MATCH search failed or fulltext not enabled (%s). Falling back to token overlap.", exc)

        # Token-overlap scoring fallback (works everywhere, including SQLite)
        all_chunks_sql = """
            SELECT c.id, c.document_id, c.chunk_index, c.chunk_text, d.filename
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
        """
        all_chunks = self.db.fetch_all(all_chunks_sql)
        scored_chunks: List[Dict[str, Any]] = []

        query_lower = clean_query.lower()

        for chunk in all_chunks:
            text_lower = chunk["chunk_text"].lower()
            score = 0.0

            # Substring exact phrase match bonus
            if query_lower in text_lower:
                score += 2.0

            # Token overlap score
            if tokens:
                matched_tokens = sum(1 for token in tokens if token in text_lower)
                score += (matched_tokens / len(tokens))

            if score > 0.0:
                scored_chunk = dict(chunk)
                scored_chunk["score"] = round(score, 3)
                scored_chunks.append(scored_chunk)

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]


class ConversationRepository:
    """Repository handling optional conversation and message persistence."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or get_db_manager()

    def get_or_create_conversation(self, conv_id: Optional[str] = None) -> str:
        """Retrieve existing conversation or create a new one."""
        if not conv_id:
            conv_id = str(uuid.uuid4())

        existing = self.db.fetch_one("SELECT id FROM conversations WHERE id = %s", (conv_id,))
        if not existing:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            self.db.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (%s, %s, %s, %s)",
                (conv_id, "New Conversation", now_str, now_str),
            )
        return conv_id

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        citations_json: Optional[str] = None,
        audio_present: bool = False,
    ) -> str:
        """Insert conversation message."""
        self.get_or_create_conversation(conv_id)
        msg_id = str(uuid.uuid4())
        sql = """
            INSERT INTO messages (id, conversation_id, role, content, citations_json, audio_present)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.db.execute(sql, (msg_id, conv_id, role, content, citations_json, 1 if audio_present else 0))
        return msg_id

    def get_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        """Retrieve ordered messages for a conversation."""
        sql = """
            SELECT id, conversation_id, role, content, citations_json, audio_present, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY created_at ASC
        """
        return self.db.fetch_all(sql, (conv_id,))

    def clear_conversation(self, conv_id: str) -> bool:
        """Purge messages for a conversation while retaining uploaded knowledge base."""
        self.db.execute("DELETE FROM messages WHERE conversation_id = %s", (conv_id,))
        count = self.db.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))
        return count > 0
