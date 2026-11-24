"""Database initialization and seed ingestion script.

Initializes schema tables and optionally ingests the default sample wellness handbook.
Usage:
    python scripts/init_db.py [--seed]
"""

import argparse
from pathlib import Path
import sys
import uuid

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.connection import get_db_manager
from src.database.repositories import DocumentRepository
from src.rag.chunker import TextChunker
from src.rag.extractor import DocumentExtractor
from src.safety.logger import get_logger

logger = get_logger("init_db")


def initialize_database(seed: bool = True) -> None:
    """Initialize tables and optionally seed the sample knowledge base."""
    db = get_db_manager()
    health = db.check_health()
    print(f"[*] Database connection verified: backend={health['db_type']}, active={health['connected']}")

    if not health["connected"]:
        print("[!] Error: Could not establish a database connection.")
        sys.exit(1)

    if seed:
        sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_knowledge_base.txt"
        if not sample_path.exists():
            print(f"[!] Sample file not found at: {sample_path}")
            return

        print(f"[*] Ingesting sample knowledge base: {sample_path.name}")
        extractor = DocumentExtractor()
        chunker = TextChunker()
        doc_repo = DocumentRepository(db)

        # Check if already ingested
        existing_docs = doc_repo.list_documents()
        for doc in existing_docs:
            if doc["filename"] == sample_path.name:
                print(f"[*] Sample document '{sample_path.name}' already indexed (ID: {doc['id']}). Skipping re-seeding.")
                return

        # Extract & chunk
        raw_text = sample_path.read_text(encoding="utf-8")
        clean_text = extractor.clean_text(raw_text)
        chunks = chunker.chunk_text(clean_text)

        doc_id = str(uuid.uuid4())
        doc_repo.create_document(
            doc_id=doc_id,
            filename=sample_path.name,
            file_type="txt",
            file_size_bytes=len(raw_text.encode("utf-8")),
            chunk_count=len(chunks),
        )

        chunk_records = [
            {
                "id": str(uuid.uuid4()),
                "document_id": doc_id,
                "chunk_index": i,
                "chunk_text": chunk,
                "char_length": len(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]
        doc_repo.insert_chunks(chunk_records)
        print(f"[+] Successfully seeded '{sample_path.name}' ({len(chunks)} chunks inserted, doc_id: {doc_id}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize AI-Smart-Chatbot database.")
    parser.add_argument("--seed", action="store_true", default=True, help="Seed sample knowledge base.")
    parser.add_argument("--no-seed", dest="seed", action="store_false", help="Skip seeding.")
    args = parser.parse_args()

    initialize_database(seed=args.seed)
