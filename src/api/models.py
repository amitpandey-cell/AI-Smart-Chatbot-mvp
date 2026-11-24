"""Pydantic schemas for FastAPI request and response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check payload."""
    status: str
    db_connected: bool
    db_type: str
    providers: Dict[str, Any]


class ChatRequest(BaseModel):
    """Chat message payload."""
    message: str = Field(..., min_length=1, max_length=4000, description="User prompt text")
    conversation_id: Optional[str] = Field(None, description="Client conversation session ID")
    use_rag: bool = Field(True, description="Whether to query knowledge base for grounding")
    synthesize_speech: bool = Field(False, description="Whether to generate TTS audio response")
    region: Optional[str] = Field(None, description="Optional region code for crisis hotlines (e.g. US, UK, EU, India)")


class CitationModel(BaseModel):
    """Citation metadata for grounded RAG excerpts."""
    document_id: str
    filename: str
    chunk_index: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    """Chat response payload."""
    reply: str
    citations: List[CitationModel] = []
    audio_base64: Optional[str] = None
    conversation_id: str
    safety_triggered: bool = False
    safety_category: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Document ingestion response."""
    document_id: str
    filename: str
    chunk_count: int
    status: str = "indexed"


class DocumentItem(BaseModel):
    """Document summary item."""
    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    created_at: Optional[Any] = None


class DeleteDocumentResponse(BaseModel):
    """Document deletion response."""
    success: bool
    deleted_id: str


class ImageUploadResponse(BaseModel):
    """Multimodal image query response."""
    reply: str
    filename: str
    conversation_id: str


class AudioUploadResponse(BaseModel):
    """Speech transcription and response payload."""
    transcribed_text: str
    reply: str
    audio_base64: Optional[str] = None
    conversation_id: str


class ClearChatRequest(BaseModel):
    """Clear chat request payload."""
    conversation_id: str


class ClearChatResponse(BaseModel):
    """Clear chat response payload."""
    cleared: bool
    conversation_id: str
