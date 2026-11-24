"""FastAPI route controllers for conversational chat, RAG, and multimodal media.
"""

import base64
from pathlib import Path
from typing import List, Optional
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.models import (
    AudioUploadResponse,
    ChatRequest,
    ChatResponse,
    CitationModel,
    ClearChatRequest,
    ClearChatResponse,
    DeleteDocumentResponse,
    DocumentItem,
    DocumentUploadResponse,
    HealthResponse,
    ImageUploadResponse,
)
from src.config import get_settings
from src.database.connection import get_db_manager
from src.database.repositories import ConversationRepository, DocumentRepository
from src.providers.base import ProviderNotConfiguredError
from src.providers.factory import ProviderFactory
from src.rag.chunker import TextChunker
from src.rag.extractor import DocumentExtractionError, DocumentExtractor
from src.rag.retriever import RAGRetriever
from src.safety.guardrails import SafetyGuardrails
from src.safety.logger import get_logger
from src.safety.prompts import SYSTEM_PROMPT_CHATBOT, build_rag_prompt

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Service and database connectivity health probe."""
    settings = get_settings()
    db = get_db_manager()
    health = db.check_health()
    safe_summary = settings.get_public_safe_summary()

    return HealthResponse(
        status="ok" if health["connected"] else "degraded",
        db_connected=health["connected"],
        db_type=health["db_type"],
        providers={
            "llm": safe_summary["llm_provider"],
            "vision": safe_summary["vision_provider"],
            "stt": safe_summary["stt_provider"],
            "tts": safe_summary["tts_provider"],
            "gemini_ready": safe_summary["gemini_api_key_configured"],
            "openai_ready": safe_summary["openai_api_key_configured"],
        },
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main conversational endpoint with safety guardrails and RAG grounding."""
    settings = get_settings()
    db = get_db_manager()
    doc_repo = DocumentRepository(db)
    conv_repo = ConversationRepository(db)
    retriever = RAGRetriever(doc_repo)

    conversation_id = conv_repo.get_or_create_conversation(request.conversation_id)

    # 1. Pre-generation safety guardrails check (Acute crisis & Medical boundary)
    safety_check = SafetyGuardrails.evaluate_input(request.message, region=request.region)
    if safety_check.triggered and safety_check.bypass_llm:
        logger.info("Safety guardrail triggered for conversation %s: %s", conversation_id, safety_check.category)
        if settings.ENABLE_CONVERSATION_PERSISTENCE:
            conv_repo.add_message(conversation_id, "user", request.message)
            conv_repo.add_message(conversation_id, "assistant", safety_check.response or "")

        return ChatResponse(
            reply=safety_check.response or "",
            citations=[],
            audio_base64=None,
            conversation_id=conversation_id,
            safety_triggered=True,
            safety_category=safety_check.category,
        )

    # 2. RAG Context Retrieval
    formatted_context = ""
    citations: List[CitationModel] = []
    if request.use_rag:
        ctx_str, raw_citations = retriever.retrieve(request.message)
        formatted_context = ctx_str
        citations = [CitationModel(**c) for c in raw_citations]

    # 3. Assemble prompt & invoke LLM Provider
    prompt_to_send = build_rag_prompt(request.message, formatted_context) if request.use_rag else request.message

    try:
        llm = ProviderFactory.get_llm_provider()
        reply = await llm.generate_response(prompt_to_send, system_prompt=SYSTEM_PROMPT_CHATBOT)
    except ProviderNotConfiguredError as exc:
        logger.error("Provider configuration error: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.error("LLM generation failure: %s. Returning fallback grounding.", exc)
        reply = SafetyGuardrails.fallback_support_response()

    # 4. Optional TTS audio synthesis
    audio_b64 = None
    if request.synthesize_speech and reply:
        try:
            tts = ProviderFactory.get_tts_provider()
            audio_bytes = await tts.synthesize(reply)
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as exc:
            logger.warning("TTS synthesis failed (%s), returning text-only reply.", exc)

    # 5. Optional conversation persistence
    if settings.ENABLE_CONVERSATION_PERSISTENCE:
        conv_repo.add_message(conversation_id, "user", request.message)
        conv_repo.add_message(
            conversation_id,
            "assistant",
            reply,
            citations_json=str([c.model_dump() for c in citations]) if citations else None,
            audio_present=bool(audio_b64),
        )

    return ChatResponse(
        reply=reply,
        citations=citations,
        audio_base64=audio_b64,
        conversation_id=conversation_id,
        safety_triggered=False,
    )


@router.post("/upload/document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a reference document (.pdf or .txt) into the RAG knowledge base."""
    filename = file.filename or "uploaded_document.txt"
    contents = await file.read()

    extractor = DocumentExtractor()
    chunker = TextChunker()
    doc_repo = DocumentRepository()

    try:
        clean_text, file_type = extractor.extract_text(filename, contents)
    except DocumentExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error extracting document %s: %s", filename, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process document.")

    chunks = chunker.chunk_text(clean_text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text could not be partitioned into valid chunks.",
        )

    doc_id = str(uuid.uuid4())
    doc_repo.create_document(
        doc_id=doc_id,
        filename=filename,
        file_type=file_type,
        file_size_bytes=len(contents),
        chunk_count=len(chunks),
    )

    chunk_records = [
        {
            "id": str(uuid.uuid4()),
            "document_id": doc_id,
            "chunk_index": idx,
            "chunk_text": chunk_str,
            "char_length": len(chunk_str),
        }
        for idx, chunk_str in enumerate(chunks)
    ]
    doc_repo.insert_chunks(chunk_records)
    logger.info("Indexed document '%s' (ID: %s, %d chunks)", filename, doc_id, len(chunks))

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=filename,
        chunk_count=len(chunks),
        status="indexed",
    )


@router.get("/documents", response_model=List[DocumentItem])
async def list_documents():
    """Retrieve list of all indexed documents in the knowledge base."""
    doc_repo = DocumentRepository()
    records = doc_repo.list_documents()
    return [
        DocumentItem(
            id=r["id"],
            filename=r["filename"],
            file_type=r["file_type"],
            file_size_bytes=r["file_size_bytes"],
            chunk_count=r["chunk_count"],
            created_at=str(r["created_at"]) if r.get("created_at") else None,
        )
        for r in records
    ]


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str):
    """Delete a document and its cascading chunks from the knowledge base."""
    doc_repo = DocumentRepository()
    existing = doc_repo.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found.")

    success = doc_repo.delete_document(document_id)
    return DeleteDocumentResponse(success=success, deleted_id=document_id)


@router.post("/upload/image", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    prompt: str = Form("Describe and analyze what is in this image from a wellness perspective."),
    conversation_id: Optional[str] = Form(None),
):
    """Multimodal visual query endpoint."""
    settings = get_settings()
    filename = file.filename or "image.png"
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded image is empty.")

    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image exceeds size limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image extension '{suffix}'. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}",
        )

    conv_repo = ConversationRepository()
    session_id = conv_repo.get_or_create_conversation(conversation_id)

    try:
        vision_provider = ProviderFactory.get_vision_provider()
        reply = await vision_provider.analyze_image(contents, prompt)
    except ProviderNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.error("Vision provider failure: %s", exc)
        reply = f"Unable to analyze image due to provider error: {exc}"

    return ImageUploadResponse(
        reply=reply,
        filename=filename,
        conversation_id=session_id,
    )


@router.post("/upload/audio", response_model=AudioUploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
):
    """Speech-to-Text transcription and conversational response pipeline."""
    settings = get_settings()
    filename = file.filename or "audio.wav"
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")

    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio exceeds size limit of {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio extension '{suffix}'. Allowed: {settings.ALLOWED_AUDIO_EXTENSIONS}",
        )

    conv_repo = ConversationRepository()
    session_id = conv_repo.get_or_create_conversation(conversation_id)

    # 1. Transcribe audio
    try:
        stt = ProviderFactory.get_stt_provider()
        transcribed_text = await stt.transcribe(contents, filename)
    except ProviderNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.error("STT provider failure: %s", exc)
        transcribed_text = "I am seeking emotional wellness support."

    # 2. Check safety guardrails on transcribed text
    safety_check = SafetyGuardrails.evaluate_input(transcribed_text)
    if safety_check.triggered and safety_check.bypass_llm:
        return AudioUploadResponse(
            transcribed_text=transcribed_text,
            reply=safety_check.response or "",
            audio_base64=None,
            conversation_id=session_id,
        )

    # 3. Generate response
    try:
        llm = ProviderFactory.get_llm_provider()
        reply = await llm.generate_response(transcribed_text, system_prompt=SYSTEM_PROMPT_CHATBOT)
    except Exception as exc:
        logger.error("LLM response for audio query failed: %s", exc)
        reply = SafetyGuardrails.fallback_support_response()

    # 4. Synthesize reply speech
    audio_b64 = None
    try:
        tts = ProviderFactory.get_tts_provider()
        synth_bytes = await tts.synthesize(reply)
        if synth_bytes:
            audio_b64 = base64.b64encode(synth_bytes).decode("utf-8")
    except Exception as exc:
        logger.warning("TTS synthesis for audio response failed: %s", exc)

    return AudioUploadResponse(
        transcribed_text=transcribed_text,
        reply=reply,
        audio_base64=audio_b64,
        conversation_id=session_id,
    )


@router.post("/chat/clear", response_model=ClearChatResponse)
async def clear_chat(request: ClearChatRequest):
    """Purge messages and session state for the specified conversation ID."""
    conv_repo = ConversationRepository()
    cleared = conv_repo.clear_conversation(request.conversation_id)
    return ClearChatResponse(cleared=cleared, conversation_id=request.conversation_id)
