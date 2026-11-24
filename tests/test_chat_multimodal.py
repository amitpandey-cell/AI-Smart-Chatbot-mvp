"""Tests for multimodal endpoints: conversational chat, image analysis, and audio speech.
"""

import base64
import io
from PIL import Image
from fastapi import status


def test_conversational_chat_without_rag(client):
    """Test standard conversational response without RAG."""
    payload = {
        "message": "I am feeling stressed and exhausted from work.",
        "use_rag": False,
        "synthesize_speech": False,
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert "reply" in data
    assert len(data["reply"]) > 10
    assert data["citations"] == []
    assert data["audio_base64"] is None


def test_conversational_chat_with_tts(client):
    """Test chat endpoint returns base64 synthesized audio when synthesize_speech is True."""
    payload = {
        "message": "Hello, can you hear me?",
        "use_rag": False,
        "synthesize_speech": True,
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["audio_base64"] is not None

    # Verify base64 audio decodes to valid WAV header ('RIFF')
    audio_bytes = base64.b64decode(data["audio_base64"])
    assert len(audio_bytes) > 44
    assert audio_bytes[:4] == b"RIFF"


def test_image_multimodal_upload(client):
    """Test image upload and visual inspection using mock vision provider."""
    # Create small synthetic test image
    img = Image.new("RGB", (120, 80), color=(70, 180, 120))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    files = {"file": ("mindfulness_setup.png", buffer.getvalue(), "image/png")}
    data = {"prompt": "What does this workspace look like?"}

    resp = client.post("/upload/image", files=files, data=data)
    assert resp.status_code == status.HTTP_200_OK
    res = resp.json()
    assert "PNG" in res["reply"]
    assert "120x80" in res["reply"]
    assert res["filename"] == "mindfulness_setup.png"


def test_audio_speech_to_text_and_response(client):
    """Test audio upload transcription and spoken response generation."""
    # Synthetic audio bytes (1 second of dummy data)
    synthetic_audio = b"RIFF" + b"\x00" * 200

    files = {"file": ("user_voice.wav", synthetic_audio, "audio/wav")}
    resp = client.post("/upload/audio", files=files)
    assert resp.status_code == status.HTTP_200_OK
    res = resp.json()
    assert "transcribed_text" in res
    assert len(res["transcribed_text"]) > 0
    assert "reply" in res
    assert res["audio_base64"] is not None


def test_document_lifecycle_and_deletion(client):
    """Test upload, list, and deletion of reference documents."""
    # Ingest document
    content = b"Ergonomic desk stipend is up to $400 for noise-cancelling headphones."
    files = {"file": ("ergonomics.txt", content, "text/plain")}
    up_resp = client.post("/upload/document", files=files)
    assert up_resp.status_code == status.HTTP_200_OK
    doc_id = up_resp.json()["document_id"]

    # List documents
    list_resp = client.get("/documents")
    assert list_resp.status_code == status.HTTP_200_OK
    docs = list_resp.json()
    assert any(d["id"] == doc_id for d in docs)

    # Delete document
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == status.HTTP_200_OK
    assert del_resp.json()["success"] is True

    # Confirm deletion
    list_resp_after = client.get("/documents")
    assert not any(d["id"] == doc_id for d in list_resp_after.json())


def test_clear_chat_endpoint(client):
    """Test clearing conversation state."""
    resp = client.post("/chat/clear", json={"conversation_id": "test-session-123"})
    assert resp.status_code == status.HTTP_200_OK
    assert resp.json()["conversation_id"] == "test-session-123"
