"""Tests for document upload validation, extension whitelisting, and size limits.
"""

from fastapi import status


def test_upload_valid_txt_document(client):
    """Test successful ingestion of a valid text document."""
    content = b"This is a valid company mental wellness policy. Take breaks every two hours."
    files = {"file": ("wellness_policy.txt", content, "text/plain")}

    resp = client.post("/upload/document", files=files)
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert data["filename"] == "wellness_policy.txt"
    assert data["chunk_count"] >= 1
    assert data["status"] == "indexed"


def test_reject_unsupported_extensions(client):
    """Test rejection of non-whitelisted file extensions (e.g., .exe, .py, .docx)."""
    disallowed = [
        ("malicious.exe", b"binary content", "application/octet-stream"),
        ("script.py", b"print('hello')", "text/x-python"),
        ("document.docx", b"word content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ]
    for filename, content, mime in disallowed:
        files = {"file": (filename, content, mime)}
        resp = client.post("/upload/document", files=files)
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unsupported file extension" in resp.json()["detail"]


def test_reject_empty_document(client):
    """Test rejection of empty 0-byte document uploads."""
    files = {"file": ("empty.txt", b"", "text/plain")}
    resp = client.post("/upload/document", files=files)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "empty" in resp.json()["detail"].lower()


def test_reject_oversized_document(client):
    """Test rejection of documents exceeding the 10MB threshold."""
    # 10.5 MB payload
    large_payload = b"A" * (11 * 1024 * 1024)
    files = {"file": ("huge_document.txt", large_payload, "text/plain")}
    resp = client.post("/upload/document", files=files)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "exceeds maximum allowed limit" in resp.json()["detail"]
