"""Tests for API health probe and secret leak prevention.
"""

from fastapi import status


def test_health_endpoint(client):
    """Validate /health returns 200 and healthy DB status without leaking secrets."""
    resp = client.get("/health")
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()

    assert data["status"] in ("ok", "degraded")
    assert data["db_connected"] is True
    assert data["db_type"] in ("sqlite", "mysql")
    assert "providers" in data

    # Verify no raw API keys or passwords are exposed
    raw_response_text = resp.text
    assert "fake_test_gemini_key_12345" not in raw_response_text
    assert "fake_test_openai_key_67890" not in raw_response_text
    assert "DB_PASSWORD" not in raw_response_text
    assert "password" not in raw_response_text.lower()
