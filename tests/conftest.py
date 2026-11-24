"""Shared pytest fixtures for isolated offline testing.
"""

import os
from pathlib import Path
import tempfile

# Force offline safe testing configuration BEFORE any src imports
os.environ["FORCE_SQLITE"] = "true"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["VISION_PROVIDER"] = "mock"
os.environ["STT_PROVIDER"] = "mock"
os.environ["TTS_PROVIDER"] = "mock"
os.environ["GEMINI_API_KEY"] = "fake_test_gemini_key_12345"
os.environ["OPENAI_API_KEY"] = "fake_test_openai_key_67890"

import pytest
from starlette.testclient import TestClient

from src.api.main import create_app
from src.database.connection import DatabaseManager, get_db_manager


@pytest.fixture(scope="session")
def temp_db_path():
    """Create a temporary SQLite database file for testing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    yield tmp.name
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_env(temp_db_path):
    """Ensure test environment uses mock providers and isolated sqlite db."""
    os.environ["SQLITE_PATH"] = temp_db_path
    db = get_db_manager(override_sqlite_path=temp_db_path)
    db._ensure_sqlite_schema()
    yield db


@pytest.fixture
def client(setup_test_env):
    """FastAPI TestClient fixture."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
