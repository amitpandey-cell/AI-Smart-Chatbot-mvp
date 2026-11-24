"""Configuration management for AI-Smart-Chatbot.

Provides typed application settings with localhost defaults, upload size limits,
extension whitelisting, and a public-safe summary method to prevent secret leaks.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import List, Set
from dotenv import load_dotenv

# Load .env file early if present
load_dotenv()


class Settings:
    """Application configuration settings with environment variable fallbacks."""

    def __init__(self):
        # Server & Network Bindings
        self.BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
        self.BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
        self.FRONTEND_PORT: int = int(os.getenv("FRONTEND_PORT", "8501"))

        # Database Configuration (MySQL with SQLite fallback)
        self.DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
        self.DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
        self.DB_USER: str = os.getenv("DB_USER", "root")
        self.DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
        self.DB_NAME: str = os.getenv("DB_NAME", "ai_smart_chatbot")
        self.DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
        self.SQLITE_PATH: str = os.getenv("SQLITE_PATH", "ai_chatbot.db")

        # AI Provider Selections (mock, gemini, openai)
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock").lower()
        self.VISION_PROVIDER: str = os.getenv("VISION_PROVIDER", "mock").lower()
        self.STT_PROVIDER: str = os.getenv("STT_PROVIDER", "mock").lower()
        self.TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "mock").lower()

        # API Keys (External providers)
        self.GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

        # RAG Configuration
        self.RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
        self.RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
        self.RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
        self.RAG_MIN_SCORE: float = float(os.getenv("RAG_MIN_SCORE", "0.1"))

        # Uploads & Security
        self.MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
        self.ALLOWED_DOCUMENT_EXTENSIONS: Set[str] = {".pdf", ".txt"}
        self.ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}
        self.ALLOWED_AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".ogg", ".m4a"}

        # Privacy & CORS
        self.ENABLE_CONVERSATION_PERSISTENCE: bool = (
            os.getenv("ENABLE_CONVERSATION_PERSISTENCE", "false").lower() in ("true", "1", "yes")
        )
        self.CORS_ORIGINS: str = os.getenv(
            "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
        )

        # Base Directories
        self.BASE_DIR: Path = Path(__file__).resolve().parent.parent
        self.DATA_DIR: Path = self.BASE_DIR / "data"

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum allowed upload file size in bytes."""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def cors_origins_list(self) -> List[str]:
        """Parsed list of allowed CORS origins."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def get_public_safe_summary(self) -> dict:
        """Return safe configuration summary scrubbed of secrets and credentials."""
        return {
            "backend_host": self.BACKEND_HOST,
            "backend_port": self.BACKEND_PORT,
            "frontend_port": self.FRONTEND_PORT,
            "db_host": self.DB_HOST,
            "db_port": self.DB_PORT,
            "db_name": self.DB_NAME,
            "llm_provider": self.LLM_PROVIDER,
            "vision_provider": self.VISION_PROVIDER,
            "stt_provider": self.STT_PROVIDER,
            "tts_provider": self.TTS_PROVIDER,
            "gemini_api_key_configured": bool(self.GEMINI_API_KEY.strip()),
            "openai_api_key_configured": bool(self.OPENAI_API_KEY.strip()),
            "max_upload_size_mb": self.MAX_UPLOAD_SIZE_MB,
            "rag_chunk_size": self.RAG_CHUNK_SIZE,
            "rag_chunk_overlap": self.RAG_CHUNK_OVERLAP,
            "rag_top_k": self.RAG_TOP_K,
            "rag_min_score": self.RAG_MIN_SCORE,
            "conversation_persistence_enabled": self.ENABLE_CONVERSATION_PERSISTENCE,
        }


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
