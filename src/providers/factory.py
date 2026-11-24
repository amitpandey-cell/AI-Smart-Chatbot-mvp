"""Provider factory resolving AI implementations with credential validation.
"""

from typing import Optional
from src.config import get_settings
from src.providers.base import (
    LLMProvider,
    VisionProvider,
    STTProvider,
    TTSProvider,
    ProviderNotConfiguredError,
)
from src.providers.mock import (
    MockLLMProvider,
    MockVisionProvider,
    MockSTTProvider,
    MockTTSProvider,
)
from src.providers.gemini import GeminiProvider
from src.providers.openai import OpenAIProvider
from src.safety.logger import get_logger

logger = get_logger(__name__)


class ProviderFactory:
    """Factory creating appropriate AI provider instances based on configuration."""

    @classmethod
    def get_llm_provider(cls, name: Optional[str] = None) -> LLMProvider:
        settings = get_settings()
        provider_name = (name or settings.LLM_PROVIDER).lower()

        if provider_name == "mock":
            return MockLLMProvider()
        elif provider_name == "gemini":
            if not settings.GEMINI_API_KEY.strip():
                raise ProviderNotConfiguredError(
                    "Gemini API key is not configured. Set GEMINI_API_KEY in .env or switch LLM_PROVIDER=mock."
                )
            return GeminiProvider()
        elif provider_name == "openai":
            if not settings.OPENAI_API_KEY.strip():
                raise ProviderNotConfiguredError(
                    "OpenAI API key is not configured. Set OPENAI_API_KEY in .env or switch LLM_PROVIDER=mock."
                )
            return OpenAIProvider()
        else:
            raise ValueError(f"Unknown LLM provider: '{provider_name}'. Supported: mock, gemini, openai.")

    @classmethod
    def get_vision_provider(cls, name: Optional[str] = None) -> VisionProvider:
        settings = get_settings()
        provider_name = (name or settings.VISION_PROVIDER).lower()

        if provider_name == "mock":
            return MockVisionProvider()
        elif provider_name == "gemini":
            if not settings.GEMINI_API_KEY.strip():
                raise ProviderNotConfiguredError(
                    "Gemini API key is not configured. Set GEMINI_API_KEY in .env or switch VISION_PROVIDER=mock."
                )
            return GeminiProvider()
        elif provider_name == "openai":
            if not settings.OPENAI_API_KEY.strip():
                raise ProviderNotConfiguredError(
                    "OpenAI API key is not configured. Set OPENAI_API_KEY in .env or switch VISION_PROVIDER=mock."
                )
            return OpenAIProvider()
        else:
            raise ValueError(f"Unknown Vision provider: '{provider_name}'. Supported: mock, gemini, openai.")

    @classmethod
    def get_stt_provider(cls, name: Optional[str] = None) -> STTProvider:
        settings = get_settings()
        provider_name = (name or settings.STT_PROVIDER).lower()

        if provider_name == "mock":
            return MockSTTProvider()
        elif provider_name == "openai":
            if not settings.OPENAI_API_KEY.strip():
                raise ProviderNotConfiguredError(
                    "OpenAI API key is not configured. Set OPENAI_API_KEY in .env or switch STT_PROVIDER=mock."
                )
            return OpenAIProvider()
        else:
            raise ValueError(f"Unknown STT provider: '{provider_name}'. Supported: mock, openai.")

    @classmethod
    def get_tts_provider(cls, name: Optional[str] = None) -> TTSProvider:
        settings = get_settings()
        provider_name = (name or settings.TTS_PROVIDER).lower()

        if provider_name == "mock":
            return MockTTSProvider()
        elif provider_name == "openai":
            if not settings.OPENAI_API_KEY.strip():
                raise ProviderNotConfiguredError(
                    "OpenAI API key is not configured. Set OPENAI_API_KEY in .env or switch TTS_PROVIDER=mock."
                )
            return OpenAIProvider()
        else:
            raise ValueError(f"Unknown TTS provider: '{provider_name}'. Supported: mock, openai.")
