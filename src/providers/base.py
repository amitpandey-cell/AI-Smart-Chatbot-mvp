"""Abstract base classes defining multimodal AI provider contracts.
"""

from abc import ABC, abstractmethod
from typing import Optional


class ProviderNotConfiguredError(Exception):
    """Raised when an external provider is selected without necessary API credentials."""
    pass


class LLMProvider(ABC):
    """Abstract interface for text generation language models."""

    @abstractmethod
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate conversational response given user prompt and optional system instructions."""
        pass


class VisionProvider(ABC):
    """Abstract interface for multimodal visual understanding."""

    @abstractmethod
    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        """Analyze image contents and answer user question."""
        pass


class STTProvider(ABC):
    """Abstract interface for Speech-to-Text transcription."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe speech audio buffer into plain text."""
        pass


class TTSProvider(ABC):
    """Abstract interface for Text-to-Speech audio synthesis."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize spoken audio bytes from text."""
        pass
