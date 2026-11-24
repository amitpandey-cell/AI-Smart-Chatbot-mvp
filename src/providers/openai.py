"""OpenAI API provider adapter for GPT-4o-mini, Whisper-1, and TTS-1.
"""

import base64
from typing import Optional
import httpx

from src.config import get_settings
from src.providers.base import (
    LLMProvider,
    VisionProvider,
    STTProvider,
    TTSProvider,
    ProviderNotConfiguredError,
)
from src.safety.logger import get_logger

logger = get_logger(__name__)

OPENAI_API_BASE = "https://api.openai.com/v1"


class OpenAIProvider(LLMProvider, VisionProvider, STTProvider, TTSProvider):
    """Interfaces with OpenAI endpoints for text, vision, transcription, and speech."""

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.OPENAI_API_KEY

        if not self.api_key or not self.api_key.strip():
            logger.info("OpenAIProvider instantiated without OPENAI_API_KEY.")

    def _verify_configured(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ProviderNotConfiguredError(
                "OpenAI API key is not configured. Set OPENAI_API_KEY in .env or switch providers to 'mock'."
            )

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
        }

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._verify_configured()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI Chat API error ({resp.status_code}): {resp.text}")

            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        self._verify_configured()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "Describe and analyze this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 500,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers=self._get_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI Vision API error ({resp.status_code}): {resp.text}")

            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        self._verify_configured()
        files = {
            "file": (filename or "audio.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "model": "whisper-1",
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{OPENAI_API_BASE}/audio/transcriptions",
                headers=self._get_headers(),
                files=files,
                data=data,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI Whisper API error ({resp.status_code}): {resp.text}")

            return resp.json().get("text", "")

    async def synthesize(self, text: str) -> bytes:
        self._verify_configured()
        payload = {
            "model": "tts-1",
            "input": text[:4000],  # Clip text if oversized
            "voice": "alloy",
            "response_format": "mp3",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OPENAI_API_BASE}/audio/speech",
                headers=self._get_headers(),
                json=payload,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI TTS API error ({resp.status_code}): {resp.text}")

            return resp.content
