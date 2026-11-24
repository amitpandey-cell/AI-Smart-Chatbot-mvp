"""Google Gemini API provider adapter for text and multimodal vision.
"""

import base64
from typing import Optional
import httpx

from src.config import get_settings
from src.providers.base import LLMProvider, VisionProvider, ProviderNotConfiguredError
from src.safety.logger import get_logger

logger = get_logger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(LLMProvider, VisionProvider):
    """Interfaces with Google Gemini for text and visual reasoning."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model

        if not self.api_key or not self.api_key.strip():
            logger.info("GeminiProvider instantiated without GEMINI_API_KEY.")

    def _verify_configured(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ProviderNotConfiguredError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in .env or switch LLM_PROVIDER=mock."
            )

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._verify_configured()
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent?key={self.api_key}"

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"Instructions: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these guidelines."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})
        payload = {"contents": contents}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")

            data = resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return "Received empty or unexpected response structure from Gemini API."

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        self._verify_configured()
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent?key={self.api_key}"

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt or "Describe and analyze this image in detail."},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_image,
                            }
                        },
                    ],
                }
            ]
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini Vision API error ({resp.status_code}): {resp.text}")

            data = resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return "Could not extract image analysis text from Gemini API response."
