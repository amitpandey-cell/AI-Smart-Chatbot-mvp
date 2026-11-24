"""Mock AI provider implementations for zero-dependency offline execution and testing.
"""

import io
import math
import struct
import wave
from typing import Optional
from PIL import Image

from src.providers.base import LLMProvider, VisionProvider, STTProvider, TTSProvider
from src.safety.prompts import GROUNDING_BOX_BREATHING_TEXT, GROUNDING_54321_TEXT, UNTRUSTED_DOCS_START


class MockLLMProvider(LLMProvider):
    """Mock LLM generating grounded answers or empathetic support."""

    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        prompt_lower = prompt.lower()

        # Check if RAG reference documents are included in the prompt
        if UNTRUSTED_DOCS_START in prompt:
            # Parse query and context
            if "stipend" in prompt_lower or "$1,200" in prompt_lower or "ben-well" in prompt_lower or "expense" in prompt_lower:
                return (
                    "According to Section 4 of the Aura Wellness Handbook, employees are eligible for up to $1,200 "
                    "annually for licensed mental health therapy and psychiatric consultations under Benefit Code BEN-WELL-2026. "
                    "Reimbursements can be filed through the internal portal using Expense Code EXP-MTH-902."
                )
            if "box breathing" in prompt_lower or "square breathing" in prompt_lower:
                return (
                    "Based on Section 2.2 of the Aura Wellness Handbook, Box Breathing is a 4-step technique:\n"
                    "1. Inhale slowly for 4 seconds.\n"
                    "2. Hold your breath for 4 seconds.\n"
                    "3. Exhale smoothly for 4 seconds.\n"
                    "4. Rest with empty lungs for 4 seconds.\n"
                    "Repeat 4 to 6 cycles to help regulate your nervous system."
                )
            if "5-4-3-2-1" in prompt_lower or "sensory" in prompt_lower:
                return (
                    "According to Section 2.1 of the Aura Wellness Handbook, the 5-4-3-2-1 sensory grounding method helps "
                    "relieve acute anxiety by identifying:\n"
                    "- 5 things you can see\n- 4 things you can feel\n- 3 things you can hear\n"
                    "- 2 things you can smell\n- 1 thing you can taste."
                )
            if "resilience day" in prompt_lower:
                return (
                    "According to the Aura Wellness Handbook, employees are entitled to 4 designated mental resilience "
                    "recharge days per calendar year in addition to regular PTO, with no medical certificate required."
                )

            return (
                "Based on the provided Aura Wellness reference documentation, I have reviewed the excerpts and synthesized "
                "the guidance relevant to your question. Please let me know if you would like me to elaborate on any specific section."
            )

        # Empathetic emotional support queries
        if any(w in prompt_lower for w in ["stress", "anxious", "anxiety", "overwhelm", "burnout", "tired"]):
            return (
                "I hear how much pressure you're feeling right now, and it is completely understandable to feel overwhelmed. "
                "Let's take a pause together. Would you like to try a gentle grounding exercise right now?\n\n"
                f"{GROUNDING_BOX_BREATHING_TEXT}\n\n"
                "I'm here with you. What feels like the heaviest part of your day right now?"
            )

        if "breath" in prompt_lower or "relax" in prompt_lower:
            return (
                "Taking a moment to breathe is a powerful way to reset. Here is a calming technique:\n\n"
                f"{GROUNDING_BOX_BREATHING_TEXT}"
            )

        # Default conversational response
        return (
            "Hello! I am Aura, your wellness and resilience assistant. I am here to provide a listening ear, "
            "help guide you through calming exercises, and answer questions about our company wellness benefits and resources. "
            "How can I support you today?"
        )


class MockVisionProvider(VisionProvider):
    """Mock Vision Provider inspecting image dimensions, format, and color profiles."""

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            img_format = image.format or "Unknown"
            mode = image.mode

            # Calculate average RGB if mode is RGB or RGBA
            color_desc = "balanced tones"
            if mode in ("RGB", "RGBA"):
                rgb_img = image.convert("RGB")
                small = rgb_img.resize((1, 1))
                r, g, b = small.getpixel((0, 0))
                if r > g and r > b:
                    color_desc = "warm reddish/amber tones"
                elif g > r and g > b:
                    color_desc = "calming green/nature tones"
                elif b > r and b > g:
                    color_desc = "cool blue/slate tones"
                else:
                    color_desc = f"neutral tones (RGB {r},{g},{b})"

            return (
                f"I have inspected the uploaded image. It is a {img_format} image with dimensions "
                f"{width}x{height} pixels in {mode} color mode, showing predominant {color_desc}.\n\n"
                f"Regarding your inquiry ('{prompt}'): The image appears calm and visually structured. "
                "If this is a workplace setup, workspace view, or mindfulness snapshot, it suggests an organized, reflective space."
            )
        except Exception as exc:
            return f"I received the image file ({len(image_bytes)} bytes), but could not parse visual metadata: {exc}"


class MockSTTProvider(STTProvider):
    """Mock Speech-to-Text provider simulating transcription."""

    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        size = len(audio_bytes)
        if size == 0:
            return ""
        # Return realistic transcribed query
        return "I am feeling stressed about my upcoming project deadline and would like some grounding advice."


class MockTTSProvider(TTSProvider):
    """Mock Text-to-Speech provider synthesizing a valid PCM WAV audio tone."""

    async def synthesize(self, text: str) -> bytes:
        # Generate a short, soft pleasant chime (440Hz A4 tone fading gently, 1 second, 16000Hz)
        sample_rate = 16000
        duration_sec = 1.0
        num_samples = int(sample_rate * duration_sec)
        frequency = 440.0  # Hz

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            samples = []
            for i in range(num_samples):
                t = i / sample_rate
                # Envelope decay to make a soft bell/chime sound
                decay = math.exp(-3.0 * t)
                value = int(32767.0 * 0.4 * decay * math.sin(2.0 * math.pi * frequency * t))
                samples.append(struct.pack("<h", max(-32768, min(32767, value))))

            wav_file.writeframes(b"".join(samples))

        return buffer.getvalue()
