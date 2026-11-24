"""Safety guardrails and emergency crisis interception.

Evaluates user inputs prior to LLM or RAG processing. Detects acute emotional crisis,
self-harm, suicide ideation, domestic violence, and inappropriate requests for medical
diagnosis or prescription medication modifications. Provides verified hotlines and
failsafe grounding support.
"""

from dataclasses import dataclass
import re
from typing import Optional, Dict


@dataclass
class GuardrailResult:
    """Evaluation result from safety guardrails check."""
    triggered: bool
    category: Optional[str] = None  # 'crisis', 'medical', None
    response: Optional[str] = None
    bypass_llm: bool = False


class SafetyGuardrails:
    """Pre-generation input interceptor and emotional safety evaluator."""

    # Crisis patterns: acute self-harm, suicide, severe domestic abuse/violence
    CRISIS_PATTERNS = [
        re.compile(r"\b(suicid(e|al)|kill\s+myself|end\s+my\s+life|want\s+to\s+die|hang\s+myself)\b", re.IGNORECASE),
        re.compile(r"\b(self[\s-]?harm|cut(ting)?\s+(myself|my\s+wrists)|overdose)\b", re.IGNORECASE),
        re.compile(r"\b(better\s+off\s+dead|don'?t\s+want\s+to\s+live(\s+anymore)?|no\s+reason\s+to\s+live)\b", re.IGNORECASE),
        re.compile(r"\b(domestic\s+abuse|partner\s+is\s+hitting\s+me|being\s+abused|physically\s+abusive)\b", re.IGNORECASE),
    ]

    # Medical & Prescription patterns: diagnosis requests and medication tampering
    MEDICAL_PATTERNS = [
        re.compile(r"\b(diagnose\s+me|do\s+I\s+have\s+(bipolar|schizophrenia|depression|adhd|autism|cancer))\b", re.IGNORECASE),
        re.compile(r"\b(should\s+I\s+(increase|decrease|stop|change|taper)\s+my\s+(dose|dosage|medication|antidepressant|adderall|lexapro|zoloft|prozac|xanax))\b", re.IGNORECASE),
        re.compile(r"\b(what\s+dose\s+of\s+\w+\s+should\s+I\s+take|prescribe\s+me)\b", re.IGNORECASE),
    ]

    # Verified emergency directories (No invented numbers)
    HOTLINES: Dict[str, str] = {
        "US_CANADA": "Dial or text 988 (Suicide & Crisis Lifeline, 24/7, free, confidential).",
        "UK": "Call 111 (NHS mental health services) or 999 for imminent emergencies.",
        "EU": "Dial 112 (European emergency services).",
        "INDIA": "Dial 14416 (Tele-MANAS mental health helpline) or 112.",
        "INTERNATIONAL": "Visit https://findahelpline.com for confidential support in over 130 countries.",
    }

    @classmethod
    def evaluate_input(cls, text: str, region: Optional[str] = None) -> GuardrailResult:
        """Evaluate text for acute crisis or medical advice requests."""
        if not text or not text.strip():
            return GuardrailResult(triggered=False)

        clean_text = text.strip()

        # Check acute crisis triggers first (highest priority)
        for pattern in cls.CRISIS_PATTERNS:
            if pattern.search(clean_text):
                return GuardrailResult(
                    triggered=True,
                    category="crisis",
                    response=cls._build_crisis_response(region),
                    bypass_llm=True,
                )

        # Check medical/prescription boundaries
        for pattern in cls.MEDICAL_PATTERNS:
            if pattern.search(clean_text):
                return GuardrailResult(
                    triggered=True,
                    category="medical",
                    response=cls._build_medical_response(),
                    bypass_llm=True,
                )

        return GuardrailResult(triggered=False)

    @classmethod
    def _build_crisis_response(cls, region: Optional[str] = None) -> str:
        """Generate compassionate crisis intervention with verified hotlines."""
        region_key = region.upper() if region and region.upper() in cls.HOTLINES else None
        specific_line = f"\n* Local Support: {cls.HOTLINES[region_key]}" if region_key else ""

        return (
            "I hear how much pain you are carrying right now, and I want you to be safe. "
            "Please know that you do not have to carry this alone, and there is compassionate, "
            "confidential help available immediately:\n\n"
            f"* United States & Canada: {cls.HOTLINES['US_CANADA']}\n"
            f"* United Kingdom: {cls.HOTLINES['UK']}\n"
            f"* European Union: {cls.HOTLINES['EU']}\n"
            f"* India: {cls.HOTLINES['INDIA']}\n"
            f"* Global Directory: {cls.HOTLINES['INTERNATIONAL']}"
            f"{specific_line}\n\n"
            "If you or someone around you is in immediate danger, please reach out to emergency "
            "services right now. There are people ready and waiting to support you through this."
        )

    @classmethod
    def _build_medical_response(self) -> str:
        """Generate firm yet empathetic rejection of medical diagnosis and dosing advice."""
        return (
            "I want to support your well-being, but as an AI assistant, I am not a doctor, "
            "psychiatrist, or healthcare professional. I cannot diagnose medical or mental health "
            "conditions, nor can I advise you to alter, start, or stop prescription medications.\n\n"
            "Please speak directly with your doctor, psychiatrist, or pharmacist. They understand "
            "your medical history and can safely guide your care and medication decisions."
        )

    @classmethod
    def fallback_support_response(cls) -> str:
        """Failsafe response providing 5-4-3-2-1 grounding protocol during outages or errors."""
        return (
            "I apologize, but I am temporarily experiencing connection difficulties. "
            "If you are feeling overwhelmed, anxious, or distressed right now, please take a moment "
            "to try this grounding exercise:\n\n"
            "**The 5-4-3-2-1 Sensory Grounding Technique:**\n"
            "• **5** things you can see around you right now.\n"
            "• **4** things you can physically feel (feet on floor, hands on your lap).\n"
            "• **3** things you can hear (room hum, distant sounds, your breath).\n"
            "• **2** things you can smell around you.\n"
            "• **1** thing you can taste or a sip of cool water.\n\n"
            "Take slow, steady breaths. If you are experiencing a crisis, confidential help is available 24/7 "
            "at 988 (US/Canada), 111/999 (UK), 112 (EU/India), or https://findahelpline.com."
        )
