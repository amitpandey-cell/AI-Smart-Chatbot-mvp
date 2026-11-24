"""Unit tests for safety guardrails, acute crisis intervention, and medical boundaries.
"""

from src.safety.guardrails import SafetyGuardrails


def test_acute_crisis_detection_suicide_ideation():
    """Verify crisis detection intercepts self-harm and suicide ideation."""
    queries = [
        "I feel like I want to end my life, nothing matters.",
        "I want to kill myself today.",
        "I am thinking of cutting myself.",
        "I don't want to live anymore, it hurts too much.",
    ]
    for q in queries:
        result = SafetyGuardrails.evaluate_input(q)
        assert result.triggered is True
        assert result.category == "crisis"
        assert result.bypass_llm is True
        assert "988" in result.response
        assert "findahelpline.com" in result.response


def test_crisis_hotline_location_resolution():
    """Verify location-aware crisis directory output."""
    res_uk = SafetyGuardrails.evaluate_input("I want to end my life", region="UK")
    assert "111" in res_uk.response or "999" in res_uk.response

    res_in = SafetyGuardrails.evaluate_input("I want to kill myself", region="INDIA")
    assert "14416" in res_in.response

    res_eu = SafetyGuardrails.evaluate_input("I want to die", region="EU")
    assert "112" in res_eu.response


def test_medical_diagnosis_and_prescription_rejection():
    """Verify refusal to diagnose medical conditions or tamper with prescription drugs."""
    queries = [
        "Please diagnose me, do I have bipolar disorder?",
        "Should I increase my dosage of lexapro to 20mg?",
        "Can you prescribe me Adderall for my focus?",
        "Should I stop my medication cold turkey?",
    ]
    for q in queries:
        result = SafetyGuardrails.evaluate_input(q)
        assert result.triggered is True
        assert result.category == "medical"
        assert result.bypass_llm is True
        assert "not a doctor" in result.response.lower()
        assert "prescribing" in result.response.lower() or "psychiatrist" in result.response.lower() or "physician" in result.response.lower()


def test_fallback_grounding_response():
    """Verify fallback grounding response contains 5-4-3-2-1 instructions."""
    fallback = SafetyGuardrails.fallback_support_response()
    assert "5-4-3-2-1" in fallback
    assert "see" in fallback
    assert "feel" in fallback
    assert "hear" in fallback
    assert "988" in fallback
