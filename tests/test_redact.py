import pytest

from logs_to_training.pii.placeholder_map import PlaceholderMap
from logs_to_training.pii.redact import redact_text


def test_placeholder_consistency_within_session():
    pmap = PlaceholderMap()
    t = "Call me at 9876543210 or 9876543210 again."
    out = redact_text(t, pmap)
    assert "<PHONE_" in out
    assert "9876543210" not in out


def test_email_and_aadhaar_patterns():
    pmap = PlaceholderMap()
    s = "Email farmer@example.com id 1234 5678 9012"
    out = redact_text(s, pmap)
    assert "farmer@example.com" not in out
    assert "<EMAIL_" in out
    assert "1234 5678 9012" not in out


# --- Presidio-specific tests (skipped when presidio_analyzer is not installed) ---

def test_indian_phone_with_country_code_presidio():
    pytest.importorskip("presidio_analyzer")
    pmap = PlaceholderMap()
    out = redact_text("Call me at +91 9876543210 for support.", pmap)
    assert "+91 9876543210" not in out
    assert "<UK_NHS_" not in out


def test_bare_10digit_indian_mobile_no_uk_nhs_label():
    pytest.importorskip("presidio_analyzer")
    pmap = PlaceholderMap()
    # Bare 10-digit number — UK_NHS recognizer would previously fire on this
    out = redact_text("Contact: 9876543210", pmap)
    assert "9876543210" not in out
    assert "<UK_NHS_" not in out
