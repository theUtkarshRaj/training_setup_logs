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
