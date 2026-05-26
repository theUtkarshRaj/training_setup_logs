"""
PII and secret redaction with Presidio when installed, regex fallback otherwise.

Design goals: deterministic placeholders, session-level consistency, minimal deps by default.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from logs_to_training.pii.placeholder_map import PlaceholderMap

if TYPE_CHECKING:
    from logs_to_training.schemas.canonical_event import Session

# --- Regex patterns (conservative; prefer false positives over leaks) ---

_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)
# E.164-ish and common local formats
_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"
    r"|\b\+?\d{10,14}\b",
)
# Aadhaar-style 12 digits, optional spaces
_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
# Generic long digit IDs (govt / account style)
_LONG_NUMERIC_ID = re.compile(r"\b\d{12,16}\b")
# API keys / tokens (broad heuristics)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]{20,}\b", re.IGNORECASE)
_HEX_KEY = re.compile(r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{20,}\b", re.IGNORECASE)
_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")

# Presidio entity-type → canonical label.  Prevents region-specific recognizers
# that fire on phone-shaped digit runs from leaking wrong labels into placeholders.
_ENTITY_TYPE_NORM: dict[str, str] = {
    "UK_NHS": "PHONE_NUMBER",
    "UK_NINO": "PHONE_NUMBER",
    "US_DRIVER_LICENSE": "PHONE_NUMBER",
    "IN_PAN": "GOVT_ID",
    "IN_AADHAAR": "GOVT_ID",
}

# Matches any placeholder token already inserted by either redaction pass, e.g. <PHONE_NUMBER_1>
_PLACEHOLDER_RE = re.compile(r"(<[A-Z][A-Z0-9_]*_\d+>)")

# Matches a bare country-code prefix (+XX/+XXX with optional separator) that immediately
# precedes a placeholder — Presidio consumed the digits but not the prefix.
_ORPHANED_CC_RE = re.compile(r"\+\d{1,3}[\s.\-]?(?=<[A-Z][A-Z0-9_]*_\d+>)")


def _apply_regex_redactions(text: str, pmap: PlaceholderMap) -> str:
    # Strip country-code prefixes stranded when Presidio replaced only the digits
    # (e.g. "+91 <PHONE_NUMBER_1>" → "<PHONE_NUMBER_1>")
    text = _ORPHANED_CC_RE.sub("", text)

    def _sub_seg(seg: str) -> str:
        def sub(pattern: re.Pattern[str], category: str, s: str) -> str:
            def repl(m: re.Match[str]) -> str:
                return pmap.placeholder_for(category, m.group(0).strip())
            return pattern.sub(repl, s)
        s = seg
        s = sub(_EMAIL, "email", s)
        s = sub(_BEARER, "token", s)
        s = sub(_JWT, "token", s)
        s = sub(_HEX_KEY, "api_key", s)
        s = sub(_AWS_KEY, "api_key", s)
        s = sub(_AADHAAR, "gov_id", s)
        s = sub(_LONG_NUMERIC_ID, "gov_id", s)
        s = sub(_PHONE, "phone", s)
        return s

    # Split on existing placeholder tokens so regexes only run on raw text segments,
    # never on already-redacted placeholders.
    parts = _PLACEHOLDER_RE.split(text)
    return "".join(part if i % 2 == 1 else _sub_seg(part) for i, part in enumerate(parts))


def _presidio_redact(text: str, pmap: PlaceholderMap) -> str | None:
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer  # type: ignore[import-untyped]
    except ImportError:
        return None

    indian_phone = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[
            # +91 or 0091 prefix with 10-digit mobile (starts 6-9)
            Pattern(name="indian_phone_intl", regex=r"(?:\+91|0091)[\s\-]?[6-9]\d{9}\b", score=0.9),
            # bare 10-digit Indian mobile — score above US_DRIVER_LICENSE (~0.65) so it wins
            Pattern(name="indian_mobile_bare", regex=r"\b[6-9]\d{9}\b", score=0.75),
        ],
    )
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(indian_phone)
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text

    # Deduplicate overlapping spans: keep only the highest-scoring result per span.
    # Without this, multiple recognizers firing on the same span cause position corruption
    # when the second replacement slices into the already-modified string.
    kept: list = []
    for r in sorted(results, key=lambda r: r.score, reverse=True):
        if not any(r.start < k.end and r.end > k.start for k in kept):
            kept.append(r)
    sorted_results = sorted(kept, key=lambda r: r.start, reverse=True)
    out = text
    for r in sorted_results:
        span = text[r.start : r.end]
        raw_type = r.entity_type or "PII"
        normalized = _ENTITY_TYPE_NORM.get(raw_type, raw_type)
        category = normalized.lower().replace(" ", "_")
        ph = pmap.placeholder_for(category, span)
        out = out[: r.start] + ph + out[r.end :]
    return out


def redact_text(text: str, pmap: PlaceholderMap) -> str:
    """
    Redact a single string. Tries Presidio first; falls back to regex rules.

    When Presidio is available, analyzer hits are replaced first, then regex
    catches patterns Presidio may miss (e.g. custom API key shapes).
    """
    if not text:
        return text
    presidio_out = _presidio_redact(text, pmap)
    base = presidio_out if presidio_out is not None else text
    return _apply_regex_redactions(base, pmap)


def redact_session_inplace(session: Session) -> Session:
    """Mutate canonical session strings in place (returns same object for chaining)."""
    pmap = PlaceholderMap()
    session.user_query = redact_text(session.user_query, pmap)
    session.final_response = redact_text(session.final_response, pmap)
    if session.persona:
        session.persona = redact_text(session.persona, pmap)
    for turn in session.turns:
        if turn.assistant_text:
            turn.assistant_text = redact_text(turn.assistant_text, pmap)
        for tc in turn.tool_calls:
            # JSON-serialize args values roughly by redacting string values only
            for k, v in list(tc.args.items()):
                if isinstance(v, str):
                    tc.args[k] = redact_text(v, pmap)
        for tr in turn.tool_returns:
            tr.content = redact_text(tr.content, pmap)
    return session
