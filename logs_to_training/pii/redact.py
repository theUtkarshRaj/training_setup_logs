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


def _apply_regex_redactions(text: str, pmap: PlaceholderMap) -> str:
    def sub(pattern: re.Pattern[str], category: str, s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            val = m.group(0).strip()
            return pmap.placeholder_for(category, val)

        return pattern.sub(repl, s)

    out = text
    out = sub(_EMAIL, "email", out)
    out = sub(_BEARER, "token", out)
    out = sub(_JWT, "token", out)
    out = sub(_HEX_KEY, "api_key", out)
    out = sub(_AWS_KEY, "api_key", out)
    out = sub(_AADHAAR, "gov_id", out)
    out = sub(_LONG_NUMERIC_ID, "gov_id", out)
    out = sub(_PHONE, "phone", out)
    return out


def _presidio_redact(text: str, pmap: PlaceholderMap) -> str | None:
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]
    except ImportError:
        return None

    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text

    sorted_results = sorted(results, key=lambda r: r.start, reverse=True)
    out = text
    for r in sorted_results:
        span = text[r.start : r.end]
        category = (r.entity_type or "PII").lower().replace(" ", "_")
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
