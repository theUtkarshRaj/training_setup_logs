"""Trajectory complexity signals for stratified sampling and curriculum hooks."""

from __future__ import annotations

import re
from typing import Literal

from logs_to_training.schemas.canonical_event import ComplexityTags, Session


def _retry_like(tool_names: list[str]) -> int:
    """Count adjacent duplicate tool names as soft proxy for retries."""
    retries = 0
    for i in range(1, len(tool_names)):
        if tool_names[i] == tool_names[i - 1]:
            retries += 1
    return retries


def _recovery_detected(session: Session, tool_names: list[str]) -> bool:
    if any("error" in (t.error or "").lower() for t in session.turns):
        return True
    for tr in [x for t in session.turns for x in t.tool_returns]:
        low = tr.content.lower()
        if any(s in low for s in ("timeout", "rate limit", "failed", "invalid", "no results")):
            return True
    return len(tool_names) >= 3 and _retry_like(tool_names) > 0


def _ambiguity_score(session: Session) -> float:
    """Lightweight lexical ambiguity proxy (no extra models)."""
    q = session.user_query.lower()
    score = 0.0
    if re.search(r"\b(?:maybe|either|unclear|not sure)\b", q):
        score += 0.35
    if re.search(r"\bor\b", q) or re.search(r"\bversus\b", q) or re.search(r"\bvs\.?\b", q):
        score += 0.2
    if "?" in q and q.count("?") > 1:
        score += 0.2
    if len(q.split()) > 80:
        score += 0.15
    vague = len(re.findall(r"\b(something|anything|stuff|thing)\b", q))
    score += min(0.3, vague * 0.1)
    return min(1.0, score)


def _tier(
    step_count: int,
    tool_count: int,
    recovery: bool,
    ambiguity: float,
) -> Literal["low", "medium", "high"]:
    if tool_count <= 1 and step_count <= 2 and ambiguity < 0.25 and not recovery:
        return "low"
    if tool_count >= 4 or recovery or ambiguity >= 0.55 or step_count >= 6:
        return "high"
    return "medium"


def compute_complexity(session: Session) -> ComplexityTags:
    tool_names: list[str] = []
    for t in session.turns:
        tool_names.extend(tc.tool_name for tc in t.tool_calls)

    tool_count = len(tool_names)
    unique_tools = len(set(tool_names))
    retry_count = _retry_like(tool_names)
    recovery = _recovery_detected(session, tool_names)
    ambiguity = _ambiguity_score(session)
    step_count = len(session.turns)

    tier = _tier(step_count, tool_count, recovery, ambiguity)
    tags = ComplexityTags(
        tool_count=tool_count,
        unique_tools=unique_tools,
        retry_count=retry_count,
        recovery_detected=recovery,
        ambiguity_score=round(ambiguity, 4),
        step_count=step_count,
        complexity_tier=tier,
    )
    session.complexity = tags
    return tags
