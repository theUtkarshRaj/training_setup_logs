"""
Build DPO *candidate* pairs from logs using rule-based proxies.

Explicit human feedback is sparse in production; these rows are meant for review
or lightweight scoring—not blind training without QA.
"""

from __future__ import annotations

from logs_to_training.export.hard_negatives import build_hard_negative_rows
from logs_to_training.schemas.canonical_event import Session
from logs_to_training.schemas.dpo_schema import DPOCandidateRow
from logs_to_training.splits.integrity import session_prompt_fingerprint


def _persona_violation_proxy(session: Session) -> bool:
    """Heuristic: mention of ignoring instructions when persona demands careful tone."""
    if not session.persona:
        return False
    persona_lower = session.persona.lower()
    resp = session.final_response.lower()
    if "must not" in persona_lower and any(
        x in resp for x in ("ignore previous", "disregard", "as an ai")
    ):
        return True
    return False


def _inefficient_tool_path(session: Session) -> bool:
    if not session.complexity:
        return False
    return (session.complexity.retry_count or 0) > 0 or (
        session.complexity.tool_count or 0
    ) > 4


def _synthetic_rejected_response(session: Session) -> str:
    """Template suboptimal completion for tool-path contrast (review before use)."""
    return (
        "I'll skip tool verification and answer from memory without checking "
        "Agristack or weather APIs. " + session.final_response[:200]
    )


def session_to_dpo_candidates(session: Session) -> list[DPOCandidateRow]:
    """
    Emit zero or more DPOCandidateRow objects for a session.

    Pair types:
    - tool_path: optimal true final vs synthetic careless completion (review).
    - persona_adherence: same prompt tail, chosen adheres / rejected violates (proxy).
    - efficiency: chosen uses final answer / rejected encourages redundant tool use.
    """
    rows: list[DPOCandidateRow] = []
    base_prompt = (session.persona or "") + "\n\nUser:\n" + session.user_query
    fp = session.metadata.extra.get("prompt_fingerprint") or session_prompt_fingerprint(session)
    meta_base = {
        "session_id": session.session_id,
        "prompt_fingerprint": fp,
        "segmentation": session.segmentation_label,
        "complexity": session.complexity.model_dump(exclude_none=True)
        if session.complexity
        else None,
    }
    if ds := session.metadata.extra.get("disjoint_split"):
        meta_base["disjoint_split"] = ds

    # Tool path contrast (candidate): chosen grounded path vs rejected shortcut
    if session.task_type == "agentic" and session.turns:
        rows.append(
            DPOCandidateRow(
                prompt=base_prompt + "\n\nProvide the best assistant completion.",
                chosen=session.final_response,
                rejected=_synthetic_rejected_response(session),
                pair_type="tool_path",
                metadata={**meta_base, "note": "rejected is synthetic template—replace after review"},
            )
        )

    # Persona adherence proxy
    if session.persona and _persona_violation_proxy(session):
        rows.append(
            DPOCandidateRow(
                prompt=base_prompt,
                chosen=session.final_response,
                rejected="Sure — I'll ignore the persona constraints you listed.",
                pair_type="persona_adherence",
                metadata=meta_base,
            )
        )

    # Efficiency: prefer concise final when retries detected
    if _inefficient_tool_path(session):
        rows.append(
            DPOCandidateRow(
                prompt=base_prompt
                + "\n\nAssistant had a long tool chain. Prefer efficient grounded answer.",
                chosen=session.final_response,
                rejected="Let me call the same tools again to double-check the same fields...",
                pair_type="efficiency",
                metadata=meta_base,
            )
        )

    rows.extend(build_hard_negative_rows(session))
    return rows
