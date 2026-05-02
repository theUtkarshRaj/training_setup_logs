"""
Curated hard-negative *templates* for DPO-style contrast (review before training).

Each template targets a distinct failure mode aligned with downstream alignment:
tool discipline, persona boundaries, factual grounding, and safety around IDs.
"""

from __future__ import annotations

import re

from logs_to_training.schemas.canonical_event import Session
from logs_to_training.schemas.dpo_schema import DPOCandidateRow
from logs_to_training.splits.integrity import session_prompt_fingerprint


def _base_meta(session: Session) -> dict:
    fp = session.metadata.extra.get("prompt_fingerprint") or session_prompt_fingerprint(session)
    meta: dict = {
        "session_id": session.session_id,
        "prompt_fingerprint": fp,
        "segmentation": session.segmentation_label,
        "complexity": session.complexity.model_dump(exclude_none=True)
        if session.complexity
        else None,
    }
    if ds := session.metadata.extra.get("disjoint_split"):
        meta["disjoint_split"] = ds
    return meta


def _user_prompt_block(session: Session) -> str:
    return (session.persona or "") + "\n\nUser:\n" + session.user_query


def rejected_tool_omission(session: Session) -> str:
    """Skips tools despite production requiring grounded tool use."""
    return (
        "I'll answer directly from general knowledge without calling your registry or "
        "weather tools, so this may not match live Agristack or forecast data. "
        f"Summary: {session.final_response[:120]}…"
    )


def rejected_extra_tool_call(session: Session) -> str:
    """Encourages redundant / noisy tool choreography."""
    tools = []
    for t in session.turns:
        tools.extend(tc.tool_name for tc in t.tool_calls)
    extra = "soil_moisture_probe" if "soil_moisture_probe" not in tools else "satellite_ndvi_timeseries"
    return (
        "Next I'll re-run the same lookup with duplicate parameters, then call "
        f"`{extra}` with default args even though the user didn't ask, then summarize. "
        "This maximizes latency while looking busy."
    )


def rejected_persona_leakage(session: Session) -> str:
    """Leaks system framing / internal instructions (bad UX + training hazard)."""
    snippet = (session.persona or "SYSTEM PROMPT REDACTED")[:180].replace("\n", " ")
    return (
        "INTERNAL NOTE — I will paste my system instructions for transparency: "
        f"«{snippet}». Now the user-visible answer: " + session.final_response[:200]
    )


def rejected_hallucinated_gov_id(session: Session) -> str:
    """Fabricates government-style identifiers (hard safety negative)."""
    return (
        "Verified: your Aadhaar on file is 1234 5678 9012 and your PM-KISAN beneficiary "
        f"id is IN-FARM-88421. {session.final_response[:160]}"
    )


def rejected_slight_factual_drift(session: Session) -> str:
    """Subtly perturbs numeric/geo facts while sounding confident."""
    text = session.final_response
    text = re.sub(r"19\.0?66", "18.412", text, count=1)
    text = re.sub(r"77\.17[0-9]", "76.900", text, count=1)
    text = re.sub(r"\b32°C\b", "29°C", text, count=1)
    if text == session.final_response:
        text = (
            session.final_response
            + " (Update: district boundary shifted slightly—use 18.4N, 76.9E for all plots.)"
        )
    return text


def build_hard_negative_rows(session: Session) -> list[DPOCandidateRow]:
    """
    Emit DPO rows where `chosen` is the logged assistant completion and `rejected`
    is a **template** illustrating a specific defect. All require human or model QA
    before merging into a preference dataset.
    """
    base = _user_prompt_block(session)
    meta = {**_base_meta(session), "review_gate": "hard_negative_template"}
    prompt_suffix = "\n\nProvide the best assistant completion."
    rows: list[DPOCandidateRow] = []

    if session.task_type == "agentic" and session.turns:
        rows.extend(
            [
                DPOCandidateRow(
                    prompt=base + prompt_suffix,
                    chosen=session.final_response,
                    rejected=rejected_tool_omission(session),
                    pair_type="hard_negative",
                    metadata={**meta, "hard_negative_kind": "tool_omission"},
                ),
                DPOCandidateRow(
                    prompt=base + prompt_suffix,
                    chosen=session.final_response,
                    rejected=rejected_extra_tool_call(session),
                    pair_type="hard_negative",
                    metadata={**meta, "hard_negative_kind": "extra_tool_call"},
                ),
            ]
        )

    # Persona / safety negatives apply broadly (not only agentic).
    rows.append(
        DPOCandidateRow(
            prompt=base + prompt_suffix,
            chosen=session.final_response,
            rejected=rejected_persona_leakage(session),
            pair_type="hard_negative",
            metadata={**meta, "hard_negative_kind": "persona_leakage"},
        )
    )
    rows.append(
        DPOCandidateRow(
            prompt=base + prompt_suffix,
            chosen=session.final_response,
            rejected=rejected_hallucinated_gov_id(session),
            pair_type="hard_negative",
            metadata={**meta, "hard_negative_kind": "hallucinated_gov_id"},
        )
    )
    rows.append(
        DPOCandidateRow(
            prompt=base + prompt_suffix,
            chosen=session.final_response,
            rejected=rejected_slight_factual_drift(session),
            pair_type="hard_negative",
            metadata={**meta, "hard_negative_kind": "slight_factual_drift"},
        )
    )

    return rows
