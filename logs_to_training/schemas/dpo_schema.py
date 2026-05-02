"""DPO candidate row: shared prompt with chosen vs rejected completions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DPOCandidateRow(BaseModel):
    """
    Preference learning row.

    `prompt` is the shared prefix (often system + user + partial trajectory).
    `chosen` / `rejected` are completion strings or serialized message tails,
    depending on trainer; we default to assistant completion text for TRL-style DPO.
    """

    prompt: str
    chosen: str
    rejected: str
    pair_type: Literal[
        "tool_path",
        "persona_adherence",
        "efficiency",
        "synthetic_contrast",
        "hard_negative",
    ] = "tool_path"
    metadata: dict[str, Any] = Field(default_factory=dict)
