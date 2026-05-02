"""Canonical session / turn models for normalized Langfuse-style traces."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CanonicalMetadata(BaseModel):
    """Provider and run metadata carried through the pipeline (non-PII by contract)."""

    run_id: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    provider_url: str | None = None
    provider_response_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A single tool invocation from the assistant."""

    tool_name: str
    tool_call_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = None
    raw_part: dict[str, Any] | None = Field(
        default=None,
        description="Original part dict for audit; strip before external export if policy requires.",
    )


class ToolReturn(BaseModel):
    """Observation returned to the model after a tool call."""

    tool_name: str
    tool_call_id: str
    content: str
    timestamp: datetime | None = None
    raw_part: dict[str, Any] | None = None


class Turn(BaseModel):
    """One logical step: assistant message and/or tool interaction."""

    turn_index: int
    assistant_text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_returns: list[ToolReturn] = Field(default_factory=list)
    metadata: CanonicalMetadata | None = None
    error: str | None = None


class ComplexityTags(BaseModel):
    """Placeholder at session level; filled by tagging.complexity."""

    tool_count: int | None = None
    unique_tools: int | None = None
    retry_count: int | None = None
    recovery_detected: bool | None = None
    ambiguity_score: float | None = None
    step_count: int | None = None
    complexity_tier: Literal["low", "medium", "high"] | None = None


class Session(BaseModel):
    """End-to-end canonical representation of one logged interaction."""

    session_id: str
    user_query: str
    final_response: str
    persona: str | None = Field(
        default=None,
        description="System / persona text if present in logs; used for adherence checks.",
    )
    task_type: Literal["qa", "agentic"] = "qa"
    turns: list[Turn] = Field(default_factory=list)
    success: bool = True
    timestamp: datetime | None = None
    complexity: ComplexityTags | None = Field(
        default=None,
        description="Populated after complexity tagging.",
    )
    metadata: CanonicalMetadata = Field(default_factory=CanonicalMetadata)
    segmentation_label: str | None = Field(
        default=None,
        description="single_turn_qa | multi_turn_qa | agentic_trajectory | failed_trajectory",
    )
    source_format: str = "langfuse_pydantic"
