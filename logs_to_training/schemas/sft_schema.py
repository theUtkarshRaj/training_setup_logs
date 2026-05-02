"""LoRA-ready SFT row schema (OpenAI-style messages)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallBlock(BaseModel):
    """Assistant tool_calls item compatible with common chat templates."""

    id: str
    type: Literal["function"] = "function"
    function: dict[str, Any]


class ChatMessage(BaseModel):
    """One message in an SFT conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCallBlock] | None = None
    tool_call_id: str | None = None


class SFTRow(BaseModel):
    """Single JSONL record for supervised fine-tuning."""

    messages: list[ChatMessage]
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Complexity tier, session_id, segmentation; safe for trainer filtering.",
    )
