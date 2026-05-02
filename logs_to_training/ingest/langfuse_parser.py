"""
Ingest raw Langfuse / Pydantic-style JSON logs into a canonical Session.

Expected shapes (flexible):
- A single dict with user_question, bot_response, optional agent_turns, persona, ids.
- A list of such dicts (each becomes one session with synthetic session_id if missing).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from logs_to_training.schemas.canonical_event import (
    CanonicalMetadata,
    Session,
    ToolCall,
    ToolReturn,
    Turn,
)


class _RawToolPart(BaseModel):
    model_config = {"extra": "allow"}

    tool_name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None
    content: str | None = None
    part_kind: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


class _RawAgentTurn(BaseModel):
    model_config = {"extra": "allow"}

    parts: list[_RawToolPart] = Field(default_factory=list)
    timestamp: datetime | None = None
    kind: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    provider_url: str | None = None
    provider_details: dict[str, Any] | None = None
    provider_response_id: str | None = None
    finish_reason: str | None = None
    run_id: str | None = None
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class _RawLogRow(BaseModel):
    model_config = {"extra": "allow"}

    user_question: str = ""
    bot_response: str = ""
    agent_turns: list[_RawAgentTurn] = Field(default_factory=list)
    persona: str | None = None
    session_id: str | None = None
    success: bool | None = None
    task_hint: Literal["qa", "agentic"] | None = None


def _parse_turns(row: _RawLogRow) -> list[Turn]:
    turns: list[Turn] = []
    for idx, agent_turn in enumerate(row.agent_turns):
        tool_calls: list[ToolCall] = []
        tool_returns: list[ToolReturn] = []
        assistant_text: str | None = None

        for part in agent_turn.parts:
            kind = (part.part_kind or "").lower().replace("_", "-")
            if kind == "tool-call":
                tid = part.tool_call_id or f"call_{uuid.uuid4().hex[:12]}"
                tool_calls.append(
                    ToolCall(
                        tool_name=part.tool_name or "unknown_tool",
                        tool_call_id=tid,
                        args=dict(part.args) if part.args else {},
                        timestamp=part.timestamp,
                        raw_part=part.model_dump(exclude_none=True),
                    )
                )
            elif kind in ("tool-return", "tool_result", "tool-result"):
                tid = part.tool_call_id or ""
                tool_returns.append(
                    ToolReturn(
                        tool_name=part.tool_name or "unknown_tool",
                        tool_call_id=tid,
                        content=part.content or "",
                        timestamp=part.timestamp,
                        raw_part=part.model_dump(exclude_none=True),
                    )
                )
            elif kind in ("text", "assistant", "message"):
                assistant_text = (assistant_text or "") + (part.content or "")
            else:
                # Some stacks put plain text without part_kind
                if part.content and not part.tool_name:
                    assistant_text = (assistant_text or "") + part.content

        meta = CanonicalMetadata(
            run_id=agent_turn.run_id,
            model_name=agent_turn.model_name,
            provider_name=agent_turn.provider_name,
            provider_url=agent_turn.provider_url,
            provider_response_id=agent_turn.provider_response_id,
            finish_reason=agent_turn.finish_reason,
            usage=agent_turn.usage,
            extra={
                "kind": agent_turn.kind,
                "provider_details": agent_turn.provider_details,
                "metadata": agent_turn.metadata,
            },
        )

        turns.append(
            Turn(
                turn_index=idx,
                assistant_text=assistant_text,
                tool_calls=tool_calls,
                tool_returns=tool_returns,
                metadata=meta,
            )
        )
    return turns


def _infer_task_type(row: _RawLogRow, turns: list[Turn]) -> Literal["qa", "agentic"]:
    if row.task_hint:
        return row.task_hint
    for t in turns:
        if t.tool_calls or t.tool_returns:
            return "agentic"
    return "qa"


def _session_timestamp(turns: list[Turn]) -> datetime | None:
    for t in reversed(turns):
        if t.metadata and t.metadata.extra:
            pass
        for tr in t.tool_returns:
            if tr.timestamp:
                return tr.timestamp
        for tc in t.tool_calls:
            if tc.timestamp:
                return tc.timestamp
    return None


def parse_langfuse_log(raw: dict[str, Any] | list[Any]) -> list[Session]:
    """
    Normalize raw JSON (one row or batch) into canonical Session objects.
    """
    rows: list[dict[str, Any]]
    if isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
    else:
        rows = [raw]

    sessions: list[Session] = []
    for item in rows:
        row = _RawLogRow.model_validate(item)
        turns = _parse_turns(row)
        task_type = _infer_task_type(row, turns)
        success = row.success if row.success is not None else True
        sid = row.session_id or str(uuid.uuid4())

        meta = CanonicalMetadata()
        if turns and turns[-1].metadata:
            meta = turns[-1].metadata.model_copy()

        sessions.append(
            Session(
                session_id=sid,
                user_query=row.user_question,
                final_response=row.bot_response,
                persona=row.persona,
                task_type=task_type,
                turns=turns,
                success=success,
                timestamp=_session_timestamp(turns),
                metadata=meta,
            )
        )
    return sessions
