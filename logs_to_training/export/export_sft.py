"""
Export canonical sessions to LoRA-ready JSONL rows (OpenAI-style messages).

Order per trajectory: system → user → (assistant [+tool_calls] → tool)* → assistant final.
"""

from __future__ import annotations

import json
from typing import Any

from logs_to_training.schemas.canonical_event import Session
from logs_to_training.schemas.sft_schema import ChatMessage, SFTRow, ToolCallBlock
from logs_to_training.splits.integrity import session_prompt_fingerprint


def session_to_sft_row(session: Session) -> SFTRow:
    messages: list[ChatMessage] = []

    if session.persona:
        messages.append(ChatMessage(role="system", content=session.persona))

    messages.append(ChatMessage(role="user", content=session.user_query))

    i = 0
    turns = session.turns
    while i < len(turns):
        turn = turns[i]
        if turn.tool_calls:
            blocks: list[ToolCallBlock] = []
            for tc in turn.tool_calls:
                blocks.append(
                    ToolCallBlock(
                        id=tc.tool_call_id,
                        function={
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.args, ensure_ascii=False),
                        },
                    )
                )
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=turn.assistant_text or None,
                    tool_calls=blocks,
                )
            )
            for tr in turn.tool_returns:
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=tr.tool_call_id or tr.tool_name,
                        name=tr.tool_name,
                        content=tr.content,
                    )
                )
            i += 1
            while i < len(turns) and not turns[i].tool_calls and turns[i].tool_returns:
                for tr in turns[i].tool_returns:
                    messages.append(
                        ChatMessage(
                            role="tool",
                            tool_call_id=tr.tool_call_id or tr.tool_name,
                            name=tr.tool_name,
                            content=tr.content,
                        )
                    )
                i += 1
            continue
        if (turn.assistant_text or "").strip():
            messages.append(ChatMessage(role="assistant", content=turn.assistant_text))
        i += 1

    messages.append(ChatMessage(role="assistant", content=session.final_response))

    fp = session.metadata.extra.get("prompt_fingerprint") or session_prompt_fingerprint(session)
    meta: dict[str, Any] = {
        "session_id": session.session_id,
        "prompt_fingerprint": fp,
        "task_type": session.task_type,
        "success": session.success,
        "segmentation": session.segmentation_label,
    }
    if session.complexity:
        meta["complexity"] = session.complexity.model_dump(exclude_none=True)
    if ds := session.metadata.extra.get("disjoint_split"):
        meta["disjoint_split"] = ds

    return SFTRow(messages=messages, metadata=meta)


def sft_row_to_jsonl_dict(row: SFTRow) -> dict[str, Any]:
    """Serialize for JSONL writing."""
    return {
        "messages": [m.model_dump(exclude_none=True) for m in row.messages],
        "metadata": row.metadata,
    }
