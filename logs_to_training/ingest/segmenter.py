"""Classify sessions into QA vs agentic trajectories and failure modes."""

from __future__ import annotations

from enum import Enum

from logs_to_training.schemas.canonical_event import Session


class SegmentLabel(str, Enum):
    SINGLE_TURN_QA = "single_turn_qa"
    MULTI_TURN_QA = "multi_turn_qa"
    AGENTIC_TRAJECTORY = "agentic_trajectory"
    FAILED_TRAJECTORY = "failed_trajectory"


def segment_session(session: Session) -> SegmentLabel:
    """
    Heuristic segmentation:

    - failed_trajectory: explicit session.success is False or any turn error.
    - agentic_trajectory: any tool call/return present.
    - multi_turn_qa: multiple assistant steps without tools.
    - single_turn_qa: one effective assistant step, no tools.
    """
    if not session.success:
        session.segmentation_label = SegmentLabel.FAILED_TRAJECTORY.value
        return SegmentLabel.FAILED_TRAJECTORY

    for t in session.turns:
        if t.error:
            session.segmentation_label = SegmentLabel.FAILED_TRAJECTORY.value
            return SegmentLabel.FAILED_TRAJECTORY

    has_tools = any(t.tool_calls or t.tool_returns for t in session.turns)
    if has_tools or session.task_type == "agentic":
        session.segmentation_label = SegmentLabel.AGENTIC_TRAJECTORY.value
        return SegmentLabel.AGENTIC_TRAJECTORY

    assistant_steps = sum(1 for t in session.turns if (t.assistant_text or "").strip())
    if assistant_steps > 1 or len(session.turns) > 1:
        session.segmentation_label = SegmentLabel.MULTI_TURN_QA.value
        return SegmentLabel.MULTI_TURN_QA

    session.segmentation_label = SegmentLabel.SINGLE_TURN_QA.value
    return SegmentLabel.SINGLE_TURN_QA
