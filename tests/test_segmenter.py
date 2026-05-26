from logs_to_training.ingest.segmenter import SegmentLabel, segment_session
from logs_to_training.schemas.canonical_event import Session, ToolCall, Turn


def test_failed_trajectory_when_success_false():
    s = Session(session_id="t1", user_query="q", final_response="a", success=False)
    assert segment_session(s) == SegmentLabel.FAILED_TRAJECTORY


def test_failed_trajectory_when_turn_has_error():
    s = Session(
        session_id="t2",
        user_query="q",
        final_response="a",
        success=True,
        turns=[Turn(turn_index=0, error="timeout")],
    )
    assert segment_session(s) == SegmentLabel.FAILED_TRAJECTORY


def test_agentic_label_requires_tool_calls():
    s_agentic = Session(
        session_id="t3",
        user_query="q",
        final_response="a",
        turns=[
            Turn(
                turn_index=0,
                tool_calls=[
                    ToolCall(tool_name="weather_forecast", tool_call_id="c1", args={}, timestamp=None)
                ],
            )
        ],
    )
    assert segment_session(s_agentic) == SegmentLabel.AGENTIC_TRAJECTORY

    s_qa = Session(
        session_id="t4",
        user_query="q",
        final_response="a",
        turns=[Turn(turn_index=0, assistant_text="hello")],
    )
    assert segment_session(s_qa) == SegmentLabel.SINGLE_TURN_QA
