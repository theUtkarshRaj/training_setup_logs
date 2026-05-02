from logs_to_training.schemas.canonical_event import Session, ToolCall, ToolReturn, Turn
from logs_to_training.validation.tool_consistency import validate_tool_consistency


def test_matching_call_return():
    s = Session(
        session_id="t1",
        user_query="q",
        final_response="a",
        turns=[
            Turn(
                turn_index=0,
                tool_calls=[
                    ToolCall(
                        tool_name="weather_forecast",
                        tool_call_id="id1",
                        args={"latitude": 1.0, "longitude": 2.0},
                        timestamp=None,
                    )
                ],
                tool_returns=[
                    ToolReturn(
                        tool_name="weather_forecast",
                        tool_call_id="id1",
                        content="ok",
                        timestamp=None,
                    )
                ],
            )
        ],
    )
    r = validate_tool_consistency(s)
    assert r.ok


def test_unknown_tool_with_registry():
    s = Session(
        session_id="t2",
        user_query="q",
        final_response="a",
        turns=[
            Turn(
                turn_index=0,
                tool_calls=[
                    ToolCall(tool_name="bad_tool", tool_call_id="x", args={}, timestamp=None)
                ],
                tool_returns=[],
            )
        ],
    )
    r = validate_tool_consistency(s, known_tools={"weather_forecast"})
    assert not r.ok
