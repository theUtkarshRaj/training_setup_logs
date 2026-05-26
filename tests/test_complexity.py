from logs_to_training.schemas.canonical_event import Session, ToolCall, ToolReturn, Turn
from logs_to_training.tagging.complexity import _retry_like, compute_complexity


def test_retry_count_only_adjacent_duplicates():
    # Only the 0→1 pair ("a"→"a") is adjacent; the 1→3 repetition is not
    assert _retry_like(["a", "a", "b", "a"]) == 1
    # No adjacent duplicates
    assert _retry_like(["a", "b", "a"]) == 0


def test_tier_is_high_when_tool_return_signals_failure():
    s = Session(
        session_id="t5",
        user_query="q",
        final_response="a",
        turns=[
            Turn(
                turn_index=0,
                tool_calls=[
                    ToolCall(tool_name="weather_forecast", tool_call_id="c1", args={}, timestamp=None)
                ],
                tool_returns=[
                    ToolReturn(
                        tool_name="weather_forecast",
                        tool_call_id="c1",
                        content="rate limit exceeded, please retry",
                        timestamp=None,
                    )
                ],
            )
        ],
    )
    tags = compute_complexity(s)
    assert tags.recovery_detected is True
    assert tags.complexity_tier == "high"
