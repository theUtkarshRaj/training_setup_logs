from logs_to_training.export.hard_negatives import build_hard_negative_rows, rejected_slight_factual_drift
from logs_to_training.ingest.segmenter import segment_session
from logs_to_training.schemas.canonical_event import Session, ToolCall, Turn
from logs_to_training.tagging.complexity import compute_complexity


def _agentic_session(session_id: str = "hn1") -> Session:
    return Session(
        session_id=session_id,
        user_query="Give a 3-day forecast.",
        final_response="The outlook is clear with highs around 31°C.",
        task_type="agentic",
        turns=[
            Turn(
                turn_index=0,
                tool_calls=[
                    ToolCall(tool_name="weather_forecast", tool_call_id="c1", args={}, timestamp=None)
                ],
            )
        ],
    )


def test_hallucinated_gov_id_in_rejected_not_chosen():
    s = _agentic_session()
    segment_session(s)
    compute_complexity(s)
    rows = build_hard_negative_rows(s)
    gov_rows = [r for r in rows if r.metadata.get("hard_negative_kind") == "hallucinated_gov_id"]
    assert len(gov_rows) == 1
    row = gov_rows[0]
    assert "1234 5678 9012" in row.rejected
    assert "1234 5678 9012" not in row.chosen


def test_factual_drift_fallback_modifies_response():
    s = Session(
        session_id="hn2",
        user_query="q",
        # No coordinate or temperature patterns — forces the fallback branch
        final_response="The weather is pleasant today.",
    )
    result = rejected_slight_factual_drift(s)
    assert result != s.final_response
