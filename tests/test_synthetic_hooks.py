from logs_to_training.expand.synthetic_hooks import SyntheticExpansionConfig, apply_synthetic_mutations
from logs_to_training.schemas.canonical_event import Session, ToolCall, Turn


def _session_with_coords(session_id: str = "syn1") -> Session:
    return Session(
        session_id=session_id,
        user_query="q",
        final_response="a",
        task_type="agentic",
        turns=[
            Turn(
                turn_index=0,
                tool_calls=[
                    ToolCall(
                        tool_name="weather_forecast",
                        tool_call_id="c1",
                        args={"latitude": 19.07, "longitude": 77.17},
                        timestamp=None,
                    )
                ],
            )
        ],
    )


def test_mutations_do_not_modify_original():
    s = _session_with_coords()
    cfg = SyntheticExpansionConfig(mutate_location=True, rng_seed=42)
    apply_synthetic_mutations(s, cfg)
    assert s.turns[0].tool_calls[0].args["latitude"] == 19.07
    assert s.turns[0].tool_calls[0].args["longitude"] == 77.17


def test_same_seed_produces_identical_mutations():
    s = _session_with_coords()
    cfg = SyntheticExpansionConfig(mutate_location=True, mutate_crop=True, rng_seed=99)
    out1 = apply_synthetic_mutations(s, cfg)
    out2 = apply_synthetic_mutations(s, cfg)
    assert out1.model_dump() == out2.model_dump()
