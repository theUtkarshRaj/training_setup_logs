import json
from pathlib import Path

from logs_to_training.export.export_dpo import session_to_dpo_candidates
from logs_to_training.ingest.langfuse_parser import parse_langfuse_log
from logs_to_training.ingest.segmenter import segment_session
from logs_to_training.schemas.canonical_event import Session, Turn
from logs_to_training.tagging.complexity import compute_complexity


def test_no_tool_path_pair_for_qa_session():
    s = Session(
        session_id="qa1",
        user_query="What is crop rotation?",
        final_response="Crop rotation is the practice of growing different crops sequentially.",
        task_type="qa",
        turns=[Turn(turn_index=0, assistant_text="Crop rotation is the practice...")],
    )
    compute_complexity(s)
    rows = session_to_dpo_candidates(s)
    assert all(r.pair_type != "tool_path" for r in rows)


def test_chosen_never_equals_rejected():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (root / "logs_to_training" / "sample_data" / "sample_log.json").read_text(encoding="utf-8")
    )
    sessions = parse_langfuse_log(raw)
    agentic = next(s for s in sessions if s.session_id == "demo-agentic-001")
    segment_session(agentic)
    compute_complexity(agentic)
    rows = session_to_dpo_candidates(agentic)
    assert rows, "expected at least one DPO row"
    for row in rows:
        assert row.chosen != row.rejected, f"chosen == rejected for pair_type={row.pair_type!r}"
