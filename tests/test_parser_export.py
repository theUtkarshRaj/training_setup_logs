import json
from pathlib import Path

from logs_to_training.export.export_sft import session_to_sft_row
from logs_to_training.ingest.langfuse_parser import parse_langfuse_log
from logs_to_training.ingest.segmenter import SegmentLabel, segment_session
from logs_to_training.pii.redact import redact_session_inplace
from logs_to_training.tagging.complexity import compute_complexity


def test_end_to_end_sample_file():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "logs_to_training" / "sample_data" / "sample_log.json").read_text(encoding="utf-8"))
    sessions = parse_langfuse_log(raw)
    assert len(sessions) == 2
    agentic = next(s for s in sessions if s.session_id == "demo-agentic-001")
    assert segment_session(agentic) == SegmentLabel.AGENTIC_TRAJECTORY
    compute_complexity(agentic)
    redact_session_inplace(agentic)
    assert "+91" not in agentic.user_query
    row = session_to_sft_row(agentic)
    roles = [m.role for m in row.messages]
    assert roles[0] == "system"
    assert "tool" in roles
    assert roles[-1] == "assistant"
