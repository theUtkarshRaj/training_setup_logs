import json
from pathlib import Path

from logs_to_training.export.export_dpo import session_to_dpo_candidates
from logs_to_training.ingest.langfuse_parser import parse_langfuse_log
from logs_to_training.ingest.segmenter import segment_session
from logs_to_training.pipeline import PipelineConfig, process_sessions
from logs_to_training.splits.integrity import (
    assign_disjoint_split,
    partition_sessions_disjoint,
    session_prompt_fingerprint,
)
from logs_to_training.tagging.complexity import compute_complexity
from logs_to_training.validation.input_validate import validate_file


def test_partition_is_disjoint_and_exhaustive():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "logs_to_training" / "sample_data" / "sample_log.json").read_text(encoding="utf-8"))
    cfg = PipelineConfig()
    sessions = process_sessions(raw, cfg)
    parts = partition_sessions_disjoint(sessions, seed=7)
    total = sum(len(v) for v in parts.values())
    assert total == len(sessions)
    fps = [session_prompt_fingerprint(s) for s in sessions]
    for fp in fps:
        splits = {assign_disjoint_split(fp, 7)}
        assert len(splits) == 1


def test_same_prompt_surface_same_fingerprint():
    raw = [
        {"user_question": "Hello", "bot_response": "Hi", "persona": "P", "agent_turns": []},
        {"user_question": "Hello", "bot_response": "Different", "persona": "P", "agent_turns": []},
    ]
    sessions = parse_langfuse_log(raw)
    assert session_prompt_fingerprint(sessions[0]) == session_prompt_fingerprint(sessions[1])


def test_hard_negative_rows_present_for_agentic():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "logs_to_training" / "sample_data" / "sample_log.json").read_text(encoding="utf-8"))
    sessions = parse_langfuse_log(raw)
    s = next(x for x in sessions if x.session_id == "demo-agentic-001")
    segment_session(s)
    compute_complexity(s)
    rows = session_to_dpo_candidates(s)
    kinds = {r.metadata.get("hard_negative_kind") for r in rows if r.pair_type == "hard_negative"}
    assert "tool_omission" in kinds
    assert "hallucinated_gov_id" in kinds


def test_validate_raw_sample_passes():
    root = Path(__file__).resolve().parents[1]
    p = root / "logs_to_training" / "sample_data" / "sample_log.json"
    rep = validate_file(
        p,
        "raw",
        {"fetch_agristack_data", "weather_forecast"},
    )
    assert rep.ok and rep.failed == 0


def test_validate_sft_jsonl_roundtrip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    from logs_to_training.pipeline import iter_sft_jsonl, load_raw, process_sessions

    raw = load_raw(root / "logs_to_training" / "sample_data" / "sample_log.json")
    sessions = process_sessions(raw, PipelineConfig())
    out = tmp_path / "sft.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in iter_sft_jsonl(sessions):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rep = validate_file(out, "sft_jsonl", None)
    assert rep.ok
