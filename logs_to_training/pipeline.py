"""Thin orchestration layer for CLI and services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from logs_to_training.expand.synthetic_hooks import SyntheticExpansionConfig, apply_synthetic_mutations
from logs_to_training.export.export_dpo import session_to_dpo_candidates
from logs_to_training.export.export_sft import session_to_sft_row, sft_row_to_jsonl_dict
from logs_to_training.ingest.langfuse_parser import parse_langfuse_log
from logs_to_training.ingest.segmenter import segment_session
from logs_to_training.pii.redact import redact_session_inplace
from logs_to_training.schemas.canonical_event import Session
from logs_to_training.schemas.dpo_schema import DPOCandidateRow
from logs_to_training.tagging.complexity import compute_complexity
from logs_to_training.validation.tool_consistency import validate_tool_consistency


@dataclass
class PipelineConfig:
    """Runtime toggles for prototype runs."""

    known_tools: set[str] | None = None
    skip_failed_validation: bool = True
    synthetic: SyntheticExpansionConfig | None = None


def load_raw(path: Path) -> dict[str, Any] | list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def process_sessions(raw: dict[str, Any] | list[Any], cfg: PipelineConfig) -> list[Session]:
    sessions = parse_langfuse_log(raw)
    out: list[Session] = []
    for s in sessions:
        segment_session(s)
        compute_complexity(s)
        if cfg.synthetic:
            s = apply_synthetic_mutations(s, cfg.synthetic)
            segment_session(s)
            compute_complexity(s)
        redact_session_inplace(s)
        report = validate_tool_consistency(s, known_tools=cfg.known_tools)
        if cfg.skip_failed_validation and not report.ok:
            s.metadata.extra["tool_validation_failed"] = True
            s.metadata.extra["tool_validation_issues"] = report.issues
            s.metadata.extra["tool_contradictions"] = report.contradictory_outputs
        out.append(s)
    return out


def iter_sft_jsonl(sessions: list[Session]) -> Iterator[dict[str, Any]]:
    for s in sessions:
        if s.metadata.extra.get("tool_validation_failed"):
            continue
        yield sft_row_to_jsonl_dict(session_to_sft_row(s))


def iter_dpo_jsonl(sessions: list[Session]) -> Iterator[dict[str, Any]]:
    for s in sessions:
        if s.metadata.extra.get("tool_validation_failed"):
            continue
        for row in session_to_dpo_candidates(s):
            d: DPOCandidateRow = row
            yield d.model_dump(exclude_none=True)
