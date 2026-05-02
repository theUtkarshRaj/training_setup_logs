"""Validate raw logs or exported JSONL against Pydantic schemas and tool rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from logs_to_training.ingest.langfuse_parser import parse_langfuse_log
from logs_to_training.ingest.segmenter import segment_session
from logs_to_training.pii.redact import redact_session_inplace
from logs_to_training.schemas.canonical_event import Session
from logs_to_training.schemas.dpo_schema import DPOCandidateRow
from logs_to_training.schemas.sft_schema import SFTRow
from logs_to_training.tagging.complexity import compute_complexity
from logs_to_training.validation.tool_consistency import validate_tool_consistency


ValidateKind = Literal["raw", "sft_jsonl", "dpo_jsonl"]


@dataclass
class ValidationReport:
    ok: bool
    checked: int = 0
    passed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _prepare_session_like_pipeline(
    raw: dict[str, Any] | list[Any],
    known_tools: set[str] | None,
) -> list[Session]:
    sessions = parse_langfuse_log(raw)
    for s in sessions:
        segment_session(s)
        compute_complexity(s)
        redact_session_inplace(s)
        report = validate_tool_consistency(s, known_tools=known_tools)
        if not report.ok:
            s.metadata.extra["tool_validation_failed"] = True
            s.metadata.extra["tool_validation_issues"] = report.issues
            s.metadata.extra["tool_contradictions"] = report.contradictory_outputs
    return sessions


def validate_raw_file(path: Path, known_tools: set[str] | None) -> ValidationReport:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rep = ValidationReport(ok=True)
    try:
        sessions = _prepare_session_like_pipeline(raw, known_tools)
    except Exception as e:  # noqa: BLE001 — surface parse errors to operator
        rep.ok = False
        rep.errors.append(f"parse_error: {e}")
        return rep

    for s in sessions:
        rep.checked += 1
        issues: list[str] = []
        if s.metadata.extra.get("tool_validation_failed"):
            issues.extend(s.metadata.extra.get("tool_validation_issues") or [])
            issues.extend(s.metadata.extra.get("tool_contradictions") or [])
        if issues:
            rep.failed += 1
            rep.errors.append(f"{s.session_id}: " + "; ".join(issues))
        else:
            rep.passed += 1

    rep.ok = rep.failed == 0
    return rep


def validate_sft_jsonl(path: Path) -> ValidationReport:
    rep = ValidationReport(ok=True)
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rep.checked += 1
        try:
            SFTRow.model_validate_json(line)
            rep.passed += 1
        except Exception as e:  # noqa: BLE001
            rep.failed += 1
            rep.errors.append(f"line {lineno}: {e}")
    rep.ok = rep.failed == 0
    return rep


def validate_dpo_jsonl(path: Path) -> ValidationReport:
    rep = ValidationReport(ok=True)
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rep.checked += 1
        try:
            DPOCandidateRow.model_validate_json(line)
            rep.passed += 1
        except Exception as e:  # noqa: BLE001
            rep.failed += 1
            rep.errors.append(f"line {lineno}: {e}")
    rep.ok = rep.failed == 0
    return rep


def validate_file(path: Path, kind: ValidateKind, known_tools: set[str] | None) -> ValidationReport:
    if kind == "raw":
        return validate_raw_file(path, known_tools)
    if kind == "sft_jsonl":
        return validate_sft_jsonl(path)
    if kind == "dpo_jsonl":
        return validate_dpo_jsonl(path)
    raise ValueError(f"unknown kind: {kind}")
