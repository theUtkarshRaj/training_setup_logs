"""
Validate tool call / return pairing and basic schema hygiene.

Flags contradictory observations when the same entity is asserted differently
across tool returns (lightweight numeric / coordinate heuristic).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from logs_to_training.schemas.canonical_event import Session, Turn


@dataclass
class ToolConsistencyReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    contradictory_outputs: list[str] = field(default_factory=list)


def _allowed_tool_name(name: str, registry: set[str] | None) -> bool:
    if not registry:
        return bool(name and name.replace("_", "").isalnum())
    return name in registry


def _args_basic_check(tool_name: str, args: dict) -> list[str]:
    problems: list[str] = []
    if not isinstance(args, dict):
        problems.append(f"{tool_name}: args must be object/dict")
        return problems
    for k, v in args.items():
        if not isinstance(k, str):
            problems.append(f"{tool_name}: non-string key in args")
        if isinstance(v, dict) and len(str(v)) > 8000:
            problems.append(f"{tool_name}: oversized nested arg {k}")
    return problems


_FLOATS = re.compile(r"-?\d+\.\d+")


def _extract_floats(text: str) -> set[tuple[int, float]]:
    out: set[tuple[int, float]] = set()
    for m in _FLOATS.finditer(text):
        try:
            out.add((m.start(), float(m.group(0))))
        except ValueError:
            continue
    return out


def _contradictions_across_returns(turns: list[Turn]) -> list[str]:
    """Flag if disjoint sets of coordinates appear across returns (possible conflict)."""
    per_tool: dict[str, list[set[tuple[int, float]]]] = {}
    for t in turns:
        for tr in t.tool_returns:
            floats = _extract_floats(tr.content)
            if len(floats) < 2:
                continue
            per_tool.setdefault(tr.tool_name, []).append(floats)

    flags: list[str] = []
    for tool, groups in per_tool.items():
        if len(groups) < 2:
            continue
        # If no intersection of rounded coords between consecutive returns, note it
        for a, b in zip(groups, groups[1:], strict=False):
            ra = {round(x[1], 3) for x in a}
            rb = {round(x[1], 3) for x in b}
            if ra and rb and ra.isdisjoint(rb):
                flags.append(
                    f"Possible contradictory numeric fields between "
                    f"consecutive {tool} returns (check lat/lon consistency)."
                )
    return flags


def validate_tool_consistency(
    session: Session,
    known_tools: set[str] | None = None,
) -> ToolConsistencyReport:
    issues: list[str] = []
    pending: dict[str, str] = {}  # tool_call_id -> tool_name

    for t in session.turns:
        for tc in t.tool_calls:
            if not tc.tool_call_id:
                issues.append(f"Tool call missing id for {tc.tool_name}")
            if not _allowed_tool_name(tc.tool_name, known_tools):
                issues.append(f"Unknown or invalid tool name: {tc.tool_name}")
            issues.extend(_args_basic_check(tc.tool_name, tc.args))
            if tc.tool_call_id:
                pending[tc.tool_call_id] = tc.tool_name

        returned_ids: list[str] = []
        for tr in t.tool_returns:
            if tr.tool_call_id:
                if tr.tool_call_id not in pending:
                    issues.append(
                        f"Tool return for id {tr.tool_call_id} without preceding call in session"
                    )
                else:
                    expected = pending.get(tr.tool_call_id)
                    if expected and expected != tr.tool_name:
                        issues.append(
                            f"Tool name mismatch for {tr.tool_call_id}: "
                            f"call {expected} vs return {tr.tool_name}"
                        )
                    returned_ids.append(tr.tool_call_id)
            elif len(t.tool_calls) == 1 and len(t.tool_returns) == 1:
                only = t.tool_calls[0]
                if only.tool_name != tr.tool_name:
                    issues.append(
                        f"Implicit tool return name mismatch: call {only.tool_name} vs return {tr.tool_name}"
                    )
                if only.tool_call_id:
                    returned_ids.append(only.tool_call_id)
            elif not tr.tool_call_id:
                issues.append(
                    f"Tool return for {tr.tool_name} missing tool_call_id (ambiguous pairing)"
                )

        for rid in returned_ids:
            pending.pop(rid, None)

    if pending:
        issues.append(f"Unmatched tool calls (no return): {sorted(pending.keys())}")

    contradictions = _contradictions_across_returns(session.turns)
    ok = not issues and not contradictions
    return ToolConsistencyReport(
        ok=ok,
        issues=issues,
        contradictory_outputs=contradictions,
    )
