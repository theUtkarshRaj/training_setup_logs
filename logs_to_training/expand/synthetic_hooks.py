"""
Differentiator: declarative mutation hooks for grounded synthetic expansion.

Mutations are **opt-in** and deterministic given a seed so runs are reproducible.
Categories align with agriculture/agent demos: location, crop, weather, tool failure,
ambiguity injection.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from logs_to_training.schemas.canonical_event import Session


@dataclass
class SyntheticExpansionConfig:
    """Toggle categories for future data synthesis from log seeds."""

    mutate_location: bool = False
    mutate_crop: bool = False
    mutate_weather: bool = False
    inject_tool_failure: bool = False
    inject_ambiguity: bool = False
    rng_seed: int = 42


def _rng_from_session(session: Session, base_seed: int) -> random.Random:
    h = hashlib.sha256(f"{session.session_id}:{base_seed}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _patch_args_and_text(
    session: Session,
    rng: random.Random,
    cfg: SyntheticExpansionConfig,
) -> Session:
    """Deep-copy session and apply shallow textual / arg tweaks."""
    s = session.model_copy(deep=True)

    if cfg.mutate_location:
        delta = rng.uniform(-0.02, 0.02)
        for t in s.turns:
            for tc in t.tool_calls:
                if "latitude" in tc.args and "longitude" in tc.args:
                    try:
                        lat = float(tc.args["latitude"]) + delta
                        lon = float(tc.args["longitude"]) - delta
                        tc.args["latitude"] = round(lat, 6)
                        tc.args["longitude"] = round(lon, 6)
                    except (TypeError, ValueError):
                        continue
            for tr in t.tool_returns:
                tr.content = tr.content.replace(
                    "Maharashtra",
                    rng.choice(["Karnataka", "Telangana", "Maharashtra"]),
                )

    if cfg.mutate_crop:
        for t in s.turns:
            for tr in t.tool_returns:
                tr.content = tr.content.replace("wheat", rng.choice(["wheat", "rice", "maize"]))

    if cfg.mutate_weather:
        for t in s.turns:
            for tr in t.tool_returns:
                if "weather" in tr.tool_name.lower():
                    tr.content += "\n[simulated: slight rain chance elevated for stress test]"

    if cfg.inject_tool_failure:
        for t in s.turns:
            for tr in t.tool_returns:
                if rng.random() < 0.25:
                    tr.content = (
                        '{"error":"tool_timeout","retryable":true,"detail":"upstream throttling"}'
                    )

    if cfg.inject_ambiguity:
        s.user_query += " " + rng.choice(
            [
                "It might be either last season or this one — not sure.",
                "Could you also clarify if this is for irrigated or rainfed plots?",
            ]
        )

    s.metadata.extra.setdefault("synthetic_mutations", [])
    s.metadata.extra["synthetic_mutations"] = list(
        {
            *s.metadata.extra.get("synthetic_mutations", []),
            *[k for k, v in cfg.__dict__.items() if k != "rng_seed" and v],
        }
    )
    return s


def apply_synthetic_mutations(session: Session, cfg: SyntheticExpansionConfig) -> Session:
    """
    Return a new Session with configured mutations applied.

    Safe default: if all flags false, returns a deep copy with only lineage metadata.
    """
    rng = _rng_from_session(session, cfg.rng_seed)
    if not any(
        getattr(cfg, name)
        for name in (
            "mutate_location",
            "mutate_crop",
            "mutate_weather",
            "inject_tool_failure",
            "inject_ambiguity",
        )
    ):
        s = session.model_copy(deep=True)
        s.metadata.extra.setdefault("synthetic_mutations", [])
        return s

    return _patch_args_and_text(session, rng, cfg)


def dump_session_fingerprint(session: Session) -> str:
    """Stable fingerprint for dedup / split integrity across augments."""
    payload = session.model_dump(mode="json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
