"""
Disjoint split assignment by **canonical prompt fingerprint**.

Guarantees a session's primary prompt key appears in **at most one** of:
`SFT_TRAIN`, `DPO_TRAIN`, or `EVAL_HOLDOUT` — preventing trivial leakage where the
same user intent is simultaneously optimized under SFT, contrasted under DPO,
and measured on eval.
"""

from __future__ import annotations

import hashlib
import unicodedata
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logs_to_training.schemas.canonical_event import Session


class DisjointSplit(str, Enum):
    SFT_TRAIN = "sft_train"
    DPO_TRAIN = "dpo_train"
    EVAL_HOLDOUT = "eval_holdout"


def normalize_prompt_key(text: str) -> str:
    """Unicode-normalize and collapse whitespace for stable dedup keys."""
    t = unicodedata.normalize("NFKC", text or "")
    return " ".join(t.split()).strip().lower()


def session_prompt_fingerprint(session: Session) -> str:
    """
    Fingerprint the *intent surface* shared across SFT/DPO/eval: persona + user query.

    Uses post-redaction session fields so PII placeholders line up with exports.
    """
    key = normalize_prompt_key((session.persona or "") + "\n" + session.user_query)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def assign_disjoint_split(fingerprint_hex: str, seed: int) -> DisjointSplit:
    """Deterministic bucket: one of three disjoint training/eval roles."""
    h = int(hashlib.sha256(f"{seed}:{fingerprint_hex}".encode()).hexdigest(), 16)
    bucket = h % 3
    if bucket == 0:
        return DisjointSplit.SFT_TRAIN
    if bucket == 1:
        return DisjointSplit.DPO_TRAIN
    return DisjointSplit.EVAL_HOLDOUT


def partition_sessions_disjoint(
    sessions: list[Session],
    seed: int,
) -> dict[DisjointSplit, list[Session]]:
    out: dict[DisjointSplit, list[Session]] = {
        DisjointSplit.SFT_TRAIN: [],
        DisjointSplit.DPO_TRAIN: [],
        DisjointSplit.EVAL_HOLDOUT: [],
    }
    for s in sessions:
        fp = session_prompt_fingerprint(s)
        split = assign_disjoint_split(fp, seed)
        s.metadata.extra["prompt_fingerprint"] = fp
        s.metadata.extra["disjoint_split"] = split.value
        out[split].append(s)
    return out
