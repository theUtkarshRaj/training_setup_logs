"""CLI entrypoint for ingest → transform → export → validate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from logs_to_training.expand.synthetic_hooks import SyntheticExpansionConfig
from logs_to_training.pipeline import PipelineConfig, iter_dpo_jsonl, iter_sft_jsonl, load_raw, process_sessions
from logs_to_training.splits.integrity import DisjointSplit, partition_sessions_disjoint
from logs_to_training.validation.input_validate import validate_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="logs2train",
        description="Logs-to-training: Langfuse-style JSON → PII-safe SFT/DPO JSONL",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="End-to-end: ingest JSON, emit SFT and/or DPO JSONL")
    run.add_argument("--input", type=Path, required=True, help="Path to JSON or JSONL array file")
    run.add_argument("--sft-out", type=Path, default=None, help="Output SFT JSONL path")
    run.add_argument("--dpo-out", type=Path, default=None, help="Output DPO candidates JSONL path")
    run.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Write disjoint splits: sft_train.jsonl, dpo_train.jsonl, eval_holdout.jsonl + manifest",
    )
    run.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for disjoint fingerprint→split assignment (reproducible splits)",
    )
    run.add_argument(
        "--known-tools",
        type=str,
        default="",
        help="Comma-separated allowlist; empty disables strict tool-name checks",
    )
    run.add_argument(
        "--include-failed-validation",
        action="store_true",
        help="Include sessions that fail tool consistency checks (not recommended)",
    )
    run.add_argument("--synth-location", action="store_true")
    run.add_argument("--synth-crop", action="store_true")
    run.add_argument("--synth-weather", action="store_true")
    run.add_argument("--synth-tool-failure", action="store_true")
    run.add_argument("--synth-ambiguity", action="store_true")
    run.add_argument("--rng-seed", type=int, default=42)

    val = sub.add_parser("validate", help="Validate raw logs or exported JSONL rows against schemas")
    val.add_argument("--input", type=Path, required=True)
    val.add_argument(
        "--kind",
        choices=["raw", "sft_jsonl", "dpo_jsonl"],
        default="raw",
        help="raw: Langfuse-style JSON array; *_jsonl: one Pydantic row per line",
    )
    val.add_argument(
        "--known-tools",
        type=str,
        default="",
        help="Comma-separated tool allowlist for raw validation (optional)",
    )
    return p


def _cmd_validate(args: argparse.Namespace) -> int:
    known = {t.strip() for t in args.known_tools.split(",") if t.strip()} or None
    rep = validate_file(args.input, args.kind, known)
    print(f"checked={rep.checked} passed={rep.passed} failed={rep.failed}")
    for err in rep.errors[:50]:
        print(err, file=sys.stderr)
    if len(rep.errors) > 50:
        print(f"... {len(rep.errors) - 50} more errors truncated", file=sys.stderr)
    return 0 if rep.ok else 1


def _write_disjoint_splits(split_dir: Path, sessions: list[object], split_seed: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    parts = partition_sessions_disjoint(sessions, split_seed)
    names = {
        DisjointSplit.SFT_TRAIN: "sft_train.jsonl",
        DisjointSplit.DPO_TRAIN: "dpo_train.jsonl",
        DisjointSplit.EVAL_HOLDOUT: "eval_holdout.jsonl",
    }
    manifest: dict = {"split_seed": split_seed, "files": {}, "prompt_fingerprint_disjoint": True}
    for split, fname in names.items():
        plist = parts[split]
        manifest["files"][fname] = {"sessions": len(plist)}
        out_path = split_dir / fname
        with out_path.open("w", encoding="utf-8") as f:
            if split == DisjointSplit.DPO_TRAIN:
                for row in iter_dpo_jsonl(plist):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                for row in iter_sft_jsonl(plist):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (split_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    raw = load_raw(args.input)
    known = {t.strip() for t in args.known_tools.split(",") if t.strip()} or None
    use_synth = any(
        (
            args.synth_location,
            args.synth_crop,
            args.synth_weather,
            args.synth_tool_failure,
            args.synth_ambiguity,
        )
    )
    synth_cfg = (
        SyntheticExpansionConfig(
            mutate_location=args.synth_location,
            mutate_crop=args.synth_crop,
            mutate_weather=args.synth_weather,
            inject_tool_failure=args.synth_tool_failure,
            inject_ambiguity=args.synth_ambiguity,
            rng_seed=args.rng_seed,
        )
        if use_synth
        else None
    )
    cfg = PipelineConfig(
        known_tools=known,
        skip_failed_validation=not args.include_failed_validation,
        synthetic=synth_cfg,
    )

    sessions = process_sessions(raw, cfg)

    if args.split_dir:
        if args.sft_out or args.dpo_out:
            print(
                "Note: --split-dir takes precedence; ignoring --sft-out/--dpo-out for this run.",
                file=sys.stderr,
            )
        _write_disjoint_splits(args.split_dir, sessions, args.split_seed)
        return 0

    wrote = False
    if args.sft_out:
        wrote = True
        args.sft_out.parent.mkdir(parents=True, exist_ok=True)
        with args.sft_out.open("w", encoding="utf-8") as f:
            for row in iter_sft_jsonl(sessions):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.dpo_out:
        wrote = True
        args.dpo_out.parent.mkdir(parents=True, exist_ok=True)
        with args.dpo_out.open("w", encoding="utf-8") as f:
            for row in iter_dpo_jsonl(sessions):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not wrote:
        print("No outputs requested; parsed sessions:", len(sessions), file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "validate":
        return _cmd_validate(args)
    if args.cmd == "run":
        return _cmd_run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
