# Logs-to-training (OpenAgriNet / DMP 2026 prototype)

Python-first pipeline that ingests **Langfuse / Pydantic-style JSON traces** (user question, bot response, `agent_turns` with `tool-call` / `tool-return` parts), strips **PII and secrets**, segments **Q&A vs agentic trajectories**, tags **compositional complexity**, validates **tool consistency**, and exports **LoRA-ready SFT JSONL** plus **DPO candidate** rows for later preference training.

Design stance: **data quality over training tricks**, **LoRA SFT first**, **DPO later**, **≤ ~50k curated rows**, logs as **seeds** (plus optional synthetic expansion hooks), **multi-turn tool trajectories** preserved.

---

## Problem statement

Production agent stacks emit rich JSON: personas, multi-step tool use, recoveries, and noisy user content (phones, government IDs). Training smaller dense models (≤ ~32B) to hit latency/tool budgets requires:

1. A **canonical schema** that is stable for exporters and validators.
2. **Deterministic PII handling** with placeholders and audit-friendly patterns.
3. **Trajectory-aware** SFT rows (system → user → assistant → tool → … → final assistant).
4. **Stratification metadata** (complexity tier, segmentation) for staged mixtures / curriculum.
5. **Guardrails** that flag broken tool graphs before data enters trainers.

This package is a **minimal working spine** for that story—not a full hosted service.

---

## Architecture

```text
Raw JSON (Langfuse-style)
    → langfuse_parser (canonical Session)
    → segmenter (QA vs agentic vs failed)
    → complexity (tiers + signals)
    → [optional] expand.synthetic_hooks (seed mutations)
    → pii.redact (Presidio if installed, else regex)
    → validation.tool_consistency
    → export_sft / export_dpo (JSONL)
    → [optional] splits/integrity (disjoint SFT / DPO / eval by prompt fingerprint)
```

| Module | Role |
|--------|------|
| `schemas/canonical_event.py` | `Session`, `Turn`, `ToolCall`, `ToolReturn`, `ComplexityTags` |
| `schemas/sft_schema.py` | `SFTRow` (`messages[...]`) |
| `schemas/dpo_schema.py` | `DPOCandidateRow` (`prompt`, `chosen`, `rejected`) |
| `ingest/langfuse_parser.py` | Normalize heterogeneous JSON into `Session` |
| `ingest/segmenter.py` | `single_turn_qa`, `multi_turn_qa`, `agentic_trajectory`, `failed_trajectory` |
| `tagging/complexity.py` | Tool counts, retries proxy, recovery heuristics, ambiguity, tier |
| `validation/tool_consistency.py` | Pairing, allowlist, arg sanity, coarse contradiction hints |
| `validation/input_validate.py` | CLI-backed validation for raw logs or exported JSONL |
| `pii/redact.py` | Presidio + regex fallbacks, session-stable placeholders |
| `export/export_sft.py` | OpenAI-style chat JSONL |
| `export/export_dpo.py` | Rule-based **candidates** + **hard-negative templates** |
| `export/hard_negatives.py` | Curated rejected completions (tool omission, extra tools, leakage, IDs, drift) |
| `splits/integrity.py` | `prompt_fingerprint` + disjoint `sft_train` / `dpo_train` / `eval_holdout` |
| `expand/synthetic_hooks.py` | **Differentiator:** opt-in mutations (location, crop, weather, failure, ambiguity) |

---

## Pipeline flow (CLI)

Install (dev):

```bash
cd training_setup_logs-main
python -m pip install -e ".[dev]"
```

Run end-to-end on bundled sample:

```bash
python -m logs_to_training.cli run \
  --input logs_to_training/sample_data/sample_log.json \
  --sft-out out/sft.jsonl \
  --dpo-out out/dpo_candidates.jsonl \
  --known-tools fetch_agristack_data,weather_forecast
```

Optional synthetic expansion (deterministic per session + seed):

```bash
python -m logs_to_training.cli run \
  --input logs_to_training/sample_data/sample_log.json \
  --sft-out out/sft_aug.jsonl \
  --synth-location --synth-ambiguity --rng-seed 7
```

### Disjoint splits (SFT vs DPO vs eval integrity)

The same **normalized persona + user query** surface gets a stable `prompt_fingerprint`. `logs2train run --split-dir …` assigns each session to **exactly one** of:

- `sft_train.jsonl` — SFT-only rows for that intent surface  
- `dpo_train.jsonl` — DPO-only rows (including hard negatives)  
- `eval_holdout.jsonl` — SFT-shaped eval rows (no overlap with the other two for that fingerprint)

This prevents trivial **cross-split leakage** where one prompt is simultaneously optimized under SFT, contrasted under DPO, and measured on eval.

```bash
python -m logs_to_training.cli run \
  --input logs_to_training/sample_data/sample_log.json \
  --split-dir out/splits \
  --split-seed 42 \
  --known-tools fetch_agristack_data,weather_forecast
```

### Schema validator (`validate`)

```bash
python -m logs_to_training.cli validate \
  --input logs_to_training/sample_data/sample_log.json \
  --kind raw \
  --known-tools fetch_agristack_data,weather_forecast

python -m logs_to_training.cli validate --input out/sft.jsonl --kind sft_jsonl
```

Exit code **1** if any record fails. Use this in CI before merging datasets or running trainer dry-runs.

### Hard-negative templates (DPO readiness)

Beyond the legacy “lazy tool use” string, `export_dpo.py` appends rows with `pair_type="hard_negative"` and `metadata.hard_negative_kind` in:

`tool_omission` · `extra_tool_call` · `persona_leakage` · `hallucinated_gov_id` · `slight_factual_drift`

Each `rejected` is a **named defect template** for reviewer scoring or semi-automatic filtering—not blind training without QA.

### Gold examples (reviewer fixtures)

See repository `gold_examples/` for **ideal agentic**, **inefficient tool use**, and **persona violation** JSON snippets you can validate or export in one command.

---

## Example I/O

**Input** (excerpt): see `sample_data/sample_log.json` — `user_question`, `agent_turns[].parts[]` with `part_kind`: `tool-call` | `tool-return`, `bot_response`, optional `persona`.

**SFT JSONL row** (shape):

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."},
    {"role": "assistant", "content": "final answer"}
  ],
  "metadata": {
    "session_id": "...",
    "prompt_fingerprint": "sha256…",
    "complexity": {"complexity_tier": "medium", "tool_count": 2, "...": "..."}
  }
}
```

**DPO JSONL candidate** (shape):

```json
{
  "prompt": "persona + user ...",
  "chosen": "grounded assistant completion",
  "rejected": "template or proxy bad completion",
  "pair_type": "hard_negative",
  "metadata": {"session_id": "...", "hard_negative_kind": "tool_omission", "review_gate": "hard_negative_template"}
}
```

Rejected strings for `tool_path` are **intentionally synthetic** placeholders so the file reads as *candidate pairs for review*, not production-ready preferences.

---

## Alignment with DMP / issue goals

| Issue theme | This prototype |
|-------------|----------------|
| Langfuse + Pydantic JSON | `langfuse_parser.py` + sample log |
| PII + audit mindset | Presidio hook + regex + placeholders; flag rows for audit downstream |
| LoRA SFT JSONL | `export_sft.py` multi-turn + tools |
| DPO later | `export_dpo.py` emits governed **candidates** |
| Complexity & diversity | `complexity.py` + metadata on every row |
| Tool integrity | `tool_consistency.py` filters (CLI skips failed rows by default) |
| Student model path | Metadata supports context/tool-aware filtering (see below) |

---

## Mid-point milestone mapping

Per issue “Goals Achieved By Mid-point”:

- **Working E2E on sample subset** → CLI + `sample_data/sample_log.json`.
- **PII-stripped SFT JSONL** → `redact_session_inplace` before export.
- **Small DPO pair set** → `export_dpo.py` (review-gated).
- **Documented schemas** → Pydantic models + this README.
- **Gold alignment for one workflow** → sample covers Agristack + weather path.
- **Automated checks** → `validate_tool_consistency` + pytest.

---

## Student-model & filtering hooks

`SFTRow.metadata` carries `complexity_tier`, `segmentation`, and counts. Downstream you can:

- Drop `high` tier early in curriculum.
- Cap tool messages to match smaller context windows.
- Restrict to a **tool allowlist** matching student capabilities (`--known-tools` in CLI).

---

## Future additions (explicitly out of scope here)

- **Synthetic data generation** at scale (mock tool executor, open-weight generator on 1× H100).
- **Mock tools** with schema-validated args and realistic Agristack/weather stubs.
- **Curriculum scheduling** driven by `complexity_tier` + domain tags.
- **Student filtering** using held-out behavioral evals comparing teacher vs student checkpoints (TRL / PEFT dry-runs).

---

## Tests

```bash
pytest -q
```

---

## Residual risk (honest)

Regex + lightweight heuristics **cannot** guarantee zero leakage. Presidio improves recall for supported languages/entities but misses domain-specific IDs. Treat exports as **conditionally safe**: sample, audit, and blocklisted patterns before LoRA/DPO runs, as the issue acceptance criteria describe.

---

## License

Apache-2.0 (see `pyproject.toml`).
