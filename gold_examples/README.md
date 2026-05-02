# Gold examples (reviewer / contributor fixtures)

Small **Langfuse-shaped** JSON arrays you can pipe through `logs2train run` or `logs2train validate`.

| File | Intent |
|------|--------|
| `ideal_agentic.json` | Clean 2-tool trajectory + grounded final answer. |
| `inefficient_trajectory.json` | Redundant duplicate tool calls (efficiency / DPO negatives). |
| `persona_violation.json` | Response explicitly conflicts with persona constraints (adherence negatives). |

```bash
logs2train validate --input gold_examples/ideal_agentic.json --kind raw \
  --known-tools fetch_agristack_data,weather_forecast
```

These are **not** private production logs; they document expected shapes and quality axes for the pipeline.
