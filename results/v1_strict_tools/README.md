# Baselines v1: tools sent with strict=true (INVALID for most change types)

Test split, seed 0, after adding ID lookup hints. Inspect's OpenAI-compatible
provider defaults to `strict_tools=True`. vLLM then grammar-constrains tool
arguments to the (stale) schema, so the model cannot emit renamed params, new
required fields, pagination cursors or undeclared endpoints. Fixed in
harness/run_baseline.py (`model_args={"strict_tools": False}` plus a smoke-test
guard).

Valid: `none` (control) and `format_change` rows only.
Invalid: rename_param, new_required_field, deprecation_with_migration,
pagination_change, error_schema_change.

| Model | control | format_change |
|-------|---------|---------------|
| Qwen3-1.7B, non-thinking, greedy | 0/30 | 1/30 |
| Qwen3-4B, non-thinking, greedy | 5/30 (17%) | 1/30 |
| Qwen3-4B, thinking, T=0.6 | 23/30 (77%) | 25/30 (83%) |

Seen in traces: in non-thinking mode, 4B copies the example ID from the param
docs (`evt_8f3a2c91d0`) instead of looking it up. Under strict decoding the
thinking model reads docs, then tries to add the new field and gets it squashed
into a string value (`"email": "a@b.org', "`).
