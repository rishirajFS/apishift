# APIShift: Training Tool-Use Agents to Adapt to Evolving APIs

## Testing rules (read first)
- NEVER write unit tests after you write code.
- Highly prefer E2E tests as the sole testing mechanism. Use them to verify complex features work. At the end of E2E tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, FIRST write all the ways it could fail, THEN write the code.

How this applies here:
- E2E test = a full episode run: task loaded, mutation applied by seed, agent loop executed (scripted agent or 1.7B model), reward computed. Artifact: trace JSON plus reward in `artifacts/e2e/`, reproducible byte-for-byte from the same seed with one command.
- The reward function and task success checks are the one place isolation is required. Write their failure modes first (e.g. predicate passes on unchanged state, reward rewards invalid calls, model text alone satisfies success), then the code.

## Goal
Build an RL environment where the APIs an agent uses have silently changed since its tool schemas were written, then show that GRPO on a 4B model teaches it to recover (read errors and docs, adapt calls, finish the task), including on change types never seen in training. Headline target: trained 4B beats a 32B+ reference model on held-out change types. Hard deadline: Oct 6, 2026. Budget: $500 Modal compute.

## Core idea
The agent is given the ORIGINAL (stale) tool schemas. The live sandbox API has been mutated. Success requires noticing failures, calling `get_api_docs(endpoint)` or reading error messages, and adapting.

## Environment (`apishift/envs/`)
- 3 domains: `calendar`, `payments`, `ecommerce`. Each is an in-process Python mock API with 8-10 endpoints, a mutable state dict, JSON schemas, and a docs string per endpoint.
- Tool calls use OpenAI function-calling JSON. Errors return realistic structured messages (status code, message, hint where a real API would give one).
- `get_api_docs(endpoint)` always reflects the CURRENT (mutated) schema.
- Mutation engine (`mutations.py`), deterministic by seed, applied per episode:
  - Train types: `rename_param`, `format_change` (e.g. date string to ISO 8601), `new_required_field`, `deprecation_with_migration` (old endpoint returns 410 plus migration notice).
  - Held-out types: `pagination_change`, `error_schema_change`.
- Tasks (`tasks/`): ~150 templated tasks, each with initial state, natural-language instruction, and a success predicate on FINAL STATE (never on model text). Splits: train / val / test, with held-out mutation types only in test.

## Agent loop and reward
- Max 8 turns. Model emits tool calls or a final answer.
- Reward: 1.0 if success predicate holds; minus 0.05 per invalid call; minus 0.02 per turn beyond the task's minimum; floor at 0.
- Log every episode as JSON trace (messages, calls, responses, reward, mutation applied).

## Models and training (`train/`)
- Debug: Qwen3-1.7B. Main: Qwen3-4B. Reference baseline: Qwen3-32B or a 72B model (verify current releases first).
- Serving and rollouts: vLLM on Modal.
- SFT: generate trajectories with the reference model on train split, keep only verified successes, LoRA (r=16) SFT on 4B.
- GRPO: multi-turn GRPO starting from the SFT checkpoint. Evaluate ART and verl for at most 2 hours, pick one. 2 seeds.
- FALLBACK if multi-turn GRPO is not learning by Sep 29: single-retry formulation (one call, one error, one retry) using TRL GRPOTrainer.
- Optional if time: DPO on success/failure trajectory pairs from the same start state.

## Evaluation (`harness/`, built on Inspect AI)
- Report for base, SFT, GRPO, and reference: success rate on seen mutations, held-out mutations, and no-mutation control; mean turns; invalid-call rate.
- BFCL subset as a regression check on general tool use.
- Auto-labeled failure taxonomy: repeated identical call, invented parameter, ignored migration notice, gave up early.
- All results written to `results/*.json` for the demo.

## Modal rules (non-negotiable)
- Every function has `timeout` and `max_containers` set. Checkpoints on a Modal Volume.
- Nothing runs on 4B until the full pipeline works on 1.7B.
- Append every job's GPU type, duration, and estimated cost to `budget_log.csv`.

## Demo (`demo/`)
Next.js static site on Vercel. Pages: (1) Arena: pick task and mutation, view base vs SFT vs GRPO traces side by side from precomputed JSON, with failures highlighted; (2) Results: charts for seen vs held-out success. Must load in under 3 seconds. Live inference (Modal endpoint, scale to zero, rate-limited) only if time allows.

## Milestones
- Sep 25: envs, mutations, tasks, reward; E2E episode test passing with reproducible trace artifact.
- Sep 26: Inspect harness, baselines, first figure.
- Sep 27: SFT done.
- Sep 28-30: GRPO (1.7B debug, then 4B, 2 seeds).
- Oct 1: full eval and failure analysis.
- Oct 2-3: demo deployed. Oct 4: README, HF model card, write-up.

## Non-goals
More than 3 domains, more than one main model, reward-model ablations, UI polish beyond clean and readable.
