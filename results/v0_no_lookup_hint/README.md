# Baselines v0: before ID lookup hints (2026-09-24)

Test split, seed 0, greedy, Qwen3 non-thinking mode, vLLM 0.30.0 on Modal L4.

| Model | control | seen | held-out |
|-------|---------|------|----------|
| Qwen3-1.7B | 0/30 | 0/120 | 0/58 |
| Qwen3-4B | 0/30 | 0/120 | 0/58 |

Neither model made a single lookup call in 208 episodes (0/208 first calls
were list/search/get/docs). They guessed IDs from names, e.g.
`cancel_event(event_id="Offsite Planning")`, got a 404, and gave up. Tool
schemas and the harness were verified correct from the raw vLLM requests.

Change made afterwards: ID and email parameter docs now show an example ID and
where to look it up (as real API docs do). The system prompt now says not to
guess IDs or emails. Neither change mentions the API changing.

Cost: $0.17 total (billing report).
