#!/usr/bin/env bash
# Sep 25 work package, run unattended on the teel-lab Modal workspace. Cap: $30 total.
#   A. extra eval seeds (1, 2) for Qwen3-4B and Qwen3-32B, thinking mode, test split
#   B. 1.7B debug pipeline: self-SFT data -> LoRA SFT -> GRPO (ART, one H100, hard timeout)
#   C. test-split eval of the SFT and GRPO checkpoints (seed 0, same protocol as baselines)
# Every job runs the budget guard first. Queued time waiting for a GPU is not billed.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="${LOG:-logs/wp0925}"
mkdir -p "$LOG"
BUDGET=(--budget-usd 30 --budget-since 2026-09-25)
QUEUE=(--queue-timeout-s 21600)
RUN=${RUN:-debug-1.7b-v3}

# A (seeds) completed 06:24; skip it when resuming.
if [[ "${SKIP_A:-0}" != "1" ]]; then
echo "A: seeds $(date)"
uv run python -m harness.run_baseline --model qwen3-4b --thinking --seeds 1,2 --max-connections 48 \
  "${BUDGET[@]}" "${QUEUE[@]}" --planned-usd 1.2 > "$LOG/a_4b.log" 2>&1
uv run python -m harness.run_baseline --model qwen3-32b --thinking --seeds 1,2 --max-connections 48 \
  "${BUDGET[@]}" "${QUEUE[@]}" --planned-usd 2.0 > "$LOG/a_32b.log" 2>&1

fi

echo "B: train $(date)"
uv run python -m train.run_debug --run-name "$RUN" --timeout-s 5400 --sft-samples 2 --grpo-lr 3e-6 "${BUDGET[@]}" \
  > "$LOG/b_train.log" 2>&1

echo "C: checkpoint evals $(date)"
read -r SFT_DIR GRPO_DIR < <(uv run python - "$RUN" <<'EOF'
import json, sys
from pathlib import Path
s = json.loads(Path(f"results/train/{sys.argv[1]}.json").read_text())
by_step = {int(Path(p).name): p for p in s["checkpoints"] if Path(p).name.isdigit()}
print(by_step[s["sft_step"]], by_step[s["best_step"] or s["final_step"]])
EOF
)
uv run python -m harness.run_baseline --model qwen3-1.7b --thinking --lora "sft-1.7b=$SFT_DIR" --max-connections 48 \
  "${BUDGET[@]}" "${QUEUE[@]}" --planned-usd 0.6 > "$LOG/c_sft.log" 2>&1
uv run python -m harness.run_baseline --model qwen3-1.7b --thinking --lora "grpo-1.7b=$GRPO_DIR" --max-connections 48 \
  "${BUDGET[@]}" "${QUEUE[@]}" --planned-usd 0.6 > "$LOG/c_grpo.log" 2>&1
echo "done $(date)"
