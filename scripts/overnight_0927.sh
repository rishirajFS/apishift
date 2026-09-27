#!/usr/bin/env bash
# Overnight chain for Sep 27, cap $12: 4B GRPO re-pilot on the adaptation-hard pool, then a
# 3-seed test eval of its best checkpoint. The Modal jobs are detached; this local chain only
# waits, launches the second job and collects. Run it so it survives sleep and a closed terminal:
#   nohup caffeinate -ims bash scripts/overnight_0927.sh > logs/overnight_0927.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
LAUNCHES=results/launches/2026-09-27.json
BUDGET=(--budget-usd 12 --budget-since 2026-09-27 --launches-file "$LAUNCHES")
RUN=grpo-4b-pool-v1

wait_done() {  # poll every 10 min, up to 10 h
  local kind=$1 out
  for _ in $(seq 1 60); do
    out=$(uv run python scripts/collect_detached.py "$LAUNCHES" --stop-finished 2>&1 | grep -E "^$kind " || true)
    echo "[$(date)] $out"
    if [[ -n "$out" ]] && ! grep -q "call=running" <<<"$out"; then
      grep -q "call=finished" <<<"$out" && return 0 || return 1
    fi
    sleep 600
  done
  return 1
}

echo "[$(date)] launching re-pilot $RUN"
uv run python scripts/launch_detached.py --repilot --repilot-name "$RUN" "${BUDGET[@]}"
wait_done pilot

echo "[$(date)] recording re-pilot cost estimate from its job record (billing reconciles later)"
uv run python - "$RUN" <<'PY'
import csv, json, sys
from pathlib import Path
run = sys.argv[1]
rec = json.loads(Path(f"results/remote/{run}/{run}.json").read_text())
gpu = rec.get("gpu_seconds", 0) / 3600 * 3.95 * 1.1
cpu = (rec.get("finished", rec["launched"]) - rec["launched"]) / 3600 * 0.063
with open("budget_log.csv") as fh:
    rows = list(csv.DictReader(fh))
for r in rows:
    if r["job"] == f"detached_pilot_{run}" and not r["actual_cost_usd"]:
        r["actual_cost_usd"] = f"{gpu + cpu:.4f}"
        r["notes"] += ";actual = job-record GPU time estimate, billing to confirm"
with open("budget_log.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"re-pilot estimate ${gpu + cpu:.2f} (gpu_seconds={rec.get('gpu_seconds')})")
PY

echo "[$(date)] launching checkpoint eval"
uv run python scripts/launch_detached.py --checkpoint-eval "$RUN" "${BUDGET[@]}"
wait_done checkpoint_eval

echo "[$(date)] collecting prompt baseline (Sep 26 launch)"
uv run python scripts/collect_detached.py results/launches/2026-09-26.json --stop-finished || true
echo "[$(date)] done"
