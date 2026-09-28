"""Train, then evaluate the best checkpoint, as ONE detached Modal job (no laptop needed).

chain_job (CPU, nonpreemptible) is the only coordinator:
  1. prefetch weights (CPU), spawn GPU training (train.modal_art.train, resumable)
  2. read the run summary, pick the best step
  3. start the vLLM eval server (L4), load that checkpoint as a runtime LoRA
  4. run the test-split eval in-process (Inspect) and write results to the Volume
If chain_job itself is ever restarted it re-attaches to the training call, or
skips training when the run already finished. Launch with
scripts/launch_detached.py --chain.
"""

from __future__ import annotations

import os

os.environ.setdefault("APISHIFT_MODEL", "qwen3-4b")
os.environ["APISHIFT_RUNTIME_LORA"] = "1"

import modal  # noqa: E402

from infra import modal_vllm as mv  # noqa: E402
from train import modal_art as ma  # noqa: E402

CHAIN_TIMEOUT_S = int(os.environ.get("APISHIFT_CHAIN_TIMEOUT_S", str(10 * 3600 + 1800)))

app = modal.App("apishift-chain")
app.include(ma.app)
app.include(mv.app)

chain_image = mv.eval_image.add_local_python_source("train")


@app.function(image=chain_image, cpu=1, memory=2048, volumes={ma.CKPT: ma.ckpt_volume},
              secrets=[modal.Secret.from_name("apishift-vllm")], timeout=CHAIN_TIMEOUT_S,
              max_containers=1, nonpreemptible=True)
def chain_job(train_cfg: dict, eval_spec: dict) -> dict:
    import json
    import time
    from pathlib import Path

    from harness.remote_eval import run_job

    run = train_cfg["run_name"]
    ma.ckpt_volume.reload()
    rec_path = Path(ma.CKPT) / "jobs" / f"{run}.json"
    summary_path = Path(ma.CKPT) / "runs" / run / "summary.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}

    if rec.get("status") == "done" and summary_path.exists():
        summary = json.loads(summary_path.read_text())  # restarted after training finished
    else:
        if rec.get("train_call_id") and rec.get("status") != "failed":
            call = modal.FunctionCall.from_id(rec["train_call_id"])
            ma._write_job_record(run, {"chain_reattached": time.time()})
        else:
            ma._write_job_record(run, {"launched": time.time(), "config": train_cfg, "status": "prefetching",
                                       "chain": True, "eval_spec": eval_spec})
            ma.prefetch.remote(train_cfg["base_model"])
            call = ma.train.spawn(train_cfg)
            ma._write_job_record(run, {"train_call_id": call.object_id})
        summary = call.get()

    step = summary.get("best_step") or summary["final_step"]
    by_step = {int(Path(c).name): c for c in summary["checkpoints"] if Path(c).name.isdigit()}
    job = {**eval_spec, "runtime_lora": {"name": eval_spec["served"], "path": by_step[step]}, "source_step": step}
    ma._write_job_record(run, {"eval_started": time.time(), "eval_step": step, "eval_name": job["name"]})
    mv.download_weights.remote()
    result = run_job(job, mv.serve.get_web_url(), ma.CKPT, mv.CFG.usd_per_hour, ma.ckpt_volume.commit)
    ma._write_job_record(run, {"chain_finished": time.time(), "eval": result})
    return {"best_step": step, "eval": result}
