"""Modal app for the ART training pipeline (one GPU container per run).

Cost rules: the GPU function's `timeout` is the hard spend cap for a run.
max_containers=1. Weights prefetch on CPU. Checkpoints and run outputs go to
the `apishift-checkpoints` Volume.
"""

from __future__ import annotations

import os

import modal

ART_VERSION = "0.5.20"
GPU = os.environ.get("APISHIFT_TRAIN_GPU", "H100")
GPU_USD_PER_HOUR = {"H100": 3.95, "A100-80GB": 2.50, "L40S": 1.95, "L4": 0.80}[GPU]
TIMEOUT_S = int(os.environ.get("APISHIFT_TRAIN_TIMEOUT_S", "9000"))
CACHE, CKPT = "/cache", "/ckpt"

app = modal.App("apishift-train")
hf_volume = modal.Volume.from_name("apishift-hf-cache", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("apishift-checkpoints", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(
        f"openpipe-art[backend]=={ART_VERSION}",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    # Install ART's pinned vLLM runtime at build time so GPU minutes are not spent on it.
    .run_commands("art runtime prepare || echo 'art runtime prepare failed at build; will prepare at run time'")
    # VLLM_USE_V2_MODEL_RUNNER=0: ART 0.5.20's policy-span plugin crashes in the V2 runner's
    # startup dummy run (KeyError on a dummy request id); the V1 runner does not route it there.
    .env({"HF_HOME": f"{CACHE}/hf", "WANDB_MODE": "disabled", "VLLM_USE_V2_MODEL_RUNNER": "0"})
    .add_local_python_source("apishift", "train")
)


@app.function(image=image, volumes={CACHE: hf_volume}, cpu=4, memory=8192, timeout=1800, max_containers=1)
def prefetch(repo: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo)
    hf_volume.commit()
    return path


def _write_job_record(run_name: str, record: dict) -> None:
    import json
    from pathlib import Path

    path = Path(CKPT) / "jobs" / f"{run_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **record}, indent=1))
    ckpt_volume.commit()


@app.function(image=image, gpu=GPU, volumes={CACHE: hf_volume, CKPT: ckpt_volume},
              timeout=TIMEOUT_S, max_containers=1)
def train(cfg: dict) -> dict:
    """GPU container: the whole pipeline. Records its own start/end so GPU time is known."""
    import asyncio
    import time

    from train.pipeline import PipelineConfig, run_pipeline

    started = time.time()
    _write_job_record(cfg["run_name"], {"gpu": GPU, "gpu_started": started, "status": "running"})
    status = "failed"
    try:
        summary = asyncio.run(run_pipeline(PipelineConfig(**cfg)))
        status = "done"
        return summary
    finally:
        ended = time.time()
        _write_job_record(cfg["run_name"], {"gpu_ended": ended, "gpu_seconds": round(ended - started),
                                            "est_gpu_usd": round((ended - started) / 3600 * GPU_USD_PER_HOUR, 2),
                                            "status": status})


ORCH_TIMEOUT_S = TIMEOUT_S + 3 * 3600 + 1800  # train cap + up to 3 h waiting for a GPU + prefetch


@app.function(image=image, cpu=1, memory=2048, volumes={CKPT: ckpt_volume},
              timeout=ORCH_TIMEOUT_S, max_containers=1)
def train_job(cfg: dict) -> dict:
    """Detached entry point: prefetch weights on CPU, then run the GPU pipeline. Lives on Modal only."""
    import time

    _write_job_record(cfg["run_name"], {"launched": time.time(), "config": cfg, "status": "prefetching"})
    prefetch.remote(cfg["base_model"])
    summary = train.remote(cfg)
    _write_job_record(cfg["run_name"], {"finished": time.time(), "best_step": summary.get("best_step"),
                                        "val": summary.get("val")})
    return summary
