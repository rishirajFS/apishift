"""vLLM OpenAI-compatible server on Modal, one app per model.

The model is picked by the APISHIFT_MODEL env var at import time, both locally
and in the container (it is passed in as a secret). App names are
per model, so `modal billing report` attributes cost per model.

Cost rules (AGENTS.md): every function sets timeout and max_containers.
Weights download on CPU into a Volume, never on a GPU container. Runs use
ephemeral apps (app.run()), which stop when the driver exits.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import modal


@dataclass(frozen=True)
class ModelConfig:
    hf_repo: str
    gpu: str
    usd_per_hour: float  # Modal list price for `gpu`, checked 2026-09-24
    max_model_len: int = 16384
    max_num_seqs: int = 64


MODELS: dict[str, ModelConfig] = {
    "qwen3-1.7b": ModelConfig("Qwen/Qwen3-1.7B", "L4", 0.80),
    "qwen3-4b": ModelConfig("Qwen/Qwen3-4B", "L4", 0.80),
    "qwen3-32b": ModelConfig("Qwen/Qwen3-32B-FP8", "H100", 3.95),
}

VLLM_VERSION = "v0.30.0"
PORT = 8000
CACHE = "/cache"
SERVER_TIMEOUT_S = 3600
STARTUP_TIMEOUT_S = 1500
SCALEDOWN_S = 120

MODEL_KEY = os.environ.get("APISHIFT_MODEL", "qwen3-1.7b")
CFG = MODELS[MODEL_KEY]
# Optional LoRA adapter "name=/ckpt/<dir>", served next to the base model.
LORA = os.environ.get("APISHIFT_LORA")
CKPT = "/ckpt"

app = modal.App(f"apishift-serve-{MODEL_KEY.replace('.', '-')}" + ("-lora" if LORA else ""))
volume = modal.Volume.from_name("apishift-hf-cache", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("apishift-checkpoints", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry(f"vllm/vllm-openai:{VLLM_VERSION}")
    .entrypoint([])
    # Modal detects the image's Python via `python`; the vLLM image only ships `python3`.
    .run_commands("ln -sf \"$(command -v python3)\" /usr/local/bin/python", "python --version")
    .env({"HF_HOME": f"{CACHE}/hf"})
)
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub==2.0.0")
    .env({"HF_HOME": f"{CACHE}/hf"})
)
model_secret = modal.Secret.from_dict({"APISHIFT_MODEL": MODEL_KEY, **({"APISHIFT_LORA": LORA} if LORA else {})})


def weights_dir(repo: str) -> str:
    return f"{CACHE}/models/{repo}"


@app.function(image=download_image, volumes={CACHE: volume}, cpu=4, memory=8192,
              timeout=1800, max_containers=1, secrets=[model_secret])
def download_weights() -> str:
    from huggingface_hub import snapshot_download

    path = weights_dir(CFG.hf_repo)
    snapshot_download(CFG.hf_repo, local_dir=path,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    volume.commit()
    return path


@app.function(
    image=vllm_image,
    gpu=CFG.gpu,
    volumes={CACHE: volume, CKPT: ckpt_volume},
    secrets=[model_secret, modal.Secret.from_name("apishift-vllm")],
    timeout=SERVER_TIMEOUT_S,
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT_S)
def serve() -> None:
    cmd = [
        "vllm", "serve", weights_dir(CFG.hf_repo),
        "--served-model-name", MODEL_KEY,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", str(CFG.max_model_len),
        "--max-num-seqs", str(CFG.max_num_seqs),
        "--gpu-memory-utilization", "0.90",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        # Keeps <think> content out of message text and tool-call parsing (no-op when thinking is off).
        "--reasoning-parser", "qwen3",
        "--seed", "0",
    ]
    if LORA:
        cmd += ["--enable-lora", "--max-lora-rank", "32", "--lora-modules", LORA]
    subprocess.Popen(cmd)


# --- Detached eval job: Inspect runs on Modal against `serve`, results go to the checkpoints Volume ---

EVAL_JOB_TIMEOUT_S = int(os.environ.get("APISHIFT_EVAL_JOB_TIMEOUT_S", str(4 * 3600 + 2 * 3600)))

eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    # same versions as uv.lock; Inspect 0.3.268's OpenAI-compatible provider needs openai>=3.1
    .uv_pip_install("inspect-ai==0.3.268", "openai==3.19.2")
    .add_local_python_source("apishift", "harness", "infra")
)


@app.function(image=eval_image, cpu=1, memory=2048, volumes={CKPT: ckpt_volume},
              secrets=[model_secret, modal.Secret.from_name("apishift-vllm")],
              timeout=EVAL_JOB_TIMEOUT_S, max_containers=1)
def eval_job(job: dict) -> dict:
    """Detached entry point: weights on CPU, then smoke gate + seeds against the vLLM server."""
    from harness.remote_eval import run_job

    download_weights.remote()
    return run_job(job, serve.get_web_url(), CKPT, CFG.usd_per_hour, ckpt_volume.commit)
