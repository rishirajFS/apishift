"""Check detached Modal jobs and pull their outputs from the checkpoints Volume.

    uv run python scripts/collect_detached.py results/launches/2026-09-26.json [--stop-finished]

For each launched job: prints status (from the function call and the job record
on the Volume), downloads the job's outputs into results/remote/<name>/, and
with --stop-finished stops the detached app once its call has returned (a
web-server app can otherwise stay registered, idle, with no containers).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VOLUME = "apishift-checkpoints"


def volume_get(remote: str, local: Path) -> bool:
    local.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", "--force", VOLUME, remote, str(local)],
                       capture_output=True, text=True)
    return r.returncode == 0


def call_status(call_id: str) -> str:
    import modal

    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        return "finished"
    except TimeoutError:
        return "running"
    except Exception as exc:  # the remote function raised
        return f"failed: {type(exc).__name__}: {str(exc)[:200]}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("launches", type=Path)
    p.add_argument("--stop-finished", action="store_true")
    p.add_argument("--modal-profile", default="teel-lab-ace-ai")
    args = p.parse_args()
    os.environ["MODAL_PROFILE"] = args.modal_profile

    launches = json.loads(args.launches.read_text())
    for kind, info in launches.items():
        spec = info["spec"]
        name = spec.get("run_name") or spec.get("name")
        status = call_status(info["call_id"])
        dest = REPO / "results" / "remote" / name
        if kind == "pilot":
            volume_get(f"jobs/{name}.json", dest)
            if status != "running":
                volume_get(f"runs/{name}", dest)
        else:
            volume_get(f"evals/{name}/job.json", dest)
            if status != "running":
                volume_get(f"evals/{name}/results", dest)
        record = {}
        for rec in list(dest.rglob(f"{name}.json")) + list(dest.rglob("job.json")):
            record = json.loads(rec.read_text())
        print(f"{kind:16s} {name:28s} call={status}  job={record.get('status')}  "
              f"gpu_s={record.get('gpu_seconds') or record.get('gpu_seconds_est')}  "
              f"est_gpu_usd={record.get('est_gpu_usd')}")
        if args.stop_finished and status != "running":
            subprocess.run(["modal", "app", "stop", "-y", info["app"]], capture_output=True)
            print(f"  stopped app {info['app']}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
