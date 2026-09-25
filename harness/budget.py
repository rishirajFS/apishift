"""Spend tracking and hard caps for Modal jobs (budget_log.csv).

Every job appends one row: an upper-bound estimate at run time, and the
actual cost filled in later from `modal billing report`. A work package is
capped by summing rows since its start time (actual when known, else the
estimate) plus the next job's planned cost.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUDGET_LOG = REPO / "budget_log.csv"
COLUMNS = ["date", "job", "modal_app", "gpu", "duration_s", "est_cost_usd", "actual_cost_usd", "notes"]


def _rows() -> list[dict]:
    if not BUDGET_LOG.exists():
        return []
    with BUDGET_LOG.open() as fh:
        return list(csv.DictReader(fh))


def row_cost(row: dict) -> float:
    return float(row["actual_cost_usd"] or row["est_cost_usd"] or 0)


def spend_since(since: str | None = None) -> float:
    """Logged spend for rows dated at or after `since` (ISO prefix, local time); all rows if None."""
    return round(sum(row_cost(r) for r in _rows() if since is None or r["date"] >= since), 4)


def guard(limit_usd: float, since: str | None, planned_usd: float, what: str) -> None:
    spent = spend_since(since)
    if spent + planned_usd > limit_usd:
        raise SystemExit(
            f"budget: {what} would bring spend since {since or 'start'} to "
            f"${spent + planned_usd:.2f} (> ${limit_usd:.2f} cap; ${spent:.2f} logged). Ask before spending more."
        )
    print(f"budget ok: ${spent:.2f} logged since {since}, planned {what} ${planned_usd:.2f}, cap ${limit_usd:.2f}")


def append(job: str, modal_app: str, gpu: str, duration_s: float, usd_per_hour: float, notes: list[str]) -> float:
    est = round(duration_s / 3600 * usd_per_hour * 1.1, 2)  # +10% for CPU/memory
    new = not BUDGET_LOG.exists()
    with BUDGET_LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({"date": dt.datetime.now().isoformat(timespec="seconds"), "job": job, "modal_app": modal_app,
                    "gpu": gpu, "duration_s": round(duration_s), "est_cost_usd": est, "actual_cost_usd": "",
                    "notes": ";".join(notes)})
    return est
