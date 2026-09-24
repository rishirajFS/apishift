"""Figure: success rate by mutation group (control / seen / held-out) per model.

    uv run python -m harness.figures

Reads results/baseline_*_test_s0.json. Writes results/figures/success_by_group.png
plus a CSV table view. Colors follow the model (entity), in a fixed slot
order validated with the dataviz palette checker (light surface). Aqua and
yellow are below 3:1 contrast, so every bar carries a value label and a table
is written.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "figures"

# Fixed entity -> slot mapping; a missing model never repaints the others.
SERIES = [
    ("qwen3-1.7b", "Qwen3-1.7B (base)", "#2a78d6"),
    ("qwen3-4b", "Qwen3-4B (base)", "#eb6834"),
    ("qwen3-32b", "Qwen3-32B (reference)", "#1baf7a"),
    ("qwen3-4b-grpo", "Qwen3-4B + GRPO", "#eda100"),
]
GROUPS = [("control", "No change\n(control)"), ("seen", "Seen change types\n(train types)"),
          ("heldout", "Held-out change types\n(pagination, error schema)")]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def load(split: str = "test", seed: int = 0) -> list[tuple[str, str, str, dict]]:
    out = []
    for key, label, color in SERIES:
        path = RESULTS / f"baseline_{key}_{split}_s{seed}.json"
        if path.exists():
            out.append((key, label, color, json.loads(path.read_text())))
    return out


def plot(rows: list[tuple[str, str, str, dict]]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(rows)
    width = 0.8 / max(n, 1)
    for i, (_, label, color, res) in enumerate(rows):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(GROUPS))]
        vals = [100 * (res["by_group"][g]["success_rate"] or 0) for g, _ in GROUPS]
        ns = [res["by_group"][g]["n"] for g, _ in GROUPS]
        ax.bar(xs, vals, width=width - 0.02, color=color, label=label, edgecolor=SURFACE, linewidth=1)
        for x, v, cnt in zip(xs, vals, ns, strict=True):
            ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
            ax.text(x, -4.5, f"n={cnt}", ha="center", va="top", fontsize=6.5, color=INK_2)
    ax.set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS], fontsize=9, color=INK)
    ax.tick_params(axis="x", pad=14, length=0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Task success rate (%)", fontsize=9, color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_2, labelsize=8, length=0)
    ax.set_title("APIShift test split: success when the API has silently changed",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    path = OUT / "success_by_group.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_table(rows: list[tuple[str, str, str, dict]]) -> Path:
    path = OUT / "success_by_group.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "group", "n", "success_rate", "mean_reward", "mean_turns", "invalid_call_rate"])
        for key, _, _, res in rows:
            for g, _ in GROUPS:
                a = res["by_group"][g]
                w.writerow([key, g, a["n"], a["success_rate"], a["mean_reward"], a["mean_turns"],
                            a["invalid_call_rate"]])
    return path


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit("no results/baseline_*.json yet")
    print(plot(rows), write_table(rows))


if __name__ == "__main__":
    main()
