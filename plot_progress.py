"""Plots how the Environment Designer's frontier evolves over iterations.

Usage: python plot_progress.py results
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(run_dir: str):
    run = Path(run_dir)
    records = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines() if l.strip()]
    archive = json.loads((run / "archive.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    colors = {"accepted": "#2a9d8f", "rejected": "#e76f51", "invalid": "#999999"}
    for status, color in colors.items():
        pts = [(r["iteration"], r["complexity"]) for r in records
               if r["status"] == status and r.get("complexity") is not None]
        if pts:
            ax.scatter(*zip(*pts), c=color, label=status, s=40, zorder=3)
    frontier_x = [e["iteration"] for e in archive]
    frontier_y = [e["complexity"] for e in archive]
    ax.step(frontier_x + [records[-1]["iteration"] if records else 0],
            frontier_y + [frontier_y[-1]], where="post", color="#264653", label="frontier", zorder=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("scenario complexity")
    ax.set_title("Environment Designer: complexity of generated scenarios")
    ax.legend()

    ax = axes[1]
    idx = list(range(len(archive)))
    ax.plot(idx, [e["metrics"]["survival_nohint"] for e in archive], "o-", label="passive operator (no hint)")
    ax.plot(idx, [e["metrics"]["survival_hinted"] for e in archive], "s-", label="greedy operator (hint + lookahead)")
    ax.plot(idx, [e["metrics"]["regret"] for e in archive], "^--", label="regret (gap)")
    ax.set_xticks(idx)
    ax.set_xticklabels([e["name"].replace("Scenario", "") for e in archive], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("fraction of episode survived")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Accepted scenarios (in order)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(run / "progress.png", dpi=150)
    print("wrote", run / "progress.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
