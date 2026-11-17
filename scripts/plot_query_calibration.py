"""Standalone diagnostic figure from the frozen independent test; never tunes a threshold."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("benchmarks/results/campaign-09")
    rows = [json.loads(s) for s in (root / "locked-records.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, label, color in [
        ("original-frozen", "Original reference", "#64748b"),
        ("continued-mixture", "Trained evaluator", "#167d8d"),
    ]:
        allrows = [r for r in rows if r["variant"] == name]
        eligible = [r for r in allrows if r["calibrated_correctness"] is not None]
        x = []
        y = []
        lo = []
        hi = []
        for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
            group = [
                r
                for r in eligible
                if a <= r["calibrated_correctness"] < b
                or b == 1
                and r["calibrated_correctness"] == 1
            ]
            if not group:
                continue
            n = len(group)
            p = sum(r["candidate"] == r["expected"] for r in group) / n
            z = 1.96
            den = 1 + z * z / n
            center = (p + z * z / (2 * n)) / den
            half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            x.append(np.mean([r["calibrated_correctness"] for r in group]))
            y.append(p)
            lo.append(p - (center - half))
            hi.append(center + half - p)
        axes[0].errorbar(x, y, yerr=[lo, hi], fmt="o-", capsize=3, label=label, color=color)
        coverage = []
        risk = []
        for threshold in [0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]:
            accepted = [
                r
                for r in eligible
                if not r["truncated"] and r["calibrated_correctness"] >= threshold
            ]
            if accepted:
                coverage.append(len(accepted) / len(allrows))
                risk.append(sum(r["candidate"] != r["expected"] for r in accepted) / len(accepted))
        axes[1].plot(coverage, risk, "o-", label=label, color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
    axes[0].set(
        xlabel="Mapped correctness estimate",
        ylabel="Observed candidate correctness",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Reliability (95% Wilson bin intervals)",
    )
    axes[1].axhline(0.2, color="#c2410c", linestyle="--", linewidth=1, label="20% risk reference")
    axes[1].set(
        xlabel="Fraction of all queries selected",
        ylabel="Empirical selected error",
        ylim=(0, 0.3),
        title="Diagnostic risk–coverage curve",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "VitaminC: 600 independent heldout queries — conditional evidence labels", fontsize=11
    )
    fig.savefig(root / "calibration-diagnostics.png", dpi=180)
    fig.savefig(root / "calibration-diagnostics.svg")
    plt.close(fig)
    (root / "calibration-figure-note.md").write_text(
        "These are descriptive frozen-test diagnostics, not threshold selection. Curves use uncapped mapped correctness; the deployed research floor remains.80 and public confidence is conservatively capped. Bin intervals are not simultaneous guarantees. Conditional human evidence labels, not current-world truth.\n"
    )


if __name__ == "__main__":
    main()
