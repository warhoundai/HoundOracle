import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("research/experiments/results")
out = Path("benchmarks/results/campaign-02")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for name, label in [("fp32", "Raw encoder"), ("mlp", "MLP head"), ("mlp-int8", "MLP head int8")]:
    report = json.loads((root / f"analysis-{name}.json").read_text())
    data = (
        report["raw_correctness_calibration"]
        if name == "fp32"
        else report["methods"][report["selected_mapping"]]["validation"]
    )
    bins = data["reliability_bins"]
    axes[0].plot(
        [b["mean_confidence"] for b in bins],
        [b["empirical_correctness"] for b in bins],
        marker="o",
        label=label,
    )
    points = [p for p in data["risk_coverage"] if p["selective_risk"] is not None]
    axes[1].plot(
        [p["coverage"] for p in points],
        [p["selective_risk"] for p in points],
        marker="o",
        label=label,
    )
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0].set(
    xlabel="Estimated correctness",
    ylabel="Observed correctness",
    title="Validation reliability (candidate answers)",
)
axes[1].axhline(0.2, color="black", linestyle="--", linewidth=1)
axes[1].set(
    xlabel="Definitive coverage", ylabel="Selective risk", title="Validation threshold trade-off"
)
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out / "calibration-and-risk.svg")
fig.savefig(out / "calibration-and-risk.png", dpi=150)
plt.close(fig)
