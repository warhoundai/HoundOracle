"""Uncertainty for frozen successor results; no threshold selection."""

import json
from pathlib import Path

from scipy.stats import beta


def interval(k, n, alpha=0.05):
    return [
        float(beta.ppf(alpha / 2, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0,
    ]


def main():
    root = Path("benchmarks/results/campaign-06")
    rows = [
        json.loads(s) for s in (root / "safe-specialist-records.jsonl").read_text().splitlines()
    ]
    output = {}
    for panel in sorted({r["panel"] for r in rows}):
        selected = [r for r in rows if r["panel"] == panel]
        skip = sum(r["additional_abstention"] for r in selected)
        false = sum(r["additional_abstention"] and r["expected"] != "MAYBE" for r in selected)
        output[panel] = {
            "n": len(selected),
            "calls_avoided": skip,
            "calls_avoided_95_binomial_interval": interval(skip, len(selected)),
            "false_abstentions": false,
            "false_abstention_fraction_95_binomial_interval": interval(false, len(selected)),
            "strata": {},
        }
        for label in ["YES", "NO", "MAYBE"]:
            cohort = [r for r in selected if r["expected"] == label]
            k = sum(r["additional_abstention"] for r in cohort)
            output[panel]["strata"][label] = {
                "n": len(cohort),
                "abstentions": k,
                "95_binomial_interval": interval(k, len(cohort)),
            }
        if panel.startswith("balanced"):
            avoided, false_bounds = [], []
            for label in ["YES", "NO", "MAYBE"]:
                cohort = [r for r in selected if r["expected"] == label]
                k = sum(r["additional_abstention"] for r in cohort)
                avoided.append(interval(k, len(cohort), alpha=0.05 / 3))
                false_bounds.append(
                    interval(k, len(cohort), alpha=0.05 / 2) if label != "MAYBE" else [0.0, 0.0]
                )
            output[panel]["calls_avoided_95_binomial_interval"] = None
            output[panel]["false_abstention_fraction_95_binomial_interval"] = None
            output[panel]["calls_avoided_95_stratified_bonferroni_interval"] = [
                sum(b[i] for b in avoided) / 3 for i in [0, 1]
            ]
            output[panel]["false_abstention_95_stratified_bonferroni_interval"] = [
                sum(b[i] for b in false_bounds) / 3 for i in [0, 1]
            ]
    output["interpretation"] = (
        "Intervals describe within-panel group-sampled proportions; not simultaneous guarantees or transport across regimes. Balanced sampling differs from natural frequencies. Exact definitive baseline preservation is structural; fallback quality/latency/cost remain unmeasured."
    )
    (root / "safe-specialist-uncertainty.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
