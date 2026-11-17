"""Assess frozen local generative development screens; no promotion on reused data."""

import argparse
import json
from pathlib import Path

from houndoracle.experiments.metrics import paired_comparison, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=int, choices=[32, 34])
    args = parser.parse_args()
    root = Path(f"/workspace/houndoracle-data/campaign-{args.campaign}")
    public = Path(f"research/experiments/campaign-{args.campaign}")
    if (public / "screen.json").exists():
        raise FileExistsError("Preserve completed screen")
    result = json.loads((root / "result.json").read_text())
    inputs = json.loads((root / "inputs.json").read_text())
    rows = result["rows"]
    if not result["completed"] or [(r["id"], r["group_id"], r["expected"]) for r in rows] != [
        (r["id"], r["group_id"], r["expected"]) for r in inputs["questions"]
    ]:
        raise ValueError("Incomplete or unpaired experiment")
    baseline = [
        r | {"decision": r["candidate"], "latency_seconds": None} for r in inputs["baseline"]
    ]
    candidate = [r | {"decision": r["candidate"]} for r in rows]
    a, b = summary(baseline), summary(candidate)
    passed = (
        b["accuracy_all_decisions"] - a["accuracy_all_decisions"] >= 0.02
        and b["definitive_coverage"] >= a["definitive_coverage"] - 0.02
    )
    report = {
        "baseline_raw": a,
        "candidate_raw": b,
        "paired_raw": paired_comparison(
            baseline, [r | {"latency_seconds": None} for r in candidate], seed=1970, replicates=2000
        ),
        "screen_passed": passed,
        "decision": "retain-research-candidate-only"
        if passed
        else "reject-before-independent-data",
        "resource_measurements": {
            k: result[k] for k in ["acquisition_seconds", "total_seconds", "peak_rss_kib"]
        },
        "runtime": result["runtime"],
        "production_qualified": False,
        "fresh_risk_opened": False,
        "locked_opened": False,
        "scope": "48balanced reused development cases with supplied evidence, not live search. No calibrated confidence, independent citation audit,99qualification or all-in comparison. Reference timings not paired.",
        "records": candidate,
    }
    (public / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
