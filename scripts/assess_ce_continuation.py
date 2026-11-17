"""Apply the declared gate to a completed CE continuation without moving thresholds."""

import json
from pathlib import Path

from houndoracle.experiments.metrics import paired_comparison, summary


def raw(rows):
    return [r | {"decision": r["candidate"], "latency_seconds": None} for r in rows]


def main():
    public = Path("research/experiments/campaign-38")
    if (public / "screen.json").exists():
        raise FileExistsError("Preserve completed screen")
    private = Path("/workspace/houndoracle-data/campaign-38")
    result = json.loads((private / "ce-continuation/result.json").read_text())
    if not result["completed"] or result["steps"] != 384:
        report = {
            "screen_passed": False,
            "decision": "reject-incomplete-training",
            "training": {k: v for k, v in result.items() if k != "losses"},
            "production_qualified": False,
        }
    else:
        base = json.loads(Path("/workspace/houndoracle-data/campaign-29/result.json").read_text())[
            "scores"
        ]["validation"]
        pilot = json.loads(
            Path("/workspace/houndoracle-data/campaign-35/ce-control/result.json").read_text()
        )["validation_scores"]
        rows = result["validation_scores"]
        keys = lambda r: [(x["id"], x["group_id"], x["expected"]) for x in r]
        if keys(rows) != keys(base) or keys(pilot) != keys(base):
            raise ValueError("Unpaired validation")
        a, b, c = summary(raw(base)), summary(raw(pilot)), summary(raw(rows))
        passed = (
            c["accuracy_all_decisions"] >= a["accuracy_all_decisions"] + 0.02
            and c["definitive_coverage"] >= a["definitive_coverage"] - 0.02
            and c["accuracy_all_decisions"] >= b["accuracy_all_decisions"]
        )
        report = {
            "baseline_raw": a,
            "pilot_raw": b,
            "candidate_raw": c,
            "paired_vs_base": paired_comparison(raw(base), raw(rows), seed=1987, replicates=2000),
            "paired_vs_pilot": paired_comparison(raw(pilot), raw(rows), seed=1987, replicates=2000),
            "training": {
                k: v for k, v in result.items() if k not in ["validation_scores", "losses"]
            },
            "screen_passed": passed,
            "decision": "retain-for-frozen-fresh-evaluation"
            if passed
            else "reject-before-new-held-out-data",
            "records": rows,
            "production_qualified": False,
            "fresh_risk_opened": False,
            "locked_opened": False,
            "scope": "207reused development queries; no independent promotion, truth confidence, citation audit or99claim.",
            "new_cloud_spend_usd": 0,
        }
    (public / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
