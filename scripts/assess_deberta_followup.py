"""Fixed-stage assessment; refuses incomplete scoring, refits and unlocked test access."""

import argparse
import hashlib
import json
from pathlib import Path

from houndoracle.experiments.fit_calibration import certify, diagnostics, predict
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.strict_goal import evaluate_goal

PUBLIC = Path("research/experiments/campaign-29")
PRIVATE = Path("/workspace/houndoracle-data/campaign-29")
DATA = Path("/workspace/houndoracle-data/campaign-19")


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def decisions(rows, mapping, threshold=0.8):
    return [
        r
        | {
            "decision": r["candidate"]
            if r["candidate"] in ["YES", "NO"]
            and not r["truncated"]
            and predict(mapping, r["raw_score"]) >= threshold
            else "MAYBE",
            "confidence": predict(mapping, r["raw_score"]),
            "citation_supported": None,
            "citation_audit_method": "not independently assessed",
        }
        for r in rows
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["risk", "locked"])
    args = parser.parse_args()
    output = PUBLIC / (args.stage + "-assessment.json")
    if output.exists():
        raise FileExistsError("Preserve fixed-stage observation")
    protocol = json.loads((PUBLIC / "independent-protocol.json").read_text())
    mapping_path = PUBLIC / "provisional-mapping.json"
    if (
        hashlib.file_digest(mapping_path.open("rb"), "sha256").hexdigest()
        != protocol["mapping_sha256"]
    ):
        raise ValueError("Frozen map changed")
    mapping = json.loads(mapping_path.read_text())
    if args.stage == "locked":
        risk = json.loads((PUBLIC / "risk-assessment.json").read_text())
        if not risk["conditional80_research_gate_passed"]:
            raise PermissionError("Independent risk gate failed; do not open locked cohort")
    split = "calibration-risk" if args.stage == "risk" else "locked"
    manifest = json.loads(
        Path("research/experiments/campaign-19/dataset-manifest.json").read_text()
    )
    data_path = DATA / (split + ".jsonl")
    if (
        hashlib.file_digest(data_path.open("rb"), "sha256").hexdigest()
        != manifest["panels"][split]["sha256"]
    ):
        raise ValueError("Sealed cohort changed")
    expected = read(data_path)
    rows = read(PRIVATE / (args.stage + "-scores.jsonl"))
    if [(r["id"], r["group_id"], r["expected"]) for r in rows] != [
        (r["id"], r["group_id"], r["expected"]) for r in expected
    ]:
        raise ValueError("Incomplete or unpaired scoring")
    # Truncation must abstain in both reported metrics and the risk gate.
    eligible = [r for r in rows if not r["truncated"]]
    gate = certify(mapping, eligible, threshold=0.8, alpha=0.025)
    report = {
        "stage": args.stage,
        "count": len(rows),
        "conditional80_research_gate": gate,
        "conditional80_research_gate_passed": gate["qualified"] and gate["selected_groups"] >= 30,
        "raw": summary([r | {"decision": r["candidate"]} for r in rows]),
        "selected80": summary(decisions(rows, mapping)),
        "correctness_calibration": diagnostics(eligible, mapping),
        "strict99_joint_goal": evaluate_goal(decisions(rows, mapping, 0.99)),
        "production_qualified": False,
        "scope": "One fixed historical supplied-evidence population. No independent citation audit, general API qualification, Gemini comparison or distribution-shift guarantee. Timings from separate runs are not paired.",
    }
    if args.stage == "locked":
        baseline = read(PRIVATE / "baseline-locked-scores.jsonl")
        if [(r["id"], r["group_id"], r["expected"]) for r in baseline] != [
            (r["id"], r["group_id"], r["expected"]) for r in rows
        ]:
            raise ValueError("Unpaired locked baseline")
        old = json.loads(
            Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
        )["mapping"]
        a, b = decisions(baseline, old), decisions(rows, mapping)
        # Separate deployment/runtime measurements cannot be treated as a paired latency trial.
        a = [r | {"latency_seconds": None} for r in a]
        b = [r | {"latency_seconds": None} for r in b]
        comparison = paired_comparison(a, b, seed=1963, replicates=2000)
        report.update(baseline_selected80=summary(a), paired_selected=comparison)
        report["research_noninferiority_passed"] = (
            comparison["accuracy_delta"]["cluster_bootstrap_95pct_interval"][0] >= -0.01
            and comparison["coverage_delta"]["cluster_bootstrap_95pct_interval"][0] >= -0.02
        )
        report["paired_raw"] = paired_comparison(
            [r | {"decision": r["candidate"], "latency_seconds": None} for r in baseline],
            [r | {"decision": r["candidate"], "latency_seconds": None} for r in rows],
            seed=1963,
            replicates=2000,
        )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
