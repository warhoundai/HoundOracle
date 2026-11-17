"""Freeze correctness calibration, then evaluate fresh risk and a gated locked cohort."""

import argparse
import hashlib
import json
from pathlib import Path

from houndoracle.experiments.fit_calibration import certify, diagnostics, fit, predict
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.paired_risk import paired_improvement_lower
from houndoracle.experiments.strict_goal import evaluate_goal

PUBLIC = Path("research/experiments/campaign-39")
PRIVATE = Path("/workspace/houndoracle-data/campaign-39")


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def selected(rows, mapping, threshold=0.8):
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
            "latency_seconds": None,
        }
        for r in rows
    ]


def checked(stage):
    rows = read(PRIVATE / (stage + "-scores.jsonl"))
    source = (
        Path("/workspace/houndoracle-data/campaign-19/calibration-fit.jsonl")
        if stage == "fit"
        else PRIVATE / (("locked" if "locked" in stage else "risk") + ".jsonl")
    )
    expected = read(source)
    keys = lambda r: [(x["id"], x["group_id"], x["expected"]) for x in r]
    if keys(rows) != keys(expected):
        raise ValueError("Incomplete or unpaired cohort observations")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fit", "risk", "locked"])
    args = p.parse_args()
    chosen = json.loads(Path("research/experiments/campaign-38/screen.json").read_text())
    if not chosen["screen_passed"]:
        raise PermissionError("No passing development candidate")
    target = PUBLIC / (
        "frozen-policy.json" if args.stage == "fit" else args.stage + "-assessment.json"
    )
    if target.exists():
        raise FileExistsError("Preserve frozen policy and stage observations")
    if args.stage == "fit":
        rows = checked("fit")
        mapping = fit("logistic", [r for r in rows if not r["truncated"]])
        result = {
            "adapter_sha256": chosen["training"]["checkpoint_sha256"],
            "base_sha256": chosen["training"]["base_checkpoint_sha256"],
            "mapping": mapping,
            "threshold": 0.8,
            "risk_alpha": 0.025,
            "minimum_risk_groups": 30,
            "ni_alpha_per_endpoint": 0.0125,
            "accuracy_noninferiority_margin": 0.01,
            "coverage_noninferiority_margin": 0.02,
            "calibration_scores_sha256": hashlib.file_digest(
                (PRIVATE / "fit-scores.jsonl").open("rb"), "sha256"
            ).hexdigest(),
            "population": "Fresh unused historical real revision source components; supplied evidence only",
            "production_qualified": False,
        }
    else:
        frozen = json.loads((PUBLIC / "frozen-policy.json").read_text())
        if frozen["adapter_sha256"] != chosen["training"]["checkpoint_sha256"]:
            raise ValueError("Frozen candidate changed")
        if (
            args.stage == "locked"
            and not json.loads((PUBLIC / "risk-assessment.json").read_text())["qualified80"]
        ):
            raise PermissionError("Risk gate failed; locked labels must remain unread")
        rows = checked(args.stage)
        mapping = frozen["mapping"]
        accepted = selected(rows, mapping)
        eligible = [r for r in rows if not r["truncated"]]
        certificate = certify(mapping, eligible, threshold=0.8, alpha=0.025)
        result = {
            "count": len(rows),
            "qualified80": certificate["qualified"] and certificate["selected_groups"] >= 30,
            "risk_certificate": certificate,
            "raw": summary(
                [r | {"decision": r["candidate"], "latency_seconds": None} for r in rows]
            ),
            "selected80": summary(accepted),
            "calibration_diagnostics": diagnostics(eligible, mapping),
            "diagnostic_nontruncated_denominator": len(eligible),
            "strict99_goal": evaluate_goal(selected(rows, mapping, 0.99)),
            "production_qualified": False,
            "new_cloud_spend_usd": 0,
            "scope": "Fresh human-labeled historical snapshot cohort, not independent citation audit or arbitrary web truth. No policy/threshold refit after risk. Separate timings are not a paired latency trial.",
        }
        if args.stage == "locked":
            baseline = checked("baseline-locked")
            oldmap = json.loads(
                Path("research/experiments/campaign-29/provisional-mapping.json").read_text()
            )
            old = selected(baseline, oldmap)
            accuracy = [
                int(b["decision"] == b["expected"]) - int(a["decision"] == a["expected"])
                for a, b in zip(old, accepted, strict=True)
            ]
            coverage = [
                int(b["decision"] in ["YES", "NO"]) - int(a["decision"] in ["YES", "NO"])
                for a, b in zip(old, accepted, strict=True)
            ]
            accuracy_lower = paired_improvement_lower(accuracy, alpha=0.0125)
            coverage_lower = paired_improvement_lower(coverage, alpha=0.0125)
            bs = summary(old)
            cs = result["selected80"]
            material = (
                cs["accuracy_all_decisions"] >= bs["accuracy_all_decisions"] + 0.02
                or cs["correct_definitive_fraction"] >= bs["correct_definitive_fraction"] + 0.02
            )
            result.update(
                baseline_selected80=bs,
                baseline_raw=summary(
                    [r | {"decision": r["candidate"], "latency_seconds": None} for r in baseline]
                ),
                paired_selected=paired_comparison(old, accepted, seed=1987, replicates=2000),
                paired_raw=paired_comparison(
                    [r | {"decision": r["candidate"], "latency_seconds": None} for r in baseline],
                    [r | {"decision": r["candidate"], "latency_seconds": None} for r in rows],
                    seed=1987,
                    replicates=2000,
                ),
                accuracy_noninferiority_lower=accuracy_lower,
                coverage_noninferiority_lower=coverage_lower,
                material_gain=material,
                research_promotion_gate_passed=material
                and accuracy_lower >= -0.01
                and coverage_lower >= -0.02,
            )
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
