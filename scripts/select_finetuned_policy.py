"""Freeze operating thresholds using old validation only, before fresh scoring."""

import json
from pathlib import Path

from houndoracle.experiments.abstention import policy_digest, would_abstain
from houndoracle.experiments.fit_calibration import fit
from houndoracle.experiments.metrics import summary
from houndoracle.experiments.pipeline import apply


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def policy_rows(rows, artifact, threshold):
    result = []
    for row in rows:
        r = apply(row, artifact)
        if r["decision"] in ["YES", "NO"] and (r["confidence"] or 0) < threshold:
            r["decision"] = "MAYBE"
        result.append(r)
    return result


def main():
    root = Path("research/experiments/campaign-06")
    out = root / "final-policy-frozen.json"
    if out.exists():
        raise FileExistsError("Policy already frozen; never retune on new outcomes")
    old = json.loads(
        Path("research/experiments/campaign-04/calibration-logistic-threads2.json").read_text()
    )
    new = json.loads((root / "calibration-finetuned.json").read_text())
    baseline = summary(
        policy_rows(
            load(Path("/workspace/houndoracle-data/scores/validation-logistic.jsonl")), old, 0.85
        )
    )
    valid = load(
        Path("/workspace/houndoracle-data/campaign-06-finetuned/validation-finetuned.jsonl")
    )
    points = []
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        measured = summary(policy_rows(valid, new, threshold))
        passed = (
            measured["accuracy_all_decisions"] >= baseline["accuracy_all_decisions"]
            and measured["definitive_coverage"] >= baseline["definitive_coverage"]
            and measured["selective_risk"] is not None
            and measured["selective_risk"] <= baseline["selective_risk"]
        )
        points.append({"threshold": threshold, "summary": measured, "passed": passed})
    eligible = [p for p in points if p["passed"]]
    chosen = (
        max(eligible, key=lambda p: p["summary"]["correct_definitive_fraction"])
        if eligible
        else None
    )
    fit_groups = set(new["fit_group_ids"])
    cal = load(
        Path("/workspace/houndoracle-data/campaign-06-finetuned/calibration-finetuned.jsonl")
    )
    fitting = [
        {**r, "candidate": "YES", "expected": "YES" if r["expected"] == "MAYBE" else "NO"}
        for r in cal
        if r["group_id"] in fit_groups and r["candidate"] == "MAYBE" and not r.get("truncated")
    ]
    mapping = fit("logistic", fitting)
    gate_candidates = []
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        gate = {"mapping": mapping, "threshold": threshold}
        selected = [r for r in valid if would_abstain(r, gate)]
        mistakes = sum(r["expected"] != "MAYBE" for r in selected)
        gate_candidates.append(
            {
                "threshold": threshold,
                "avoided": len(selected),
                "false_abstentions": mistakes,
                "passed_old_validation": len(selected) / len(valid) >= 0.05
                and mistakes / len(valid) <= 0.01,
            }
        )
    allowed = [g for g in gate_candidates if g["passed_old_validation"]]
    selected_gate = max(allowed, key=lambda g: g["avoided"]) if allowed else None
    frozen = {
        "model_variant": "finetuned-logistic",
        "minimum_confidence": chosen["threshold"] if chosen else None,
        "model_config": new["model_config"],
        "validation_policy_passed": chosen is not None,
        "insufficiency_gate": {
            "mapping": mapping,
            "threshold": selected_gate["threshold"] if selected_gate else 0.99,
            "model_config": new["model_config"],
            "verification_policy_digest": policy_digest(new),
            "risk_certificate": {"qualified": False},
            "production_qualified": False,
        },
        "insufficiency_gate_screen_passed": selected_gate is not None,
        "fresh_risk_and_locked_scored": False,
        "production_qualified": False,
    }
    out.write_text(json.dumps(frozen, indent=2) + "\n")
    (root / "final-policy-validation.json").write_text(
        json.dumps(
            {
                "baseline": baseline,
                "truth_threshold_points": points,
                "gate_candidates": gate_candidates,
                "selected_truth_threshold": frozen["minimum_confidence"],
                "selected_gate": selected_gate,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "baseline": baseline,
                "truth_candidates": [
                    {k: v for k, v in p.items() if k != "summary"}
                    | {
                        "coverage": p["summary"]["definitive_coverage"],
                        "risk": p["summary"]["selective_risk"],
                    }
                    for p in points
                ],
                "selected_truth": frozen["minimum_confidence"],
                "gate_candidates": gate_candidates,
                "selected_gate": selected_gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
