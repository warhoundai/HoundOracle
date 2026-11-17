"""Freeze a preselected parent/continued research policy before any locked predictions."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("research/experiments/campaign-08/round-2")
    target = root / "locked-authorization.json"
    if target.exists():
        raise FileExistsError("Selection already frozen")
    names = ["parent-frozen", "parent-mixture", "continued-mixture"]
    reports = {n: json.loads((root / (n + "-analysis.json")).read_text()) for n in names}

    def dominates(candidate, references):
        raw = candidate["raw_validation"]
        cal = candidate["calibrated_validation_080"]
        good = (
            raw["accuracy_all_decisions"]
            >= max(r["raw_validation"]["accuracy_all_decisions"] for r in references)
            and raw["complete_case_accuracy"]
            >= max(r["raw_validation"]["complete_case_accuracy"] for r in references)
            and all(
                cal[k] >= max(r["calibrated_validation_080"][k] for r in references)
                for k in [
                    "accuracy_all_decisions",
                    "correct_definitive_fraction",
                    "definitive_coverage",
                ]
            )
        )
        material = (
            raw["accuracy_all_decisions"]
            - max(r["raw_validation"]["accuracy_all_decisions"] for r in references)
            >= 0.02
            or cal["correct_definitive_fraction"]
            - max(r["calibrated_validation_080"]["correct_definitive_fraction"] for r in references)
            >= 0.02
        )
        return bool(good and material)

    eligible = []
    decisions = {}
    for name in names:
        report = reports[name]
        qualifies = report["certificate"]["qualified"] is True
        improvement = (
            True
            if name == "parent-frozen"
            else dominates(
                report,
                [reports["parent-frozen"]]
                if name == "parent-mixture"
                else [reports["parent-frozen"], reports["parent-mixture"]],
            )
        )
        decisions[name] = {
            "risk_qualified": qualifies,
            "quality_and_material_improvement": improvement,
        }
        if qualifies and improvement:
            eligible.append(name)
    selected = (
        max(
            eligible,
            key=lambda n: (
                reports[n]["calibrated_validation_080"]["correct_definitive_fraction"],
                reports[n]["raw_validation"]["complete_case_accuracy"],
                reports[n]["raw_validation"]["accuracy_all_decisions"],
            ),
        )
        if eligible
        else None
    )
    files = ["dataset-manifest.json", "dataset-manifest-extended.json", "trained-manifest.json"] + [
        n + suffix
        for n in names
        for suffix in ["-head.json", "-calibration.json", "-analysis.json"]
    ]
    output = {
        "qualified": selected is not None,
        "model_selected": selected,
        "decisions": decisions,
        "locked_predictions_seen": False,
        "production_qualified": False,
        "file_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in files},
        "scope": "One-shot conditional revision-evidence research test; no cloud expenditure, public release or general-world truth authorization.",
    }
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
