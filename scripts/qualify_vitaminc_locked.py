"""Freeze exactly qualified policies before one-shot locked scoring; never reads locked predictions."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("research/experiments/campaign-08")
    target = root / "locked-authorization.json"
    if target.exists():
        raise FileExistsError("Locked authorization already frozen")
    names = ["original-mixture", "original-frozen", "fever-finetuned-frozen", "contrastive-mixture"]
    analyses = {n: json.loads((root / (n + "-analysis.json")).read_text()) for n in names}
    candidate = analyses["contrastive-mixture"]
    baselines = [v for k, v in analyses.items() if k != "contrastive-mixture"]
    raw = candidate["raw_validation"]
    calibrated = candidate["calibrated_validation_080"]
    best_raw = max(v["raw_validation"]["accuracy_all_decisions"] for v in baselines)
    best_case = max(v["raw_validation"]["complete_case_accuracy"] for v in baselines)
    best_def = max(v["calibrated_validation_080"]["correct_definitive_fraction"] for v in baselines)
    quality_gate = (
        raw["accuracy_all_decisions"] >= best_raw
        and raw["complete_case_accuracy"] >= best_case
        and calibrated["correct_definitive_fraction"] >= best_def
        and calibrated["accuracy_all_decisions"]
        >= max(v["calibrated_validation_080"]["accuracy_all_decisions"] for v in baselines)
        and calibrated["definitive_coverage"]
        >= max(v["calibrated_validation_080"]["definitive_coverage"] for v in baselines)
    )
    material = (
        raw["accuracy_all_decisions"] - best_raw >= 0.02
        or calibrated["correct_definitive_fraction"] - best_def >= 0.02
    )
    passed = bool(quality_gate and material and candidate["certificate"]["qualified"])
    bound_files = ["dataset-manifest.json", "trained-manifest.json"] + [
        n + suffix
        for n in names
        for suffix in ["-head.json", "-calibration.json", "-analysis.json"]
    ]
    result = {
        "qualified": passed,
        "quality_gate": quality_gate,
        "material_improvement": material,
        "independent_risk_gate": candidate["certificate"]["qualified"],
        "model_selected": "contrastive-mixture",
        "file_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in bound_files
        },
        "locked_predictions_seen": False,
        "production_qualified": False,
        "meaning": "Authorization to evaluate the frozen research configuration once, not permission for cloud spending or publication.",
    }
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
