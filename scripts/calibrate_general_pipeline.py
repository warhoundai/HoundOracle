"""Freeze primary map before validation; report other methods without selecting on risk."""

import argparse
import json
import time
from pathlib import Path

from houndoracle.calibration import GeneralCalibration
from houndoracle.experiments.fit_calibration import certify, diagnostics, fit, predict
from houndoracle.experiments.metrics import summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "validation"])
    args = parser.parse_args()
    public = Path("research/experiments/campaign-47")
    private = Path("/workspace/houndoracle-data/campaign-47")
    if args.stage == "fit":
        if (public / "frozen-map.json").exists():
            raise FileExistsError("Preserve frozen calibration policy")
        rows = [
            json.loads(s) for s in (private / "calibration-scores.jsonl").read_text().splitlines()
        ]
        if len(rows) != 1000:
            raise ValueError("Incomplete calibration observations")
        fitting, risk = rows[:500], rows[500:]
        if {r["group_id"] for r in fitting} & {r["group_id"] for r in risk}:
            raise ValueError("Calibrationfit/risk overlap")
        runtime = json.loads((private / "calibration-runtime.json").read_text())
        reports = {}
        maps = {}
        for method in ["temperature", "logistic", "isotonic"]:
            mapping = fit(method, [r for r in fitting if not r["failure"]])
            maps[method] = mapping
            certificate = certify(mapping, risk, threshold=0.8, alpha=0.05 / 3)
            reports[method] = {
                "certificate": certificate,
                "raw_diagnostics": diagnostics(risk),
                "mapped_diagnostics": diagnostics(risk, mapping),
            }
        primary = reports["logistic"]["certificate"]
        passed = primary["qualified"] and primary["selected_groups"] >= 30
        frozen = {
            "policy_id": runtime["policy_id"],
            "maps": maps,
            "primary_method": "logistic",
            "threshold": 0.8,
            "primary_research_risk_passed": passed,
            "source_audits_pending": True,
            "production_qualified": False,
        }
        (public / "frozen-map.json").write_text(json.dumps(frozen, indent=2) + "\n")
        (public / "risk-assessment.json").write_text(json.dumps(reports, indent=2) + "\n")
        if primary["selected_groups"] >= 30:
            from houndoracle.experiments.qualification95 import digest

            artifact = GeneralCalibration(
                policy_id=runtime["policy_id"],
                population="Historical supplied VitaminC real revision components only; independent sourceaudit pending",
                supported_kinds=["prose", "not"],
                expires_at=time.time() + 30 * 86400,
                dataset_digest=digest([r["id"] for r in rows]),
                weight=maps["logistic"]["weight"],
                bias=maps["logistic"]["bias"],
                risk_lower_bound=primary["group_correctness_lower_bound"],
                risk_groups=primary["selected_groups"],
                qualification_status="research",
            )
            (public / "research-calibration.json").write_text(
                artifact.model_dump_json(indent=2) + "\n"
            )
        print(
            json.dumps(
                {
                    "primary_research_risk_passed": passed,
                    "certificate": primary,
                    "production_qualified": False,
                }
            )
        )
    else:
        frozen = json.loads((public / "frozen-map.json").read_text())
        if not frozen["primary_research_risk_passed"]:
            raise PermissionError("Risk failed")
        target = public / "validation-assessment.json"
        if target.exists():
            raise FileExistsError("Preserve validation result")
        rows = [
            json.loads(s) for s in (private / "validation-scores.jsonl").read_text().splitlines()
        ]
        if len(rows) != 500:
            raise ValueError("Incomplete validation observations")
        mapping = frozen["maps"]["logistic"]
        selected = []
        for r in rows:
            probability = (
                predict(mapping, r["raw_score"]) if r["candidate"] in ["YES", "NO"] else None
            )
            selected.append(
                r
                | {
                    "decision": r["candidate"]
                    if probability is not None and probability >= 0.8 and not r["failure"]
                    else "MAYBE",
                    "research_estimated_correctness": probability,
                }
            )
        result = {
            "raw_candidates": summary([r | {"decision": r["candidate"]} for r in rows]),
            "research_selected80": summary(selected),
            "accepted_API": summary(rows),
            "calibration": diagnostics(rows, mapping),
            "production_qualified": False,
            "qualification95_passed": False,
            "note": "500caseprovisional-source validation diagnostics. Not either1000case acceptance gate; API remains research-unqualified. No locked cases read.",
        }
        target.write_text(json.dumps(result, indent=2) + "\n")
        out = Path("benchmarks/results/campaign-47")
        out.mkdir(parents=True, exist_ok=True)
        (out / "validation-metadata.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in selected)
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
