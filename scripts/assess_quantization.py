"""Controlled numeric change: frozen head/map, fresh independent risk, no fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
from quantization_common import campaign_index, variant_names

from houndoracle.experiments.fit_calibration import certify
from houndoracle.experiments.head import head_digest, infer_head
from houndoracle.experiments.metrics import summary
from houndoracle.experiments.pipeline import apply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    number = campaign_index(args)
    names = variant_names(args)
    public = Path(f"research/experiments/campaign-{number:02d}")
    root = Path(f"/workspace/houndoracle-data/campaign-{number:02d}/scores")
    seal = json.loads((public / "dataset-manifest.json").read_text())
    parent = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-calibration.json").read_text()
    )
    head = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
    )
    mapping = parent["mapping"]
    scores = {}
    reports = {}
    for name in names:
        target = public / (name + "-analysis.json")
        if target.exists():
            raise FileExistsError("Do not retune completed quantization risk")
        scored = {}
        for split in ["validation", "calibration-risk"]:
            rows = [
                json.loads(s) for s in (root / name / (split + ".jsonl")).read_text().splitlines()
            ]
            if (
                len(rows) != len({r["id"] for r in rows})
                or {r["id"] for r in rows} != set(seal["panels"][split]["ids"])
                or len({r["group_id"] for r in rows}) != len(rows)
            ):
                raise ValueError("Incomplete/nonindependent quantization trace")
            scored[split] = [
                infer_head(r, head, evidence_group_count=r["evidence_group_count"]) for r in rows
            ]
        risk = [r for r in scored["calibration-risk"] if not r.get("truncated")]
        certificate = certify(mapping, risk, alpha=0.05 / len(names))
        certificate["qualified"] = bool(
            certificate["qualified"] and certificate["selected_rows"] >= 30
        )
        certificate["row_risk_upper"] = 1 - certificate["group_correctness_lower_bound"]
        certificate["risk_unit"] = (
            "One independent query per component; not all-pair case correctness"
        )
        config = json.loads((root / name / "model.json").read_text()) | {
            "aggregation_head": name,
            "head_digest": head_digest(head),
        }
        artifact = parent | {
            "model_config": config,
            "certificate": certificate,
            "population": f"vitaminc-component-uniform-query-quantization-{number}",
            "production_qualified": False,
        }
        (public / (name + "-calibration.json")).write_text(json.dumps(artifact, indent=2) + "\n")
        (public / (name + "-head.json")).write_text(json.dumps(head, indent=2) + "\n")
        val = [apply(r, artifact) for r in scored["validation"]]
        raw = [r | {"decision": r["candidate"]} for r in scored["validation"]]
        a, b = summary(raw), summary(val)
        a["latency_seconds"] = None
        b["latency_seconds"] = None
        reports[name] = {
            "variant": name,
            "raw_validation": a,
            "calibrated_validation": b,
            "certificate": certificate,
            "locked_scored": False,
        }
        scores[name] = scored["validation"]
    base = {r["id"]: r for r in scores["fp32"]}
    for name, report in reports.items():
        same = np.mean([r["candidate"] == base[r["id"]]["candidate"] for r in scores[name]])
        report["label_agreement_fp32"] = float(same)
        report["validation_quality_passed"] = bool(
            same >= 0.98
            and report["raw_validation"]["accuracy_all_decisions"]
            >= reports["fp32"]["raw_validation"]["accuracy_all_decisions"] - 0.01
            and report["calibrated_validation"]["definitive_coverage"]
            >= reports["fp32"]["calibrated_validation"]["definitive_coverage"] - 0.02
        )
        (public / (name + "-analysis.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                n: {
                    "risk_qualified": r["certificate"]["qualified"],
                    "agreement": r["label_agreement_fp32"],
                    "quality_passed": r["validation_quality_passed"],
                }
                for n, r in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
