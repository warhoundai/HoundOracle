"""Calibrate per-question correctness on independent component queries; exact selective risk."""

import argparse
import json
from pathlib import Path

from fit_vitaminc_evaluators import quality

from houndoracle.experiments.fit_calibration import certify, diagnostics, fit
from houndoracle.experiments.head import head_digest, infer_head
from houndoracle.experiments.pipeline import apply

SPECS = {
    "original-frozen": (
        "original",
        "research/experiments/results/trained-aggregators.json",
        "logistic",
    ),
    "fever-frozen": ("fever", "research/experiments/campaign-06/finetuned-head.json", None),
    "parent-frozen": (
        "parent",
        "research/experiments/campaign-08/contrastive-mixture-head.json",
        None,
    ),
    "parent-mixture": (
        "parent",
        "research/experiments/campaign-08/round-2/parent-mixture-head.json",
        None,
    ),
    "continued-mixture": (
        "continued",
        "research/experiments/campaign-08/round-2/continued-mixture-head.json",
        None,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=SPECS)
    args = parser.parse_args()
    name = args.variant
    model, path, key = SPECS[name]
    public = Path("research/experiments/campaign-09")
    private = Path("/workspace/houndoracle-data/campaign-09/scores") / model
    target = public / (name + "-analysis.json")
    if target.exists():
        raise FileExistsError("Do not retune a finished risk assessment")
    checkpoint = json.loads(Path(path).read_text())
    checkpoint = checkpoint[key] if key else checkpoint
    seal = json.loads((public / "dataset-manifest.json").read_text())
    raw = {}
    for split in ["validation", "calibration-fit", "calibration-risk"]:
        rows = [json.loads(s) for s in (private / (split + ".jsonl")).read_text().splitlines()]
        if len(rows) != len({r["id"] for r in rows}) or {r["id"] for r in rows} != set(
            seal["panels"][split]["ids"]
        ):
            raise ValueError("Incomplete query trace")
        if len(rows) != len({r["group_id"] for r in rows}):
            raise ValueError("Query-risk units are not independent components")
        raw[split] = [
            infer_head(r, checkpoint, evidence_group_count=r["evidence_group_count"]) for r in rows
        ]
    methods = {}
    for method in ["temperature", "logistic", "isotonic"]:
        try:
            mapping = fit(method, [r for r in raw["calibration-fit"] if not r.get("truncated")])
            methods[method] = {
                "mapping": mapping,
                "validation": diagnostics(
                    [r for r in raw["validation"] if not r.get("truncated")], mapping
                ),
            }
        except ValueError as exc:
            methods[method] = {"failure": str(exc)}
    valid = {
        k: v for k, v in methods.items() if "mapping" in v and v["validation"]["brier"] is not None
    }
    chosen = min(valid, key=lambda k: valid[k]["validation"]["brier"])
    mapping = valid[chosen]["mapping"]
    risk = [r for r in raw["calibration-risk"] if not r.get("truncated")]
    certificate = certify(mapping, risk, alpha=0.05 / 5)
    certificate["qualified"] = bool(certificate["qualified"] and certificate["selected_rows"] >= 30)
    certificate["row_risk_upper"] = 1 - certificate["group_correctness_lower_bound"]
    certificate["risk_unit"] = (
        "One query per independent component; case-any-error equals query error only for this sampling design."
    )
    certificate["row_weighted_risk_guaranteed"] = True
    certificate["assumptions"] = (
        "Only for the declared component-uniform query population, independent/exchangeable components and a fixed pre-risk selection rule. Not all-pairs row weighting, per-input truth, adversarial retrieval or arbitrary shift."
    )
    config = json.loads((private / "model.json").read_text()) | {
        "aggregation_head": name,
        "head_digest": head_digest(checkpoint),
    }
    artifact = {
        "schema_version": 1,
        "population": "vitaminc-component-uniform-query-1922",
        "model_config": config,
        "mapping": mapping,
        "certificate": certificate,
        "production_qualified": False,
        "scope": "Conditional human evidence labels on independently sampled revision queries.",
    }
    (public / (name + "-head.json")).write_text(json.dumps(checkpoint, indent=2) + "\n")
    (public / (name + "-calibration.json")).write_text(json.dumps(artifact, indent=2) + "\n")
    scored = {split: [apply(r, artifact) for r in rows] for split, rows in raw.items()}
    dest = private / name
    dest.mkdir()
    for split, rows in scored.items():
        (dest / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    raw_quality = quality([r | {"decision": r["candidate"]} for r in raw["validation"]])
    cal_quality = quality(scored["validation"])
    raw_quality["complete_case_accuracy"] = None
    cal_quality["complete_case_accuracy"] = None
    report = {
        "variant": name,
        "certificate": certificate,
        "selected_mapping": chosen,
        "methods": methods,
        "raw_validation": quality([r | {"decision": r["candidate"]} for r in raw["validation"]]),
        "calibrated_validation": cal_quality,
        "risk_calibration": diagnostics(risk, mapping),
        "locked_scored": False,
        "production_qualified": False,
        "query_population": "component-uniform one question per article/claim component; not all-case correctness",
    }
    target.write_text(json.dumps(report, indent=2) + "\n")
    records = [
        {
            k: r.get(k)
            for k in [
                "id",
                "group_id",
                "case_id",
                "expected",
                "candidate",
                "decision",
                "confidence",
                "raw_score",
                "truncated",
            ]
        }
        | {"variant": name, "split": split}
        for split, rows in scored.items()
        for r in rows
    ]
    Path("benchmarks/results/campaign-09", name + "-records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    print(
        json.dumps(
            {
                "variant": name,
                "qualified": certificate["qualified"],
                "risk_selected": certificate["selected_rows"],
                "risk_lower_correctness": certificate["group_correctness_lower_bound"],
                "raw_validation": report["raw_validation"]["accuracy_all_decisions"],
                "coverage": report["calibrated_validation"]["definitive_coverage"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
