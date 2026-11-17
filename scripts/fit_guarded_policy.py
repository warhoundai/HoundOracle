"""A new architecture fit on retired diagnostic data; fresh successor splits remain unopened."""

import json
from pathlib import Path

from scipy.stats import beta

from houndoracle.experiments.fit_calibration import fit
from houndoracle.experiments.guarded import GuardedCompiler, proposed
from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.nli import OnnxNLI


def main():
    root = Path("research/experiments/campaign-06")
    out = root / "guarded-frozen.json"
    if out.exists():
        raise FileExistsError("Guarded policy already frozen")
    manifest = json.loads(
        Path("research/model-comparisons/finetuned-minilm-provenance.json").read_text()
    )
    fine = HeadNLI(
        OnnxNLI(
            "/workspace/houndoracle-models/minilm-fever-1912",
            variant="fp32",
            threads=2,
            reviewed_manifest=manifest,
        ),
        json.loads((root / "finetuned-head.json").read_text()),
        "finetuned-logistic",
    )
    base = OnnxNLI("/workspace/houndoracle-models/minilm", variant="fp32", threads=2)
    model = GuardedCompiler(fine, base)
    mappings = {}
    counts = {}
    private = Path("/workspace/houndoracle-data/campaign-06-guard-fit")
    private.mkdir(exist_ok=True)
    for regime, path in [
        ("balanced", "/workspace/houndoracle-data/campaign-06-balanced/risk-extended.jsonl"),
        ("strict-pages", "/workspace/houndoracle-data/campaign-06/risk.jsonl"),
    ]:
        rows = []
        with (private / (regime + ".jsonl")).open("w") as f:
            for i, line in enumerate(Path(path).read_text().splitlines()):
                example = json.loads(line)
                r = model.predict(example["question"], example["evidence"])
                row = {
                    **r,
                    "id": example["id"],
                    "expected": example["expected"],
                    "group_id": example["group_id"],
                }
                f.write(json.dumps(row) + "\n")
                rows.append(row)
                if (i + 1) % 400 == 0:
                    print(json.dumps({"regime": regime, "compiled": i + 1}), flush=True)
        eligible = [r for r in rows if r["semantic_guard_passed"]]
        errors = sum(r["candidate"] != r["expected"] for r in eligible)
        if len({r["candidate"] == r["expected"] for r in eligible}) == 2:
            mapping = fit("logistic", eligible)
        else:
            lower = (
                float(beta.ppf(0.05, len(eligible) - errors, errors + 1))
                if eligible and not errors
                else 0.0
            )
            mapping = {"method": "isotonic", "x": [0.0, 1.0], "y": [lower, lower]}
        mappings[regime] = mapping
        counts[regime] = {"rows": len(rows), "guarded_eligible": len(eligible), "errors": errors}
    artifact = {
        "model_config": model.config,
        "regime_mappings": mappings,
        "minimum_confidence": 0.85,
        "population": "historical-fact-and-page-regime-guarded-v1",
        "certificate": {"qualified": False, "reason": "New successor risk sets not evaluated"},
        "production_qualified": False,
        "note": "Worst of two separately fitted correctness estimates; assumes calibration regimes bracket deployment. No guarantee under arbitrary shift. Semantic agreement is not source independence.",
    }
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    # Diagnostic use of retired outcomes is explicit; these are no longer held-out tests for this successor.
    stats = {}
    for regime in mappings:
        rows = [json.loads(s) for s in (private / (regime + ".jsonl")).read_text().splitlines()]
        selected = [r for r in rows if proposed(r, artifact)["decision"] in ["YES", "NO"]]
        stats[regime] = {
            "local_coverage": len(selected) / len(rows),
            "selected": len(selected),
            "errors": sum(r["candidate"] != r["expected"] for r in selected),
        }
    (root / "guarded-fit.json").write_text(
        json.dumps(
            {"fit_counts": counts, "retired_data_diagnostics": stats, "mappings": mappings},
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"fit_counts": counts, "retired_diagnostics": stats, "mappings": mappings}, indent=2
        )
    )


if __name__ == "__main__":
    main()
