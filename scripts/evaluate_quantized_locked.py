"""Fresh paired noninferiority test of the preselected quantized policy versus FP32."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from quantization_common import campaign_index
from scipy.stats import beta

from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import summary
from houndoracle.experiments.paired_risk import paired_improvement_lower
from houndoracle.experiments.pipeline import apply


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    number = campaign_index(args)
    public = Path(f"research/experiments/campaign-{number:02d}")
    safe = Path(f"benchmarks/results/campaign-{number:02d}")
    target = safe / "locked-summary.json"
    if target.exists():
        raise FileExistsError("Locked quantization study already evaluated")
    gate = json.loads((public / "locked-authorization.json").read_text())
    selected = gate["model_selected"]
    if gate.get("qualified") is not True or selected == "fp32":
        raise PermissionError("No quantized improvement qualified; preserve locked data")
    for f, sha in gate["file_sha256"].items():
        if hashlib.sha256((public / f).read_bytes()).hexdigest() != sha:
            raise ValueError("Policy changed after selection")
    expected = set(
        json.loads((public / "dataset-manifest.json").read_text())["panels"]["locked"]["ids"]
    )
    systems = {}
    rows_by = {}
    records = []
    for name in ["fp32", selected]:
        private = Path(f"/workspace/houndoracle-data/campaign-{number:02d}/scores") / name
        raw = [json.loads(line) for line in (private / "locked.jsonl").read_text().splitlines()]
        if (
            len(raw) != len(expected)
            or {r["id"] for r in raw} != expected
            or len({r["group_id"] for r in raw}) != len(raw)
        ):
            raise ValueError("Incomplete or dependent query test")
        head = json.loads((public / (name + "-head.json")).read_text())
        artifact = json.loads((public / (name + "-calibration.json")).read_text())
        if any(
            artifact["model_config"][k] != v
            for k, v in json.loads((private / "model.json").read_text()).items()
        ):
            raise ValueError("Encoder/calibration mismatch")
        predictions = [
            infer_head(r, head, evidence_group_count=r["evidence_group_count"]) for r in raw
        ]
        cal = [apply(r, artifact) for r in predictions]
        raw_rows = [r | {"decision": r["candidate"]} for r in predictions]
        a, b = summary(raw_rows), summary(cal)
        a["latency_seconds"] = None
        b["latency_seconds"] = None
        accepted = [r for r in cal if r["decision"] in ["YES", "NO"]]
        errors = sum(r["decision"] != r["expected"] for r in accepted)
        upper = (
            float(beta.ppf(1 - 0.05 / 2, errors + 1, len(accepted) - errors))
            if accepted and errors < len(accepted)
            else 1.0
        )
        systems[name] = {
            "raw": a,
            "calibrated": b,
            "selected_errors": errors,
            "selected_answers": len(accepted),
            "selective_error_family_upper": upper,
        }
        rows_by[name] = {
            "raw": {r["id"]: r for r in raw_rows},
            "calibrated": {r["id"]: r for r in cal},
        }
        records.extend(
            {
                k: r.get(k)
                for k in [
                    "id",
                    "group_id",
                    "expected",
                    "candidate",
                    "decision",
                    "confidence",
                    "raw_score",
                    "truncated",
                ]
            }
            | {"variant": name}
            for r in cal
        )
    keys = sorted(expected)
    b = rows_by["fp32"]
    q = rows_by[selected]
    n = len(keys)
    rng = np.random.default_rng(1925)
    idx = rng.integers(0, n, size=(10000, n))
    accuracy = np.array(
        [
            int(q["raw"][k]["decision"] == q["raw"][k]["expected"])
            - int(b["raw"][k]["decision"] == b["raw"][k]["expected"])
            for k in keys
        ]
    )
    coverage = np.array(
        [
            int(q["calibrated"][k]["decision"] in ["YES", "NO"])
            - int(b["calibrated"][k]["decision"] in ["YES", "NO"])
            for k in keys
        ]
    )
    acc = accuracy[idx].mean(axis=1)
    cov = coverage[idx].mean(axis=1)
    lower_a = paired_improvement_lower(accuracy)
    lower_k = paired_improvement_lower(coverage)
    ni = bool(lower_a >= -0.01 and lower_k >= -0.02)
    risk = systems[selected]["selective_error_family_upper"] <= 0.2
    output = {
        "selected_configuration": selected,
        "systems": systems,
        "accuracy_delta": float(accuracy.mean()),
        "accuracy_delta_95_paired_interval": np.quantile(acc, [0.025, 0.975]).tolist(),
        "accuracy_one_sided_95_lower": lower_a,
        "coverage_delta": float(coverage.mean()),
        "coverage_delta_95_paired_interval": np.quantile(cov, [0.025, 0.975]).tolist(),
        "coverage_one_sided_95_lower": lower_k,
        "noninferiority_passed": ni,
        "risk_passed": risk,
        "conditional_candidate_validated": bool(ni and risk),
        "production_qualified": False,
        "interpretation": "Margins epsilon=.01 raw accuracy and delta=.02 coverage fixed before test. Both one-sided conditions required (intersection-union). Gate uses conservative exact marginal discordance bounds with a union bound, not a degenerate zero-discordance bootstrap. Bootstrap intervals remain descriptive. No Gemini, live retrieval or general-world truth claim.",
    }
    target.write_text(json.dumps(output, indent=2) + "\n")
    (safe / "locked-records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
