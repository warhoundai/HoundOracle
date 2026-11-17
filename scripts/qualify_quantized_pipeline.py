"""Select a qualified smaller/faster variant before its fresh locked test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from quantization_common import campaign_index, variant_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    number = campaign_index(args)
    root = Path(f"research/experiments/campaign-{number:02d}")
    safe = Path(f"benchmarks/results/campaign-{number:02d}")
    target = root / "locked-authorization.json"
    if target.exists():
        raise FileExistsError("Quantized selection already frozen")
    names = variant_names(args)
    reports = {n: json.loads((root / (n + "-analysis.json")).read_text()) for n in names}
    profiles = {n: json.loads((safe / (n + "-profile.json")).read_text()) for n in names}
    eligible = ["fp32"] if reports["fp32"]["certificate"]["qualified"] else []
    comparisons = {}
    base = profiles["fp32"]

    def matrix(name):
        rows = [
            json.loads(s)
            for s in (safe / (name + "-profile-records.jsonl")).read_text().splitlines()
        ]
        by = {}
        for r in rows:
            by.setdefault(r["id"], {})[r["repeat"]] = r["seconds"]
        return {id: [r[i] for i in range(3)] for id, r in by.items()}

    b = matrix("fp32")
    keys = sorted(b)
    bm = np.array([b[k] for k in keys])
    rng = np.random.default_rng(1923)
    idx = rng.integers(0, len(keys), size=(3000, len(keys)))
    for name in names[1:]:
        q = matrix(name)
        if q.keys() != b.keys():
            raise ValueError("Unpaired performance workload")
        qm = np.array([q[k] for k in keys])
        bounds = {}
        for quantile, label in [(0.5, "p50"), (0.95, "p95")]:
            reduction = 1 - np.quantile(qm[idx], quantile, axis=(1, 2)) / np.quantile(
                bm[idx], quantile, axis=(1, 2)
            )
            bounds[label + "_reduction_95_question_bootstrap_interval"] = np.quantile(
                reduction, [0.025, 0.975]
            ).tolist()
        p = profiles[name]
        p50 = 1 - p["p50_seconds"] / base["p50_seconds"]
        p95 = 1 - p["p95_seconds"] / base["p95_seconds"]
        size = 1 - p["model_bytes"] / base["model_bytes"]
        passed = bool(
            reports[name]["certificate"]["qualified"]
            and reports[name]["validation_quality_passed"]
            and p50 >= 0.2
            and p95 >= 0.2
            and size >= 0.3
            and all(v[0] > 0 for v in bounds.values())
        )
        comparisons[name] = {
            "p50_reduction": p50,
            "p95_reduction": p95,
            "file_size_reduction": size,
            "passed": passed,
            **bounds,
        }
        if passed:
            eligible.append(name)
    selected = min(eligible, key=lambda n: profiles[n]["model_bytes"]) if eligible else None
    files = (
        ["dataset-manifest.json"]
        + [
            n + suffix
            for n in names
            for suffix in ["-analysis.json", "-head.json", "-calibration.json"]
        ]
        + [n + "-manifest.json" for n in names[1:]]
    )
    output = {
        "qualified": selected is not None,
        "model_selected": selected,
        "comparisons": comparisons,
        "file_sha256": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
        "locked_predictions_seen": False,
        "production_qualified": False,
        "scope": "Local one-thread conditional evidence path; bootstrap captures workload sampling, not all host/environment variability. No end-to-end network/Gemini claim.",
    }
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
