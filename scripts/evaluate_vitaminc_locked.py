"""One-shot frozen semantic subsystem evaluation; no fitting or threshold selection here."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from fit_vitaminc_evaluators import quality
from scipy.stats import beta

from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import cluster_interval
from houndoracle.experiments.pipeline import apply


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round2", action="store_true")
    args = parser.parse_args()
    root = Path("research/experiments/campaign-08")
    safe = Path("benchmarks/results/campaign-08")
    if args.round2:
        root = root / "round-2"
        safe = safe / "round-2"
    target = safe / "locked-summary.json"
    if target.exists():
        raise FileExistsError("Locked test already evaluated; no repeats/tuning")
    gate = json.loads((root / "locked-authorization.json").read_text())
    if gate.get("qualified") is not True:
        raise PermissionError("Candidate did not qualify")
    for name, sha in gate["file_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen policy changed")
    expected = set(
        json.loads((root / "dataset-manifest.json").read_text())["panels"]["locked"]["ids"]
    )
    specs = {
        "original-mixture": "original",
        "original-frozen": "original",
        "fever-finetuned-frozen": "fever-finetuned",
        "contrastive-mixture": "contrastive",
    }
    if args.round2:
        specs = {
            "parent-frozen": "contrastive",
            "parent-mixture": "contrastive",
            "continued-mixture": "continued",
        }
    selected_name = gate["model_selected"] if args.round2 else "contrastive-mixture"
    all_rows = {}
    reports = {}
    records = []
    for name, model in specs.items():
        path = Path(
            "/workspace/houndoracle-data/campaign-08-round2/scores"
            if args.round2
            else "/workspace/houndoracle-data/campaign-08/scores",
            model,
        )
        raw = load(path / "locked.jsonl")
        if len(raw) != len(expected) or {r["id"] for r in raw} != expected:
            raise ValueError("Incomplete locked trace")
        head = json.loads((root / (name + "-head.json")).read_text())
        artifact = json.loads((root / (name + "-calibration.json")).read_text())
        config = json.loads((path / "model.json").read_text())
        if any(artifact["model_config"][k] != v for k, v in config.items()):
            raise ValueError("Calibration belongs to a different encoder configuration")
        predictions = [
            infer_head(r, head, evidence_group_count=r["evidence_group_count"]) for r in raw
        ]
        accepted = [apply(r, artifact) for r in predictions]
        for r in accepted:
            if r["decision"] in ["YES", "NO"]:
                assert (
                    not r.get("truncated") and r["confidence"] >= 0.8 and r["calibration_qualified"]
                )
        at85 = [
            r | {"decision": r["decision"] if (r.get("confidence") or 0) >= 0.85 else "MAYBE"}
            for r in accepted
        ]
        all_rows[name] = {
            "raw": [r | {"decision": r["candidate"]} for r in predictions],
            "calibrated": accepted,
        }
        cases = {}
        for r in all_rows[name]["raw"]:
            cases.setdefault(r["group_id"], []).append(r)
        k = sum(all(r["decision"] == r["expected"] for r in rows) for rows in cases.values())
        n = len(cases)
        reports[name] = {
            "raw": quality(all_rows[name]["raw"]),
            "calibrated080": quality(accepted),
            "calibrated085": quality(at85),
            "complete_case_correct": k,
            "complete_cases": n,
            "complete_case_accuracy_95_exact_interval": [
                float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0,
                float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0,
            ],
            "calibrated_group_bootstrap_intervals": {
                metric: cluster_interval(accepted, metric, seed=1920)
                for metric in ["accuracy_all_decisions", "definitive_coverage", "selective_risk"]
            },
            "production_qualified": False,
        }
        records.extend(
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
            | {"variant": name}
            for r in accepted
        )
    comparisons = {}
    candidate = all_rows[selected_name]
    for name, base in all_rows.items():
        if name == selected_name:
            continue
        comparison = {}
        for policy in ["raw", "calibrated"]:
            b = {r["id"]: r for r in base[policy]}
            c = {r["id"]: r for r in candidate[policy]}
            groups = {}
            for id, r in b.items():
                groups.setdefault(r["group_id"], []).append(id)
            keys = sorted(groups)
            rng = random.Random(1920)
            acc = []
            correct = []
            for _ in range(2000):
                ids = [id for g in rng.choices(keys, k=len(keys)) for id in groups[g]]
                acc.append(
                    sum(
                        (c[id]["decision"] == c[id]["expected"])
                        - (b[id]["decision"] == b[id]["expected"])
                        for id in ids
                    )
                    / len(ids)
                )
                correct.append(
                    sum(
                        (
                            c[id]["decision"] == c[id]["expected"]
                            and c[id]["decision"] in ["YES", "NO"]
                        )
                        - (
                            b[id]["decision"] == b[id]["expected"]
                            and b[id]["decision"] in ["YES", "NO"]
                        )
                        for id in ids
                    )
                    / len(ids)
                )
            comparison[policy] = {
                "accuracy_delta_95_cluster_interval": np.quantile(acc, [0.025, 0.975]).tolist(),
                "correct_definitive_fraction_delta_95_cluster_interval": np.quantile(
                    correct, [0.025, 0.975]
                ).tolist(),
            }
        comparisons[name] = comparison
    output = {
        "selected_configuration": selected_name,
        "systems": reports,
        "paired_comparisons": comparisons,
        "scope": "Conditional human-labeled semantic subsystem test on fixed revision evidence. No live retrieval or Gemini workflow, all-in cost, source freshness or independent citation review measured.",
        "latency_comparison": None,
        "gemini_comparison": None,
        "all_in_cost": None,
    }
    target.write_text(json.dumps(output, indent=2) + "\n")
    (safe / "locked-records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
