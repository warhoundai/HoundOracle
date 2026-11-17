"""Single controlled context-head experiment, using cached encoder scores; never reads evaluation."""

import json
import time
from pathlib import Path

import numpy as np
from analyze_campaign import analyze
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.features import SCHEMA, context_features
from houndoracle.experiments.head import head_digest, infer_head


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    started = time.perf_counter()
    source = Path("/workspace/houndoracle-data/scores")
    out = Path("/workspace/houndoracle-data/campaign-04-scores")
    out.mkdir(exist_ok=True)
    data = Path("/workspace/houndoracle-data/campaign-02")
    pairs = {}
    for split in ["development", "validation", "calibration"]:
        examples = {r["id"]: r for r in load(data / f"{split}.jsonl")}
        scores = load(source / f"{split}-fp32.jsonl")
        assert len(scores) == len(examples) and len({r["id"] for r in scores}) == len(scores)
        pairs[split] = [
            (r, context_features(examples[r["id"]]["question"], examples[r["id"]]["evidence"]))
            for r in scores
        ]

    def vector(row, extra):
        return (
            (row["logits"] or [0, 0, 1])
            + [np.log1p(row.get("model_input_tokens") or 0), np.log1p(row["evidence_group_count"])]
            + extra
        )

    train = pairs["development"]
    x = np.array([vector(*pair) for pair in train])
    y = np.array([{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r, _ in train])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1, max_iter=500, random_state=1729).fit(scaler.transform(x), y)
    checkpoint = {
        "feature_schema": SCHEMA,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": model.coef_.tolist(),
        "biases": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
    }
    results = Path("research/experiments/campaign-04")
    (results / "context-head.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    config = json.loads((source / "model-fp32.json").read_text()) | {
        "aggregation_head": "context-logistic",
        "head_digest": head_digest(checkpoint),
    }
    (out / "model-context-logistic.json").write_text(json.dumps(config) + "\n")
    for split, rows in pairs.items():
        scored = []
        for row, extra in rows:
            result = infer_head(
                row,
                checkpoint,
                evidence_group_count=row["evidence_group_count"],
                extra_features=extra,
            )
            scored.append(
                result
                | {
                    "engine_config": config,
                    "engine": "context-logistic",
                    "citation_supported": None,
                }
            )
        (out / f"{split}-context-logistic.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in scored)
        )
    _, report = analyze(out, results, "context-logistic")
    private_trace = results / "validation-calibrated-context-logistic.jsonl"
    (out / private_trace.name).write_bytes(private_trace.read_bytes())
    private_trace.unlink()
    report["timing_note"] = (
        "Cached encoder timings, not newly measured end-to-end candidate latency."
    )
    (results / "analysis-context-logistic.json").write_text(json.dumps(report, indent=2) + "\n")
    baseline = json.loads(Path("research/experiments/results/analysis-logistic.json").read_text())
    b, c = baseline["calibrated_summary"], report["calibrated_summary"]
    keep = (
        c["accuracy_all_decisions"] >= b["accuracy_all_decisions"]
        and c["definitive_coverage"] >= b["definitive_coverage"]
        and c["selective_risk"] is not None
        and b["selective_risk"] is not None
        and c["selective_risk"] <= b["selective_risk"]
    )
    b = b | {"citation_supported_definitive_fraction": None, "latency_seconds": None}
    c = c | {"citation_supported_definitive_fraction": None, "latency_seconds": None}
    result = {
        "hypothesis": "Context features improve the calibrated evidence head.",
        "seed": 1729,
        "C": 1,
        "max_iter": 500,
        "training_rows": len(train),
        "validation_rows": len(pairs["validation"]),
        "baseline": b,
        "candidate": c,
        "decision": "retain-research-candidate" if keep else "reject",
        "training_and_analysis_seconds": time.perf_counter() - started,
        "new_cloud_cost_usd": 0,
        "limitations": "Validation selection only; no locked/fresh evaluation. Existing baseline at .80 matched by candidate .80; production remains unqualified.",
    }
    (results / "context-head-decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
