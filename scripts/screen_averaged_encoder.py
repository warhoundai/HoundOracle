"""Equally supervised head for the fixed averaged encoder; development only."""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import summary
from houndoracle.experiments.nli import OnnxNLI


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    public = Path("research/experiments/campaign-16")
    private = Path("/workspace/houndoracle-data/campaign-16")
    if private.exists():
        raise FileExistsError("Preserve all completed/partial score traces")
    private.mkdir()
    manifest = json.loads((public / "averaged-manifest.json").read_text())
    model = OnnxNLI(
        "/workspace/houndoracle-models/minilm-vitc-average-1932",
        variant="fp32",
        threads=1,
        max_length=256,
        reviewed_manifest=manifest,
    )
    (private / "model.json").write_text(json.dumps(model.config, indent=2) + "\n")
    predictions = {}
    for split, path in [
        ("development", "/workspace/houndoracle-data/campaign-08-round2/development.jsonl"),
        ("validation", "/workspace/houndoracle-data/campaign-09/validation.jsonl"),
    ]:
        rows = load(path)
        output = []
        with (private / (split + ".jsonl")).open("w") as target:
            for i, row in enumerate(rows):
                result = model.predict(row["question"], row["evidence"])
                record = (
                    {k: row[k] for k in ["id", "group_id", "expected"]}
                    | result
                    | {"evidence_group_count": len({e["page"] for e in row["evidence"]})}
                )
                output.append(record)
                target.write(json.dumps(record) + "\n")
                if (i + 1) % 400 == 0:
                    print(json.dumps({"split": split, "rows": i + 1}), flush=True)
        predictions[split] = output
    if {r["group_id"] for r in predictions["development"]} & {
        r["group_id"] for r in predictions["validation"]
    }:
        raise ValueError("Training/validation leakage")
    train = predictions["development"]
    x = np.array(
        [
            (r["logits"] or [0, 0, 1])
            + [np.log1p(r.get("token_count") or 0), np.log1p(r["evidence_group_count"])]
            for r in train
        ]
    )
    y = [{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r in train]
    scaler = StandardScaler().fit(x)
    head = LogisticRegression(C=1, max_iter=500, random_state=1932).fit(scaler.transform(x), y)
    checkpoint = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": head.coef_.tolist(),
        "biases": head.intercept_.tolist(),
        "classes": head.classes_.tolist(),
    }
    (public / "averaged-head.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    scored = [
        infer_head(r, checkpoint, evidence_group_count=r["evidence_group_count"])
        for r in predictions["validation"]
    ]
    rows = [r | {"decision": r["candidate"], "latency_seconds": None} for r in scored]
    metrics = summary(rows)
    baseline = json.loads(Path("research/experiments/campaign-15/screen.json").read_text())[
        "baseline"
    ]
    delta = metrics["accuracy_all_decisions"] - baseline["accuracy_all_decisions"]
    passed = (
        delta >= 0.02 and metrics["definitive_coverage"] >= baseline["definitive_coverage"] - 0.02
    )
    result = {
        "raw_validation": metrics,
        "baseline_raw_validation": baseline,
        "raw_accuracy_delta": delta,
        "screen_passed": passed,
        "locked_scored": False,
        "decision": "qualify-fresh-risk" if passed else "reject-before-fresh-risk",
        "scope": "Fixed checkpoint average; reused validation screen only. No independent gain, live retrieval or Gemini claim.",
    }
    (public / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    safe = [
        {k: r.get(k) for k in ["id", "group_id", "expected", "candidate", "raw_score", "truncated"]}
        for r in scored
    ]
    Path("benchmarks/results/campaign-16/development-records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in safe)
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
