"""Calibrate the one frozen encoder/head using old partitions; no fresh or locked data."""

import json
import time
from pathlib import Path

import numpy as np
from analyze_campaign import analyze
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from houndoracle.experiments.head import head_digest, infer_head
from houndoracle.experiments.nli import OnnxNLI


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    public = Path("research/experiments/campaign-06")
    private = Path("/workspace/houndoracle-data/campaign-06-finetuned")
    private.mkdir(exist_ok=True)
    if (public / "finetuned-head.json").exists():
        raise FileExistsError("Candidate already fitted; preserve it")
    manifest = json.loads(
        Path("research/model-comparisons/finetuned-minilm-provenance.json").read_text()
    )
    model = OnnxNLI(
        "/workspace/houndoracle-models/minilm-fever-1912",
        variant="fp32",
        threads=2,
        reviewed_manifest=manifest,
    )
    scored = {}
    started = time.perf_counter()
    for split in ["development", "validation", "calibration"]:
        scored[split] = []
        for i, row in enumerate(
            load(Path(f"/workspace/houndoracle-data/campaign-02/{split}.jsonl"))
        ):
            prediction = model.predict(row["question"], row["evidence"])
            scored[split].append(
                {k: row[k] for k in ["id", "group_id", "expected", "category"]}
                | prediction
                | {
                    "latency_seconds": prediction["model_seconds"],
                    "evidence_group_count": len({e["page"] for e in row["evidence"]}),
                    "model_input_tokens": prediction.get("token_count"),
                    "citation_supported": None,
                    "all_in_cost_usd": None,
                }
            )
            if (i + 1) % 500 == 0:
                print(
                    json.dumps(
                        {"split": split, "rows": i + 1, "seconds": time.perf_counter() - started}
                    ),
                    flush=True,
                )
        (private / f"{split}-raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in scored[split])
        )

    def vector(r):
        return (r["logits"] or [0, 0, 1]) + [
            np.log1p(r.get("model_input_tokens") or 0),
            np.log1p(r["evidence_group_count"]),
        ]

    train = scored["development"]
    x = np.array([vector(r) for r in train])
    y = np.array([{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r in train])
    scaler = StandardScaler().fit(x)
    head = LogisticRegression(C=1, max_iter=500, random_state=1912).fit(scaler.transform(x), y)
    checkpoint = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": head.coef_.tolist(),
        "biases": head.intercept_.tolist(),
        "classes": head.classes_.tolist(),
    }
    (public / "finetuned-head.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    config = model.config | {
        "aggregation_head": "finetuned-logistic",
        "head_digest": head_digest(checkpoint),
    }
    (private / "model-finetuned.json").write_text(json.dumps(config) + "\n")
    for split, rows in scored.items():
        (private / f"{split}-finetuned.jsonl").write_text(
            "".join(
                json.dumps(
                    infer_head(r, checkpoint, evidence_group_count=r["evidence_group_count"])
                )
                + "\n"
                for r in rows
            )
        )
    _, report = analyze(private, public, "finetuned")
    trace = public / "validation-calibrated-finetuned.jsonl"
    (private / trace.name).write_bytes(trace.read_bytes())
    trace.unlink()
    baseline = json.loads(Path("research/experiments/results/analysis-logistic.json").read_text())[
        "calibrated_summary"
    ]
    candidate = report["calibrated_summary"]
    keep = (
        candidate["accuracy_all_decisions"] >= baseline["accuracy_all_decisions"]
        and candidate["definitive_coverage"] >= baseline["definitive_coverage"]
        and candidate["selective_risk"] is not None
        and candidate["selective_risk"] <= baseline["selective_risk"]
    )
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "decision": "retain-research-candidate" if keep else "reject-at-original-.80-policy",
        "training_log_loss": float(log_loss(y, head.predict_proba(scaler.transform(x)))),
        "C": 1,
        "seed": 1912,
        "evaluation_seconds": time.perf_counter() - started,
        "fresh_and_locked_scored": False,
    }
    for k in ["baseline", "candidate"]:
        result[k] = result[k] | {
            "latency_seconds": None,
            "citation_supported_definitive_fraction": None,
        }
    (public / "finetuned-decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
