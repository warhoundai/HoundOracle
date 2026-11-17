"""Single controlled two-evaluator architecture experiment; never opens fresh risk or locked splits."""

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
from houndoracle.experiments.pooling import POLICY, pooled_features


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    started = time.perf_counter()
    output = Path("research/experiments/campaign-06")
    private = Path("/workspace/houndoracle-data/campaign-06-pooling")
    private.mkdir(exist_ok=True)
    if (output / "pool-head.json").exists():
        raise FileExistsError("Preserve experiment; do not silently overwrite fitted candidate")
    model = OnnxNLI("/workspace/houndoracle-models/minilm", variant="fp32", threads=2)
    compiled = {}
    extra_calls = 0
    for split in ["development", "validation", "calibration"]:
        examples = {
            r["id"]: r for r in load(Path(f"/workspace/houndoracle-data/campaign-02/{split}.jsonl"))
        }
        rows = load(Path(f"/workspace/houndoracle-data/scores/{split}-fp32.jsonl"))
        compiled[split] = []
        for row in rows:
            example = examples[row["id"]]
            extra, truncated, calls = pooled_features(
                model, example["question"], example["evidence"], row
            )
            extra_calls += calls
            compiled[split].append((row | {"truncated": truncated}, extra))
        print(
            json.dumps(
                {
                    "split": split,
                    "compiled": len(rows),
                    "extra_calls": extra_calls,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    def vector(row, extra):
        return (
            (row["logits"] or [0, 0, 1])
            + [np.log1p(row.get("model_input_tokens") or 0), np.log1p(row["evidence_group_count"])]
            + extra
        )

    train = compiled["development"]
    valid = compiled["validation"]
    x = np.array([vector(*p) for p in train])
    y = np.array([{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r, _ in train])
    vx = np.array([vector(*p) for p in valid])
    vy = np.array([{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r, _ in valid])
    scaler = StandardScaler().fit(x)
    model_head = LogisticRegression(C=1, max_iter=500, random_state=1911).fit(
        scaler.transform(x), y
    )
    checkpoint = {
        "evidence_policy": POLICY,
        "max_passages": 8,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": model_head.coef_.tolist(),
        "biases": model_head.intercept_.tolist(),
        "classes": model_head.classes_.tolist(),
    }
    (output / "pool-head.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    config = model.config | {
        "aggregation_head": "passage-pool-logistic",
        "evidence_policy": POLICY,
        "max_passages": 8,
        "head_digest": head_digest(checkpoint),
    }
    (private / "model-passage-pool.json").write_text(json.dumps(config) + "\n")
    for split, rows in compiled.items():
        scored = [
            infer_head(
                row,
                checkpoint,
                evidence_group_count=row["evidence_group_count"],
                extra_features=extra,
            )
            | {"citation_supported": None}
            for row, extra in rows
        ]
        (private / f"{split}-passage-pool.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in scored)
        )
    _, report = analyze(private, output, "passage-pool")
    trace = output / "validation-calibrated-passage-pool.jsonl"
    (private / trace.name).write_bytes(trace.read_bytes())
    trace.unlink()
    baseline = json.loads(Path("research/experiments/results/analysis-logistic.json").read_text())[
        "calibrated_summary"
    ]
    candidate = report["calibrated_summary"]
    retain = (
        candidate["accuracy_all_decisions"] >= baseline["accuracy_all_decisions"]
        and candidate["definitive_coverage"] >= baseline["definitive_coverage"]
        and candidate["selective_risk"] is not None
        and candidate["selective_risk"] <= baseline["selective_risk"]
    )
    result = {
        "seed": 1911,
        "C": 1,
        "max_iter": 500,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "training_log_loss": float(log_loss(y, model_head.predict_proba(scaler.transform(x)))),
        "validation_log_loss": float(log_loss(vy, model_head.predict_proba(scaler.transform(vx)))),
        "additional_encoder_calls": extra_calls,
        "elapsed_seconds": time.perf_counter() - started,
        "baseline": baseline,
        "candidate": candidate,
        "decision": "retain-research-candidate" if retain else "reject",
        "note": "No locked/fresh scoring; archived base timings are not a new latency experiment. Source citations require independent adjudication.",
    }
    for key in ["baseline", "candidate"]:
        result[key] = result[key] | {
            "latency_seconds": None,
            "citation_supported_definitive_fraction": None,
        }
    (output / "pool-decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
