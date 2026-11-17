"""One prespecified encoder/head challenger; frozen live Gemini campaign is untouched."""

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
    started = time.perf_counter()
    data = Path("/workspace/houndoracle-data/campaign-02")
    private = Path("/workspace/houndoracle-data/campaign-04-deberta")
    private.mkdir(exist_ok=True)
    public = Path("research/experiments/campaign-04")
    model = OnnxNLI("/workspace/houndoracle-models/deberta-small", variant="int8", threads=2)
    valid = load(data / "validation.jsonl")
    subset = [
        r
        for label in ["YES", "NO", "MAYBE"]
        for r in [x for x in valid if x["expected"] == label][:40]
    ]
    old = {
        r["id"]: r for r in load(Path("/workspace/houndoracle-data/scores/validation-fp32.jsonl"))
    }
    cached = {r["id"]: model.predict(r["question"], r["evidence"]) for r in subset}
    base_accuracy = sum(old[r["id"]]["candidate"] == r["expected"] for r in subset) / len(subset)
    accuracy = sum(cached[r["id"]]["candidate"] == r["expected"] for r in subset) / len(subset)
    screen = {
        "rows": len(subset),
        "baseline_raw_accuracy": base_accuracy,
        "candidate_raw_accuracy": accuracy,
        "seconds": time.perf_counter() - started,
        "decision": "continue" if accuracy >= base_accuracy else "reject",
    }
    (public / "deberta-screen.json").write_text(json.dumps(screen, indent=2) + "\n")
    print(json.dumps(screen), flush=True)
    if accuracy < base_accuracy:
        return
    scores = {}
    for split in ["development", "validation", "calibration"]:
        scored = []
        for i, row in enumerate(load(data / f"{split}.jsonl")):
            pred = cached.get(row["id"]) or model.predict(row["question"], row["evidence"])
            result = {k: row[k] for k in ["id", "group_id", "expected", "category"]} | pred
            result.update(
                latency_seconds=pred["model_seconds"],
                evidence_group_count=len({e["page"] for e in row["evidence"]}),
                model_input_tokens=pred.get("token_count"),
                citation_supported=None,
                all_in_cost_usd=None,
            )
            scored.append(result)
            if (i + 1) % 400 == 0:
                print(
                    json.dumps(
                        {"split": split, "rows": i + 1, "seconds": time.perf_counter() - started}
                    ),
                    flush=True,
                )
        scores[split] = scored
        (private / f"{split}-encoder.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in scored)
        )

    def vector(r):
        return (r["logits"] or [0, 0, 1]) + [
            np.log1p(r.get("model_input_tokens") or 0),
            np.log1p(r["evidence_group_count"]),
        ]

    train = scores["development"]
    x = np.array([vector(r) for r in train])
    y = np.array([{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r in train])
    scaler = StandardScaler().fit(x)
    head = LogisticRegression(C=1, max_iter=500, random_state=1729).fit(scaler.transform(x), y)
    checkpoint = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": head.coef_.tolist(),
        "biases": head.intercept_.tolist(),
        "classes": head.classes_.tolist(),
    }
    (public / "deberta-head.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    config = model.config | {
        "aggregation_head": "deberta-logistic",
        "head_digest": head_digest(checkpoint),
    }
    (private / "model-deberta-logistic.json").write_text(json.dumps(config) + "\n")
    for split, rows in scores.items():
        (private / f"{split}-deberta-logistic.jsonl").write_text(
            "".join(
                json.dumps(
                    infer_head(r, checkpoint, evidence_group_count=r["evidence_group_count"])
                )
                + "\n"
                for r in rows
            )
        )
    _, report = analyze(private, public, "deberta-logistic")
    trace = public / "validation-calibrated-deberta-logistic.jsonl"
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
        "decision": "retain-research-only" if keep else "reject",
        "baseline": baseline,
        "candidate": candidate,
        "seed": 1729,
        "C": 1,
        "training_rows": len(train),
        "model_config": model.config,
        "training_log_loss": float(log_loss(y, head.predict_proba(scaler.transform(x)))),
        "seconds": time.perf_counter() - started,
        "new_cloud_cost_usd": 0,
        "limitations": [
            "Validation selection, not independent superiority. No locked/fresh scoring or Gemini comparison.",
            "Timing is encoder trace only and host/thread settings differ; no causal speed comparison with archived MiniLM.",
            "No claim of independent citation adjudication or production qualification.",
        ],
    }
    for key in ["baseline", "candidate"]:
        result[key] = result[key] | {
            "citation_supported_definitive_fraction": None,
            "latency_seconds": None,
        }
    (public / "deberta-decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
