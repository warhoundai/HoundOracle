"""Paired CPU threading experiment; no provider calls or locked evaluation."""

import json
import random
from pathlib import Path

import numpy as np

from houndoracle.experiments.head import HeadNLI
from houndoracle.experiments.metrics import paired_comparison, summary
from houndoracle.experiments.nli import OnnxNLI
from houndoracle.experiments.pipeline import apply


def main():
    rows = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/campaign-02/validation.jsonl")
        .read_text()
        .splitlines()
    ]
    subset = [
        r
        for label in ["YES", "NO", "MAYBE"]
        for r in [x for x in rows if x["expected"] == label][:40]
    ]
    head = json.loads(Path("research/experiments/results/trained-aggregators.json").read_text())[
        "logistic"
    ]
    artifact = json.loads(
        Path("research/experiments/results/calibration-logistic.json").read_text()
    )
    models = {
        str(t): HeadNLI(
            OnnxNLI("/workspace/houndoracle-models/minilm", variant="fp32", threads=t),
            head,
            "logistic",
        )
        for t in [1, 2]
    }
    records = {t: [] for t in models}
    delta = 0.0
    drift = 0
    rng = random.Random(1908)
    for repeat in range(3):
        rng.shuffle(subset)
        for row in subset:
            order = list(models)
            rng.shuffle(order)
            outputs = {}
            for t in order:
                result = models[t].predict(row["question"], row["evidence"])
                outputs[t] = apply(result, artifact)
                decision = outputs[t]["decision"]
                if decision in ["YES", "NO"] and (outputs[t]["confidence"] or 0) < 0.85:
                    decision = "MAYBE"
                records[t].append(
                    {
                        "id": row["id"] + ":" + str(repeat),
                        "group_id": row["group_id"],
                        "expected": row["expected"],
                        "decision": decision,
                        "latency_seconds": result["model_seconds"],
                        "all_in_cost_usd": None,
                    }
                )
            delta = max(
                delta,
                float(np.max(np.abs(np.array(outputs["1"]["logits"]) - outputs["2"]["logits"]))),
            )
            drift += outputs["1"]["decision"] != outputs["2"]["decision"]
        print(json.dumps({"repeat": repeat}), flush=True)
    summaries = {t: summary(r) for t, r in records.items()}
    reduction = (
        1 - summaries["2"]["latency_seconds"]["p95"] / summaries["1"]["latency_seconds"]["p95"]
    )
    comparison = paired_comparison(records["1"], records["2"], replicates=1000)
    result = {
        "questions": len(subset),
        "repetitions": 3,
        "summaries": summaries,
        "paired": comparison,
        "maximum_aggregate_logit_absolute_difference": delta,
        "calibrated_decision_drift": int(drift),
        "observed_p95_reduction": reduction,
        "decision": "retain-runtime-experiment-only"
        if reduction >= 0.2 and not drift
        else "reject",
        "model_configs": {t: m.config for t, m in models.items()},
        "limitations": [
            "Local CPU inference only. Two models resident; no per-model memory claim.",
            "Calibration reused only for diagnostic comparison; changed thread configuration is not production-qualified.",
            "New held-out calibration and live Gemini measurements required before promotion.",
        ],
    }
    dest = Path("benchmarks/results/campaign-04")
    (dest / "threads-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (dest / "threads-records.jsonl").write_text(
        "".join(json.dumps({"threads": t, **r}) + "\n" for t, rr in records.items() for r in rr)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
