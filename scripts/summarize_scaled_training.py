"""Record actual optimization diagnostics and safe per-query development outcomes."""

import json
import math
from pathlib import Path

from houndoracle.experiments.fit_calibration import predict
from houndoracle.experiments.head import infer_head


def main():
    public = Path("research/experiments/campaign-19")
    training = json.loads((public / "semantic-training.json").read_text())
    report = json.loads((public / "screen.json").read_text())
    head = json.loads((public / "candidate-head.json").read_text())
    artifact = json.loads((public / "candidate-provisional-calibration.json").read_text())
    rows = [
        json.loads(s)
        for s in Path("/workspace/houndoracle-data/campaign-19/scores/candidate/validation.jsonl")
        .read_text()
        .splitlines()
    ]
    encoder_ce = []
    head_ce = []
    records = []
    for row in rows:
        scored = infer_head(row, head, evidence_group_count=row["evidence_group_count"])
        index = {"NO": 0, "YES": 1, "MAYBE": 2}[row["expected"]]
        if row.get("probabilities") is not None:
            encoder_ce.append(-math.log(max(1e-12, row["probabilities"][index])))
        if scored.get("probabilities") is not None:
            head_ce.append(-math.log(max(1e-12, scored["probabilities"][index])))
        probability = (
            predict(artifact["mapping"], scored["raw_score"])
            if scored["candidate"] in ["YES", "NO"] and scored.get("raw_score") is not None
            else None
        )
        decision = (
            scored["candidate"]
            if probability is not None and probability >= 0.8 and not scored.get("truncated")
            else "MAYBE"
        )
        records.append(
            {
                k: scored.get(k)
                for k in ["id", "group_id", "expected", "candidate", "raw_score", "truncated"]
            }
            | {
                "provisional_decision": decision,
                "mapped_correctness": probability,
                "risk_qualified": False,
                "latency_seconds": None,
                "all_in_cost_usd": None,
            }
        )
    diagnostic = {
        "training_rows": training["rows"],
        "steps": training["steps"],
        "training_seconds": training["training_seconds"],
        "mean_training_batch_loss": sum(training["losses"]) / len(training["losses"]),
        "mean_first25_training_loss": sum(training["losses"][:25]) / 25,
        "mean_last25_training_loss": sum(training["losses"][-25:]) / 25,
        "encoder_validation_cross_entropy": sum(encoder_ce) / len(encoder_ce),
        "head_validation_cross_entropy": sum(head_ce) / len(head_ce),
        "cross_entropy_eligible_queries": len(encoder_ce),
        "validation_queries": len(rows),
        "screen_passed": report["screen_passed"],
        "raw_accuracy_delta": report["raw_accuracy_delta"],
        "correct_definitive_delta": report["correct_definitive_delta"],
        "decision": "REJECTED_NO_STABLE_REPLACEMENT",
        "locked_scored": False,
        "risk_scored": False,
        "inference_cost_comparison": None,
        "note": "End-of-run reused development diagnostics, not checkpoint selection or independent improvement. Training batch mean and query validation CE have different weighting/populations. CPU8 training timing does not measure query serving latency.",
    }
    (public / "training-diagnostics.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    Path("benchmarks/results/campaign-19/development-records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
