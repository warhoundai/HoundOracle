"""Fixed nonlinear aggregation challengers with equally supervised logistic reference."""

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from houndoracle.experiments.head import infer_head
from houndoracle.experiments.metrics import summary


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def score(rows, head):
    return [infer_head(r, head, evidence_group_count=r["evidence_group_count"]) for r in rows]


def quality(rows):
    result = summary([r | {"decision": r["candidate"]} for r in rows])
    result["latency_seconds"] = None
    return result


def main():
    public = Path("research/experiments/campaign-15")
    target = public / "screen.json"
    if target.exists():
        raise FileExistsError("Do not retune the completed head screen")
    training_path = Path(
        "/workspace/houndoracle-data/campaign-08-round2/scores/continued/development.jsonl"
    )
    training = load(training_path)
    validation = load("/workspace/houndoracle-data/campaign-09/scores/continued/validation.jsonl")
    if {r["group_id"] for r in training} & {r["group_id"] for r in validation}:
        raise ValueError("Training/validation leakage")

    def vector(row):
        return (row["logits"] or [0, 0, 1]) + [
            np.log1p(row.get("model_input_tokens") or row.get("token_count") or 0),
            np.log1p(row["evidence_group_count"]),
        ]

    x = np.array([vector(r) for r in training])
    y = [{"NO": 0, "YES": 1, "MAYBE": 2}[r["expected"]] for r in training]
    scaler = StandardScaler().fit(x)
    base = json.loads(
        Path("research/experiments/campaign-09/continued-mixture-head.json").read_text()
    )
    baseline = quality(score(validation, base))
    outcomes = {}
    for hidden in (16, 32):
        name = f"mlp{hidden}"
        start = time.perf_counter()
        model = MLPClassifier(
            hidden_layer_sizes=(hidden,),
            activation="relu",
            solver="lbfgs",
            alpha=0.01,
            max_iter=500,
            random_state=1931,
        )
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(scaler.transform(x), y)
        head = {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "weights": [w.tolist() for w in model.coefs_],
            "biases": [b.tolist() for b in model.intercepts_],
            "classes": model.classes_.tolist(),
        }
        # Verify the safe JSON inference implementation against the fitted reference.
        scored = score(validation, head)
        reference = model.predict_proba(scaler.transform(np.array([vector(r) for r in validation])))
        usable = [i for i, r in enumerate(validation) if r.get("logits") is not None]
        error = float(
            np.max(
                np.abs(reference[usable] - np.array([scored[i]["probabilities"] for i in usable]))
            )
        )
        if error > 1e-10:
            raise ValueError("Serialized nonlinear head does not reproduce sklearn")
        metrics = quality(scored)
        delta = metrics["accuracy_all_decisions"] - baseline["accuracy_all_decisions"]
        passed = (
            delta >= 0.02
            and metrics["definitive_coverage"] >= baseline["definitive_coverage"] - 0.02
        )
        (public / (name + "-head.json")).write_text(json.dumps(head, indent=2) + "\n")
        outcomes[name] = {
            "raw_validation": metrics,
            "raw_accuracy_delta": delta,
            "screen_passed": passed,
            "training_seconds": time.perf_counter() - start,
            "iterations": model.n_iter_,
            "training_loss": float(model.loss_),
            "warnings": [str(w.message) for w in caught],
            "numeric_checkpoint_probability_max_error": error,
        }
    result = {
        "baseline": baseline,
        "variants": outcomes,
        "training_rows": len(training),
        "training_source_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "locked_scored": False,
        "new_cloud_cost_usd": 0,
        "decision": "qualify-fresh-risk"
        if any(r["screen_passed"] for r in outcomes.values())
        else "reject-all-before-fresh-risk",
        "scope": "Development screen only; no inference timing, independent quality or Gemini claim.",
    }
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
